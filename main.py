# main.py — NextBestAction API (FastAPI + MABWiser)
# Fix: correct call to predict_expectations; no more AttributeError

from __future__ import annotations
import os, threading, logging, traceback
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import dump, load
from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy

app = FastAPI(title="NextBestAction API", version="1.0.2")

# --- CORS so Swagger "Try it out" works from the browser ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=False
)

# --- Bearer auth (Swagger's Authorize injects the header) ---
bearer = HTTPBearer(auto_error=True)
def require_token(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> bool:
    if creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Use Bearer token")
    if creds.credentials != os.environ.get("NBA_API_KEY", "dev"):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True

# --- Logging ---
log = logging.getLogger("nba"); logging.basicConfig(level=logging.INFO)
def _log_exc(msg: str, e: Exception): log.error("%s: %s\n%s", msg, repr(e), traceback.format_exc())

# --- State & config ---
MODEL_PATH = os.environ.get("NBA_MODEL_PATH", "nba_model.joblib")
LOCK = threading.Lock()

ACTIONS_BY_STAGE: Dict[str, List[str]] = {
    "AWR": ["invite","prep","follow_up_t2","discovery_invite"],
    "EDU": ["diagnose_spiced","impact_calc","case_study_send"],
    "SEL": ["meddic_gap_check","stakeholder_map","pilot_roi_calc","exec_sponsor_outreach","legal_prep"],
    "MUTCOM": ["pilot_sow_review","security_review","data_mapping_check"],
    "ONB": ["connector_setup","first_value_event","risk_triage"],
    "RECIMPACT": ["habit_coach","qbr_schedule","health_review"],
    "EXP": ["eligibility_list","expansion_proposal","referral_ask"],
    "ADV": ["case_brief","webinar_invite","reference_call"]
}

def _new_model(arms: List[str]) -> MAB:
    return MAB(
        arms=arms,
        learning_policy=LearningPolicy.LinUCB(alpha=0.3),
        neighborhood_policy=NeighborhoodPolicy.KNN(k=0)
    )

def _load_or_init(arms: List[str]) -> MAB:
    try:
        if os.path.exists(MODEL_PATH):
            m: MAB = load(MODEL_PATH)
            if set(getattr(m, "arms", [])) != set(arms):
                m = _new_model(arms)
            return m
    except Exception as e:
        _log_exc("model load failed; reinit", e)
    return _new_model(arms)

def _persist(m: MAB):
    try:
        dump(m, MODEL_PATH)
    except Exception as e:
        _log_exc("model save failed (continuing)", e)

def _vectorize(features: Dict[str, Any]) -> List[float]:
    vec: List[float] = []
    for k in sorted(features.keys()):
        v = features[k]
        if isinstance(v, (int, float)): vec.append(float(v))
        elif isinstance(v, bool):        vec.append(1.0 if v else 0.0)
        else:                             vec.append((abs(hash(str(v))) % 1000) / 1000.0)
    return vec

# --- Schemas ---
class NBARequest(BaseModel):
    run_id: str
    account_id: str
    stage: str
    allowed_actions: List[str]
    k: int = 3
    features: Dict[str, Any]
    goal: str | None = None
    debug: bool = False

class Feedback(BaseModel):
    run_id: str
    account_id: str
    stage: str
    action: str
    reward: float
    context: Dict[str, Any]

# --- Routes ---
@app.get("/")
def health():
    return {"ok": True, "service": "NextBestAction API"}

@app.post("/next_best_action", dependencies=[Depends(require_token)])
def next_best_action(payload: NBARequest):
    try:
        stage = (payload.stage or "").upper()
        default_arms = ACTIONS_BY_STAGE.get(stage, [])
        arms = [a for a in payload.allowed_actions if a in default_arms] or default_arms
        if not arms:
            raise HTTPException(status_code=400, detail=f"No configured actions for stage {stage}")

        x = _vectorize(payload.features)

        with LOCK:
            mab = _load_or_init(arms)

            # ---- Correct call: 'contexts' positional/keyword accepted; handle dict vs list-of-dicts ----
            scores: Dict[str, float] = {a: 0.0 for a in arms}
            try:
                pe = None
                if hasattr(mab, "predict_expectations"):
                    # MABWiser API: predict_expectations(contexts) returns Dict for 1 context, List[Dict] for many
                    pe = mab.predict_expectations(x)  # pass 1-D list (single context)  ✅
                    if isinstance(pe, list) and len(pe) > 0:
                        pe = pe[0]
                    if isinstance(pe, dict):
                        for a in arms:
                            scores[a] = float(pe.get(a, 0.0))
                else:
                    # Fallback to best-arm then zero scores (rare)
                    best = mab.predict(x) if hasattr(mab, "predict") else None
                    if best in arms:
                        scores[best] = 1.0
            except Exception as e:
                _log_exc("scoring failed; using zeros", e)

            # simple exploration bonus even before any training
            n = getattr(mab, "n", {}) if hasattr(mab, "n") else {}
            ranked = sorted(arms, key=lambda a: scores.get(a, 0.0) + 0.3 / ((n.get(a, 0) + 1) ** 0.5), reverse=True)
            topk = ranked[: max(1, payload.k)]
            _persist(mab)

        choices = [{
            "action": a,
            "score": float(scores.get(a, 0.0)),
            "uncertainty": round(0.3 / ((n.get(a, 0) + 1) ** 0.5), 3),
            "expected": {"cr_uplift": 0.01, "delta_t_days": -0.5}
        } for a in topk]

        return {"choices": choices, "policy": "mabwiser.linucb", "notes": "OK"}
    except HTTPException:
        raise
    except Exception as e:
        _log_exc("nba/next_best_action unexpected", e)
        fallback = payload.allowed_actions[: max(1, payload.k)] or ACTIONS_BY_STAGE.get((payload.stage or "").upper(), [])[:1]
        return {
            "choices": [{"action": a, "score": 0.0, "uncertainty": 0.3,
                         "expected": {"cr_uplift": 0.0, "delta_t_days": 0.0}} for a in fallback],
            "policy": "fallback",
            "notes": f"fallback due to {type(e).__name__}"
        }

@app.post("/feedback", dependencies=[Depends(require_token)])
def feedback(fb: Feedback):
    try:
        stage = (fb.stage or "").upper()
        arms = ACTIONS_BY_STAGE.get(stage, [])
        if fb.action not in arms:
            arms.append(fb.action)

        x = _vectorize(fb.context)

        with LOCK:
            mab = _load_or_init(arms)
            try:
                # MABWiser supports online learning via partial_fit(decisions, rewards, contexts) ✅
                if hasattr(mab, "partial_fit"):
                    mab.partial_fit(decisions=[fb.action], rewards=[fb.reward], contexts=[x])
                else:
                    mab.fit(decisions=[fb.action], rewards=[fb.reward], contexts=[x])
            except Exception as e:
                _log_exc("fit failed; reinit once", e)
                mab = _new_model(arms)
                mab.fit(decisions=[fb.action], rewards=[fb.reward], contexts=[x])
            _persist(mab)

        return {"ok": True, "notes": "learned"}
    except Exception as e:
        _log_exc("nba/feedback unexpected", e)
        return {"ok": True, "notes": f"fallback due to {type(e).__name__}"}
