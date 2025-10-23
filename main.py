# main.py — NextBestAction API (rock-solid)
from __future__ import annotations
import os, math, threading, traceback
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import dump, load
from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy

# ---------- app ----------
app = FastAPI(title="NextBestAction API", version="1.1.0")

# CORS so Swagger never shows "Failed to fetch"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

# ---------- auth (Swagger Authorize button injects the header) ----------
bearer = HTTPBearer(auto_error=True)
def require_token(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> bool:
    if creds.scheme.lower() != "bearer":
        raise HTTPException(401, "Use Bearer token")
    if creds.credentials != os.environ.get("NBA_API_KEY", "dev"):
        raise HTTPException(403, "Invalid API key")
    return True

# ---------- config ----------
NBA_MODEL_PATH = os.environ.get("NBA_MODEL_PATH", "nba_model.joblib")
LOCK = threading.Lock()
DEBUG = os.environ.get("DEBUG", "0") == "1"

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

# ---------- helpers ----------
def _new_model(arms: List[str]) -> MAB:
    return MAB(
        arms=arms,
        learning_policy=LearningPolicy.LinUCB(alpha=0.3),
        neighborhood_policy=NeighborhoodPolicy.KNN(k=0)
    )

def _load_or_init(arms: List[str]) -> MAB:
    if os.path.exists(NBA_MODEL_PATH):
        try:
            m: MAB = load(NBA_MODEL_PATH)
            if set(getattr(m, "arms", [])) != set(arms):
                m = _new_model(arms)
            return m
        except Exception:
            return _new_model(arms)
    return _new_model(arms)

def _vectorize(features: Dict[str, Any]) -> List[float]:
    vec: List[float] = []
    for k in sorted(features.keys()):
        v = features[k]
        if isinstance(v, (int, float)): vec.append(float(v))
        elif isinstance(v, bool): vec.append(1.0 if v else 0.0)
        else: vec.append((abs(hash(str(v))) % 1000) / 1000.0)
    return vec

def _err_note(e: Exception) -> str:
    msg = f"{type(e).__name__}: {e}"
    if DEBUG:
        msg += " | " + traceback.format_exc(limit=1).strip().replace("\n", " / ")
    return msg

# ---------- schemas ----------
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

# ---------- routes ----------
@app.get("/")
def health():
    return {"ok": True, "service": "NextBestAction API", "version": "1.1.0"}

@app.post("/next_best_action", dependencies=[Depends(require_token)])
def next_best_action(payload: NBARequest):
    stage = (payload.stage or "").upper()
    default_arms = ACTIONS_BY_STAGE.get(stage, [])
    arms = [a for a in payload.allowed_actions if a in default_arms] or default_arms
    if not arms:
        raise HTTPException(400, f"No configured actions for stage {stage}")

    x = _vectorize(payload.features)
    notes = "ok"

    with LOCK:
        try:
            mab = _load_or_init(arms)
        except Exception as e:
            # absolute fallback — never 500
            choices = [{"action": a, "score": 0.0, "uncertainty": 0.3,
                        "expected": {"cr_uplift": 0.0, "delta_t_days": 0.0}} for a in arms[:max(1,payload.k)]]
            return {"choices": choices, "policy": "mabwiser.linucb", "notes": "model load fail: " + _err_note(e)}

        # try to score safely
        scores = {a: 0.0 for a in arms}
        try:
            if hasattr(mab, "predict_expectations"):
                pe = mab.predict_expectations(context=x)  # dict arm->float
                for a in arms:
                    scores[a] = float(pe.get(a, 0.0))
            elif hasattr(mab, "predict"):
                try:
                    best = mab.predict(context=x)  # may raise before any fit
                    for a in arms:
                        scores[a] = 1.0 if a == best else 0.0
                except Exception as e:
                    notes = "cold-start (predict) -> zeros | " + _err_note(e)
            else:
                notes = "no predict API; zeros"
        except Exception as e:
            notes = "score error -> zeros | " + _err_note(e)

        # exploration bonus (works even pre-training)
        try:
            n = getattr(mab, "n", {}) if hasattr(mab, "n") else {}
        except Exception:
            n = {}
        ranked = sorted(arms, key=lambda a: scores.get(a, 0.0) + 0.3 / ((n.get(a, 0) + 1) ** 0.5), reverse=True)
        topk = ranked[: max(1, payload.k)]

        # persist snapshot (don’t care if first-run)
        try:
            dump(mab, NBA_MODEL_PATH)
        except Exception as e:
            notes += " | persist warn: " + _err_note(e)

    choices = [{
        "action": a,
        "score": float(scores.get(a, 0.0)),
        "uncertainty": round(0.3 / ((n.get(a, 0) + 1) ** 0.5), 3) if isinstance(n, dict) else 0.3,
        "expected": {"cr_uplift": 0.01, "delta_t_days": -0.5}
    } for a in topk]

    return {"choices": choices, "policy": "mabwiser.linucb", "notes": notes}

@app.post("/feedback", dependencies=[Depends(require_token)])
def feedback(fb: Feedback):
    stage = (fb.stage or "").upper()
    arms = ACTIONS_BY_STAGE.get(stage, [])
    if fb.action not in arms:
        arms.append(fb.action)

    x = _vectorize(fb.context)
    notes = "ok"

    with LOCK:
        try:
            mab = _load_or_init(arms)
            # train on single sample — guard all paths
            if hasattr(mab, "partial_fit"):
                try:
                    mab.partial_fit(decisions=[fb.action], rewards=[fb.reward], contexts=[x])
                except Exception as e:
                    notes = "partial_fit failed; trying fit | " + _err_note(e)
                    mab.fit(decisions=[fb.action], rewards=[fb.reward], contexts=[x])
            else:
                mab.fit(decisions=[fb.action], rewards=[fb.reward], contexts=[x])
            try:
                dump(mab, NBA_MODEL_PATH)
            except Exception as e:
                notes += " | persist warn: " + _err_note(e)
        except Exception as e:
            # never 500; return ok:false + reason
            return {"ok": False, "notes": "feedback error: " + _err_note(e)}

    return {"ok": True, "notes": notes}
