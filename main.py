# main.py  — NextBestAction API (Swagger Bearer auth + cold-start safe)
from __future__ import annotations
import os, math, threading
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from joblib import dump, load
from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy

app = FastAPI(title="NextBestAction API", version="1.0.0")

# --- Bearer auth (makes Swagger's Authorize button inject the header) ---
bearer = HTTPBearer(auto_error=True)
def require_token(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> bool:
    if creds.scheme.lower() != "bearer":
        raise HTTPException(401, "Use Bearer token")
    if creds.credentials != os.environ.get("NBA_API_KEY", "dev"):
        raise HTTPException(403, "Invalid API key")
    return True

NBA_MODEL_PATH = os.environ.get("NBA_MODEL_PATH", "nba_model.joblib")
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
    return MAB(arms=arms,
               learning_policy=LearningPolicy.LinUCB(alpha=0.3),
               neighborhood_policy=NeighborhoodPolicy.KNN(k=0))

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

@app.get("/")
def health(): return {"ok": True, "service": "NextBestAction API"}

@app.post("/next_best_action", dependencies=[Depends(require_token)])
def next_best_action(payload: NBARequest):
    stage = (payload.stage or "").upper()
    default_arms = ACTIONS_BY_STAGE.get(stage, [])
    arms = [a for a in payload.allowed_actions if a in default_arms] or default_arms
    if not arms: raise HTTPException(400, f"No configured actions for stage {stage}")
    x = _vectorize(payload.features)

    with LOCK:
        mab = _load_or_init(arms)
        scores = {a: 0.0 for a in arms}
        try:
            if hasattr(mab, "predict_expectations"):
                pe = mab.predict_expectations(context=x)
                for a in arms: scores[a] = float(pe.get(a, 0.0))
            else:
                for a in arms:
                    try: scores[a] = float(mab.predict_expectation(context=x, arm=a))
                    except Exception: scores[a] = 0.0
        except Exception:
            scores = {a: 0.0 for a in arms}

        n = getattr(mab, "n", {}) if hasattr(mab, "n") else {}
        ranked = sorted(arms, key=lambda a: scores.get(a,0.0)+0.3/((n.get(a,0)+1)**0.5), reverse=True)
        topk = ranked[: max(1, payload.k)]
        dump(mab, NBA_MODEL_PATH)

    choices = [{"action": a,
                "score": float(scores.get(a,0.0)),
                "uncertainty": round(0.3/((n.get(a,0)+1)**0.5),3),
                "expected": {"cr_uplift": 0.01, "delta_t_days": -0.5}} for a in topk]
    return {"choices": choices, "policy": "mabwiser.linucb", "notes": "cold-start safe; send /feedback to learn"}

@app.post("/feedback", dependencies=[Depends(require_token)])
def feedback(fb: Feedback):
    stage = (fb.stage or "").upper()
    arms = ACTIONS_BY_STAGE.get(stage, [])
    if fb.action not in arms: arms.append(fb.action)
    x = _vectorize(fb.context)

    with LOCK:
        mab = _load_or_init(arms)
        try:
            if hasattr(mab, "partial_fit"):
                mab.partial_fit(decisions=[fb.action], rewards=[fb.reward], contexts=[x])
            else:
                mab.fit(decisions=[fb.action], rewards=[fb.reward], contexts=[x])
        except Exception:
            mab = _new_model(arms)
            mab.fit(decisions=[fb.action], rewards=[fb.reward], contexts=[x])
        dump(mab, NBA_MODEL_PATH)

    return {"ok": True}
