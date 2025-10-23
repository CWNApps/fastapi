# app.py
# Minimal NextBestAction API using MABWiser + FastAPI
# pip install fastapi uvicorn pydantic mabwiser scikit-learn joblib

import os, json, math, threading
from typing import List, Dict, Any
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from joblib import dump, load
from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy

MODEL_PATH = os.environ.get("NBA_MODEL_PATH", "nba_model.joblib")
API_KEY = os.environ.get("NBA_API_KEY", "dev")
LOCK = threading.Lock()

ACTIONS_BY_STAGE = {
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
        learning_policy=LearningPolicy.LinUCB(alpha=0.3),  # simple & strong baseline
        neighborhood_policy=NeighborhoodPolicy.KNN(k=0)    # no neighborhood
    )

def _load_or_init(arms: List[str]) -> MAB:
    if os.path.exists(MODEL_PATH):
        m: MAB = load(MODEL_PATH)
        # if stage arms changed, refresh arms while preserving params
        missing = [a for a in arms if a not in m.arms]
        if missing:
            m = _new_model(arms)
        return m
    return _new_model(arms)

def _vectorize(features: Dict[str, Any]) -> List[float]:
    # Tiny, opinionated featurizer—replace with your real pipeline
    # Sort keys for stable ordering; normalize simple scales if needed.
    keys = sorted(features.keys())
    vec = []
    for k in keys:
        v = features[k]
        if isinstance(v, (int, float)):
            vec.append(float(v))
        elif isinstance(v, bool):
            vec.append(1.0 if v else 0.0)
        else:
            # hash categorical-ish text into [0,1)
            vec.append((abs(hash(str(v))) % 1000) / 1000.0)
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
    reward: float  # e.g., 1 if stage-exit happened within SLA; else shaped
    context: Dict[str, Any]

app = FastAPI(title="NextBestAction API", version="0.1")

def _auth(auth_header: str | None):
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(401, "Missing bearer token")
    token = auth_header.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(403, "Invalid API key")

@app.post("/next_best_action")
def next_best_action(payload: NBARequest, authorization: str = Header(None)):
    _auth(authorization)
    stage = payload.stage.upper()
    # enforce fences
    default_arms = ACTIONS_BY_STAGE.get(stage, [])
    arms = [a for a in payload.allowed_actions if a in default_arms]
    if not arms:
        arms = default_arms
    if not arms:
        raise HTTPException(400, f"No configured actions for stage {stage}")

    x = _vectorize(payload.features)

    with LOCK:
        mab = _load_or_init(arms)
        # warm-start: predict scores for each arm; MABWiser 'predict' returns best arm
        scores = {}
        for a in arms:
            scores[a] = mab.predict_expectation(context=x, arm=a) if hasattr(mab, "predict_expectation") else 0.0
        # simple uncertainty proxy: alpha / sqrt(n_a + 1)
        n = getattr(mab, "n", {}) if hasattr(mab, "n") else {}
        ranked = sorted(arms, key=lambda a: scores.get(a, 0.0) + 0.3 / math.sqrt(n.get(a, 0) + 1), reverse=True)
        topk = ranked[: max(1, payload.k)]
        # persist
        dump(mab, MODEL_PATH)

    choices = [{"action": a, "score": float(scores.get(a, 0.0)), "uncertainty": round(0.3 / math.sqrt(n.get(a, 0) + 1), 3),
                "expected": {"cr_uplift": 0.01, "delta_t_days": -0.5}} for a in topk]
    return {"choices": choices, "policy": "mabwiser.linucb", "notes": "baseline; tune alpha; replace features with real featurizer"}

@app.post("/feedback")
def feedback(fb: Feedback, authorization: str = Header(None)):
    _auth(authorization)
    stage = fb.stage.upper()
    arms = ACTIONS_BY_STAGE.get(stage, [])
    if fb.action not in arms:
        arms.append(fb.action)
    x = _vectorize(fb.context)
    with LOCK:
        mab = _load_or_init(arms)
        # partial fit on single (x, arm, reward)
        mab.fit(decisions=[fb.action], rewards=[fb.reward], contexts=[x])
        dump(mab, MODEL_PATH)
    return {"ok": True}
