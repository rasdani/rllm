from __future__ import annotations

import uuid
import threading
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rllm.environments.swe.swe import SWEEnv
try:
    # Only needed if you want to construct Actions directly (function_name + parameters path)
    from r2egym.agenthub.action import Action
except ImportError:
    Action = None  # Will be validated at runtime if used

app = FastAPI(title="SWEEnv Service", version="0.1.0")

class CreateEnvRequest(BaseModel):
    # If provided, used directly. Otherwise the environment will load the default dataset and pick idx.
    entry: Optional[Dict[str, Any]] = None
    idx: Optional[int] = None

    # SWEEnv kwargs
    step_timeout: int = 90
    reward_timeout: int = 300
    backend: str = "docker"  # or "kubernetes"
    delete_image: bool = False
    verbose: bool = False
    scaffold: str = "r2egym"  # or "sweagent"

class CreateEnvResponse(BaseModel):
    env_id: str

class ResetResponse(BaseModel):
    instruction: str

class StepRequest(BaseModel):
    # Either send the XML-like action string the LLM produced:
    action_str: Optional[str] = Field(default=None, description="XML-like action string")
    # Or send structured function call that will be converted to an Action:
    function_name: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    timeout: Optional[int] = None  # Optional override for step timeout

class StepResponse(BaseModel):
    observation: str
    reward: float
    done: bool
    info: Dict[str, Any]

class RewardResponse(BaseModel):
    reward: float

class OkResponse(BaseModel):
    ok: bool = True

class CommandsResponse(BaseModel):
    commands: list[str] = Field(default_factory=list)

class EnvSession:
    def __init__(self, env: SWEEnv):
        self.env = env
        self.lock = threading.Lock()

_SESSIONS: Dict[str, EnvSession] = {}

def _get_session_or_404(env_id: str) -> EnvSession:
    session = _SESSIONS.get(env_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"env_id {env_id} not found")
    return session

@app.get("/health", response_model=OkResponse)
def health() -> OkResponse:
    return OkResponse()

@app.post("/envs", response_model=CreateEnvResponse)
def create_env(req: CreateEnvRequest) -> CreateEnvResponse:
    env = SWEEnv(
        entry=req.entry,
        dataset=None,  # SWEEnv loads default dataset if entry=None
        idx=req.idx,
        step_timeout=req.step_timeout,
        reward_timeout=req.reward_timeout,
        backend=req.backend,
        delete_image=req.delete_image,
        verbose=req.verbose,
        scaffold=req.scaffold,
    )
    env_id = str(uuid.uuid4())
    _SESSIONS[env_id] = EnvSession(env)
    return CreateEnvResponse(env_id=env_id)

@app.post("/envs/{env_id}/reset", response_model=ResetResponse)
def reset_env(env_id: str) -> ResetResponse:
    session = _get_session_or_404(env_id)
    with session.lock:
        instruction, _ = session.env.reset()
    return ResetResponse(instruction=instruction)

@app.get("/envs/{env_id}/commands", response_model=CommandsResponse)
def list_commands(env_id: str) -> CommandsResponse:
    """
    Returns available command names after reset(). If you haven't called reset(),
    this may return an empty list.
    """
    session = _get_session_or_404(env_id)
    cmds = []
    # RepoEnv is created on first reset() in SWEEnv.reset(); if not created yet, cmds empty.
    if session.env and getattr(session.env, "env", None) and getattr(session.env.env, "commands", None):
        cmds = [c.name for c in session.env.env.commands]
    return CommandsResponse(commands=cmds)

@app.post("/envs/{env_id}/step", response_model=StepResponse)
def step_env(env_id: str, req: StepRequest) -> StepResponse:
    session = _get_session_or_404(env_id)

    # Build the action object or string
    action_to_send: Any
    if req.action_str:
        action_to_send = req.action_str
    else:
        if not req.function_name:
            raise HTTPException(
                status_code=400,
                detail="Provide either action_str or function_name (+ parameters).",
            )
        if Action is None:
            raise HTTPException(
                status_code=500,
                detail="r2egym Action class not available; please install R2E-Gym in this runtime.",
            )
        action_to_send = Action(req.function_name, req.parameters or {})

    with session.lock:
        # SWEEnv.step signature returns: (observation_str, reward, done, info)
        obs, reward, done, info = session.env.step(action_to_send)

    return StepResponse(
        observation=obs,
        reward=float(reward or 0.0),
        done=bool(done),
        info=info or {},
    )

@app.post("/envs/{env_id}/compute_reward", response_model=RewardResponse)
def compute_reward(env_id: str) -> RewardResponse:
    session = _get_session_or_404(env_id)
    with session.lock:
        reward = session.env.compute_final_reward()
    return RewardResponse(reward=float(reward or 0.0))

@app.delete("/envs/{env_id}", response_model=OkResponse)
def delete_env(env_id: str) -> OkResponse:
    session = _get_session_or_404(env_id)
    with session.lock:
        try:
            session.env.close()
        finally:
            _SESSIONS.pop(env_id, None)
    return OkResponse()

# Optional: run with `python -m rllm.environments.swe.service`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rllm.environments.swe.service:app", host="0.0.0.0", port=8000, reload=False)
