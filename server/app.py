# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Smartpayenv Environment.

This module creates an HTTP server that exposes the SmartpayenvEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 7860

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    # Try package-style relative import
    from ..models import SmartpayenvAction, SmartpayenvObservation
    from .SmartPayEnv_environment import SmartpayenvEnvironment
except (ImportError, ValueError):
    # Fallback to local import (for uvicorn server.app:app)
    from models import SmartpayenvAction, SmartpayenvObservation
    from server.SmartPayEnv_environment import SmartpayenvEnvironment


# ── Singleton env so custom endpoints share state with openenv ─────────
# Different openenv versions store the env in different places
# (app.env, app.state.env, per-request factory, etc.). Rather than
# guessing, we use a singleton subclass: no matter how many times
# openenv instantiates the env class, it always gets the same object,
# and we can always reach it via _SHARED_ENV.
_SHARED_ENV: SmartpayenvEnvironment | None = None


class SharedSmartpayenvEnvironment(SmartpayenvEnvironment):
    """Singleton subclass — always returns the same env instance."""

    def __new__(cls, *args, **kwargs):
        global _SHARED_ENV
        if _SHARED_ENV is None:
            inst = super().__new__(cls)
            super(SharedSmartpayenvEnvironment, inst).__init__(*args, **kwargs)
            inst._singleton_initialized = True  # type: ignore[attr-defined]
            _SHARED_ENV = inst
        return _SHARED_ENV

    def __init__(self, *args, **kwargs):  # noqa: D401
        # Already initialised by __new__ on first construction; subsequent
        # constructions are no-ops so we don't reset the env.
        if getattr(self, "_singleton_initialized", False):
            return
        super().__init__(*args, **kwargs)
        self._singleton_initialized = True


def _get_env() -> SmartpayenvEnvironment:
    """Return the shared env, creating it if openenv hasn't yet."""
    global _SHARED_ENV
    if _SHARED_ENV is None:
        SharedSmartpayenvEnvironment()  # populates _SHARED_ENV
    assert _SHARED_ENV is not None
    return _SHARED_ENV


# Create the app with web interface and README integration
app = create_app(
    SharedSmartpayenvEnvironment,
    SmartpayenvAction,
    SmartpayenvObservation,
    env_name="SmartPayEnv",
    max_concurrent_envs=1,
)


@app.post("/simulate", response_model=SmartpayenvObservation)
async def simulate(action: SmartpayenvAction):
    """Simulates an action without advancing the true environment state."""
    return _get_env().simulate(action)


# ── Theme-4 co-evolution endpoints ────────────────────────────────────
from typing import Optional
from pydantic import BaseModel


class AdversaryConfig(BaseModel):
    """Parametric fraud-agent policy. Any field may be omitted."""
    intensity: Optional[float] = None
    noise_boost: Optional[float] = None
    pattern_rate: Optional[float] = None
    strategy: Optional[str] = None  # "mixed" | "fraud_surge" | "stealth_fraud" | "velocity_attack"


class SeededReset(BaseModel):
    difficulty: int = 0
    seed: Optional[int] = None


@app.post("/configure_adversary")
async def configure_adversary(cfg: AdversaryConfig):
    """Set the learnable fraud agent's behaviour. Returns the active config."""
    return _get_env().configure_adversary(
        intensity=cfg.intensity,
        noise_boost=cfg.noise_boost,
        pattern_rate=cfg.pattern_rate,
        strategy=cfg.strategy,
    )


@app.post("/reset_seeded", response_model=SmartpayenvObservation)
async def reset_seeded(req: SeededReset):
    """Deterministic reset: same `seed` => same starting trajectory.
    Useful for GRPO so all completions in a group share the same state."""
    return _get_env().reset(difficulty=int(req.difficulty), seed=req.seed)


def main():
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 7860
        python -m SmartPayEnv.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 7860)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn SmartPayEnv.server.app:app --workers 4
    """

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()