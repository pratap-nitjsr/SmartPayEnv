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

from fastapi.responses import RedirectResponse

try:
    # Try package-style relative import
    from ..models import SmartpayenvAction, SmartpayenvObservation
    from .SmartPayEnv_environment import SmartpayenvEnvironment
except (ImportError, ValueError):
    # Fallback to local import (for uvicorn server.app:app)
    from models import SmartpayenvAction, SmartpayenvObservation
    from server.SmartPayEnv_environment import SmartpayenvEnvironment


# Create the app with web interface and README integration
app = create_app(
    SmartpayenvEnvironment,
    SmartpayenvAction,
    SmartpayenvObservation,
    env_name="SmartPayEnv",
    max_concurrent_envs=1,
    # enable_web=True,
)


@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")


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