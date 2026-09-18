"""
Local dev entrypoint for the FastAPI backend.

The README's documented `uvicorn src.backend.main:app --reload` command
never sets LOCAL_MODE, so it silently falls through to real-AWS mode and
spams "Unable to locate credentials" for Timestream on every telemetry
tick when no AWS account is configured. This wrapper sets the same
LOCAL_MODE=true default docker-compose.yml already uses, so a plain
local run behaves the way the README implies it should.
"""

import os
import sys

os.environ.setdefault("LOCAL_MODE", "true")
os.environ.setdefault("IOT_PUBLISH_INTERVAL_S", "1.0")
os.environ.setdefault("AWS_REGION", "us-east-1")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.backend.main:app", host="0.0.0.0", port=8000, reload=True, app_dir=PROJECT_ROOT)
