"""Entry point: ``python -m pandaro.main`` runs the API server on port 9090."""

from __future__ import annotations

import uvicorn

from .config import get_settings


def main() -> None:
    s = get_settings()
    uvicorn.run(
        "pandaro.api.app:app",
        host=s.host,
        port=s.port,
        log_level="info",
        ws="websockets",
    )


if __name__ == "__main__":
    main()
