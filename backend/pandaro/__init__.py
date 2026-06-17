"""Pandaro backend package.

Pandaro is a single-user, Polish-language tool for analysing recordings of phone
calls and meetings. The backend is a *stateless* GPU compute engine: it loads
models on demand, processes audio, streams results, and persists nothing between
runs. The browser SPA holds the canonical, ephemeral session state.
"""

__version__ = "0.1.0"
