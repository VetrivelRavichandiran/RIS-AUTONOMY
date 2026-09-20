"""Structured logging setup.

``setup_logging`` configures a consistent format and an optional file handler
(idempotently, so it can be called from both the CLI and the dashboard).
"""
from __future__ import annotations

import logging
import os

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_configured = False


def setup_logging(level: str = "INFO", file: str | None = None) -> None:
    """Configure root logging once.

    Parameters
    ----------
    level:
        Log level name (e.g. ``"INFO"``).
    file:
        Optional log file path (appended).
    """
    global _configured
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if _configured:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(handler)
    if file:
        os.makedirs(os.path.dirname(file) or ".", exist_ok=True)
        fh = logging.FileHandler(file)
        fh.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(fh)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger."""
    return logging.getLogger(name)