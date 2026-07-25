"""Pytest configuration for consistent local test artifacts."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_OUTPUT_DIR = PROJECT_ROOT / ".test_outputs"


def _resolve_test_output_dir(*parts: str) -> Path:
    base = Path(os.environ.get("CORTADO_TEST_OUTPUT_DIR", DEFAULT_TEST_OUTPUT_DIR))
    output_dir = base.joinpath(*parts) if parts else base
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def pytest_configure() -> None:
    os.environ.setdefault("CORTADO_TEST_OUTPUT_DIR", str(DEFAULT_TEST_OUTPUT_DIR))


def pytest_sessionstart(session) -> None:  # noqa: ARG001
    _resolve_test_output_dir("pytest")
