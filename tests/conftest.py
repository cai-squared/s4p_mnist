"""
Pytest configuration and shared fixtures.
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture
def sample_config() -> dict:
    """Provide a sample configuration for tests."""
    return {
        "model_type": "test",
        "batch_size": 32,
        "learning_rate": 0.001,
    }
