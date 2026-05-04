"""Tests for the MNIST CNN ``Model``."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

from S4P_MNIST.models.base import BaseModel
from S4P_MNIST.models.model import Model


class TestModel:
    def test_is_base_model(self) -> None:
        assert issubclass(Model, BaseModel)

    def test_default_config_is_empty_dict(self) -> None:
        assert Model().config == {}

    def test_custom_config_is_stored(self) -> None:
        cfg = {"learning_rate": 0.01, "epochs": 5}
        assert Model(cfg).config == cfg

    def test_predict_before_fit_raises(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.random((4, 784)).astype(np.float32)
        with pytest.raises(RuntimeError):
            Model().predict(x)

    def test_fit_predict_small_synthetic(self) -> None:
        rng = np.random.default_rng(1)
        n = 400
        x = rng.random((n, 784)).astype(np.float32)
        y = rng.integers(0, 10, size=(n,), dtype=np.int64)
        m = Model(
            {
                "epochs": 3,
                "batch_size": 64,
                "learning_rate": 1e-2,
                "val_fraction": 0.15,
                "seed": 0,
            }
        )
        m.fit(x, y)
        p = m.predict(x[:8])
        assert p.shape == (8,)
        assert p.dtype == np.int64

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(2)
        n = 200
        x = rng.random((n, 784)).astype(np.float32)
        y = rng.integers(0, 10, size=(n,), dtype=np.int64)
        path = tmp_path / "model.joblib"
        original = Model({"epochs": 2, "batch_size": 32, "seed": 1})
        original.fit(x, y)
        original.save(path)

        loaded = Model.load(path)
        assert isinstance(loaded, Model)
        assert np.array_equal(loaded.predict(x[:5]), original.predict(x[:5]))

    def test_load_rejects_wrong_bundle(self, tmp_path: Path) -> None:
        path = tmp_path / "not_a_model.joblib"
        joblib.dump({"kind": "other", "state_dict": {}}, path)

        with pytest.raises(TypeError):
            Model.load(path)
