"""Tests for the MNIST CNN ``Model``."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest
from omegaconf import OmegaConf

from s4p_mnist import predict_model as predict_mod
from s4p_mnist import train_model as train_mod
from s4p_mnist.models.base import BaseModel
from s4p_mnist.models.model import Model


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


def _train_hydra_cfg() -> object:
    return OmegaConf.create(
        {
            "paths": {"data_processed": "data/processed", "models_dir": "models"},
            "data": {"val_fraction": 0.1},
            "training": {
                "epochs": 18,
                "batch_size": 128,
                "learning_rate": 0.001,
                "weight_decay": 1e-4,
                "dropout": 0.3,
                "seed": 42,
            },
            "predict": {
                "model_file": "models/model.joblib",
                "input_dir": "data/processed",
                "output_file": "predictions.csv",
            },
        }
    )


class TestHydraConfigChecks:
    def test_train_cfg_ok(self) -> None:
        train_mod._check_train_cfg(_train_hydra_cfg())

    @pytest.mark.parametrize(
        "path,val",
        [
            ("training.epochs", 0),
            ("training.batch_size", 0),
            ("training.learning_rate", 0.0),
            ("training.weight_decay", -1.0),
            ("training.dropout", 1.5),
            ("data.val_fraction", 0.0),
            ("data.val_fraction", 1.0),
        ],
    )
    def test_train_cfg_rejects_bad_numbers(self, path: str, val: object) -> None:
        cfg = _train_hydra_cfg()
        OmegaConf.update(cfg, path, val, merge=True)
        with pytest.raises(ValueError):
            train_mod._check_train_cfg(cfg)

    def test_predict_cfg_ok(self) -> None:
        predict_mod._check_predict_cfg(_train_hydra_cfg())

    def test_predict_cfg_requires_section(self) -> None:
        cfg = OmegaConf.create({"paths": {}, "data": {}, "training": {}})
        with pytest.raises(ValueError):
            predict_mod._check_predict_cfg(cfg)


class TestApi:
    def test_health_without_weights(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import api.main as api_main

        monkeypatch.setenv("S4P_SKIP_MODEL_LOAD", "1")
        monkeypatch.setenv("S4P_MODEL_PATH", str(tmp_path / "missing.joblib"))
        api_main._model = None
        from fastapi.testclient import TestClient

        client = TestClient(api_main.app)
        out = client.get("/health")
        assert out.status_code == 200
        body = out.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is False
