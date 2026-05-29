from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from s4p_mnist.config import PROJECT_ROOT
from s4p_mnist.data.loaders import load_processed
from s4p_mnist.logging_config import get_logger, setup_logging
from s4p_mnist.models.model import Model

logger = get_logger("s4p_mnist.predict_model")


def _check_predict_cfg(cfg: DictConfig) -> None:
    if not OmegaConf.is_config(cfg):
        raise TypeError("cfg must be an OmegaConf config")
    if "predict" not in cfg:
        raise ValueError("predict section is required in config")
    p = cfg.predict
    for key in ("model_file", "input_dir", "output_file"):
        if key not in p or not str(p[key]).strip():
            raise ValueError(f"predict.{key} must be a non-empty string")
    out = Path(str(p.output_file))
    if out.exists() and out.is_dir():
        raise ValueError("predict.output_file must be a file path, not a directory")


def predict(model_path: Path, input_path: Path, output_path: Path) -> None:
    logger.info("Loading model from %s", model_path)
    model = Model.load(model_path)

    logger.info("Scoring %s", input_path)
    _, _, X_test_img, y_test_u8 = load_processed(input_path)
    preds = model.predict(X_test_img)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"prediction": preds})
    out.to_csv(output_path, index=False)
    logger.info("Wrote %d rows to %s", len(out), output_path)


def _resolve_under_root(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    _check_predict_cfg(cfg)
    setup_logging()

    p = cfg.predict
    model_path = _resolve_under_root(str(p.model_file))
    input_path = _resolve_under_root(str(p.input_dir))
    output_path = _resolve_under_root(str(p.output_file))

    predict(model_path, input_path, output_path)
    logger.info("Prediction complete")


if __name__ == "__main__":
    main()
