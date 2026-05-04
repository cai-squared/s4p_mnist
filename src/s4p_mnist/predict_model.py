from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from s4p_mnist.config import MODELS_DIR, PROCESSED_DATA_DIR
from s4p_mnist.logging_config import get_logger, setup_logging
from s4p_mnist.models.model import Model

logger = get_logger(__name__)


def _build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    drop = {"label", "y", "target"}
    cols = [c for c in df.columns if str(c).strip().lower() not in drop]
    if not cols:
        raise ValueError("Input CSV has no feature columns after removing labels.")
    x = df[cols].to_numpy(dtype=np.float32)
    if x.shape[1] != 784:
        logger.warning(
            "Expected 784 pixel columns; got %d. Predictions may be wrong.",
            x.shape[1],
        )
    return cast(np.ndarray, x)


def predict(model_path: Path, input_path: Path, output_path: Path) -> None:
    logger.info("Loading model from %s", model_path)
    model = Model.load(model_path)

    logger.info("Scoring %s", input_path)
    df = pd.read_csv(input_path)
    x = _build_feature_matrix(df)
    preds = model.predict(x)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"prediction": preds})
    out.to_csv(output_path, index=False)
    logger.info("Wrote %d rows to %s", len(out), output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=MODELS_DIR / "model.joblib")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DATA_DIR / "input.csv",
    )
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()

    setup_logging()
    predict(args.model_path, args.input, args.output)
    logger.info("Prediction complete")


if __name__ == "__main__":
    main()
