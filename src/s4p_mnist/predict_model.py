from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from s4p_mnist.config import MODELS_DIR, PROCESSED_DATA_DIR
from s4p_mnist.data.loaders import load_processed
from s4p_mnist.logging_config import get_logger, setup_logging
from s4p_mnist.models.model import Model

logger = get_logger(__name__)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=MODELS_DIR / "model.joblib")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DATA_DIR,
    )
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()

    setup_logging()
    predict(args.model_path, args.input, args.output)
    logger.info("Prediction complete")


if __name__ == "__main__":
    main()
