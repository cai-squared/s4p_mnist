# Configs

This folder is what Hydra reads. There is only one yaml we use day to day, `config.yaml`, and it covers both training the CNN and running batch prediction.

`training`, `data`, and `paths` are for the train script. `predict` tells the predict script where the joblib file is, where the processed tensors live, and what csv to write. Paths are from the repo root unless you paste in an absolute path.

```bash
python -m s4p_mnist.train_model
python -m s4p_mnist.train_model training.epochs=4 training.batch_size=256
python -m s4p_mnist.predict_model predict.output_file=tmp/preds.csv
```

Hydra merges yaml plus anything you pass on the command line, then the scripts validate before doing work.
