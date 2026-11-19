# Crop Disease Detection

An image classification system that identifies plant diseases from leaf photographs. It includes a modular PyTorch training pipeline and a FastAPI service that accepts an image upload and returns the predicted disease class with confidence scores.

## Features

- Transfer learning with a pretrained torchvision backbone (default: ResNet-50 style CNN, configurable)
- Official train / test splits preserved on disk; the test split is strictly held out (see below)
- Stratified validation split carved from the official train split, with a fixed seed
- Metrics: accuracy, per-class precision, recall, F1, macro and weighted averages, and confusion matrix
- YAML-driven configuration for data, model, and training hyperparameters
- FastAPI inference endpoint with image upload, input validation, and top-k predictions
- Dataset download with SHA-256 verification from a pinned, immutable dataset revision
- Unit tests for the data pipeline, split isolation, metrics, and API

## Architecture

```
            +-------------+       +-----------------+       +------------------+
  images -> | data loader | ----> | model (CNN)     | ----> | checkpoint (.pt)|
            +-------------+       +-----------------+       +------------------+
                  |                      |                          |
                  v                      v                          v
         train / val (from train)   metrics (P/R/F1,         FastAPI service
         test (official, held out)   confusion matrix)         POST /predict
```

## Dataset Sources

The dataset is PlantVillage, published on Hugging Face as `BrandonFors/Plant-Diseases-PlantVillage-Dataset` under the CC0-1.0 license. The repository is pinned to commit `96bdc27bd804297b76f373cba1c935e52076e107` in `scripts/download_dataset.py`, so results are reproducible. Each parquet shard is checked against its expected SHA-256 and byte size before use.

The official split is:

- `train`: two parquet shards, used to create the training and validation sets
- `test`: one parquet shard, held out for final evaluation

Dataset files are not committed to this repository. Download and export them with:

```bash
python scripts/download_dataset.py --out data/raw
```

This writes an ImageFolder layout with separate split directories:

```
data/raw/
  train/
    Tomato___Early_blight/
    Tomato___Late_blight/
    Tomato___healthy/
    ...
  test/
    Tomato___Early_blight/
    Tomato___Late_blight/
    Tomato___healthy/
    ...
  MANIFEST.json      # revision, shard hashes, and per-split class counts
```

If `data/raw/` contains an older flat layout (class folders directly under `data/raw/`), the download script refuses to run. Delete the old class folders first.

## Test Split Isolation

The official test split (`data/raw/test/`) is strictly held out:

- It is loaded only by the final evaluation step in `train.py`, after training has finished and the best checkpoint has been selected.
- It is never used for validation, early stopping, hyperparameter tuning, model selection, or fitting any preprocessing statistics. Normalization uses fixed ImageNet mean and standard deviation values, not statistics computed from the data.
- The validation set is created only from `data/raw/train/`, so no test image can enter training or validation.
- `tests/test_data.py` checks that no test file appears in the train or validation subsets and that a missing test directory fails loudly.

Do not copy, move, or sample from `data/raw/test/` to create other splits.

## Project Layout

```
crop-disease-detection/
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI app
├── configs/
│   └── train.yaml           # training and data configuration
├── scripts/
│   └── download_dataset.py  # pinned, verified download and split-preserving export
├── src/
│   └── crop_disease/
│       ├── __init__.py
│       ├── config.py        # config loading
│       ├── data.py          # train/val split from train, held-out test loader
│       ├── validate.py      # label, corruption, duplicate, and balance checks
│       ├── model.py         # model construction
│       ├── metrics.py       # precision, recall, F1, confusion matrix
│       ├── inference.py     # load checkpoint and predict
│       └── train.py         # training loop and CLI entry point
├── tests/
│   ├── test_data.py         # split isolation
│   ├── test_metrics.py
│   ├── test_validate.py
│   └── test_api.py
├── requirements.txt
├── requirements-data.txt
├── pyproject.toml
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install -r requirements-data.txt
```

## Data Preparation

1. Download and export the dataset (see Dataset Sources).
2. Validate the splits:

```bash
python -m crop_disease.validate data/raw/train --report artifacts/dataset_report_train.json
python -m crop_disease.validate data/raw/test --report artifacts/dataset_report_test.json
```

## Training

1. Confirm `data/raw/train/` and `data/raw/test/` exist.
2. Adjust `configs/train.yaml` as needed.
3. Run:

```bash
python -m crop_disease.train --config configs/train.yaml
```

The script splits `data/raw/train/` into train and validation sets, trains with early stopping on validation macro F1, and writes the best checkpoint to `artifacts/best.pt`. After training, it evaluates the best checkpoint once on the official test split and writes `artifacts/metrics.json` and `artifacts/confusion_matrix.csv`.

## Inference API

Start the server after training:

```bash
uvicorn api.main:app --reload --port 8000
```

Endpoints:

- `GET /health` returns service status and whether a model is loaded.
- `POST /predict` accepts a multipart form field `file` (JPEG or PNG) and returns:

```json
{
  "predicted_class": "Tomato___Late_blight",
  "confidence": 0.94,
  "top_k": [
    {"class_name": "Tomato___Late_blight", "probability": 0.94},
    {"class_name": "Tomato___Early_blight", "probability": 0.04}
  ]
}
```

Example request:

```bash
curl -X POST -F "file=@leaf.jpg" http://localhost:8000/predict
```

The model checkpoint path can be set with the `MODEL_PATH` environment variable (default `artifacts/best.pt`).

## Testing

```bash
pytest -q
```

## Disclaimer

Predictions are intended for research and decision support. They are not a substitute for diagnosis by an agronomist or plant pathologist.

## License

To be determined by the repository owner. The dataset itself is licensed CC0-1.0 by its publisher.
