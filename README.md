# Crop Disease Detection

An image classification system that identifies plant diseases from leaf photographs. It includes a modular PyTorch training pipeline and a FastAPI service that accepts an image upload and returns the predicted disease class with confidence scores.

## Features

- Transfer learning with a pretrained torchvision backbone (default: ResNet-50 style CNN, configurable)
- Stratified train / validation / test splits with a fixed seed for reproducibility
- Metrics: accuracy, per-class precision, recall, F1, macro and weighted averages, and confusion matrix
- YAML-driven configuration for data, model, and training hyperparameters
- FastAPI inference endpoint with image upload, input validation, and top-k predictions
- Unit tests for the data pipeline, metrics, and API

## Architecture

```
            +-------------+       +-----------------+       +------------------+
  images -> | data loader | ----> | model (CNN)     | ----> | checkpoint (.pt)|
            +-------------+       +-----------------+       +------------------+
                  |                      |                          |
                  v                      v                          v
            train / val          metrics (P/R/F1,           FastAPI service
              splits              confusion matrix)         POST /predict
```

## Dataset Sources

The pipeline expects an ImageFolder layout, one subdirectory per class:

```
data/raw/
  Tomato___Early_blight/
  Tomato___Late_blight/
  Tomato___healthy/
  ...
```

Suitable public sources include the PlantVillage dataset and the PlantDoc dataset. Check each dataset's license before use. Dataset files are not committed to this repository.

## Project Layout

```
crop-disease-detection/
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI app
├── configs/
│   └── train.yaml           # training and data configuration
├── src/
│   └── crop_disease/
│       ├── __init__.py
│       ├── config.py        # config loading
│       ├── data.py          # datasets, transforms, splits
│       ├── model.py         # model construction
│       ├── metrics.py       # precision, recall, F1, confusion matrix
│       ├── inference.py     # load checkpoint and predict
│       └── train.py         # training loop and CLI entry point
├── tests/
│   ├── test_metrics.py
│   └── test_api.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Training

1. Place the dataset under `data/raw/` in ImageFolder layout.
2. Adjust `configs/train.yaml` as needed.
3. Run:

```bash
python -m crop_disease.train --config configs/train.yaml
```

The script creates stratified train, validation, and test splits, trains with early stopping on validation macro F1, and writes the best checkpoint to `artifacts/best.pt` along with `artifacts/metrics.json` and `artifacts/confusion_matrix.csv`.

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

To be determined by the repository owner.
