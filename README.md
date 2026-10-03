# AI-Media Guard

The static upload interface and local Python API are connected. The API analyzes JPG, PNG, and WEBP images and samples up to eight frames from MP4/MOV videos. It returns an estimated `real` or `ai_generated` verdict from one EfficientNetB0 image classifier. It does not currently have a separate manipulation/deepfake model.

## Setup

Use Python 3.10–3.12, then from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The CIFAKE dataset is in `archive (1)` in this workspace, with `train/FAKE`, `train/REAL`, `test/FAKE`, and `test/REAL` directories. From the VS Code terminal, with `.venv` activated, train and evaluate it:

```bash
python train.py "archive (1)"
```

The default CPU-friendly run trains on a shuffled 12,000-image subset for up to 3 epochs and evaluates on a shuffled 5,000-image test subset. It uses a 10% validation split from the training data. If your dataset is in a different location, pass that folder instead. If you omit the dataset argument, the script searches `archive (1)`, `dataset`, and `data`. To train using all training and test images, pass `--max-train-samples 0 --max-test-samples 0` (this will take much longer on CPU).

The best validation checkpoint is saved to `models/AI_Media_Guard_Image_Final.keras`; final test loss, accuracy, and AUC are printed. The app expects this model path by default. Override it with `MODEL_PATH=/path/to/model.keras` if needed.

## Run the combined app

```bash
uvicorn app:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), not the `index.html` file directly. The page and `/api/analyze` run on the same local server. `/api/health` reports whether a model file is present. Files are processed in memory or a temporary video file that is deleted after analysis; uploads are not persisted.

## Model limits

Training is binary (CIFAKE `FAKE` vs `REAL`). Video analysis applies that same image classifier to sampled frames and averages the scores; it is not a temporal video model. The model's confidence is a score from this dataset/model, not proof of provenance, and its performance can vary on media outside the training distribution. Training runs on CPU if TensorFlow has no supported GPU available and can take a while.
