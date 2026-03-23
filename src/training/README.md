# Training

Object detection on BDD100K using Faster R-CNN (torchvision).

## Files

| File | Purpose |
|---|---|
| `model.py` | Builds Faster R-CNN with BDD100K head |
| `train.py` | Training loop with AMP + gradient accumulation |
| `inference.py` | Run inference with COCO pretrained weights |
| `evaluate.py` | Compute mAP and per-class AP on validation set |
| `data.py` | BDDDataset class |

## Setup

```bash
pip install torch torchvision torchmetrics pandas pillow
```

## Data

Expected folder structure:

```
data/assignment_data_bdd/
    bdd100k_labels_release/bdd100k/labels/
        filtered_train.parquet
        filtered_validation.parquet
    bdd100k_images_100k/bdd100k/images/100k/
        train/
        val/
```

## Training

```bash
python train.py
```

Trains only the classification head (backbone frozen). Saves best checkpoint
to `fasterrcnn_bdd100k_best.pth`.

To limit training to a subset of images (faster iteration):
```python
# in train.py
train_ds = BDDDataset(..., max_samples=5000)
```

## Inference

```bash
python inference.py path/to/image.jpg
```

Uses COCO pretrained weights out of the box — no checkpoint needed.
Filters detections to BDD-relevant classes only.

## Evaluation

```bash
python evaluate.py
```

Loads `fasterrcnn_bdd100k_best.pth` and reports mAP@0.5, mAP@0.5:0.95,
and per-class AP/AR on the validation set.

## Notes

- Labels must be 1-indexed (1–10) in `BDDDataset` — 0 is background
- Effective batch size = `BATCH_SIZE × ACCUM_STEPS` (default: 4 × 4 = 16)
- Reduced `min_size` / `max_size` in `build_model()`, Automatic mixed precision and gradient accumulation to lower VRAM usage