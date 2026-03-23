"""
evaluate.py
Computes mAP@0.5, mAP@0.5:0.95, per-class AP, precision, recall for
the fine-tuned Faster R-CNN on BDD100K validation set.

Uses torchmetrics — clean, correct, no reimplementation needed.
    pip install torchmetrics
"""

import torch
import pathlib
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchvision

from dataset import BDDDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_CLASSES = 11  # 10 + background
CONF_THRESH = 0.5
CHECKPOINT = "checkpoints/fasterrcnn_bdd100k_best.pth"

LABEL_DIR = pathlib.Path(
    "../../data/assignment_data_bdd/bdd100k_labels_release/bdd100k/labels"
)
IMAGE_DIR = pathlib.Path(
    "../../data/assignment_data_bdd/bdd100k_images_100k/bdd100k/images/100k"
)

IDX_TO_CLASS = {v: k for k, v in BDDDataset.CLASS_TO_IDX.items()}


def load_model(checkpoint_path=None):
    """ """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None, min_size=480, max_size=640
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(ckpt["state_dict"])
    return model.to(DEVICE).eval()


def get_transforms():
    return T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
        ]
    )


def collate_fn(
    batch,
):  # else different number of bbox in each image will cause issue. :smile:
    images, targets = zip(*batch)
    return list(images), list(targets)


@torch.no_grad()
def evaluate(model, dataloader):
    """
    Checking the accuracy of the model using metrics like AP.
    """
    # copy paste from the torchmetrics
    metric = MeanAveragePrecision(
        iou_type="bbox",
        iou_thresholds=None,
        class_metrics=True,
    )

    for images, targets in tqdm(dataloader):
        images = [img.to(DEVICE) for img in images]
        preds = model(images)

        # format preds for torchmetrics
        preds = [
            {
                "boxes": p["boxes"].cpu(),
                "scores": p["scores"].cpu(),
                "labels": p["labels"].cpu(),
            }
            for p in preds
        ]

        # format targets for torchmetrics
        targets = [
            {
                "boxes": t["boxes"].cpu(),
                "labels": t["labels"].cpu(),
            }
            for t in targets
        ]

        metric.update(preds, targets)

    return metric.compute()


def print_results(results):
    """
    neat formatting using chatgpt :smile:
    """
    print("\n" + "=" * 55)
    print(f"  mAP@0.5          : {results['map_50'].item():.4f}")
    print(f"  mAP@0.5:0.95     : {results['map'].item():.4f}")
    print(f"  mAP@0.75         : {results['map_75'].item():.4f}")
    print(f"  mean Recall      : {results['mar_100'].item():.4f}")
    print("=" * 55)

    # per-class AP
    if "map_per_class" in results and results["map_per_class"].numel() > 0:
        print("\nPer-class AP@0.5:0.95:")
        print(f"  {'class':15s}  {'AP':>6s}  {'AR':>6s}")
        print("  " + "-" * 32)
        per_class_ap = results["map_per_class"]
        per_class_ar = results["mar_100_per_class"]
        classes = results["classes"]

        for cls_id, ap, ar in zip(classes, per_class_ap, per_class_ar):
            name = IDX_TO_CLASS.get(cls_id.item(), f"cls_{cls_id.item()}")
            print(f"  {name:15s}  {ap.item():.4f}  {ar.item():.4f}")

    print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    val_df = pd.read_parquet(LABEL_DIR / "filtered_validation.parquet")
    val_ds = BDDDataset(
        df=val_df,
        image_dir=IMAGE_DIR / "val",
        transforms=get_transforms(),
        max_samples=1000,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model   = load_model(CHECKPOINT)
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # model.to(DEVICE).eval()

    results = evaluate(model, val_loader)
    print_results(results)
