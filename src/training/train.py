
import torch
import pathlib
import pandas as pd
from tqdm import tqdm
from torchvision.transforms import v2 as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import torchvision

from dataset import BDDDataset

# CONFIG
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20
BATCH_SIZE = 24
ACCUM_STEPS = 4 
LR = 3e-3
WEIGHT_DECAY = 1e-4
USE_AMP = True
NUM_CLASSES = 11  # 10 BDD classes + background 

LABEL_DIR = pathlib.Path(
    "../../data/assignment_data_bdd/bdd100k_labels_release/bdd100k/labels"
)
IMAGE_DIR = pathlib.Path(
    "../../data/assignment_data_bdd/bdd100k_images_100k/bdd100k/images/100k"
)

def build_model(checkpoint_path=None):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT",
        min_size=480,   
        max_size=640,
    )
    # resizing is handled internally by torchvision class :)

    # freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # replace head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    
    # load weights from local
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        print(
            f"Loaded checkpoint from epoch {ckpt['epoch']} "
            f"(val_loss={ckpt['val_loss']:.4f})"
        )
    # get model summary.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable: {trainable:,} / {total:,} params ({100 * trainable / total:.1f}%)"
    )

    return model


def get_transforms(train=True):
    ops = [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ]
    if train:
        ops += [T.RandomHorizontalFlip(0.5)]
    return T.Compose(ops)


def collate_fn(batch): # stacking logic for dataloader else we get error when dataloader tries to stack targets of different len
    images, targets = zip(*batch)
    return list(images), list(targets)


def train_one_epoch(model, loader, optimizer, scaler, device, accum_steps):
    model.train()
    total = cls_l = box_l = 0.0
    optimizer.zero_grad()

    for step, (images, targets) in enumerate(tqdm(loader, desc="  train", leave=False)):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with autocast(device_type=device, enabled=USE_AMP):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values()) / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total += loss.item() * accum_steps
        cls_l += loss_dict["loss_classifier"].item()
        box_l += loss_dict["loss_box_reg"].item()

    # flush remaining grads for the last remaining items
    if len(loader) % accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    n = len(loader)
    return total / n, cls_l / n, box_l / n


@torch.no_grad()
def validate(model, loader, device):
    model.eval()  
    total = 0.0
    for images, targets in tqdm(loader, desc="  val  ", leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with autocast(device_type=device, enabled=USE_AMP):
            loss_dict = model(images, targets)
            total += sum(loss_dict.values()).item()
    
    return total / len(loader)

if __name__ == "__main__":
    # ── data ──────────────────────────────────────────────────────────────
    train_df = pd.read_parquet(LABEL_DIR / "filtered_train.parquet")
    val_df = pd.read_parquet(LABEL_DIR / "filtered_validation.parquet")

    train_ds = BDDDataset(
        df=train_df,
        image_dir=IMAGE_DIR / "train",
        transforms=get_transforms(train=True),
        max_samples=10000,
    )
    val_ds = BDDDataset(
        df=val_df,
        image_dir=IMAGE_DIR / "val",
        transforms=get_transforms(train=False),
        max_samples=300,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    model = build_model().to(DEVICE)
    scaler = GradScaler(enabled=USE_AMP)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"AMP: {USE_AMP} | effective batch: {BATCH_SIZE * ACCUM_STEPS}\n")

    best_val = 100000

    for epoch in range(1, EPOCHS + 1):
        tr, cls_l, box_l = train_one_epoch(
            model, train_loader, optimizer, scaler, DEVICE, ACCUM_STEPS
        )
        val = validate(model, val_loader, DEVICE)
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"train {tr:.4f} (cls {cls_l:.3f}  box {box_l:.3f}) | "
            f"val {val:.4f}"
        )

        if val < best_val:
            # keep updating the best model based on the val loss .
            best_val = val
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict(), "val_loss": val},
                "checkpoints/fasterrcnn_bdd100k_best.pth",
            )
            print(f"  ✅ saved (val={val:.4f})")

        # save other checkpoints so that we can test different states of the model later. 
        torch.save(
            {"epoch": epoch, "state_dict": model.state_dict(), "val_loss": val},
            f"checkpoints/fasterrcnn_bdd100k_{epoch}.pth",
        )

    print(f"Done. Best val loss: {best_val:.4f}")
