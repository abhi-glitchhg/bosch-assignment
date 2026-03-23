import torch
from torch.utils.data import Dataset
import cv2
import os, json
import numpy as np
import pathlib
import pandas as pd
from PIL import Image

from torchvision.tv_tensors import BoundingBoxes


class BDDDataset(Dataset):
    CLASS_TO_IDX = {
        "person": 1,
        "rider": 2,
        "car": 3,
        "truck": 4,
        "bus": 5,
        "train": 6,
        "bike": 7,
        "motor": 8,
        "traffic light": 9,
        "traffic sign": 10,
    }

    def __init__(self, df, image_dir, transforms=None, max_samples=None):
        self.df = df

        self.grouped = df.groupby("image")
        self.image_dir = image_dir
        self.transforms = transforms

        self.images = df["image"].unique()
        if max_samples is not None:
            rng = np.random.default_rng(42)
            self.images = rng.choice(
                self.images, size=min(max_samples, len(self.images)), replace=False
            )

        # map class names to ids
        self.classes = sorted(self.CLASS_TO_IDX.keys())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        records = self.grouped.get_group(img_name)

        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path)

        if image is None:
            raise ValueError(f"Image not found: {img_path}")

        boxes = []
        labels = []

        for _, row in records.iterrows():
            boxes.append([row["x1"], row["y1"], row["x2"], row["y2"]])
            labels.append(self.CLASS_TO_IDX[row["category"]])

        if len(boxes) == 0:  # edge case
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": BoundingBoxes(
                boxes, format="XYXY", canvas_size=(image.size[1], image.size[0])
            ),
            "labels": labels,
        }

        if self.transforms:
            sample = {
                "image": image,
                "boxes": target["boxes"],
                "labels": target["labels"],
            }

            sample = self.transforms(sample)

            image = sample["image"]
            target = {
                "boxes": sample["boxes"],
                "labels": sample["labels"],
            }
        return image, target


def visualize_sample(img, target, class_names, key="window"):
    # convert tensor → numpy

    boxes = target["boxes"]
    labels = target["labels"]

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box.int().tolist()

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        class_name = class_names[label.item()]
        cv2.putText(
            img, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )
    cv2.imshow(key, img[..., ::-1])
    cv2.waitKey(0)


if __name__ == "__main__":
    label_data_path = pathlib.Path(
        "../../data/assignment_data_bdd/bdd100k_labels_release/bdd100k/labels"
    )
    image_dir = pathlib.Path(
        "../../data/assignment_data_bdd/bdd100k_images_100k/bdd100k/images/100k"
    )
    train_data_df = pd.read_parquet(label_data_path / "filtered_train.parquet")
    validation_data_df = pd.read_parquet(
        label_data_path / "filtered_validation.parquet"
    )

    val_dataset = BDDDataset(
        df=validation_data_df,
        image_dir=image_dir / "val",
    )
    train_dataset = BDDDataset(
        df=train_data_df,
        image_dir=image_dir / "train",
    )
    idx_to_class = {v: k for k, v in BDDDataset.CLASS_TO_IDX.items()}

    for i in range(5):
        image, target = val_dataset[i]
        visualize_sample(image, target, idx_to_class, key="val")

    for i in range(5):
        image, target = train_dataset[i]
        visualize_sample(image, target, idx_to_class, key="train")
