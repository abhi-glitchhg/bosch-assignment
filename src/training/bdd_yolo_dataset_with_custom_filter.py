import json
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# ----------------------------
# CONFIG
# ----------------------------

BASE_PATH = Path("../../data/assignment_data_bdd")

TRAIN_JSON = BASE_PATH / "bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"
VAL_JSON = BASE_PATH / "bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"

OUTPUT_LABELS = BASE_PATH / "labels"

CLASS_NAMES = [
    "person", "rider", "car", "truck", "bus",
    "train", "bike", "motor",
    "traffic light", "traffic sign"
]

CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}



# Thresholds are ratios w.r.t. full image area.
# Example: 0.0002 means keep boxes only if bbox_area > 0.02% of image area.
#
# These are starting values. Tune after checking how many boxes get removed.

MIN_AREA_RATIO_BY_CLASS = {
    # keep smaller ones
    "traffic light": 0.0002,
    "traffic sign": 0.0002,

    # medium threshold
    "person": 0.0008,
    "rider": 0.0008,
    "bike": 0.0008,
    "motor": 0.0008,

    # stricter for large vehicle classes
    "car": 0.001,
    "truck": 0.001,
    "bus": 0.001,

    "train": 0.030, # maybe we can remove this class itself.
}


MIN_W_NORM_BY_CLASS = {
    "traffic light": 0.003,
    "traffic sign": 0.004,
    "person": 0.001,
    "rider": 0.001,
    "bike": 0.006,
    "motor": 0.006,
    "car": 0.010,
    "truck": 0.012,
    "bus": 0.015,
    "train": 0.015,
}

MIN_H_NORM_BY_CLASS = {
    "traffic light": 0.008,
    "traffic sign": 0.008,
    "person": 0.015,
    "rider": 0.018,
    "bike": 0.012,
    "motor": 0.012,
    "car": 0.012,
    "truck": 0.015,
    "bus": 0.018,
    "train": 0.018,
}

# ----------------------------
# LOAD + PARSE
# ----------------------------

def load_bdd(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    rows = []

    for img in tqdm(data, desc=f"Parsing {json_path.name}"):
        img_name = img["name"]

        # BDD100K image size is typically 1280x720; use fallback if resolution missing
        width = img.get("attributes", {}).get("resolution", {}).get("width", 1280)
        height = img.get("attributes", {}).get("resolution", {}).get("height", 720)

        if "labels" not in img:
            continue

        for obj in img["labels"]:
            if "box2d" not in obj:
                continue

            cls = obj.get("category")
            if cls not in CLASS_MAP:
                continue

            box = obj["box2d"]

            x1, y1 = box["x1"], box["y1"]
            x2, y2 = box["x2"], box["y2"]

            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)

            rows.append({
                "image": img_name,
                "class": cls,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": width,
                "height": height,
                "bbox_w": w,
                "bbox_h": h,
                "area": w * h
            })

    return pd.DataFrame(rows)

# ----------------------------
# FILTER
# ----------------------------

def remove_small_bboxes_per_class(df):
    """
    Per-class bbox filtering based on:
    1. minimum area ratio wrt image area
    2. optional minimum normalized width
    3. optional minimum normalized height
    """
    df = df.copy()

    df["img_area"] = df["width"] * df["height"]
    df["area_ratio"] = df["area"] / df["img_area"]
    df["w_norm"] = df["bbox_w"] / df["width"]
    df["h_norm"] = df["bbox_h"] / df["height"]

    keep_mask = []

    for _, row in df.iterrows():
        cls = row["class"]

        min_area_ratio = MIN_AREA_RATIO_BY_CLASS.get(cls, 0.0002)
        min_w_norm = MIN_W_NORM_BY_CLASS.get(cls, 0.0)
        min_h_norm = MIN_H_NORM_BY_CLASS.get(cls, 0.0)

        keep = (
            row["area_ratio"] >= min_area_ratio and
            row["w_norm"] >= min_w_norm and
            row["h_norm"] >= min_h_norm
        )
        keep_mask.append(keep)

    return df[keep_mask].drop(columns=["img_area", "area_ratio", "w_norm", "h_norm"])

def print_filter_stats(before_df, after_df, split_name):
    print(f"\n{split_name} before filtering: {len(before_df)}")
    print(f"{split_name} after filtering : {len(after_df)}")
    print(f"{split_name} removed          : {len(before_df) - len(after_df)}")

    before_counts = before_df["class"].value_counts().sort_index()
    after_counts = after_df["class"].value_counts().sort_index()

    stats = pd.DataFrame({
        "before": before_counts,
        "after": after_counts
    }).fillna(0).astype(int)

    stats["removed"] = stats["before"] - stats["after"]
    stats["removed_pct"] = (100.0 * stats["removed"] / stats["before"].clip(lower=1)).round(2)

    print("\nPer-class filtering stats:")
    print(stats.sort_values("removed_pct", ascending=False).to_string())

# ----------------------------
# CONVERT TO YOLO
# ----------------------------

def convert_to_yolo(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    grouped = df.groupby("image")

    for img_name, group in tqdm(grouped, desc=f"Writing {output_dir}"):
        label_path = Path(output_dir) / img_name.replace(".jpg", ".txt")

        with open(label_path, "w") as f:
            for _, row in group.iterrows():
                cls_id = CLASS_MAP[row["class"]]

                x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
                W, H = row["width"], row["height"]

                # YOLO format (normalized)
                x = ((x1 + x2) / 2) / W
                y = ((y1 + y2) / 2) / H
                w = (x2 - x1) / W
                h = (y2 - y1) / H

                # safety clamp
                x = min(max(x, 0.0), 1.0)
                y = min(max(y, 0.0), 1.0)
                w = min(max(w, 0.0), 1.0)
                h = min(max(h, 0.0), 1.0)

                if w <= 0 or h <= 0:
                    continue

                f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

# ----------------------------
# MAIN PIPELINE
# ----------------------------

def process_split(json_path, split_name):
    df_before = load_bdd(json_path)
    df_after = remove_small_bboxes_per_class(df_before)

    print_filter_stats(df_before, df_after, split_name)

    convert_to_yolo(
        df_after,
        OUTPUT_LABELS / split_name
    )

def main():
    process_split(TRAIN_JSON, "train")
    process_split(VAL_JSON, "val")

    print("\n✅ DONE: YOLO labels ready")

if __name__ == "__main__":
    main()