import json
from pathlib import Path
from collections import defaultdict, Counter

import cv2
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

BASE_PATH = Path("../../data/assignment_data_bdd")
split = "val"  # or 'train'
EXP_NAME= "baseline" 
data_strategy_key = "baseline" # or improved_small_boxes


VAL_JSON = BASE_PATH / f"bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_{split}.json"
VAL_IMAGES_DIR = BASE_PATH / f"bdd100k_images_100k/bdd100k/images/100k/{split}"

# Change this to your trained weights
MODEL_WEIGHTS = f"runs/detect/bdd_yolov8_{EXP_NAME}/weights/best.pt"

    # Output folder
OUTPUT_DIR = Path(f"./eval_bdd_failures_{data_strategy_key}_{split}")

CLASS_NAMES = [
    "person", "rider", "car", "truck", "bus",
    "train", "bike", "motor", "traffic light", "traffic sign"
]
CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {i: name for name, i in CLASS_MAP.items()}

REMOVE_SMALL_BBOX_RATIO = 0.0002

# Evaluation params
CONF_THRES = 0.25
IOU_MATCH_THRES = 0.50
MAX_IMAGES = None # for fast iteration. 
SAVE_EXAMPLES_PER_BUCKET = 50

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def xyxy_iou(box1, box2):
    """
    box format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def area_ratio(box, img_w, img_h):
    bw = max(0.0, box[2] - box[0])
    bh = max(0.0, box[3] - box[1])
    return (bw * bh) / float(img_w * img_h)


def size_bucket(box, img_w, img_h):
    r = area_ratio(box, img_w, img_h)
    if r < 0.001:
        return "small"
    elif r < 0.01:
        return "medium"
    else:
        return "large"


def draw_box(img, box, label, color, thickness=2, delta=5):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        img,
        label,
        (x1 - delta, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


# ============================================================
# LOAD BDD VAL DATA
# ============================================================

def load_bdd_val_annotations(json_path: Path, remove_small_ratio=0.0002):
    """
    Returns:
      image_meta: dict[image_name] -> metadata dict
      gt_by_image: dict[image_name] -> list of GT objects
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    image_meta = {}
    gt_by_image = defaultdict(list)

    for img in tqdm(data, desc="Loading validation annotations"):
        img_name = img["name"]

        width = img.get("attributes", {}).get("resolution", {}).get("width", 1280)
        height = img.get("attributes", {}).get("resolution", {}).get("height", 720)
        weather = img.get("attributes", {}).get("weather", "unknown")
        timeofday = img.get("attributes", {}).get("timeofday", "unknown")
        scene = img.get("attributes", {}).get("scene", "unknown")

        image_meta[img_name] = {
            "width": width,
            "height": height,
            "weather": weather,
            "timeofday": timeofday,
            "scene": scene,
        }

        if "labels" not in img:
            continue

        for obj in img["labels"]:
            if "box2d" not in obj:
                continue

            cls_name = obj.get("category")
            if cls_name not in CLASS_MAP:
                continue

            box = obj["box2d"]
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            img_area = width * height

            if area <= img_area * remove_small_ratio:
                continue

            occluded = int(bool(obj.get("attributes", {}).get("occluded", False)))
            truncated = int(bool(obj.get("attributes", {}).get("truncated", False)))

            gt_by_image[img_name].append({
                "class_id": CLASS_MAP[cls_name],
                "class_name": cls_name,
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "area": area,
                "occluded": occluded, # we need to plot these details as well now.
                "truncated": truncated,
            })

    return image_meta, gt_by_image


# ============================================================
# MATCHING
# ============================================================

def match_predictions_to_gt(preds, gts, iou_thresh=0.5):
    """
    Greedy matching, same-class first.
    preds: list of dict {class_id, class_name, conf, box}
    gts:   list of dict {class_id, class_name, box}

    Returns:
      matches: list of (pred_idx, gt_idx, iou)
      unmatched_pred_indices
      unmatched_gt_indices
      best_wrong_class_matches: list of dict for confusion analysis
    """
    matches = []
    used_preds = set()
    used_gts = set()

    pred_order = sorted(range(len(preds)), key=lambda i: preds[i]["conf"], reverse=True)

    for pi in pred_order:
        pred = preds[pi]
        best_iou = -1.0
        best_gi = None

        for gi, gt in enumerate(gts):
            if gi in used_gts:
                continue
            if pred["class_id"] != gt["class_id"]:
                continue

            iou = xyxy_iou(pred["box"], gt["box"])
            if iou >= iou_thresh and iou > best_iou:
                best_iou = iou
                best_gi = gi

        if best_gi is not None:
            matches.append((pi, best_gi, best_iou))
            used_preds.add(pi)
            used_gts.add(best_gi)

    unmatched_pred_indices = [i for i in range(len(preds)) if i not in used_preds]
    unmatched_gt_indices = [i for i in range(len(gts)) if i not in used_gts]

    best_wrong_class_matches = []
    for pi in unmatched_pred_indices:
        pred = preds[pi]
        best_iou = 0.0
        best_gi = None

        for gi in unmatched_gt_indices:
            gt = gts[gi]
            iou = xyxy_iou(pred["box"], gt["box"])
            if iou >= iou_thresh and iou > best_iou:
                best_iou = iou
                best_gi = gi

        if best_gi is not None:
            gt = gts[best_gi]
            if pred["class_id"] != gt["class_id"]:
                best_wrong_class_matches.append({
                    "pred_idx": pi,
                    "gt_idx": best_gi,
                    "pred_class": pred["class_name"],
                    "gt_class": gt["class_name"],
                    "iou": best_iou,
                })

    return matches, unmatched_pred_indices, unmatched_gt_indices, best_wrong_class_matches



def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(OUTPUT_DIR / "examples_false_negative")
    ensure_dir(OUTPUT_DIR / "examples_false_positive")
    ensure_dir(OUTPUT_DIR / "examples_confusion")

    print("Loading model...")
    model = YOLO(MODEL_WEIGHTS)

    print("Loading validation annotations...")
    image_meta, gt_by_image = load_bdd_val_annotations(
        VAL_JSON,
        remove_small_ratio=REMOVE_SMALL_BBOX_RATIO,
    )

    image_names = sorted(image_meta.keys())
    if MAX_IMAGES is not None:
        image_names = image_names[:MAX_IMAGES]

    # Aggregates
    per_class = {
        name: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0}
        for name in CLASS_NAMES
    }

    attr_stats = {
        "non_occluded": {"gt": 0, "tp": 0, "fn": 0},
        "occluded": {"gt": 0, "tp": 0, "fn": 0},
        "non_truncated": {"gt": 0, "tp": 0, "fn": 0},
        "truncated": {"gt": 0, "tp": 0, "fn": 0},
    }

    failure_by_weather = Counter()
    failure_by_timeofday = Counter()
    failure_by_scene = Counter()

    misses_by_size = Counter()
    fp_by_size = Counter()
    tp_by_size = Counter()

    confusion_matrix = defaultdict(int)

    image_level_rows = []
    pred_rows = []
    miss_rows = []
    confusion_rows = []

    saved_fn = 0
    saved_fp = 0
    saved_conf = 0

    print("Running evaluation on validation images...")
    for img_name in tqdm(image_names):
        meta = image_meta[img_name]
        img_w = meta["width"]
        img_h = meta["height"]
        img_path = VAL_IMAGES_DIR / img_name

        if not img_path.exists():
            continue

        gts = gt_by_image.get(img_name, [])

        for gt in gts:
            per_class[gt["class_name"]]["gt"] += 1

            if gt["occluded"] == 1:
                attr_stats["occluded"]["gt"] += 1
            else:
                attr_stats["non_occluded"]["gt"] += 1

            if gt["truncated"] == 1:
                attr_stats["truncated"]["gt"] += 1
            else:
                attr_stats["non_truncated"]["gt"] += 1

        result = model.predict(
            source=str(img_path),
            conf=CONF_THRES,
            verbose=False,
            imgsz=540,
            device=0,
        )[0]

        preds = []
        if result.boxes is not None:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes_xyxy, confs, clss):
                if cls_id not in ID_TO_CLASS:
                    continue
                preds.append({
                    "class_id": int(cls_id),
                    "class_name": ID_TO_CLASS[int(cls_id)],
                    "conf": float(conf),
                    "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                })

        for pred in preds:
            per_class[pred["class_name"]]["pred"] += 1

        matches, unmatched_pred_idx, unmatched_gt_idx, wrong_class_matches = match_predictions_to_gt(
            preds, gts, iou_thresh=IOU_MATCH_THRES
        )

        # Update TP
        for pi, gi, iou in matches:
            pred = preds[pi]
            gt = gts[gi]
            per_class[gt["class_name"]]["tp"] += 1
            tp_by_size[size_bucket(gt["box"], img_w, img_h)] += 1

            if gt["occluded"] == 1:
                attr_stats["occluded"]["tp"] += 1
            else:
                attr_stats["non_occluded"]["tp"] += 1

            if gt["truncated"] == 1:
                attr_stats["truncated"]["tp"] += 1
            else:
                attr_stats["non_truncated"]["tp"] += 1

            pred_rows.append({
                "image": img_name,
                "status": "tp",
                "pred_class": pred["class_name"],
                "gt_class": gt["class_name"],
                "conf": pred["conf"],
                "iou": iou,
                "weather": meta["weather"],
                "timeofday": meta["timeofday"],
                "scene": meta["scene"],
                "size_bucket": size_bucket(gt["box"], img_w, img_h),
                "gt_occluded": gt["occluded"],
                "gt_truncated": gt["truncated"],
            })

        # Update FN
        for gi in unmatched_gt_idx:
            gt = gts[gi]
            per_class[gt["class_name"]]["fn"] += 1
            misses_by_size[size_bucket(gt["box"], img_w, img_h)] += 1

            if gt["occluded"] == 1:
                attr_stats["occluded"]["fn"] += 1
            else:
                attr_stats["non_occluded"]["fn"] += 1

            if gt["truncated"] == 1:
                attr_stats["truncated"]["fn"] += 1
            else:
                attr_stats["non_truncated"]["fn"] += 1

            failure_by_weather[meta["weather"]] += 1
            failure_by_timeofday[meta["timeofday"]] += 1
            failure_by_scene[meta["scene"]] += 1

            miss_rows.append({
                "image": img_name,
                "missed_class": gt["class_name"],
                "weather": meta["weather"],
                "timeofday": meta["timeofday"],
                "scene": meta["scene"],
                "size_bucket": size_bucket(gt["box"], img_w, img_h),
                "box_x1": gt["box"][0],
                "box_y1": gt["box"][1],
                "box_x2": gt["box"][2],
                "box_y2": gt["box"][3],
                "occluded": gt["occluded"],
                "truncated": gt["truncated"],
            })

        # Update FP
        for pi in unmatched_pred_idx:
            pred = preds[pi]
            per_class[pred["class_name"]]["fp"] += 1
            fp_by_size[size_bucket(pred["box"], img_w, img_h)] += 1

            pred_rows.append({
                "image": img_name,
                "status": "fp",
                "pred_class": pred["class_name"],
                "gt_class": None,
                "conf": pred["conf"],
                "iou": None,
                "weather": meta["weather"],
                "timeofday": meta["timeofday"],
                "scene": meta["scene"],
                "size_bucket": size_bucket(pred["box"], img_w, img_h),
                "gt_occluded": None,
                "gt_truncated": None,
            })

        # Confusions
        for row in wrong_class_matches:
            confusion_matrix[(row["gt_class"], row["pred_class"])] += 1
            gt = gts[row["gt_idx"]]
            confusion_rows.append({
                "image": img_name,
                "gt_class": row["gt_class"],
                "pred_class": row["pred_class"],
                "iou": row["iou"],
                "weather": meta["weather"],
                "timeofday": meta["timeofday"],
                "scene": meta["scene"],
                "gt_occluded": gt["occluded"],
                "gt_truncated": gt["truncated"],
            })

        # Save some example images
        save_this_image = (
            (len(unmatched_gt_idx) > 0 and saved_fn < SAVE_EXAMPLES_PER_BUCKET) or
            (len(unmatched_pred_idx) > 0 and saved_fp < SAVE_EXAMPLES_PER_BUCKET) or
            (len(wrong_class_matches) > 0 and saved_conf < SAVE_EXAMPLES_PER_BUCKET)
        )

        if save_this_image:
            img = cv2.imread(str(img_path))
            if img is not None:
                if len(unmatched_gt_idx) > 0 and saved_fn < SAVE_EXAMPLES_PER_BUCKET:
                    img_fn = img.copy()
                    for gi in unmatched_gt_idx:
                        gt = gts[gi]
                        occ_tag = "occ" if gt["occluded"] else "clear"
                        trunc_tag = "trunc" if gt["truncated"] else "full"
                        draw_box(
                            img_fn,
                            gt["box"],
                            f"MISS:{gt['class_name']} {occ_tag} {trunc_tag}",
                            (0, 0, 255),
                        )
                    out = OUTPUT_DIR / "examples_false_negative" / f"{saved_fn:03d}_{img_name}"
                    cv2.imwrite(str(out), img_fn)
                    saved_fn += 1

                if len(unmatched_pred_idx) > 0 and saved_fp < SAVE_EXAMPLES_PER_BUCKET:
                    img_fp = img.copy()
                    for pi in unmatched_pred_idx:
                        pred = preds[pi]
                        draw_box(img_fp, pred["box"], f"FP:{pred['class_name']} {pred['conf']:.2f}", (255, 0, 0))
                    out = OUTPUT_DIR / "examples_false_positive" / f"{saved_fp:03d}_{img_name}"
                    cv2.imwrite(str(out), img_fp)
                    saved_fp += 1

                if len(wrong_class_matches) > 0 and saved_conf < SAVE_EXAMPLES_PER_BUCKET:
                    img_conf = img.copy()
                    for row in wrong_class_matches:
                        pred = preds[row["pred_idx"]]
                        gt = gts[row["gt_idx"]]
                        occ_tag = "occ" if gt["occluded"] else "clear"
                        trunc_tag = "trunc" if gt["truncated"] else "full"
                        draw_box(img_conf, gt["box"], f"GT:{gt['class_name']} {occ_tag} {trunc_tag}", (0, 255, 0), delta=40)
                        draw_box(img_conf, pred["box"], f"PRED:{pred['class_name']}", (0, 165, 255), delta=-40)
                    out = OUTPUT_DIR / "examples_confusion" / f"{saved_conf:03d}_{img_name}"
                    cv2.imwrite(str(out), img_conf)
                    saved_conf += 1

        image_level_rows.append({
            "image": img_name,
            "num_gt": len(gts),
            "num_pred": len(preds),
            "num_tp": len(matches),
            "num_fp": len(unmatched_pred_idx),
            "num_fn": len(unmatched_gt_idx),
            "weather": meta["weather"],
            "timeofday": meta["timeofday"],
            "scene": meta["scene"],
        })

    # logging the metrics
    
    class_rows = []
    for cls_name in CLASS_NAMES:
        tp = per_class[cls_name]["tp"]
        fp = per_class[cls_name]["fp"]
        fn = per_class[cls_name]["fn"]
        gt = per_class[cls_name]["gt"]
        pred = per_class[cls_name]["pred"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        class_rows.append({
            "class": cls_name,
            "gt_count": gt,
            "pred_count": pred,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision@0.5IoU": precision,
            "recall@0.5IoU": recall,
            "f1@0.5IoU": f1,
        })

    attr_rows = []
    for attr_name, vals in attr_stats.items():
        gt = vals["gt"]
        tp = vals["tp"]
        fn = vals["fn"]
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        attr_rows.append({
            "bucket": attr_name,
            "gt_count": gt,
            "tp": tp,
            "fn": fn,
            "recall@0.5IoU": recall,
        })

    df_class = pd.DataFrame(class_rows).sort_values("recall@0.5IoU")
    df_attr = pd.DataFrame(attr_rows).sort_values("bucket")
    df_image = pd.DataFrame(image_level_rows).sort_values(["num_fn", "num_fp"], ascending=False)
    df_preds = pd.DataFrame(pred_rows)
    df_misses = pd.DataFrame(miss_rows)
    df_confusions = pd.DataFrame(confusion_rows)

    df_fail_weather = pd.DataFrame(
        [{"weather": k, "miss_count": v} for k, v in failure_by_weather.items()]
    ).sort_values("miss_count", ascending=False)

    df_fail_timeofday = pd.DataFrame(
        [{"timeofday": k, "miss_count": v} for k, v in failure_by_timeofday.items()]
    ).sort_values("miss_count", ascending=False)

    df_fail_scene = pd.DataFrame(
        [{"scene": k, "miss_count": v} for k, v in failure_by_scene.items()]
    ).sort_values("miss_count", ascending=False)

    confusion_rows_out = []
    for (gt_class, pred_class), count in sorted(confusion_matrix.items(), key=lambda x: x[1], reverse=True):
        confusion_rows_out.append({
            "gt_class": gt_class,
            "pred_class": pred_class,
            "count": count,
        })
    df_confusion_matrix = pd.DataFrame(confusion_rows_out)

    df_size = pd.DataFrame([
        {"bucket": "small", "tp": tp_by_size["small"], "fp": fp_by_size["small"], "fn": misses_by_size["small"]},
        {"bucket": "medium", "tp": tp_by_size["medium"], "fp": fp_by_size["medium"], "fn": misses_by_size["medium"]},
        {"bucket": "large", "tp": tp_by_size["large"], "fp": fp_by_size["large"], "fn": misses_by_size["large"]},
    ])

    # Helpful subgroup summaries from missed GTs
    if len(df_misses) > 0:
        df_miss_occ = (
            df_misses.groupby("occluded")
            .size()
            .reset_index(name="miss_count")
            .sort_values("miss_count", ascending=False)
        )
        df_miss_trunc = (
            df_misses.groupby("truncated")
            .size()
            .reset_index(name="miss_count")
            .sort_values("miss_count", ascending=False)
        )
    else:
        df_miss_occ = pd.DataFrame(columns=["occluded", "miss_count"])
        df_miss_trunc = pd.DataFrame(columns=["truncated", "miss_count"])

    # Save CSVs
    df_class.to_csv(OUTPUT_DIR / "per_class_metrics.csv", index=False)
    df_attr.to_csv(OUTPUT_DIR / "occlusion_truncation_metrics.csv", index=False)
    df_image.to_csv(OUTPUT_DIR / "per_image_summary.csv", index=False)
    df_preds.to_csv(OUTPUT_DIR / "predictions_tp_fp.csv", index=False)
    df_misses.to_csv(OUTPUT_DIR / "missed_ground_truths.csv", index=False)
    df_confusions.to_csv(OUTPUT_DIR / "raw_confusions.csv", index=False)
    df_fail_weather.to_csv(OUTPUT_DIR / "misses_by_weather.csv", index=False)
    df_fail_timeofday.to_csv(OUTPUT_DIR / "misses_by_timeofday.csv", index=False)
    df_fail_scene.to_csv(OUTPUT_DIR / "misses_by_scene.csv", index=False)
    df_confusion_matrix.to_csv(OUTPUT_DIR / "confusion_pairs.csv", index=False)
    df_size.to_csv(OUTPUT_DIR / "size_bucket_summary.csv", index=False)
    df_miss_occ.to_csv(OUTPUT_DIR / "misses_by_occluded_flag.csv", index=False)
    df_miss_trunc.to_csv(OUTPUT_DIR / "misses_by_truncated_flag.csv", index=False)

    # Print quick summary
    print("\n=== Per-class metrics ===")
    print(df_class.to_string(index=False))

    print("\n=== Occlusion / Truncation metrics ===")
    print(df_attr.to_string(index=False))

    print("\n=== Top confusion pairs ===")
    if len(df_confusion_matrix) > 0:
        print(df_confusion_matrix.head(20).to_string(index=False))
    else:
        print("No major same-location class confusions found.")

    print("\n=== Misses by weather ===")
    if len(df_fail_weather) > 0:
        print(df_fail_weather.to_string(index=False))

    print("\n=== Misses by timeofday ===")
    if len(df_fail_timeofday) > 0:
        print(df_fail_timeofday.to_string(index=False))

    print("\n=== Misses by scene ===")
    if len(df_fail_scene) > 0:
        print(df_fail_scene.to_string(index=False))

    print("\n=== Size bucket summary ===")
    print(df_size.to_string(index=False))

    print("\n=== Misses by occluded flag ===")
    if len(df_miss_occ) > 0:
        print(df_miss_occ.to_string(index=False))

    print("\n=== Misses by truncated flag ===")
    if len(df_miss_trunc) > 0:
        print(df_miss_trunc.to_string(index=False))

    print(f"\nSaved outputs to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()