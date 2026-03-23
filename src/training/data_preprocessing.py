# here we cleanup and remove unwanted data.
import pathlib, json
import pandas as pd


LABELS_DATA_PATH = pathlib.Path(
    "../../data/assignment_data_bdd/bdd100k_labels_release/bdd100k/labels"
)
TRAIN_LABELS_PATH = LABELS_DATA_PATH / "bdd100k_labels_images_train.json"
VAL_LABELS_PATH = LABELS_DATA_PATH / "bdd100k_labels_images_val.json"


class BDDMetadata:
    def __init__(self, file_path: pathlib.Path):
        self.raw_data = json.load(open(file_path, "r"))
        self.build_dataframes()

    def build_dataframes(self):
        img_rows = []
        object_rows = []

        for img in self.raw_data:
            img_name = img.get("name")

            img_rows.append(
                {
                    "image": img_name,
                    "weather": img.get("attributes", {}).get("weather"),
                    "timeofday": img.get("attributes", {}).get("timeofday"),
                    "scene": img.get("attributes", {}).get("scene"),
                }
            )

            if "labels" not in img:  # nno bbox case
                continue

            for bbox in img["labels"]:
                if "box2d" not in bbox:
                    continue

                box = bbox["box2d"]

                width = box["x2"] - box["x1"]
                height = box["y2"] - box["y1"]

                object_rows.append(
                    {
                        "image": img_name,
                        "category": bbox.get("category"),
                        "x1": box["x1"],
                        "y1": box["y1"],
                        "x2": box["x2"],
                        "y2": box["y2"],
                        "width": width,
                        "height": height,
                        "area": width * height,
                        "occluded": bbox.get("attributes", {}).get("occluded"),
                        "truncated": bbox.get("attributes", {}).get("truncated"),
                    }
                )

        self.images_df = pd.DataFrame(img_rows)
        self.bbox_df = pd.DataFrame(object_rows)


def remove_small_bboxes(df, pixel_threshold=200):
    """
    As discussed in the EDA, we have found very small bboxes are really bad quality.
    This function acts as a simple filter that will remove the small bboxes.
    This function can be later extended such that we have different criterias for different classes.
    Here, i have decided to remove the bboxes with area smaller than 0.02% of the image area.
    With more inspection one can decide the ideal value for these thresholds.
    """
    # 0.02% of image size = 0.0002 * 720 * 1280 approx == 200sq pixel.

    return df[df["area"] > pixel_threshold]


if __name__ == "__main__":
    train_data = BDDMetadata(TRAIN_LABELS_PATH)
    val_data = BDDMetadata(VAL_LABELS_PATH)

    print(f"Training data len before removing small bbxoes {len(train_data.bbox_df)}")
    filtered_train__df = remove_small_bboxes(train_data.bbox_df)
    print(f"Training data len after removing small bbxoes {len(filtered_train__df)}")

    print(f"Validation data len before removing small bbxoes {len(val_data.bbox_df)}")
    filtered_val_df = remove_small_bboxes(val_data.bbox_df)
    print(f"Validation data len after removing small bbxoes {len(filtered_val_df)}")

    filtered_train__df.to_parquet(
        "../../data/assignment_data_bdd/bdd100k_labels_release/bdd100k/labels/filtered_train.parquet"
    )
    filtered_val_df.to_parquet(
        "../../data/assignment_data_bdd/bdd100k_labels_release/bdd100k/labels/filtered_validation.parquet"
    )
