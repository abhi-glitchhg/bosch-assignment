import torch
import cv2
from evaluate import load_model
from PIL import Image
import torchvision
from torchvision.transforms import v2 as T
from argparse import ArgumentParser

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COCO_CLASSES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

BDD_RELEVANT = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "train",
    "truck",
    "traffic light",
    "stop sign",
}


def infer(model, image_path, conf_thresh=0.5):
    # transforms
    image = Image.open(image_path).convert("RGB")
    tensor = T.ToTensor()(image).to(DEVICE)

    with torch.no_grad():
        preds = model([tensor])[0]

    boxes = preds["boxes"].cpu()
    scores = preds["scores"].cpu()
    labels = [COCO_CLASSES[i] for i in preds["labels"].tolist()]

    return [
        (box, score, label)
        for box, score, label in zip(boxes, scores, labels)
        if score > conf_thresh and label in BDD_RELEVANT
    ]


# ── visualise ─────────────────────────────────────────────────────────────
def visualise(image_path, conf_thresh=0.5):
    results = infer(image_path, conf_thresh)
    img = cv2.imread(image_path)

    for box, score, label in results:
        x1, y1, x2, y2 = box.int().tolist()
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{label} {score:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    cv2.imshow("detections", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--image", type=str, default="../../data/assignment_data_bdd/bdd100k_images_100k/bdd100k/images/100k/train/0a0a0b1a-7c39d841.jpg", )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint . If not provided, uses COCO pretrained weights.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_args()
    model = load_model(args.checkpoint)
    results = infer(model, args.image)
    for box, score, label in results:
        print(f"{label:15s}  {score:.2f}  {box.int().tolist()}")

    visualise(args.image)
