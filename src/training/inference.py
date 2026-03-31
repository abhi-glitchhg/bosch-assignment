import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO


def draw_predictions(image, result):
    names = result.names

    if result.boxes is None or len(result.boxes) == 0:
        return image

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        label = f"{names[cls_id]} {conf:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return image


def main():
    parser = argparse.ArgumentParser(description="Run YOLO inference on a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str,default='runs/detect/bdd_yolov8_baseline/weights/best.pt' , help="Path to YOLO weights (.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device to use, e.g. "0", "cpu", "0,1"',
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Optional output image path. If not given, saves next to input image.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the output image in a window",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    weights_path = Path(args.weights)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = YOLO(str(weights_path))

    results = model.predict(
        source=str(image_path),
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )

    if len(results) == 0:
        raise RuntimeError("No results returned by model.")

    result = results[0]

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    output = draw_predictions(image, result)

    if args.save_path is not None:
        save_path = Path(args.save_path)
    else:
        save_path = image_path.with_name(f"{image_path.stem}_pred{image_path.suffix}")

    cv2.imwrite(str(save_path), output)
    print(f"Saved output to: {save_path}")

    if result.boxes is None or len(result.boxes) == 0:
        print("No detections found.")
    else:
        print("\nDetections:")
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            class_name = result.names[cls_id]
            print(
                f"class={class_name}, conf={conf:.4f}, "
                f"xyxy=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
            )

    if args.show:
        cv2.imshow("Prediction", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()