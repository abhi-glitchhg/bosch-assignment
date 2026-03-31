from ultralytics import YOLO
import argparse 


def main(args):
    model = YOLO("yolov8s.pt")  # pretrained on COCO
    training_args = {"data" :"bdd.yaml",
        "epochs" :50,
        "imgsz ": 540,
        "batch":36,
        "device":0,        
        "workers":8,
        "amp":True,
        "cache":True,
        "name":f"bdd_yolov8_{args.exp_name}",
        "freeze":10 }
    
    if args.resume:
        model = YOLO(f"runs/detect/bdd_yolov8_{args.exp_name}/weights/last.pt")  # pretrained on COCO
        training_args = {"resume": True}
    
    model.train(
        **training_args
    )


def parse_args():
    args = argparse.ArgumentParser(description="Train YOLOv8 on BDD100K dataset")
    args.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    args.add_argument("--exp-name", required=True, help="Experiment name for saving checkpoints and logs")

    return args.parse_args()  
if __name__ == "__main__":
    args = parse_args()
    main(args)