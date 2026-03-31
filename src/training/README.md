# Training

Object detection on BDD100K using Faster R-CNN (torchvision).

## Files

| File | Purpose |
|---|---|
| `bdd_yolo_data.py` | dataset preprocessing for baseline exp |
| `bdd_yolo_data_with_custom_filter.py` | dataset preprocessing for exp1 |
| `yolo_training.py` | Simple training script using ultralytics. |
| `evaluate.py` | Compute mAP and per-class AP on validation set |
| `data.py` | BDDDataset class |
| `data.py` | BDDDataset class |

## Setup

```bash
pip install torch torchvision torchmetrics pandas pillow ultralytics
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

TO train the model on bdd data, we first need to make the data in compatible yolo format. (this is needed for using ultralytics code base). We also filter out the data using different strategies in our experiments. In baseline we remove bboxes with area less than  0.02 % of the img size, In later exp. we change this threshold for every class specific. 

To clean and preprocess the data for yolo, 


```bash

python bdd_yolo_data.py # for baseline exp...

```

After doing this, we can train our model using yolo class. Because of resource contrains we will use a YoloV8s (s stands for small.). Many times while training, my laptop sometimes crashed due to unknown reasons (maybe heating??). Hence ive added a resume flag, if the training crashed midway, then we can apppend --resume flag which will load the last checkpoint and continue model training procedure. 

```bash
python yolo_training.py --exp-name baseline #optionlly --resume
```

## Inference
Running this script will run inference on an image. The prediction bboxes will be drawn on the image and will be saved on the disk. 

```bash
python inference.py --image path/to/image.jpg
```

