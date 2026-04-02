# Bosch assignment

This document gives idea about the repo structure and setup. Read `report.md` for more details. I decided to keep this assignment with minimal requirements and keep it simple. 

## Setup 

While running the docker image, i would suggest to mount bdd data to the /app/data directory. The expected mounted file structure would look like

 - /app
 - /app/data
 -   /app/data/asignment_data_bdd
        



## EDA 

Exploratory data analysis for the dataset is in `src/EDA` folder. It contains a jupyter notebook where you can plot the graphs and try to gauge the properties and trends in the  dataset we are going to work with.
We have found some interesting cases where the labels are incorrect or the bounding boxes are too small or too big. Just run the jupyter notebook once and then you will find such interesting cases in  `src/EDA/generated_data` folder.


## Training 

~~As discussed in the `report.md` we spent some time deciding on the model and unfortunately some trials were unfruitful. Finally we decided to finetune a coco-pretrained model from torchvision.~~ 

As training model from the scratch was difficult and time and resource constrains were there hence i chose yolo model series from `ultralytics`  which provides me great range of models and a nice training framework. I chose yolov8 small model. Yolo models are popular and their tensorrt engine files are also available hence one can easily use them in prod. 

To train the model, you will find the script in `src/training/train.py`. You can choose to change some of the hyperparameters to suit your resources. (like batch size, AMP etc)

```py
cd src/training

python yolo_training.py --exp-name baseline 
```

To infer model on an image, you can use inference script in the same directory.  

```py
python inference.py --path <your parth>  
```


## Evaluation and Improvements.  
   

To check model performance using metrics, we have a script for evaluation. We use widely used torchmetrics library to calculate mAP metric. It allso gives us on which samples our model failed along with FN,FP and confusion examplles. 

```py
cd ../evaluation 

python evaluate_bdd_failures.py 
```

For more detailed info, please read [report.md](report.md)

Note: Ruff was used for linting of the code. 
