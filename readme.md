# Bosch assignmen

This document gives idea about the repo structure and setup. Read `report.md` for more details. I decided to keep this assignment with minimal requirements and keep it simple. 

## Setup 

While running the docker image, i would suggest to mount bdd data to the /app/data directory. The expected mounted file structure would look like

 - /app 
    -/data 
        -/assignment_data_bdd
        



## EDA 

Exploratory data analysis for the dataset is in `src/EDA` folder. It contains a jupyter notebook where you can plot the graphs and try to gauge the properties and trends in the  dataset we are going to work with.
We have found some interesting cases where the labels are incorrect or the bounding boxes are too small or too big. Just run the jupyter notebook once and then you will find such interesting cases in  `src/EDA/generated_data` folder.


## Training 

As discussed in the `report.md` we spent some time deciding on the model and unfortunately some trials were unfruitful. Finally we decided to finetune a coco-pretrained model from torchvision. 

To train the model, you will find the script in `src/training/train.py`. You can choose to change some of the hyperparameters to suit your resources. (like batch size, AMP etc)

```py
python train.py
```

To infer model on an image, you can use inference script in the same directory.  

```py
python inference.py --path <your parth> --checkpoint <optional checkpoint> 
```

To check model performance using metrics, we have a script for evaluation. We use widely used torchmetrics library to calculate mAP metric. 

```py
python evaluate.py
```


## Improvements 

As model finetuning was not done properly, the i couldnt do much with this. In the `reports.md` i have noted how i would improve the model.  