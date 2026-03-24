# Object Detection on BDD100K: Exploration, Fine-Tuning, and Analysis

This report presents an object detection pipeline for the BDD100K dataset where I document my model selection process (including approaches explored and abandoned) before settling on a COCO-pretrained Faster R-CNN fine-tuned on BDD100K. I compare a zero-shot pretrained baseline against a head-only fine-tuned model, report per-class AP and mAP, and outline concrete steps to improve performance.

---

## 1. Dataset Exploration

The EDA focuses on label metadata only — image-level attributes and bounding box statistics — as full image-level analysis would require significantly more compute.

### Image-Level Attributes

Each image is annotated with **weather**, **time of day**, and **scene type**. The distributions are consistent between train and validation splits, which is a positive indicator of a well-constructed benchmark.

![Weather distribution](assets/weather_dist.png)

![Time of day and scene distributions](assets/tod_dist.png)
![](assets/scene_dist.png)

### Class Distribution

BDD100K exhibits a long-tailed class distribution. *Car* dominates with nearly half of all annotations, while *train* and *motorcycle* are severely underrepresented. This imbalance directly impacts model training — rare classes receive fewer gradient updates and are consistently missed at inference time.

![Class distribution across train and validation splits](assets/class_dist.png)

### Occlusion and Truncation

Moving object classes — *bicycle, bus, motorcycle* — show higher occlusion rates. Truncation is more uniformly distributed across classes. Both factors are known to reduce detection recall and are worth tracking separately in evaluation.

![Occlusion distribution](assets/occlusion_dist.png)
![Truncation distribution](assets/trunc_dist.png)

### Bounding Box Spatial Distribution

Bounding box centers show strong spatial clustering per class — *cars* concentrate near the centre-horizon, *traffic lights* in the upper-centre. This geometric bias means the model may struggle with objects appearing in atypical spatial positions.

![Normalised bounding box center heatmaps per class](assets/center_heatmap.png)

### Bounding Box Size and Annotation Quality

Exploration of bounding box sizes revealed two annotation quality concerns at the extremes of the size distribution.

**Very small boxes** are frequently incorrect — they tend to correspond to partially visible or ambiguous objects where the annotator marked a region too small to be reliably detected. Including these during training introduces noisy supervision and can hurt model performance. For example, many small bounding boxes in the *traffic sign* category are incorrect, raising questions about annotation quality that would need to be addressed before full-scale training.

![Top 10 smallest bounding boxes for the car class](assets/small_bbox_car.png)

**Very large boxes** also suffer from quality issues — oversized annotations often loosely wrap around objects, include significant background, or cover multiple instances under a single box. This leads to imprecise localisation targets and contributes to mAP@0.75 degradation.

![Top 10 largest bounding boxes for the traffic signal class](assets/large_bbox_traffic_singal.png)

A simple area-based filtering step — dropping boxes below a minimum pixel area threshold and reviewing boxes above a maximum threshold — would be a low-effort data cleaning step with meaningful impact on localisation quality.

---

## 2. Approach and Model Selection

### Approaches Explored but Not Used

Several approaches were explored before arriving at the final implementation.

**YOLOP** was the most promising candidate — a multitask convolutional network natively pretrained on BDD100K via PyTorch Hub. However, its detection head is fundamentally class-agnostic (single objectness score, no class labels), and the canonical multi-class BDD100K weights from the SysCV ETH model zoo were inaccessible due to the hosting server being offline. Significant time was spent attempting to replace the detection head and fine-tune for 10 classes, but the codebase proved difficult to work with and the effort did not yield usable results.

**FCOS** was also considered as an anchor-free torchvision alternative, but suffers the same absence of publicly available BDD100K pretrained weights.

**DETR** was ruled out as transformer-based, being compute intensive and difficult to train under the given constraints.

A personal project [Faster R-CNN on HuggingFace](https://huggingface.co/HugoHE/faster-rcnn-bdd-finetune) was also evaluated but the results were not satisfactory.

### Final Approach: Faster R-CNN

Settled on a **COCO-pretrained Faster R-CNN ResNet50-FPN** from torchvision, replacing the final `FastRCNNPredictor` head for BDD100K's 10 classes. The backbone and FPN were kept frozen; only the head was trained. Gradient accumulation and mixed precision training were used to maximise effective batch size within an 8GB VRAM budget.

Also we removed small bboxes which had area less than a threshold based on observations from EDA. We chose that threshold to be 200pixels. 

| Hyperparameter | Value |
|---|---|
| Backbone | ResNet50-FPN (frozen) |
| Pretrained weights | COCO |
| Num classes | 10 + background |
| Input size | 480 × 640 |
| Effective batch | 16 (4 × 4 grad. accum.) |
| Optimiser | AdamW, lr=1e-3 |
| LR schedule | Cosine annealing |
| Mixed precision | AMP (float16) |

---

## 3. Results

Two configurations are evaluated:
1. **COCO pretrained** — zero-shot baseline with BDD head but no BDD training
2. **Fine-tuned** — head trained on a BDD100K subset due to limited time and compute budget

### Overall Metrics

| Metric | Pretrained | Fine-tuned |
|---|---|---|
| mAP@0.5 | 0.1218 | 0.2629 |
| mAP@0.5:0.95 | 0.0618 | 0.1342 |
| mAP@0.75 | 0.0557 | 0.1190 |
| mean Recall@100 | 0.0861 | 0.2335 |

Fine-tuning more than doubles mAP@0.5 (0.12 → 0.26) and nearly triples recall, confirming the value of even limited domain adaptation.

### Per-class Results (IoU=0.5:0.95)

| Class | Pretrained AP | Pretrained AR | Fine-tuned AP | Fine-tuned AR |
|---|---|---|---|---|
| person | 0.2690 | 0.3641 | 0.1821 | 0.2761 |
| rider | 0.0000 | 0.0025 | 0.1053 | 0.2020 |
| car | 0.3489 | 0.4431 | 0.3050 | 0.3916 |
| truck | 0.0000 | 0.0000 | 0.1911 | 0.3869 |
| bus | 0.0000 | 0.0000 | 0.2274 | 0.3928 |
| train | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bicycle | 0.0000 | 0.0000 | 0.0921 | 0.1698 |
| motorcycle | 0.0000 | 0.0250 | 0.0744 | 0.1807 |
| traffic light | 0.0000 | 0.0000 | 0.0941 | 0.1976 |
| traffic sign | 0.0004 | 0.0260 | 0.0707 | 0.1379 |
| **mean** | **0.0618** | **0.0861** | **0.1342** | **0.2335** |

**Key observations:**
- Fine-tuning recovers classes absent in the pretrained baseline — truck (0.00→0.19), bus (0.00→0.23), rider (0.00→0.11)
- *Person* and *car* regress slightly as the COCO-optimised head is diluted by BDD-specific training on a small subset
- *Train* scores 0.00 in both models — a data scarcity problem (0.1% of BDD annotations), not a model capacity issue
- The mAP@0.5 to mAP@0.75 drop points to loose localisation, expected when the RPN remains frozen and optimised for COCO geometry

---

## 4. Improvements and Future Work

I know assignment has asked me to *actually* do these things but finetuning model required significant training resources. As my model isnt doing really well on metrics side with very little finetuning i will mention the procedures i would have tried if i had chance. :) 

**Full fine-tuning.** Unfreezing the backbone and FPN would be the single highest-impact change. Even training on a subset of 10k samples improved model performance, and loss curves were still decreasing — indicating headroom. A two-phase approach (head only → full network at lower lr=1e-4) would adapt low-level features to BDD's lighting and weather conditions.

**Dataset cleaning.** A non-trivial number of annotations have incorrect or imprecise bounding boxes. An active learning strategy — training on current data and identifying high-loss samples for re-annotation — would surface the most problematic examples efficiently. Class-wise feature clustering with outlier detection and human-in-the-loop verification would also help.

**Full dataset + balanced sampling.** Current training used a subset. Training on all images with a class-frequency-weighted sampler would directly address the 0.00 AP for rare classes like *train* and *rider*.

**Better augmentation.** Mosaic augmentation, random brightness/contrast (to simulate night/rain), and multi-scale jitter would improve robustness to BDD's diverse conditions.

**Better loss functions.** Focal loss would improve performance on rare classes where the model currently fails entirely.
