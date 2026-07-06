CAR DAMAGE DETECTION (CNN + TRANSFER LEARNING)

AIMS:
-1. (v1) Streamlit app where user can drag and drop an image and the app gives out one of the 6 prediction classes for car damage detection 
    - Build CNN from scratch
    - Use various Nets, like Res-NET, Elastic-NET, etc
    - Accuracy Aim: 75%    
    
-2. (v2) 
    - Highlight the damaged regions on the car (based on YOLO)
    - Unfreezze and re-train the last layer of other architectures in Transfer Learning

CLASSES (6 classifications):
-Front Normal, Front Crushed (minor), Front Breakage (major),
-Front: Normal/Crushed/Breakage => 500/400/500

-Back Normal,  Back Crushed (minor),  Back Breakage (major)
-Back:  Normal/Crushed/Breakage => 300/300/300

GENERAL INFO
-Images are taken at an angle so it can detect issues on the side of the car as well as well as the detection becomes easier.
-2300 RBG images from Hugging Faces.
-75% - 12.5% - 12.5% split for train - validation - test runs
-Train vs eval/test seperate transformations used
-Used best weights from 20 epochs only
-Local computer training on cpu (not gpu), so light weight models used..

-NOTE: THE DEFINITION OF CLASSIFICATION CLASSES IS VISUALLY SLIGHTLY UNCLEAR.
       See images at 1400 (R_Brakage) and 1700 (R_Crushed) locations, they look visually similar.

PUBLICATIONS:
 -https://www.sciencedirect.com/science/article/pii/S2405844024100473
 -https://ieeexplore.ieee.org/document/11115206/metrics#metrics
 -https://www.mdpi.com/2076-3417/14/20/9560

DATA:
-HUGGING FACES: https://huggingface.co/datasets/SaiVaibhavS/comprehensive-car-damage (USED HERE)
-KAGGLE:        https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset/data
-GutHub:        https://github.com/dxlabskku/TQVCD (INCOMPLETE DATASET)

TRANSFER LEARNING:
-Small training set (1725 images = 75% of 2300) → models above ~10M params carry high overfitting risk even with transfer learning
-Params/Training Image ratio -> key metric used to justify inclusion/exclusion. Models above ~6,500 params per image excluded where possible
-All selected models pretrained on ImageNet (1.2M real-world photos) -> since ImageNet classification and car damage detection both involve real-world photographs, learned features transfer well — making it a strong domain match
-Inception V3 excluded on two grounds -> requires 299x299 input (pipeline change) AND overfitting risk
-Custom CNN included as baseline -> essential to demonstrate the value of transfer learning
-Transfer learning mitigates overfitting -> most layers frozen, only final classifier retrained > but risk remains for very large models on very small datasets

Model             | Params | Versions Available   | Version Chosen    | Params/Training Image | Why This Version                        | Use? | Reason for Inclusion/Exclusion
------------------|--------|----------------------|-------------------|-----------------------|-----------------------------------------|------|---------------------------------------------
Custom CNN        | ~0.5M  | -                    | -                 | 290                   | Baseline, built from scratch            | YES  | Essential to show value of transfer learning
ResNet18          | 11M    | 18,34,50,101,152     | 18 (not 50/152)   | 6,377                 | Smallest ver, sufficient for small data | YES  | Industry standard baseline
EfficientNet-B0   | 5.3M   | B0-B7                | B0 (not B3/B7)    | 3,072                 | Best accuracy/size ratio for small data | YES  | Most efficient architecture
MobileNetV3-Small | 5.4M   | V2,V3-Small,V3-Large | Small (not Large) | 3,130                 | Lightweight, real-world deployment      | YES  | Shows deployment awareness
DenseNet121       | 8M     | 121,169,201          | 121 (not 169/201) | 4,638                 | Dense connections help small datasets   | YES  | Best suited for small data
ResNet50          | 25M    | 18,34,50,101,152     | -                 | 14,493                | Too many params for 1725 training images| NO   | High overfitting risk
ConvNeXt-Tiny     | 28M    | Tiny,Small,Base      | -                 | 16,231                | Too many params for 1725 training images| NO   | High overfitting risk
DenseNet201       | 20M    | 121,169,201          | -                 | 11,594                | Too many params for 1725 training images| NO   | High overfitting risk
VGG16             | 138M   | 11,13,16,19          | -                 | 80,000+               | Extremely large, outdated, slow         | NO   | Not worth compute cost
Inception V3      | 27M    | V3                   | -                 | 15,652                | Requires 299x299 input                  | NO   | Incompatible pipeline + overfitting risk
ViT-B/16          | 86M    | B/16,L/16            | -                 | 49,855                | Needs 100k+ images to shine             | NO   | Insufficient data
EfficientNet-B7   | 66M    | B0-B7                | -                 | 38,260                | Overkill for 1725 training images       | NO   | Overfitting risk
ResNet152         | 60M    | 18,34,50,101,152     | -                 | 34,782                | Too large for 1725 training images      | NO   | Overfitting risk
------------------|--------|-------------------|-------------------------------------------|------|-------------------------------------------------------------------------------------------

SUMMARY from notebooks:
-mobilenet_v3_small Accuracy based on test data is :         75.69 %
-mobilenet_v3_large Accuracy based on test data is :        75.00 %
-efficientnet_b0_rwightman Accuracy based on test data is :  70.83 %
-densenet121 Accuracy based on test data is :               69.79 %
-resnet18 Accuracy based on test data is :                  68.40 %
-Custom CNN Accuracy based on test data is :                52.43 %

-------------------------------------------------------------------------------------------------------------------------------
# Car Damage Detection (CNN + Transfer Learning)

A computer vision project designed to automate vehicle damage assessment using Deep Learning. This project evaluates custom CNN architectures alongside Transfer Learning models to classify car damage from multi-angle images, deploying the final model as an interactive web application.

---

## Project Overview & Goals

This project is structured into two development phases:

### Phase 1 (v1) - Current Release
*   **Interactive Web App:** A Streamlit application featuring drag-and-drop image uploads.
*   **Classification:** Accurately categorises images into one of 6 distinct vehicle condition classes.
*   **Architectures:** Baseline custom CNN built from scratch evaluated against Transfer Learning networks (e.g., ResNet, EfficientNet variants).
*   **v1 Strategy:** Freeze all base layers and retrain the classification head only.
*   **Target Performance:** Minimum accuracy threshold of **75%**.

### Phase 2 (v2) - Future Roadmap
*   **Object Detection:** Integrate YOLO to detect and highlight exact pixel regions containing damage.
*   **Fine-Tuning:** Unfreeze and retrain the deep feature extraction layers of selected Transfer Learning networks to improve classification boundaries.

---

## Dataset & Visual Challenges

### Data Breakdown
The model utilizes a total of **2,300 RGB images** sourced from Hugging Face, split across 6 classes:
*   **Front View (1,400 images total):** Normal (500) | Crushed/Minor (400) | Breakage/Major (500)
*   **Rear View (900 images total):** Normal (300) | Crushed/Minor (300) | Breakage/Major (300)

### Technical Constraints & Strategy
*   **Camera Angles:** Images are captured at specific angles to ensure side damage is visible, streamlining the extraction of complex visual features.
*   **Data Split:** Structured using a **75% Train / 12.5% Validation / 12.5% Test** partition strategy.
*   **Augmentation:** Dedicated, independent augmentation pipelines are used for training vs. evaluation/test runs.
*   **Class Ambiguity Note:** Visual overlap exists between specific classes in the dataset. For instance, images around index 1400 (`Rear Breakage`) and index 1700 (`Rear Crushed`) present highly similar damage patterns, creating a complex classification boundary.

---

## 🧠 Model Selection & Transfer Learning Strategy

### Resource Constraints
Training was conducted entirely on local **CPU infrastructure**. Consequently, the architecture selection leans heavily toward lightweight, computationally efficient models.

### Parameters-to-Image Ratio Evaluation
With only 1,725 training images (75% of the dataset), architectures exceeding ~10M parameters pose an extreme risk of overfitting. 
*   **Selection Metric:** Models maintaining a strict ratio below **~6,500 parameters per available training image** were prioritized.
*   **Inception V3 Exclusion:** Explicitly excluded due to its rigid $299 \times 299$ input dimensions (requiring pipeline restructuring) and high overfitting probability.
*   **Baseline Control:** A custom CNN was developed from scratch to serve as a baseline, directly measuring the performance jump provided by pretrained feature extractors.

### Why Transfer Learning?
Selected models are pretrained on **ImageNet** (1.2 million real-world images). Because ImageNet shares a dense feature domain with real-world automotive photography, the low-level edge and texture detectors transfer seamlessly. Freezing these core layers drastically reduces trainable parameters, successfully mitigating overfitting on our small dataset.

---

## Data Sources & References

### Sourced Datasets
*   **Used in Project:** [Hugging Face - Comprehensive Car Damage](https://huggingface.co/datasets/SaiVaibhavS/comprehensive-car-damage)
*   **Alternative Reference:** [Kaggle - Car Damage Severity Dataset](https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset/data)
*   **Incomplete Set:** [GitHub - TQVCD Dataset](https://github.com/dxlabskku/TQVCD)

### Academic Foundations
*    Automated vehicle damage classification using the three-quarter view car damage dataset and deep learning approaches
*   [ScienceDirect - Journal Reference (S2405844024100473)](https://www.sciencedirect.com/science/article/pii/S2405844024100473)
*   Deep Learning based Car Damage Detection for Insurance Claim Settlement
*   [IEEE Xplore - Literature Analysis](https://ieeexplore.ieee.org/document/11115206/metrics#metrics)
*   Automated Car Damage Assessment Using Computer Vision: Insurance Company Use Case
*   [MDPI - Applied Sciences Publication](https://www.mdpi.com/2076-3417/14/20/9560)

---

## Model Performance & Selection Matrix

The table below details the evaluation strategy for both included and excluded architectures, alongside their final verified test accuracies where applicable.


| Model | Params | Versions Available | Version Chosen | Params/Training Image | Why This Version | Use? | Reason for Inclusion/Exclusion | Test Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Custom CNN** | ~0.5M | - | - | 290 | Baseline, built from scratch | **YES** | Essential to show value of transfer learning | **52.43%** |
| **MobileNetV3-Small** | 5.4M | V2,V3-Small,V3-Large | Small | 3,130 | Lightweight, real-world deployment | **YES** | Shows deployment awareness | **75.69%** |
| **MobileNetV3-Large** | ~5.5M | V2,V3-Small,V3-Large | Large | ~3,188 | Slightly higher param count than small ver | **YES** | Evaluated for lightweight edge performance | **75.00%** |
| **EfficientNet-B0** | 5.3M | B0-B7 | B0 (not B3/B7) | 3,072 | Best accuracy/size ratio for small data | **YES** | Most efficient architecture | **70.83%** |
| **DenseNet121** | 8M | 121,169,201 | 121 (not 169/201) | 4,638 | Dense connections help small datasets | **YES** | Best suited for small data | **69.79%** |
| **ResNet18** | 11M | 18,34,50,101,152 | 18 (not 50/152) | 6,377 | Smallest ver, sufficient for small data | **YES** | Industry standard baseline | **68.40%** |
| **ResNet50** | 25M | 18,34,50,101,152 | - | 14,493 | Too many params for 1725 training images | **NO** | High overfitting risk | N/A |
| **ConvNeXt-Tiny** | 28M | Tiny,Small,Base | - | 16,231 | Too many params for 1725 training images | **NO** | High overfitting risk | N/A |
| **DenseNet201** | 20M | 121,169,201 | - | 11,594 | Too many params for 1725 training images | **NO** | High overfitting risk | N/A |
| **VGG16** | 138M | 11,13,16,19 | - | 80,000+ | Extremely large, outdated, slow | **NO** | Not worth compute cost | N/A |
| **Inception V3** | 27M | V3 | - | 15,652 | Requires 299x299 input | **NO** | Incompatible pipeline + overfitting risk | N/A |
| **ViT-B/16** | 86M | B/16,L/16 | - | 49,855 | Needs 100k+ images to shine | **NO** | Insufficient data | N/A |
| **EfficientNet-B7** | 66M | B0-B7 | - | 38,260 | Overkill for 1725 training images | **NO** | Overfitting risk | N/A |
| **ResNet152** | 60M | 18,34,50,101,152 | - | 34,782 | Too large for 1725 training images | **NO** | Overfitting risk | N/A |

---

## Key Findings & Insights
*   **Target Achieved:** Only the **MobileNetV3** family crossed our desired project baseline target of **75% accuracy**, with **MobileNetV3-Small** emerging as the top-performing model at **75.69%**.
*   **Transfer Learning Success:** The custom baseline CNN scored just 52.43%, proving that pretrained weights provide a substantial performance boost (~23% absolute accuracy gain) on small data domains.

