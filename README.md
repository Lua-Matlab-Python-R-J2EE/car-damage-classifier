CAR DAMAGE DETECTION (CNN + TRANSFER LEARNING)

AIMS:
-1. (v1) Streamlit app where user can drag and drop an image and the app gives out one of the 6 prediction classes for car damage detection 
    - Build CNN from scratch
    - Use various Nets, like Res-NET, Elastic-NET, etc
    - Accuracy Aim: 75%    
-2. (v2) Highlight the damaged regions on the car (based on YOLO)

CLASSES:
-Front Normal, Front Crushed (minor), Front Breakage (major),
-Front: Normal/Crushed/Breakage => 500/400/500

-Back Normal,  Back Crushed (minor),  Back Breakage (major)
-Back:  Normal/Crushed/Breakage => 300/300/300

-Images are taken at an angle so it can detect issues on the side of the car as well 
 as well as the detection becomes easier.
-2300 total

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
-Params/Training Image ratio → key metric used to justify inclusion/exclusion. Models above ~6,500 params per image excluded where possible
-All selected models pretrained on ImageNet (1.2M real-world photos) → since ImageNet classification and car damage detection both involve real-world photographs, learned features transfer well — making it a strong domain match
-Inception V3 excluded on two grounds — requires 299x299 input (pipeline change) AND overfitting risk
-Custom CNN included as baseline — essential to demonstrate the value of transfer learning
-Transfer learning mitigates overfitting — most layers frozen, only final classifier retrained — but risk remains for very large models on very small datasets

Model             | Params | Versions Available   | Version Chosen    | Params/Training Image | Why This Version                        | Use? | Reason for Inclusion/Exclusion
------------------|--------|----------------------|-------------------|-----------------------|-----------------------------------------|------|--------------------------------
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
------------------|--------|-------------------|-------------------------------------------|------|-----------------------------------------------------------------------------
