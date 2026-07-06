# Car Damage Detection Pipeline (CNN + Transfer Learning)

A comprehensive, production-grade deep learning pipeline designed to automatically classify structural vehicle damage into 6 distinct categories. This repository features a decoupled architecture separating the model core, an asynchronous FastAPI backend server, and an interactive Streamlit web application dashboard.

---

## Live AI Production Deployments
* **[Car Damage Classifier App](https://car-damage-classifier-v1.streamlit.app/)** - Production-grade deep learning pipeline classifying vehicle damage into 6 categories.

---

## Repository Architecture

The project is structured to keep data processing, backend microservices, and presentation layers modular, enabling clean cross-platform local execution:

```text
.
├── config.py                       # Global paths, device overrides, and hyperparameters
├── core/
│   └── model_utils.py              # Shared production model loading and prediction core
├── fastapi_server/
│   ├── server.py                   # Asynchronous REST API serving model predictions
│   ├── requirements.txt            # Dedicated backend server API dependencies
│   └── README.md                   # Backend operational server guides
├── data/
│   └── car_damage/                 # Hugging Face Dataset cache binaries (.arrow structures)
├── models/                         # Stored model architecture weights (.pth binaries)
├── notebooks/                      # Model development workspaces and automation runners
│   ├── run_all.py                  # Automated sequential notebook batch executor
│   ├── run_all_py_instructions.txt # Background process operational reference guide
│   └── set_up_checks.ipynb         # Local path and package environment scratchpad
├── src/
│   └── helpers.py                  # Custom CNN architecture classes & training loop routines
├── streamlit_app/
│   ├── app.py                      # Interactive frontend web dashboard client
│   └── requirements.txt            # Dedicated frontend cloud application dependencies
└── requirements.txt                # Master repository workstation dependencies file
```

---

## Project Overview & Goals

This computer vision project is engineered to evaluate custom convolutional networks against established pre-trained transfer learning architectures to maximize performance under strict workstation constraints.

### Phase 1 (v1) - Current Release
*   **Interactive Web App:** A Streamlit application featuring drag-and-drop image uploads.
*   **Classification:** Accurately categorises images into one of 6 distinct vehicle condition classes.
*   **Architectures:** Baseline custom CNN built from scratch evaluated against Transfer Learning networks (e.g., ResNet, EfficientNet variants).
*   **v1 Strategy:** Freeze all base layers and retrain the classification head only using the best weights tracking over 20 training epochs.
*   **Target Performance:** Minimum accuracy threshold of **75%**.

### Phase 2 (v2) - Future Roadmap
*   **Object Detection:** Integrate YOLO frameworks to automatically isolate and highlight exact pixel regions containing vehicle structural damage.
*   **Fine-Tuning:** Unfreeze and retrain the deep feature extraction layers of selected transfer learning networks to sharpen classification boundary lines.

---

## Dataset & Visual Challenges

### Data Breakdown
The model utilizes a total of **2,300 RGB images** sourced from Hugging Face, partitioned cleanly using a **75% Train (1,725 images) / 12.5% Validation / 12.5% Test** split.
*   **Front View (1,400 images total):** Normal (500) | Crushed/Minor (400) | Breakage/Major (500)
*   **Rear View (900 images total):** Normal (300) | Crushed/Minor (300) | Breakage/Major (300)

### Technical Constraints & Layout Strategy
*   **Camera Angles:** Images are taken at specific three-quarter view angles so the pipeline can detect structural issues on the side of the car as well, streamlining feature extraction.
*   **Pipeline Engineering:** Dedicated, independent augmentation pipelines are used for training vs. evaluation/test runs.
*   **Class Ambiguity Note:** The definition of classification classes contains inherent visual ambiguity. For instance, images around index 1400 (`Rear Breakage`) and index 1700 (`Rear Crushed`) present highly similar visual damage patterns, making the classification boundary highly complex.

---

## Model Selection & Transfer Learning Strategy

### Resource Constraints
Training was conducted entirely on local **CPU infrastructure**. Consequently, the architecture selection leans heavily toward lightweight, computationally efficient networks.

### Parameters-to-Image Ratio Evaluation
With only 1,725 training images available, architectures exceeding ~10M parameters pose an extreme risk of overfitting. 
*   **Selection Metric:** Models maintaining a strict ratio below **~6,500 parameters per training image** were prioritized.
*   **Inception V3 Exclusion:** Explicitly excluded due to its rigid $299 \times 299$ input dimensions (requiring pipeline restructuring) and high overfitting probability.
*   **Baseline Control:** A custom CNN was developed from scratch to serve as a baseline, directly measuring the performance jump provided by pretrained feature extractors.

### Why Transfer Learning?
Selected models are pretrained on **ImageNet** (1.2 million real-world images). Because ImageNet shares a dense feature domain with real-world automotive photography, the low-level edge and texture detectors transfer seamlessly. Freezing these core layers drastically reduces trainable parameters, successfully mitigating overfitting on our small dataset.

---

## Model Performance & Selection Matrix

The table below details the structural evaluation strategy for both included and excluded architectures, alongside their final verified test accuracies where applicable.

| Model | Params | Versions Available | Version Chosen | Params/Training Image | Why This Version | Use? | Reason for Inclusion/Exclusion | Test Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Custom CNN** | ~0.5M | - | - | 290 | Baseline, built from scratch | **YES** | Essential to show value of transfer learning | **52.43%** |
| **MobileNetV3-Small** | 5.4M | V2,V3-Small,V3-Large | Small | 3,130 | Lightweight, real-world deployment | **YES** | Shows deployment awareness | **75.69%** |
| **MobileNetV3-Large** | ~5.5M | V2,V3-Small,V3-Large | Large | ~3,188 | Slightly higher param count than small ver | **YES** | Evaluated for lightweight edge performance | **75.00%** |
| **EfficientNet-B0** | 5.3M | B0-B7 | B0 (not B3/B7) | 3,072 | Best accuracy/size ratio for small data | **YES** | Most efficient architecture | **70.83%** |
| **DenseNet121** | 8M | 121,169,201 | 121 (not 169/201) | 4,638 | Dense connections help small datasets | **YES** | Best suited for small data | **69.79%** |
| **ResNet18** | 11M | 18,34,50,101,152 | 18 (not 50/152) | 6,377 | Smallest ver, sufficient for small data | **YES** | Industry standard baseline | **68.40%** |
| *ResNet50* | 25M | 18,34,50,101,152 | - | 14,493 | Too many params for 1725 training images | **NO** | High overfitting risk | N/A |
| *ConvNeXt-Tiny* | 28M | Tiny,Small,Base | - | 16,231 | Too many params for 1725 training images | **NO** | High overfitting risk | N/A |
| *DenseNet201* | 20M | 121,169,201 | - | 11,594 | Too many params for 1725 training images | **NO** | High overfitting risk | N/A |
| *VGG16* | 138M | 11,13,16,19 | - | 80,000+ | Extremely large, outdated, slow | **NO** | Not worth compute cost | N/A |
| *Inception V3* | 27M | V3 | - | 15,652 | Requires 299x299 input | **NO** | Incompatible pipeline + overfitting risk | N/A |
| *ViT-B/16* | 86M | B/16,L/16 | - | 49,855 | Needs 100k+ images to shine | **NO** | Insufficient data | N/A |
| *EfficientNet-B7* | 66M | B0-B7 | - | 38,260 | Overkill for 1725 training images | **NO** | Overfitting risk | N/A |
| *ResNet152* | 60M | 18,34,50,101,152 | - | 34,782 | Too large for 1725 training images | **NO** | Overfitting risk | N/A |

### Key Findings & Insights
*   **Target Achieved:** Only the **MobileNetV3** family successfully crossed our desired baseline target of **75% accuracy**, with **MobileNetV3-Small** emerging as the top-performing production model at **75.69%**.
*   **Transfer Learning Success:** The custom baseline CNN scored just 52.43%, proving that pretrained weights provide a substantial performance boost (~23% absolute accuracy gain) on small data domains.

---

## Workstation Setup & Local Execution

### Environment Installation
Activate your local virtual environment and execute the master requirement installation from the root directory:
```bash
pip install -r requirements.txt
```

### Running the Streamlit Web UI Locally
To launch the interactive front-end web app client:
```bash
python3 -m streamlit run streamlit_app/app.py
```

### Running the FastAPI Backend Server Locally
To fire up the standalone asynchronous REST API server layer:
```bash
uvicorn fastapi_server.server:app --reload
```

### Automated Overnight Batch Training
To launch the automated batch-processing subsystem to execute all individual model notebooks sequentially in the background without manual oversight:
```bash
cd notebooks
nohup python3 run_all.py > output.log 2>&1 &
```
*For complete error logs review or emergency cancellation procedures, consult `notebooks/run_all_py_instructions.txt` directly.*

---

## Data Sources & Literature Context

### Sourced Datasets
*   **Used in Project:** [Hugging Face - Comprehensive Car Damage](https://huggingface.co/datasets/SaiVaibhavS/comprehensive-car-damage)
*   **Alternative Reference:** [Kaggle - Car Damage Severity Dataset](https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset/data)
*   **Incomplete Set:** [GitHub - TQVCD Dataset](https://github.com/dxlabskku/TQVCD)

### Academic Foundations & Literature Baselines
*   Lee, D., Lee, J., & Park, E. (2024). Automated vehicle damage classification using the three-quarter view car damage dataset and deep learning approaches. Heliyon, 10(14), e34016. https://doi.org/10.1016/j.heliyon.2024.e34016


