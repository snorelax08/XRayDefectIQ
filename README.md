<div align="center">

<pre>
██╗  ██╗██████╗  █████╗ ██╗   ██╗██████╗ ███████╗███████╗███████╗ ██████╗████████╗██╗ ██████╗
╚██╗██╔╝██╔══██╗██╔══██╗╚██╗ ██╔╝██╔══██╗██╔════╝██╔════╝██╔════╝██╔════╝╚══██╔══╝██║██╔═══██╗
 ╚███╔╝ ██████╔╝███████║ ╚████╔╝ ██║  ██║█████╗  █████╗  █████╗  ██║        ██║   ██║██║   ██║
 ██╔██╗ ██╔══██╗██╔══██║  ╚██╔╝  ██║  ██║██╔══╝  ██╔══╝  ██╔══╝  ██║        ██║   ██║██║▄▄ ██║
██╔╝ ██╗██║  ██║██║  ██║   ██║   ██████╔╝███████╗██║     ███████╗╚██████╗   ██║   ██║╚██████╔╝
╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═════╝ ╚══════╝╚═╝     ╚══════╝ ╚═════╝   ╚═╝   ╚═╝ ╚══▀▀═╝
</pre>

### AI-Powered Industrial Radiographic Defect Detection

[![CI](https://github.com/Sanskar121543/XRayDefectIQ/actions/workflows/ci.yml/badge.svg)](https://github.com/Sanskar121543/XRayDefectIQ/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![YOLOv8](https://img.shields.io/badge/YOLO-Ultralytics-00FFFF?style=flat-square&logo=yolo&logoColor=black)](https://ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)
[![MySQL](https://img.shields.io/badge/MySQL-8.x-4479A1?style=flat-square&logo=mysql&logoColor=white)](https://mysql.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

<br>

*Manual radiographic inspection is slow, inconsistent, and impossible to scale. XRayDefectIQ changes that.*

![Overall Process](https://raw.githubusercontent.com/Sanskar121543/XRayDefectIQ/main/assets/screenshots/Overall%20process.jpeg)

</div>

---

## Table of Contents

- [What It Does](#what-it-does)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Tech Stack](#tech-stack)
- [Use Cases](#use-cases)
- [Roadmap](#roadmap)
- [License](#license)

---

## What It Does

XRayDefectIQ is an end-to-end industrial inspection platform that ingests X-ray images, runs them through a trained YOLO detection engine, isolates and characterizes every defect found, and commits structured records to a MySQL database — all through a single interactive UI. No CV expertise required at runtime.

It's built for real NDT workflows: weld inspection, casting analysis, aerospace radiography, and automated manufacturing QA pipelines where accuracy, traceability, and speed are non-negotiable.

---

## System Architecture

![System Diagram](https://raw.githubusercontent.com/Sanskar121543/XRayDefectIQ/main/assets/screenshots/System%20diagram.jpeg)

<details>
<summary>Text representation</summary>

```
┌─────────────────────────────────────────────────────────────┐
│                      X-ray Image Input                       │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│               Image Enhancement Pipeline                     │
│   Grayscale · CLAHE · Contrast · Sharpen · Denoise · Edge   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   YOLO Detection Engine                      │
│          Bounding Boxes · Confidence Scores · Multi-defect  │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Defect ROI + Feature Extraction                 │
│   Texture · Entropy · Skewness · Kurtosis · Pixel Stats     │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 Inspection Decision Layer                    │
│              Accept / Reject · Inspector Remarks            │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│               MySQL Defect Record Storage                    │
│      Full traceability · Defect ID · Component · Date       │
└─────────────────────────────────────────────────────────────┘
```

</details>

| Architecture | Application Flow | Class Structure |
|:---:|:---:|:---:|
| ![Architecture Diagram](https://raw.githubusercontent.com/Sanskar121543/XRayDefectIQ/main/assets/screenshots/Architecture%20diagram.jpeg) | ![Flow Diagram](https://raw.githubusercontent.com/Sanskar121543/XRayDefectIQ/main/assets/screenshots/Flow%20diagram.jpeg) | ![Class Diagram](https://raw.githubusercontent.com/Sanskar121543/XRayDefectIQ/main/assets/screenshots/Class%20diagram.jpeg) |

---

## Features

### 🔍 Automated Defect Detection
- Upload any X-ray image; YOLO handles the rest
- Annotated output with bounding boxes and per-defect confidence scores
- Multi-defect support — catches everything in a single pass
- ROI extraction per detected defect for downstream analysis

### 🖼️ Image Processing Suite

Eight enhancement modes to maximize defect visibility before inference:

| Mode | Purpose |
|---|---|
| Grayscale Conversion | Normalize radiographic input |
| Histogram Equalization | Global contrast correction |
| CLAHE | Localized adaptive contrast enhancement |
| Contrast Adjustment | Manual intensity tuning |
| Sharpening | Edge accentuation for subtle defects |
| Noise Reduction | Remove sensor and scatter artifacts |
| Blur Filtering | Smooth out unwanted high-frequency noise |
| Edge Detection | Highlight structural boundaries |

### 📐 Feature Extraction

For every detected defect region, the platform computes:
- **Statistical metrics** — Mean, Variance, Skewness, Kurtosis
- **Texture descriptors** — Entropy, intensity distribution
- **Spatial geometry** — Bounding box dimensions, centroid, area

### 🧠 Model Training Console

Full model lifecycle management without leaving the app:
- Upload custom labeled datasets directly from the UI
- Trigger YOLO training jobs and watch live progress
- Save trained checkpoints and hot-swap between models
- Supports iterative retraining as defect libraries expand

### 🗄️ Inspection Database

Every inspection writes a structured, queryable record:

| Field | Description |
|---|---|
| Defect ID | Unique identifier per detection event |
| Component Name | Part under inspection |
| Asset / Satellite | System or assembly context |
| Component Code | Internal part numbering |
| Defect Type | Classification label from the model |
| Accept / Reject | Final disposition |
| Inspector Remarks | Free-text annotation |
| Timestamp | Full datetime of inspection |

---

## Model Performance

Detection metrics from the trained YOLO model evaluated on the held-out test set.

| Training Results | Results Overview |
|:---:|:---:|
| ![Training Results](https://raw.githubusercontent.com/Sanskar121543/XRayDefectIQ/main/assets/screenshots/Training%20results.jpeg) | ![Results](https://raw.githubusercontent.com/Sanskar121543/XRayDefectIQ/main/assets/screenshots/Results.jpeg) |

**Confusion Matrix**

| Raw | Normalized |
|:---:|:---:|
| ![Confusion Matrix](https://raw.githubusercontent.com/Sanskar121543/XRayDefectIQ/main/assets/screenshots/Confusion%20Matrix.jpeg) | ![Confusion Matrix Normalized](https://raw.githubusercontent.com/Sanskar121543/XRayDefectIQ/main/assets/screenshots/Confusion%20matrix%20normalized.jpeg) |

**Confidence & Recall Curves**

| F1–Confidence | Precision–Confidence |
|:---:|:---:|
| ![F1 Confidence Curve](https://raw.githubusercontent.com/Sanskar121543/XRayDefectIQ/main/assets/screenshots/F1%20Confidence%20curve.jpeg) | ![Precision Confidence Curve](https://raw.githubusercontent.com/Sanskar121543/XRayDefectIQ/main/assets/screenshots/Precision%20Confidence%20Curve.jpeg) |

| Recall–Confidence | Precision–Recall |
|:---:|:---:|
| ![Recall Confidence Curve](https://raw.githubusercontent.com/Sanskar121543/XRayDefectIQ/main/assets/screenshots/Recall%20Confidence%20curve.jpeg) | ![Precision Recall Curve](https://raw.githubusercontent.com/Sanskar121543/XRayDefectIQ/main/assets/screenshots/Precision%20Recall%20Curve.jpeg) |

---

## Project Structure

```
XRayDefectIQ/
│
├── app.py                  # Main Streamlit interface — all UI and orchestration logic
├── training_handler.py     # YOLO training pipeline and model management
├── mysql_handler.py        # Database connection, queries, and record insertion
├── DATABASE_SETUP.sql      # Schema creation — run once to initialize
│
├── models/                 # Trained model weights (.pt files)
├── Training images/        # Labeled dataset for custom training runs
└── assets/screenshots/     # Architecture diagrams and performance metrics
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- MySQL 8.x (running locally or remotely)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/Sanskar121543/XRayDefectIQ.git
cd XRayDefectIQ

# 2. Install dependencies
pip install -r requirements.txt
```

### Configuration

Open `mysql_handler.py` and update your MySQL credentials before running anything:

```python
host     = "localhost"
user     = "your_user"
password = "your_password"
database = "xraydefectiq"
```

### Database Setup

```bash
mysql -u root -p < DATABASE_SETUP.sql
```

### Launch

```bash
streamlit run app.py
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Interface | Streamlit |
| Detection | Ultralytics YOLO (YOLOv8) |
| Image Processing | OpenCV, NumPy |
| Data Handling | Pandas |
| Storage | MySQL |
| Language | Python 3.9+ |

---

## Use Cases

- **Weld Inspection** — Detect porosity, cracks, and incomplete fusion in weld joints
- **Casting Analysis** — Identify voids, inclusions, and shrinkage defects in metal castings
- **Aerospace Radiography** — Audit structural components against stringent tolerance specs
- **Manufacturing QA** — Integrate into production lines for automated pass/fail decisions
- **NDT Research** — Build and benchmark custom defect detection datasets

---

## Roadmap

- [ ] Cloud deployment (AWS / GCP / Azure)
- [ ] Real-time camera and live feed inspection
- [ ] PDF inspection report generation
- [ ] Role-based access control (RBAC)
- [ ] Explainable AI — Grad-CAM heatmap overlays
- [ ] Multi-model ensemble inference

---

## Why This Exists

Manual radiographic inspection has three problems: it's slow, it's subjective, and it doesn't scale. A single inspector reviewing hundreds of images per shift introduces fatigue-driven variance. There's no audit trail. There's no feedback loop into process improvement.

XRayDefectIQ is built to fix all three. Fast inference over any X-ray image. Consistent, model-driven detection that doesn't drift with fatigue. And a database-backed record for every inspection, so quality teams can actually learn from the data they're collecting.

---

## License

[MIT](LICENSE) — use it, fork it, build on it.

---

<div align="center">

Built by **Sanskar Shimpi**

</div>
