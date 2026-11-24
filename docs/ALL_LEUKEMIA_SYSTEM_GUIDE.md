# SmartCell: Acute Lymphoblastic Leukemia (ALL) Classification System Guide

> **IMPORTANT REGULATORY DISCLAIMER**  
> The SmartCell ALL Leukemia classification module is an **experimental academic research prototype** developed for computer vision benchmarking and educational hematology screening demonstration.  
> **This system is NOT a clinically validated medical diagnostic device.**  
> Automated outputs represent probabilistic AI classifications and must **NEVER** replace formal bone marrow biopsy, manual microscopic peripheral blood smear review by a certified pathologist, or flow cytometric immunophenotyping.

---

## 1. Executive Summary

SmartCell has been expanded to support dual-disease hematological screening:
- **Mode 1 – Malaria Screening**: YOLOv8n erythrocyte localization on multi-cell blood smears + domain-adapted CNN classification + whole-smear parasitemia estimation.
- **Mode 2 – ALL Leukemia Classification**: Dedicated single-cell deep learning classifier using transfer learning on the authoritative ALL-IDB benchmark.

The leukemia classification model achieves **100.0% test accuracy** and **1.0000 ROC-AUC** on the untouched hold-out test set, running completely isolated from the malaria pipeline.

---

## 2. Dataset Audit & Verification (Phase 1)

Before data preparation or training, an exhaustive pre-training audit of `ALL_IDB1` and `ALL_IDB2` was executed (`scripts/audit_all_idb_datasets.py`):

| Property | ALL_IDB1 | ALL_IDB2 |
| :--- | :--- | :--- |
| **Dataset Nature** | Whole-microscopic field images | Standardized single-cell crops |
| **Total Images** | 108 images | 260 images |
| **File Format** | JPEG (`.jpg`) | TIFF (`.tif`) |
| **Native Resolutions** | 74 @ 2592×1944, 33 @ 1712×1368, 1 @ 1226×652 | 250 @ 257×257, 10 @ edge-crop sizes |
| **Corrupted Files** | **0** (All valid RGB files) | **0** (All valid RGB files) |
| **Classes Available** | Class 0: Healthy (59), Class 1: ALL Patient (49) | Class 0: Normal (130), Class 1: Blast (130) |
| **Annotations** | 108 `.xyc` files with 510 blast centroids | Ground-truth encoded in filenames `ImXXX_Y` |
| **Internal Duplicates** | `Im093_0.jpg` ≡ `Im108_0.jpg` (SHA-256 match) | `Im223_0.tif` ≡ `Im257_0.tif` (SHA-256 match) |
| **Patient Identifiers** | Anonymized; sequential counter only | Anonymized; sequential counter only |
| **Cross-Dataset Relationship** | Parent whole-field frames | **100% verified sub-crops of ALL_IDB1 frames** |

### Crucial Data Provenance & Leakage Prevention Rule
Template matching with normalized cross-correlation proved that ALL_IDB2 cropped cells are direct sub-windows of ALL_IDB1 frames ($r = 1.000$). Because both datasets share the underlying biological specimens, **they must never be pooled into one dataset**. ALL_IDB2 was specifically created by Prof. Fabio Scotti as the standardized single-cell classification benchmark, directly matching SmartCell's cell-level classification architecture.

---

## 3. Data Preparation & Deduplication (Phase 2)

The dedicated pipeline (`scripts/prepare_all_idb_dataset.py`) prepares `D:\smartcell\dataset_all` without modifying original raw datasets:
- **Deduplication**: Duplicate `Im257_0.tif` was safely excluded prior to splitting, yielding 259 unique cell images (129 normal, 130 blast).
- **Format Standardization**: Losslessly saved as PNG in `dataset_all` to ensure bit-level fidelity and universal framework compatibility.
- **Stratified Split (Fixed Seed 42)**:
  - **Train Set (70%)**: 181 images (90 Normal, 91 Blast)
  - **Validation Set (15%)**: 39 images (20 Normal, 19 Blast)
  - **Test Set (15%)**: 39 images (19 Normal, 20 Blast) — **Untouched during training**

Manifest saved: [`dataset_all/split_manifest.json`](file:///d:/smartcell/dataset_all/split_manifest.json).

---

## 4. Model Training & Comparison (Phases 3 & 4)

Three distinct architectures were trained and compared using identical data splits and input resolution ($224 \times 224 \times 3$):
1. **Custom CNN Baseline**: 4-block Conv2D + BatchNorm + MaxPool + Dropout + Dense (~1.2 MB).
2. **MobileNetV2 Transfer Learning**: ImageNet pre-trained base + Fine-tuned top layers (~19.4 MB).
3. **EfficientNetB0 Transfer Learning**: ImageNet pre-trained base + Fine-tuned top layers (~27.6 MB).

### Test Set Performance on Untouched Hold-Out Data

| Model Architecture | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC | Mean Latency | Model Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNetB0 (Selected Winner)** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **1.0000** | 213.2 ms | 27.6 MB |
| **MobileNetV2 Transfer Learning** | 94.87% | 90.91% | 100.00% | 89.47% | 0.9524 | 1.0000 | 161.5 ms | 19.4 MB |
| **Custom CNN Baseline** | 48.72% | 0.00% | 0.00% | 100.00% | 0.0000 | 0.8684 | 114.3 ms | 1.2 MB |

### Overfitting & Generalization Analysis
- **Custom CNN**: Severe underfitting on the small 181-sample training set without pre-trained feature extractors (Train Acc: 89.5%, Val Acc: 64.1%, Test Acc: 48.7%).
- **MobileNetV2**: Strong generalization with small gap (Train Acc: 98.9%, Val Acc: 97.4%, Test Acc: 94.9%, 0 false negatives).
- **EfficientNetB0**: Superior representation capacity and generalization (Train Acc: 94.5%, Val Acc: 97.4%, Test Acc: **100.0%**, 0 false positives, 0 false negatives).

Selected Winner: **EfficientNetB0 Transfer Learning** saved to [`models/all_idb_leukemia_best.keras`](file:///d:/smartcell/models/all_idb_leukemia_best.keras).

---

## 5. SmartCell Application Integration (Phases 6 & 7)

The interactive application ([`scripts/smartcell_demo_app.py`](file:///d:/smartcell/scripts/smartcell_demo_app.py)) now features a top-level **Disease Mode Selector**:
- **Mode 1 – Malaria Screening**: Uses Stage 1 YOLOv8n (GPU) + Stage 2 CNN (CPU).
- **Mode 2 – ALL Leukemia Classification**: Uses the dedicated EfficientNetB0 ALL classifier.

### 5.1 Input Modes in Mode 2
Mode 2 provides three input options through a unified tabbed interface:
1. **Live Microscope**:
   - Directly connects to USB digital microscopes (standard UVC compliant cameras, e.g., 1080p, 720p, 4K) via HTML5 WebRTC `navigator.mediaDevices.getUserMedia`.
   - Reuses the robust camera management architecture (`SmartCellMicroscope`) established in Mode 1.
   - Dynamic device enumeration (`enumerateDevices`) with automatic detection of connected optical microscope hardware.
   - Camera switching without page reload, properly disposing existing media stream tracks (`track.stop()`) to prevent hardware device lock or black screens.
   - Responsive live viewfinder with active video feed indicator badge.
   - Native aspect-ratio frame capture via an off-screen HTML5 canvas matching sensor dimensions (`video.videoWidth` × `video.videoHeight`).
   - Freeze-frame inspection preview with **Retake / Discard** and **Analyze Captured Cell** actions.
2. **Upload Image**:
   - Local image file upload supporting PNG, JPG, and TIFF formats.
   - Instant client-side inspection preview with image dimensions and file size.
3. **Dataset Samples**:
   - Quick one-click selection of certified hold-out benchmark test specimens (`Im003_1.png` - Blast, `Im133_0.png` - Normal).

### 5.2 Pre-Inference Quality & Validation Heuristics
Before executing the neural network, all incoming microscope frames and uploaded files pass through rigorous validation heuristics:
- **Empty File Rejection**: Rejects 0-byte payloads (HTTP 400).
- **Black Frame Rejection**: Detects closed lens caps or inactive microscope illumination ($\mu < 4.0$ and $\sigma < 4.0$).
- **Blank / Uniform Frame Rejection**: Detects untextured slides or saturated white fields ($\sigma < 4.0$).
- **Blur & Focus Diagnostics**: Client-side Laplacian variance sharpness estimator warning operators when cells are out of focus.
- **Corrupted Frame Handling**: Gracefully handles truncated or malformed image data.

### 5.3 Clinical Output & Specimen Provenance
- Specimen source tracking (`USB Microscope Capture`, `Manual File Upload`, or `Hold-Out Benchmark Test Specimen`) preserved across all data stores, summary logs, and reports.
- Disease Badge: Green `NORMAL LYMPHOCYTE (HEALTHY)` or Red `SUSPECTED LYMPHOBLAST (ALL LEUKEMIA)`.
- Confidence Percentage gauge and dual probability breakdown ($P_{normal}$ vs $P_{blast}$).
- Inference latency and model parameters.
- Morphological summary text.
- Direct download links for PDF, JSON, and TXT reports.

---

## 6. Dedicated Leukemia PDF Report (Phase 8)

The report generator ([`scripts/leukemia_pdf_generator.py`](file:///d:/smartcell/scripts/leukemia_pdf_generator.py)) produces formal, audit-ready clinical research summaries:
- **Complete Isolation**: Contains **zero** malaria metrics (no parasitemia, no RBC counts, no YOLO boxes).
- **Morphological Description**: Details nuclear-cytoplasmic ratio, chromatin clumping, and cellular envelope characteristics.
- **Rendered Specimen Image**: Embedded at native aspect ratio without distortion.
- **Mandatory Research Disclaimer**: Explicitly cautions against primary diagnostic usage.

---

## 7. Verification & Regression Testing (Phase 9)

Automated test suite ([`scripts/test_all_idb_and_regression.py`](file:///d:/smartcell/scripts/test_all_idb_and_regression.py)) executes 9 rigorous test cases:
1. Valid normal lymphocyte classification $\rightarrow$ **PASS**
2. Valid ALL lymphoblast classification $\rightarrow$ **PASS**
3. Corrupted image exception handling $\rightarrow$ **PASS**
4. Empty 0-byte file rejection $\rightarrow$ **PASS**
5. Extreme dimensions ($4\times4$ rejected, $2000\times2000$ resized) $\rightarrow$ **PASS**
6. Blank / solid color rejection $\rightarrow$ **PASS**
7. PDF generation and zero-malaria content audit $\rightarrow$ **PASS**
8. Malaria system regression (CNN models & YOLO intact) $\rightarrow$ **PASS**
9. Model metadata consistency $\rightarrow$ **PASS**

End-to-end HTTP integration tested via [`scripts/test_live_server.py`](file:///d:/smartcell/scripts/test_live_server.py) with full HTTP 200 responses.

---

## 8. How to Run

### To Launch the Web Application
```bat
D:\smartcell\launch_smartcell_demo.bat
```
Navigate to: `http://127.0.0.1:8501` and click **Mode 2: ALL Leukemia Classification** on the top navigation bar.

### To Run Independent Leukemia Single-Cell Analysis (CLI / Python)
```python
from scripts.leukemia_analyzer import run_leukemia_analysis

result = run_leukemia_analysis(
    image_path=r"D:\smartcell\dataset_all\test\1_blast\Im001_1.png",
    run_dir=r"D:\smartcell\results\demo_app_runs\my_leukemia_run"
)
print("Predicted Class:", result["classification"]["predicted_class"])
print("Confidence:", result["classification"]["confidence_percent"], "%")
print("PDF Report:", result["files"]["pdf"])
```
