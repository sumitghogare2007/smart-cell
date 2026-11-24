# SmartCell — College Project Live Demonstration Guide (5–7 Minute Sequence)

> **RESEARCH PROTOTYPE DISCLAIMER**  
> **SmartCell** is an experimental academic research prototype developed for a college engineering project. It is **not** a clinically validated medical diagnostic device and must never be used as a substitute for professional microscopic examination or clinical diagnosis.

---

## 1. Executive Presentation Summary: Two-Stage Architecture & Dual-Environment Design

When introducing the system to evaluators, use this concise architectural explanation (approx. 45 seconds):

1. **Why a Two-Stage Pipeline (`YOLOv8n` $\rightarrow$ `Custom CNN`)?**
   - **Stage 1 — Candidate Cell Localization (`YOLOv8n`)**: Full-resolution thin blood-smear images ($5312 \times 2988\text{ px}$) contain $150\text{–}300$ closely packed cells per microscope field. On the NIH-NLM Thin Blood Smears dataset (`33` patients, `165` images, `34,264` polygons), YOLOv8n achieves **96.44% $\text{mAP@50}$** on Uninfected RBCs and localizes **97.93% (`4,695 / 4,794`)** of all ground-truth RBCs on the 25 held-out test images, but struggles to directly classify rare *Plasmodium falciparum* ring forms (`2.07%` class-specific $\text{mAP@50}$ due to a `29.6 : 1` class imbalance).
   - **Stage 2 — High-Specificity Infection Classification (`Custom CNN`)**: Instead of trusting YOLO's class label for infection status, Stage 1 acts as a class-agnostic erythrocyte localizer (excluding `class 2 [WBC]` detections and suppressing duplicate boxes at $\text{IoU} \ge 0.50$). Each native-resolution RBC crop ($+4\text{ px}$ context padding) is passed to our dedicated Stage 2 CNN classifier (`malaria_cnn_finetuned_raw_best.keras`), achieving **98.35% accuracy**, **96.28% sensitivity**, **98.53% specificity**, and **0.9016 Parasitized F1** on ground-truth full-smear test crops (and **1.12% Mean Absolute Error** on end-to-end parasitemia across all 25 test images).

2. **Why Separate GPU (`venv_yolo_gpu`) and CPU (`venv`) Environments?**
   - **Native Windows Framework Compatibility**: TensorFlow $\ge 2.11$ (`TensorFlow 2.21.0` in `D:\smartcell\venv`) dropped native Windows CUDA GPU support (requiring WSL2 for GPU execution), whereas PyTorch `2.14.1+cu130` (`D:\smartcell\venv_yolo_gpu`) natively supports the laptop's **NVIDIA GeForce RTX 2050 GPU (`CUDA device=0`, 4 GB VRAM)**.
   - **Zero-Conflict Isolation**: Keeping `venv_yolo_gpu` (PyTorch + Ultralytics on CUDA:0) and `venv` (TensorFlow 2.21.0 + Keras on CPU) strictly isolated avoids `numpy`/`protobuf`/`typing-extensions` dependency conflicts and guarantees deterministic reproduction of every phase checkpoint. The orchestrator bridges both environments seamlessly via structured JSON and PNG crop artifacts in ~16 seconds per full-resolution blood smear.

---

## 2. Pre-Demo Checklist (30 Seconds Before Evaluators Arrive)

1. Open a terminal or File Explorer at `D:\smartcell`.
2. Confirm no other service is occupying port `8501`.
3. Keep these two sample test images ready to mention (or upload directly from `D:\smartcell\yolo_malaria\images\test\`):
   - **High Parasitemia Smear**: `D:\smartcell\yolo_malaria\images\test\IMG_20150819_133447.jpg` (Held-out test Patient `209C68P29N_ThinF`; Ground Truth: `71` Parasitized / `135` Uninfected / `3` WBCs = `206` RBCs, **`34.47%` parasitemia**).
   - **Healthy Uninfected Control Smear**: `D:\smartcell\yolo_malaria\images\test\IMG_20151207_114038.jpg` (Held-out test Patient `377C238NThinF`; Ground Truth: `0` Parasitized / `208` Uninfected = **`0.00%` parasitemia**).

---

## 3. Step-by-Step 5–7 Minute Live Demonstration Sequence

### Minute 0:00 – 0:45 | Step 1: Launch the Application & Verify Hardware Status
1. Double-click **`D:\smartcell\launch_smartcell_demo.bat`** (or run in PowerShell):
   ```powershell
   D:\smartcell\venv\Scripts\python.exe D:\smartcell\scripts\smartcell_demo_app.py --host 127.0.0.1 --port 8501
   ```
2. Open **`http://127.0.0.1:8501`** in the browser.
3. **Point out to evaluators**:
   - The red **Research Prototype Notice** banner at the top confirming this is an academic screening prototype, not a clinical device.
   - The live hardware badges in the top-right header:
     - `Stage 1 GPU: NVIDIA GeForce RTX 2050 (3299.7/4095.5 MB VRAM)`
     - `Stage 2: TensorFlow 2.21.0 (venv CPU)`

---

### Minute 0:45 – 1:45 | Step 2: Upload a Full-Resolution Blood-Smear Image & Explain Stage 1 YOLO Detection
1. In **Card 1 (Blood-Smear Image Input)** on the left sidebar, click **Choose File** and upload:
   `D:\smartcell\yolo_malaria\images\test\IMG_20150819_133447.jpg`
   *(Alternatively, select `IMG_20150819_133447.jpg` from the **Preset Held-Out Test Smear** dropdown).*
2. **Explain Stage 1 while configuring**:
   - The input image is a raw $5312 \times 2988\text{ px}$ thin blood smear microscope field.
   - Stage 1 runs YOLOv8n (`best.pt`) on the RTX 2050 GPU (`device=0`) with confidence threshold `0.25`, NMS IoU `0.45`, and class-agnostic deduplication IoU `0.50`.
   - Any detected `class 2 (WBC)` boxes are separated and excluded from the RBC count, and any tiny/clipped artifacts ($< 16\text{ px}$) are rejected before Stage 2.

---

### Minute 1:45 – 2:45 | Step 3: Select CNN Variant, Crop Mode & Thresholds (Domain Adaptation Demo)
1. In **Card 2 (Stage 2 Classifier & Crop Mode)**:
   - Show the three selectable CNN checkpoints:
     1. **`Fine-Tuned Raw CNN (Phase 6 Domain-Adapted — Recommended)`** (`malaria_cnn_finetuned_raw_best.keras`)
     2. **`Fine-Tuned Masked CNN (Phase 6 Elliptical-Masked)`** (`malaria_cnn_finetuned_masked_best.keras`)
     3. **`Original Baseline CNN (Phase 1 Single-Cell Dataset)`** (`malaria_cnn_best.keras`)
   - Keep **`Fine-Tuned Raw CNN`** and **`raw — Rectangular RGB Crop (4px padding)`** selected.
2. **Briefly explain the Phase 6 Domain Adaptation breakthrough**:
   - The original Phase 1 CNN was trained on segmented single cells with pure black `[0, 0, 0]` backgrounds, so bright pink plasma corners and neighboring cell margins in raw rectangular full-smear crops caused false positives (`108 / 195 = 55.38%` on this image).
   - Fine-tuning on `24,112` domain-matched rectangular crops from the 23 training patients resolved this corner sensitivity, cutting false positives by **83.4%** across the test set without needing artificial corner masking.

---

### Minute 2:45 – 4:15 | Step 4: Run Analysis, Explain Separate Counts, Parasitemia & Confidence Scores
1. Click **"Run Two-Stage Blood-Smear Analysis"** (`~16 seconds` total: `~6.8s` GPU Stage 1 + `~9.3s` CPU Stage 2).
2. Walk through the **6 Summary KPI Cards** at the top of the right panel:
   - **Predicted Parasitized RBCs**: **`74`**
   - **Predicted Uninfected RBCs**: **`121`** (out of **`195`** valid candidate RBC crops)
   - **Experimental Parasitemia**: **`37.95%`** (`74 / 195 classified RBCs`), closely tracking the ground-truth `34.47%` (`71 / 206` RBCs).
   - **Excluded WBC Detections**: **`0`** (shown separately so leukocytes never distort parasitemia).
   - **Rejected / Duplicate Boxes**: **`0`** (shown separately).
   - **Processing Time**: **`~16.1s`** (`S1: ~6.8s` on RTX 2050 using only **`23.73 MB` Peak VRAM** | `S2: ~9.3s` on CPU).
3. Point out the **Active Configuration & Confidence Telemetry** panel:
   - **YOLO Detection Confidence**: Mean **`63.6%`** (`25.0%–92.3%`).
   - **CNN Overall Confidence**: Mean **`97.2%`**, Median **`100.0%`**.
   - **Audit Comparison**: Shows both the **Raw Crop (`37.95%`, `74` Par)** and **Elliptical-Masked Crop (`36.41%`, `71` Par)** estimates side by side.

---

### Minute 4:15 – 5:15 | Step 5: Inspect Visual Outputs & Download Complete Analysis Report (PDF)
1. **Tab 1 — Annotated Full-Smear Image**:
   - Show the full microscope field with **Red bounding boxes + confidence labels** on predicted Parasitized RBCs and **Green bounding boxes** on Uninfected RBCs.
2. **Tab 2 — Candidate RBC Contact Sheet**:
   - Show the 25-cell high-resolution zoomed grid (`contact_sheet_IMG_20150819_133447.jpg`) where intracellular ring trophozoites inside `cell_0001`, `cell_0003`, `cell_0008`, `cell_0009`, `cell_0011` are clearly visible.
3. **Tab 3 — Per-Cell Predictions & Confidence (`195` cells)**:
   - Filter by **"Parasitized RBCs Only"** to inspect individual cell thumbnails, `[x1, y1, x2, y2]` coordinates, YOLO confidence, and CNN probabilities.
4. **Download & Review the Primary PDF Analysis Report (`SmartCell_Report_<image_name>.pdf`)**:
   - Point out the green **PDF Report Ready** confirmation banner and click the primary **"Download Complete Report (PDF) — SmartCell_Report_IMG_20150819_133447.pdf"** button.
   - Open the downloaded concise **2-page PDF** (**Section F per-cell table omitted** so the report is compact and immediately readable; per-cell rows remain in the CSV export):
     - **Page 1**: Report Title & Research Prototype Notice, **A. Report & Specimen Identification**, **B. Executive Summary** (`195` valid RBCs, `74` Parasitized, `121` Uninfected, `37.95%` parasitemia, excluded WBCs, rejected boxes, processing duration), **C. Model & Configuration Details**, **D. Confidence Statistics & Processing Telemetry**, and **E. Annotated Blood-Smear Field Overview** (`Figure 1` aspect-ratio-preserved blood-smear image with AI prediction caption).
     - **Page 2 (`Page X of Y` footer)**: **G. Audit & Exclusion Summary**, **H. Dataset Ground-Truth Reference Comparison** (`71 / 206 = 34.47%` GT vs. `74 / 195 = 37.95%` predicted, included only when verified annotations exist), and the prominent **Final AI Screening Result** section (*"The AI flagged 74 potentially parasitized red blood cells. Estimated parasitemia: 37.95%."* followed immediately by *"This AI screening result does not confirm or rule out malaria in a patient. Clinical assessment and appropriate diagnostic testing are required."*).
5. **Optional Technical Exports (CSV / JSON / TXT / JPG)**:
   - Point out that **Download Per-Cell CSV (`cell_predictions_IMG_20150819_133447.csv`)** and **Download Summary JSON (`field_summary_IMG_20150819_133447.json`)** remain available directly below the PDF button for debugging, research auditing, and downstream statistical analysis.

---

### Minute 5:15 – 6:15 | Step 6: Demonstrate Edge-Case Handling (Zero Detections, Single-Cell Mode, or Invalid Box / WBC Exclusion)
1. **Option A — Zero-Detection Edge Case (`2 seconds`)**:
   - Check **"Simulate Zero YOLO Detections (`--test-zero-detections`)"** in Card 3 and click **Run Two-Stage Blood-Smear Analysis**.
   - Point out that the application safely handles `0` detections without division-by-zero and **never treats zero valid detections as a `0.00%` negative malaria result**, instead displaying **`"No reliable result is available for this image."`** and `N/A (0 valid RBCs)`.
2. **Option B — Single-Cell Mode (`C100P61ThinF_IMG_20150918_144104_cell_162.png`)**:
   - In **Card 1**, switch **Specimen Analysis Mode** from **`Full-Smear Mode (Stage 1 YOLOv8n + Stage 2 CNN)`** to **`Single-Cell Mode (Stage 2 CNN Only — Segmented RBC)`** and select or upload `C100P61ThinF_IMG_20150918_144104_cell_162.png` (`142x148 px`).
   - Click **Run Single-Cell CNN Classification** (`~4.5 seconds`): Stage 1 YOLOv8n is bypassed, Stage 2 CNN directly classifies the cell as **`Parasitized` (`99.92%` confidence)**, whole-smear parasitemia is reported as **`N/A (Single-Cell Mode)`**, and both the UI and the 2-page PDF preserve the native `142x148 px` cell image with a thin `2 px` border (never upscaled or covered by a full-smear banner).
3. **Option C — WBC Exclusion & Invalid Box Rejection**:
   - In **Full-Smear Mode**, uncheck Zero Detections, check **"Inject Synthetic WBC & Invalid Boxes (`--inject-test-invalid-boxes`)"**, and run analysis.
   - Open **Tab 4 (Excluded WBCs & Rejected Boxes)** to show `1` excluded `WBC (ID 2)` detection (`[500, 500, 720, 720]`) and `2` rejected invalid/out-of-bounds boxes (`5x4 px < 16 px` and `0x0 px < 16 px`), while the `195` valid RBCs still yield the exact same `74` Parasitized / `121` Uninfected counts.
4. **Option D — Invalid / Blank / Black / Corrupted File Upload**:
   - Upload any non-image, corrupted, all-black, or all-white blank file to show the structured red `InvalidImageError` banner (`HTTP 400`) without server interruption.

---

### Minute 6:15 – 7:00 | Step 7: Conclude with Honest Scientific Limitations & Future Work
Wrap up the presentation by highlighting five engineering and scientific takeaways:
1. **Severe Source Class Imbalance**: The NIH-NLM full-smear dataset has a `29.6 : 1` Uninfected-to-Parasitized ratio (`33,071` vs. `1,142` RBCs) and only `51` WBCs total (`37` in training), which is why a two-stage architecture (`YOLO` localization + domain-adapted `CNN` classification) dramatically outperforms single-stage YOLO classification (`0.8698` end-to-end Parasitized F1 vs. `2.07%` YOLO Parasitized $\text{mAP@50}$).
2. **Stage 1 Detection Misses in Crowded Fields**: Across all 25 test images (`4,794` GT RBCs), YOLOv8n localizes `96.28%` (`362 / 376`) of Parasitized RBCs and misses `14` overlapping/faint cells (`FN_Miss`), bounding end-to-end recall at `96.28%`.
3. **Limited Held-Out Patient Cohort**: The test split comprises `25` images from `5` held-out patients from a single *P. falciparum* thin-smear archive; multi-center validation across diverse Giemsa staining protocols and *Plasmodium* species (*P. vivax*, *P. malariae*, *P. ovale*) is required for broader generalization.
4. **Single-Field vs. Clinical Multi-Field Counting**: WHO microscopy protocols require counting $\ge 500\text{–}2,000$ RBCs across multiple fields, whereas a single microscope image contains $\sim 165\text{–}215$ RBCs.
5. **Hardware & Memory Constraints**: On the 4 GB RTX 2050 laptop GPU, full-resolution ($5312 \times 2988$) inference is run at `batch=1` (`~24–30 MB` peak VRAM) for stability.

---

## 4. USB Microscope Live Preview, Capture and Analyze Guide

When demonstrating SmartCell with a **USB digital microscope** or **USB UVC eyepiece camera** connected to the Windows laptop:

### Step 1: Connect the USB Microscope & Open Live Microscope Mode
1. Plug the USB digital microscope into an available USB port on the Windows laptop and wait 2–3 seconds for Windows to register the UVC camera device.
2. Launch SmartCell (`D:\smartcell\launch_smartcell_demo.bat` or `http://127.0.0.1:8501`).
3. In **Card 1 (Blood-Smear Image Input)** on the left sidebar, click the **"Live Microscope"** tab at the top of the card (next to **"Manual Upload / Preset"**).
   - *Note*: SmartCell does **not** request camera access automatically on page load; the camera remains completely off until you click **Start Camera**.

### Step 2: Grant Browser Camera Permission & Select the Microscope Device
1. Click **"Start Camera"**.
2. When Chrome or Edge prompts *"Allow 127.0.0.1:8501 to use your camera?"*, click **Allow**.
3. Once permission is granted, the **Camera / USB Microscope Device** dropdown automatically lists all connected video devices with their hardware names (for example, `Integrated HD Webcam` vs. `USB 2.0 Digital Microscope`).
4. Select your **USB Microscope** from the dropdown. The live preview switches immediately to the selected camera and displays the active stream resolution badge (e.g., `1920x1080` or `1280x720`) in the status bar.

### Step 3: Focus the Slide & Use Capture & Analyze
1. Adjust the microscope mechanical stage, condenser illumination, and fine focus knob until individual red blood cells have crisp membrane borders in the live preview.
2. You have two capture options:
   - **One-Click Workflow (Recommended)**: Click the prominent green **"Capture & Analyze"** button. SmartCell freezes the current native-resolution frame from the video stream, performs pre-inference image quality checks, and immediately submits the captured JPEG frame (`quality = 0.96`) into the two-stage `YOLOv8n` + `CNN` pipeline.
   - **Two-Step Preview & Retake Workflow**: Click **"Capture Image"** first to inspect the frozen still frame and review any pre-inference quality diagnostics (brightness/contrast and Laplacian focus sharpness score). If the slide moved or appears blurry, click **"Retake"** to resume the live video stream; otherwise click **"Capture & Analyze"**.
3. When finished with live microscopy, click **"Stop Camera"** or **"Return to Manual Upload"**—both actions immediately stop all `MediaStreamTrack` video tracks and release the USB camera hardware.

### Step 4: Inspect Captured Results & Download the 2-Page PDF Report
1. Once analysis completes (`~8–16s`), the right panel displays:
   - The **Final AI Screening Result** banner with predicted parasitized RBC count, estimated parasitemia percentage, and clinical screening disclaimer.
   - All **6 Summary KPI Cards** (`Predicted Parasitized RBCs`, `Predicted Uninfected RBCs`, `Experimental Parasitemia`, `Excluded WBC Detections`, `Rejected / Duplicate Boxes`, and `Processing Time`).
   - Any **Pre-Inference Capture & Quality Warnings** (for example, dark/under-illuminated field, over-exposed/blank field, low Laplacian sharpness indicating out-of-focus optics, or non-standard microscope magnification/staining advisories).
   - **Visual Output Tabs**:
     - **Tab 1 (`Annotated Full-Smear Image`)**: Full frame with **Red** (`Parasitized`) and **Green** (`Uninfected`) bounding boxes.
     - **Tab 2 (`Original Captured / Uploaded Image`)**: The unmodified raw frame captured from the USB microscope (`original_microscope_capture_<timestamp>_<id>.jpg`).
     - **Tab 3 (`Candidate RBC Contact Sheet`)**, **Tab 4 (`Per-Cell Predictions`)**, **Tab 5 (`Excluded WBCs & Rejected Boxes`)**, and **Tab 6 (`Formatted Text Report`)**.
   - **Primary Report Export**: Click **"Download Complete Report (PDF)"** to download the concise 2-page `SmartCell_Report_microscope_capture_<timestamp>_<id>.pdf` (containing Sections A–E, Section G, and the Final AI Screening Result, without the multi-page Section F per-cell table).

### Troubleshooting USB Microscope Camera Issues
| Symptom / Error Message | Cause | Resolution |
| :--- | :--- | :--- |
| **`Camera permission was denied`** | Browser blocked camera access for `127.0.0.1:8501`. | Click the camera/tune icon on the left side of the browser address bar, set **Camera** to **Allow**, and click **Start Camera** again. |
| **`No USB microscope or camera was detected`** | Microscope is unplugged or disabled in Windows Settings. | Check the USB cable connection, verify the camera appears under **Windows Device Manager $\rightarrow$ Cameras / Imaging devices**, and ensure **Windows Settings $\rightarrow$ Privacy & security $\rightarrow$ Camera** is turned **On**. |
| **`The selected USB microscope/camera is currently in use`** | Another app (Windows Camera, AmCap, OBS, Zoom, or another tab) holds an exclusive lock on the UVC stream. | Close the other camera application, wait 2 seconds, and click **Start Camera**. |
| **`The active USB microscope/camera was disconnected`** | USB cable was unplugged or lost power mid-preview. | Reconnect the USB cable, click **Start Camera**, and re-select the microscope from the dropdown. |
| **Laptop webcam opens instead of USB microscope** | Laptop built-in webcam is index `0` by default. | Keep the camera running and choose the USB microscope from the **Camera / USB Microscope Device** dropdown. |
| **Black, blank, or blurry capture warning** | Microscope LED/condenser light is off, objective is between click-stops, or focus is off. | Adjust condenser illumination and fine focus, click **Retake**, and click **Capture & Analyze** once cell boundaries are sharp. |

---

## 5. Full-Smear Mode vs. Single-Cell Mode Guide

SmartCell supports two distinct specimen input modes selectable in **Card 1 (`Specimen Analysis Mode`)**:

1. **`Full-Smear Mode (Stage 1 YOLOv8n + Stage 2 CNN)`** *(Default)*:
   - Designed for multi-cell thin blood-smear microscope fields (such as `D:\smartcell\yolo_malaria\images\test\IMG_20150819_133447.jpg`, $5312 \times 2988\text{ px}$).
   - Runs **Stage 1 YOLOv8n** (`best.pt`) to localize candidate RBCs, excludes `class 2 (WBC)` detections, rejects tiny/invalid boxes ($< 16\text{ px}$), classifies every valid RBC crop with the **Stage 2 CNN**, and calculates experimental whole-smear parasitemia only when $\ge 1$ valid RBC crop is classified.
   - If `0` valid RBCs are detected, SmartCell never reports a misleading `0.00%` negative result; instead it reports **`"No reliable result is available for this image."`**.
   - If a user selects a small single-cell image ($\le 300 \times 300\text{ px}$) while in **Full-Smear Mode**, the browser prompts the user for explicit confirmation before switching to **Single-Cell Mode** (and never silently auto-switches modes without confirmation).

2. **`Single-Cell Mode (Stage 2 CNN Only — Segmented RBC)`**:
   - Designed for pre-segmented single-cell RBC images (such as `D:\smartcell\Testing_images\cell_images\Parasitized\C100P61ThinF_IMG_20150918_144104_cell_162.png`, $142 \times 148\text{ px}$, or `D:\smartcell\cell_images` / `D:\smartcell\dataset`).
   - Bypasses Stage 1 YOLOv8n and feeds the single-cell image directly into the selected **Stage 2 CNN** (`finetuned_raw`, `finetuned_masked`, or `original`).
   - Reports the **Single-Cell Predicted Class (`Parasitized` or `Uninfected`)**, **Classification Confidence (`99.92%`)**, and **Raw vs. Elliptical-Masked Audit** while explicitly marking **Whole-Smear Parasitemia as `N/A`** (since parasitemia cannot be computed from 1 isolated cell).
   - Preserves the native resolution of the single-cell image ($142 \times 148\text{ px}$) with a thin $2\text{ px}$ colored perimeter border in both the web UI and the 2-page PDF report without upscaling or banner occlusion.


