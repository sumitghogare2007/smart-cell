# SmartCell AI Screening Platform — Cloud Deployment & Production Guide

This guide details the complete deployment architecture for hosting the **SmartCell AI Blood Cell Screening Research Platform** on public cloud infrastructure (Hugging Face Spaces, Render, or Railway) with full backend AI inference, real-time WebRTC USB microscope capture, and automated clinical-grade PDF reporting.

---

## 1. Cloud Architecture Overview

SmartCell is engineered as a unified, dual-mode medical research application:
- **Mode 1 — Malaria Blood-Smear Screening**: Stage 1 YOLOv8n Candidate RBC Localization + Stage 2 Domain-Adapted CNN Classification (`malaria_cnn_finetuned_raw_best.keras` / `best.pt`).
- **Mode 2 — ALL Leukemia Screening**: Deep Transfer Learning White Blood Cell Classifier using EfficientNetB0 (`all_idb_leukemia_best.keras`) on ALL-IDB2 benchmark standards.
- **Microscope Subsystem**: Browser-native WebRTC live video streaming, device switching, high-resolution frame capture, and client-side quality validation (rejection of corrupted, blank, or black frames).
- **Export Engine**: Vectorized multi-page PDF reporting (using ReportLab), JSON metrics, TXT diagnostic summaries, and annotated high-resolution smear visualization.

### Dynamic Cross-Platform Bridge
The codebase automatically adapts its runtime environment:
- **Local Windows Dev**: Uses isolated virtual environments (`venv`, `venv_yolo_gpu`, `venv_yolo`) to separate conflicting dependencies and leverage local NVIDIA RTX GPU acceleration.
- **Cloud Linux / Docker**: Dynamically detects single-container environments, binding `HOST=0.0.0.0` and `PORT=$PORT`, running real CPU-optimized PyTorch and TensorFlow inference seamlessly in Python 3.11.

---

## 2. Recommended Cloud Hosting Platforms

| Platform | Recommended Plan | Free Tier Specs | Cloud URL / HTTPS | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **Hugging Face Spaces** *(Top Recommendation)* | Docker Space (Free) | **2 vCPU, 16 GB RAM**, Unlimited bandwidth | Automatic SSL (`https://*.hf.space`) | **Best overall** — 16 GB RAM easily accommodates PyTorch + TensorFlow without OOM risks. Free WebRTC camera access. |
| **Render** | Web Service (Starter Plan - $7/mo) | 512 MB (Free - *Insufficient for AI inference*) / 1-2 GB (Starter) | Automatic SSL (`https://*.onrender.com`) | Production SLA, auto-deploy from GitHub. |
| **Railway** | Docker Web Service | $5 trial credits / Pay-as-you-go (~$3-5/mo) | Automatic SSL (`https://*.up.railway.app`) | Extremely fast builds, custom domain support. |

> [!IMPORTANT]
> **HTTPS Requirement for Digital Microscope Capture**:
> Modern web browsers (Chrome, Edge, Firefox, Safari) strictly block camera and microphone access (`navigator.mediaDevices.getUserMedia`) over insecure HTTP connections on remote hosts. All three recommended platforms automatically provision free SSL/TLS certificates, ensuring the USB microscope capture functions seamlessly for all external users.

---

## 3. Option A: Deploying to Hugging Face Spaces (Recommended — Free 16GB RAM)

Hugging Face Spaces provides **16 GB RAM for free**, which is ideal for dual-framework AI applications running PyTorch and TensorFlow simultaneously.

### Step 1: Create a Space on Hugging Face
1. Log in to [Hugging Face](https://huggingface.co).
2. Click **New Space** (`https://huggingface.co/new-space`).
3. Set:
   - **Space Name**: `smartcell-ai-platform` (or your preferred name)
   - **License**: `mit` or `openrail`
   - **Space SDK**: Select **Docker** -> **Blank**
   - **Space Hardware**: Free (2 vCPU, 16 GB RAM)
4. Click **Create Space**.

### Step 2: Push SmartCell Code to Your Space
In your local command prompt or PowerShell at `D:\smartcell`:

```powershell
# 1. Initialize and commit your clean repository (if not already done)
git add .
git commit -m "feat: complete production cloud deployment package"

# 2. Add your Hugging Face Space as a remote
# Replace YOUR_HF_USERNAME with your actual Hugging Face username:
git remote add space https://huggingface.co/spaces/YOUR_HF_USERNAME/smartcell-ai-platform

# 3. Push to Hugging Face
git push --force space main:main
```

Hugging Face will automatically detect the `Dockerfile`, build the container, install requirements, and launch your application at:
`https://YOUR_HF_USERNAME-smartcell-ai-platform.hf.space`

---

## 4. Option B: Deploying to Render

Render is a robust cloud platform that connects directly to GitHub.

### Step 1: Push SmartCell to GitHub
1. Create a new repository on [GitHub](https://github.com/new) (e.g., `smartcell`).
2. Push your local repository to GitHub:
   ```powershell
   git remote add origin https://github.com/YOUR_GITHUB_USERNAME/smartcell.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy on Render
1. Log in to [Render Dashboard](https://dashboard.render.com).
2. Click **New +** -> **Web Service**.
3. Select **Build and deploy from a Git repository** and connect your `smartcell` repo.
4. Render will automatically detect `render.yaml` and configure:
   - **Environment**: Docker
   - **Instance Type**: **Starter** (1 vCPU, 1-2 GB RAM - *Do not use Free 512 MB tier as TensorFlow + PyTorch inference requires ~900 MB RAM*)
   - **Health Check Path**: `/api/health`
5. Click **Create Web Service**.

---

## 5. Option C: Deploying to Railway

Railway provides instantaneous Docker builds with automatic zero-configuration SSL.

### Step 1: One-Click Deploy via GitHub
1. Push your repository to GitHub.
2. Go to [Railway](https://railway.app) and create a **New Project**.
3. Choose **Deploy from GitHub repo** and select your `smartcell` repository.
4. Railway will automatically detect `railway.json` and `Dockerfile`, build the image, and assign a public domain with HTTPS.

---

## 6. Verification and Health Endpoints

Once deployed, you can verify your cloud deployment using the built-in REST API endpoints:

| Endpoint | HTTP Method | Expected Output | Purpose |
| :--- | :--- | :--- | :--- |
| `/api/health` | `GET` / `HEAD` | `{"status": "healthy", "service": "SmartCell Research Platform", ...}` | Container health checks, load balancer pings |
| `/api/models/status` | `GET` | JSON status of all 3 Malaria CNNs, YOLOv8 detector, and GPU/CPU telemetry | Model validation & readiness verification |
| `/api/preset_image?filename=...` | `GET` | Image bytes (PNG/JPG) | Held-out benchmark preset delivery |
| `/api/analyze` | `POST` | Full analysis payload (detections, classifications, parasitemia, PDF paths) | Mode 1 & Mode 2 Real AI Inference |
| `/api/file?run_id=...&file=...` | `GET` | Generated clinical PDF reports, CSVs, TXT, and crops | Downloadable clinical export artifacts |

---

## 7. Safety and Dataset Protection Guarantee

- **Zero Retraining**: All existing weights in `models/` and `results/yolo_malaria_run/weights/best.pt` are preserved verbatim.
- **Dataset Security**: Raw gigabyte datasets (`ALL_IDB1`, `ALL_IDB2`, `cell_images`, `Polygon Set`, `Point Set`, `Testing_images`) are completely excluded from cloud artifacts via `.dockerignore` and `.gitignore`.
- **Local Dev Untouched**: Windows virtual environments and `launch_smartcell_demo.bat` remain 100% operational.
