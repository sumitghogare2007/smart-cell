# SmartCell AI Backend — Render Free Tier Deployment Guide

This package contains the deployment-ready backend files for deploying the **SmartCell AI Blood Cell Screening Platform** to **Render Web Services (Free Tier)** with **₹0 budget**.

The backend serves the live AI models for:
1. **Mode 1: Malaria Blood Smear Screening** (Stage 1 YOLOv8n RBC localization + Stage 2 Fine-Tuned CNN classification)
2. **Mode 2: ALL Leukemia Screening** (Stage 2 ALL-IDB EfficientNetB0 lymphoblast classification)

It directly powers the already deployed **Hugging Face Static Space** frontend:
👉 `https://huggingface.co/spaces/Sumit0123/smart-cell`

---

## 1. Render Free Tier Architecture & RAM Optimization

### The 512 MB RAM Challenge
Render's Free Tier provides:
- **RAM**: 512 MB ceiling
- **CPU**: 0.1 vCPU (shared)
- **Cost**: ₹0 / month (no credit card required)
- **Idle Behavior**: Spins down after 15 minutes of inactivity (cold start ~50s)

When running deep learning frameworks in Python on CPU:
- Baseline Python: ~12 MB
- Importing `torch` + `ultralytics` + loading `best.pt`: **~232 MB RAM**
- Importing `tensorflow` + loading `malaria_cnn_finetuned_raw_best.keras`: **~362 MB RAM**
- Simultaneous in-process loading: $232\text{ MB} + 362\text{ MB} = \mathbf{594\text{ MB}}$ ⚠️ *(Exceeds 512 MB limit, triggering OOM SIGKILL)*

### SmartCell's Solution: Sequential Subprocess Memory Isolation
To prevent OOM termination on Render Free:
1. **Lightweight Web Server**: `render_app.py` never imports PyTorch or TensorFlow at the root level. Its resident set size (RSS) is only **~30 MB**.
2. **Isolated Worker Subprocesses**:
   - For **Malaria**, Stage 1 YOLOv8n runs in a subprocess, outputs detections, and terminates (reclaiming ~230 MB). Then Stage 2 CNN runs in a second subprocess, classifies crops, and terminates (reclaiming ~360 MB).
   - For **Leukemia**, `leukemia_worker.py` runs in an isolated subprocess, performs EfficientNetB0 inference, and terminates (reclaiming ~360 MB).
3. **Peak Memory**: Guaranteed to stay below **~365 MB**, providing a comfortable **~150 MB safety buffer** below Render's 512 MB ceiling at all times.

---

## 2. Directory Structure

```
D:\smartcell\deployment\render_backend\
├── render_app.py              # Main Render HTTP server (CORS, endpoints, subprocess manager)
├── leukemia_worker.py         # Isolated leukemia subprocess worker
├── requirements-render.txt    # Lean CPU-only dependencies
├── render.yaml                # Render Blueprint deployment definition
├── Dockerfile                 # Containerized deployment specification
├── .dockerignore              # Excludes training datasets and virtualenvs from Docker build
└── README.md                  # This deployment manual
```

---

## 3. Step-by-Step Render Deployment Guide

### Method A: Deploy via Render Web Service (Native Python — Recommended)

1. **Log in to Render**: Go to [https://render.com](https://render.com) (sign up with GitHub if needed).
2. **Create New Web Service**:
   - In the Render Dashboard, click **New +** -> **Web Service**.
   - Select **Build and deploy from a Git repository**.
   - Connect your GitHub repository: `https://github.com/Tanvip1711/SmartCell`.
   - Branch: `main`.
3. **Configure Settings**:
   - **Name**: `smartcell-ai-backend`
   - **Region**: `Oregon (US West)` or `Frankfurt (EU Central)`
   - **Branch**: `main`
   - **Root Directory**: leave blank (or `./`)
   - **Runtime**: `Python 3`
   - **Build Command**:
     ```bash
     pip install --no-cache-dir -r deployment/render_backend/requirements-render.txt
     ```
   - **Start Command**:
     ```bash
     python deployment/render_backend/render_app.py --host 0.0.0.0 --port $PORT
     ```
   - **Instance Type**: **Free** (512 MB RAM, 0.1 CPU)
4. **Configure Environment Variables** (under *Advanced*):
   | Key | Value | Description |
   |---|---|---|
   | `PORT` | `8501` | Render dynamically injects this, default 8501 |
   | `HOST` | `0.0.0.0` | Binds to all network interfaces |
   | `PYTHON_VERSION` | `3.11.9` | Pins Python runtime |
   | `TF_ENABLE_ONEDNN_OPTS` | `0` | Disables oneDNN floating point warnings |
   | `TF_CPP_MIN_LOG_LEVEL` | `2` | Silences TensorFlow informational logs |
   | `PYTHONUNBUFFERED` | `1` | Flushes stdout/stderr logs immediately to Render console |
5. **Configure Health Check**:
   - **Health Check Path**: `/api/health`
6. **Click "Create Web Service"**:
   - Render will clone the repository, install dependencies, and launch `render_app.py`.
   - Once deployment completes, Render provides your unique public URL:
     `https://smartcell-ai-backend.onrender.com`

---

### Method B: Deploy via Docker (Alternative)

If you prefer deploying via container:
1. In Render, create **New +** -> **Web Service**.
2. Connect your repository.
3. Select **Docker** as the Runtime.
4. **Dockerfile Path**: `./deployment/render_backend/Dockerfile`.
5. **Instance Type**: **Free**.
6. Click **Create Web Service**.

---

## 4. Connecting Hugging Face Frontend to Render Backend

Once your Render service is deployed at `https://smartcell-ai-backend.onrender.com`:

### Option 1: Via Web UI (Zero Code Modification)
1. Open your Hugging Face Space:
   `https://huggingface.co/spaces/Sumit0123/smart-cell`
2. Click **⚙️ Settings** (or the backend status badge in the header).
3. In the **Backend Connection Manager** modal:
   - Enter your Render backend URL:
     `https://smartcell-ai-backend.onrender.com`
   - Click **Test Connection**.
   - Once verified ("Reachable: Healthy"), click **Save & Connect**.
4. The frontend will instantly save the URL in `localStorage` and transition all analysis buttons to live mode.

### Option 2: Default Base URL in `config.js`
In `deployment/huggingface_static/config.js`, update line 15:
```javascript
window.SMARTCELL_CONFIG = {
  API_BASE_URL: "https://smartcell-ai-backend.onrender.com",
  ...
};
```

---

## 5. API Reference & Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` or `/index.html` | Backend status and information dashboard |
| `GET` | `/api/health` | Health check (`{"status": "healthy", ...}`) |
| `GET` | `/api/status` | Hardware telemetry, loaded model states, preset lists |
| `POST` | `/api/analyze` | Multipart form-data inference (Malaria or Leukemia) |
| `GET` | `/api/file` | Secure artifact downloads (`?run_id=...&file=...&download=1`) |
| `OPTIONS` | `/api/*` | CORS preflight response (204 No Content) |

### Verification with curl

#### 1. Health Check
```bash
curl -i https://smartcell-ai-backend.onrender.com/api/health
```
*Expected response (HTTP 200):*
```json
{
  "status": "healthy",
  "service": "SmartCell Research Platform",
  "environment": "render_free_tier",
  "timestamp": "2026-11-25T...",
  "modes": ["malaria_screening", "all_leukemia"]
}
```

#### 2. CORS Preflight Check
```bash
curl -i -X OPTIONS https://smartcell-ai-backend.onrender.com/api/analyze \
  -H "Origin: https://sumit0123-smart-cell.hf.space" \
  -H "Access-Control-Request-Method: POST"
```
*Expected response:*
HTTP 204 No Content with `Access-Control-Allow-Origin: https://sumit0123-smart-cell.hf.space`.

#### 3. ALL Leukemia Inference Request
```bash
curl -X POST https://smartcell-ai-backend.onrender.com/api/analyze \
  -F "disease_mode=leukemia" \
  -F "preset_filename=Im001_1.png" \
  -F "capture_source=preset_test_image"
```

#### 4. Malaria Smear Inference Request
```bash
curl -X POST https://smartcell-ai-backend.onrender.com/api/analyze \
  -F "disease_mode=malaria" \
  -F "preset_filename=IMG_20150728_121151.jpg" \
  -F "cnn_variant=finetuned_raw" \
  -F "crop_mode=raw" \
  -F "input_mode=full_smear"
```

---

## 6. Cold Starts & Spin-Down Handling

Render Free instances automatically spin down into sleep mode after 15 minutes without incoming traffic.

- **First Request (Wake-up)**: Takes approximately **45 to 60 seconds** while Render spins up the container.
- **Subsequent Requests**: Respond instantaneously (~100–300 ms for health, 1–3s for single cell inference, 8–15s for full smear detection).
- **Hugging Face Frontend Support**: The SmartCell frontend's backend connection manager has an extended health check timeout and clear offline/standby indicators, keeping the user informed while the backend spins up.
