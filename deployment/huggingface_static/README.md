---
title: SmartCell AI Blood Cell Screening Platform
emoji: 🔬
colorFrom: blue
colorTo: red
sdk: static
pinned: false
---

# SmartCell — Hugging Face Static Deployment Guide

**Space URL:** [https://huggingface.co/spaces/Sumit0123/smart-cell](https://huggingface.co/spaces/Sumit0123/smart-cell)  
**Space Type:** Hugging Face Space  
**Space SDK:** `static`  
**Template:** Blank  
**Target Environment:** Client-Side Web Application (HTML5 / CSS3 / Vanilla JavaScript)  

---

## 1. Purpose of This Deployment

This directory contains the standalone, client-side static frontend for the **SmartCell AI-Assisted Blood Cell Screening Platform**. It is specifically configured for deployment to a **Hugging Face Static Space** to provide evaluators, faculty reviewers, viva examiners, and researchers with an interactive, publicly accessible medical interface showcasing:
- **Dual-Disease Screening Interfaces:** Malaria Blood-Smear Field Screening (Two-Stage YOLOv8n + CNN) and Acute Lymphoblastic Leukemia (ALL) Single-Cell Classification (EfficientNetB0).
- **Interactive Multi-Mode Input Workflows:** High-resolution local image file upload, drag-and-drop support, format validation, and direct WebRTC digital microscope video streaming.
- **Client-Side Optical Quality Assurance:** Real-time luminance and Laplacian variance computation to reject black frames, oversaturated fields, and defocused/blurred captures before submitting for inference.
- **Configurable Deep-Learning Backend Routing:** Direct REST API integration with a separate SmartCell AI inference engine for running PyTorch YOLOv8n and TensorFlow CNN models.

> [!IMPORTANT]
> **Static Space Execution Model:**  
> Hugging Face Static Spaces execute **pure client-side code** (HTML, CSS, and JavaScript in the user's web browser). Static spaces do **not** run a Python server, PyTorch, CUDA, or TensorFlow runtime. Therefore, this frontend operates as the presentation and acquisition layer. Live deep-learning inference, parasitemia quantification, and PDF report generation require connecting to an external SmartCell Python backend.

---

## 2. Directory Structure

```text
deployment/huggingface_static/
├── index.html        # Main responsive single-page medical dashboard
├── styles.css        # Complete laboratory dark-theme styling & responsive rules
├── app.js            # Client-side controller, WebRTC camera manager & API client
├── config.js         # Backend URL configuration & pre-inference quality thresholds
├── assets/           # Supporting static assets and icons
└── README.md         # Deployment instructions & Hugging Face Space documentation
```

---

## 3. Feature Support Matrix

| Feature | Works in Static Space Alone | Requires SmartCell AI Backend | Implementation Details |
| :--- | :---: | :---: | :--- |
| **Disease Mode Switching** | ✅ Yes | ❌ No | Toggles between Malaria and ALL Leukemia workflows instantly in the browser. |
| **File Drag & Drop Upload** | ✅ Yes | ❌ No | Validates formats (`.jpg`, `.png`, `.tif`, `.bmp`), checks size limits (<35 MB), and renders client preview. |
| **Digital Microscope Streaming** | ✅ Yes | ❌ No | Uses native WebRTC `navigator.mediaDevices.getUserMedia()`, detects UVC cameras, and renders live video feed. |
| **Microscope Snapshot Capture** | ✅ Yes | ❌ No | Captures uncompressed frames from `<video>` to an off-screen `<canvas>` with retake and discard controls. |
| **Pre-Inference Quality Validation** | ✅ Yes | ❌ No | Computes mean luminance (flags black frames $<15$ or oversaturated fields $>245$) and Laplacian variance ($\sigma^2 < 85$) client-side. |
| **Backend Connection Monitor** | ✅ Yes | ❌ No | Automatically pings `${API_BASE_URL}/api/health` and displays connection status in the header. |
| **YOLOv8n Cell Localization** | ❌ No | ✅ Yes | Requires PyTorch, Ultralytics, and trained weights (`best.pt`) running on the backend. |
| **CNN Parasitemia Quantification** | ❌ No | ✅ Yes | Requires TensorFlow and domain-adapted CNN (`malaria_cnn_finetuned_raw_best.keras`). |
| **EfficientNetB0 ALL Classification** | ❌ No | ✅ Yes | Requires TensorFlow and ALL model (`all_idb_leukemia_best.keras`). |
| **Clinical PDF Report Generation** | ❌ No | ✅ Yes | Generated server-side using ReportLab and downloaded via `/api/file`. |

---

## 4. How to Configure the AI Backend URL

By default, `config.js` leaves `API_BASE_URL` unconfigured (`""`) so the Space can be deployed safely without broken connections.

### Method A: Via Web UI Settings Modal (Recommended)
1. Open the deployed Hugging Face Space: [https://huggingface.co/spaces/Sumit0123/smart-cell](https://huggingface.co/spaces/Sumit0123/smart-cell).
2. Click the **"Backend: Not Configured"** pill or the **"Settings"** button in the top navigation banner.
3. In the **Backend Connection Settings** dialog, enter your running backend URL (e.g. `http://127.0.0.1:8501` for a local machine, or a deployed backend URL).
4. Click **"Test Connection"** to verify that `/api/health` answers with HTTP 200.
5. Click **"Save & Connect"**. The URL is stored securely in your browser's `localStorage` and will persist across sessions.

### Method B: Via `config.js` (Permanent Deployment Configuration)
Before pushing to Hugging Face, edit `config.js`:
```javascript
window.SMARTCELL_CONFIG = {
  API_BASE_URL: "https://your-deployed-smartcell-backend.domain.com",
  // ...
};
```

### Method C: Running the Local Backend on Your Laptop
To provide real inference to this frontend from your local computer:
1. Open `D:\smartcell` in terminal or double-click `launch_smartcell_demo.bat`.
2. The server starts at `http://127.0.0.1:8501`.
3. In the Hugging Face static frontend settings, set Backend URL to `http://127.0.0.1:8501`.
   *(Note: Ensure browser mixed-content permissions allow HTTP calls if testing from an HTTPS Space, or use a secure tunnel like Cloudflare Tunnel / ngrok).*

---

## 5. Camera & HTTPS Requirements

- **Browser Security Context:** Modern browsers restrict `navigator.mediaDevices.getUserMedia()` to **Secure Contexts** (`HTTPS` or `http://localhost` / `http://127.0.0.1`).
- **Hugging Face Spaces Native Support:** Hugging Face Spaces are served over **HTTPS by default**, which satisfies the browser's security requirement for direct digital microscope video streaming.
- **Physical Microscope Compatibility:** Compatible with any standard USB Video Class (UVC) digital microscope (e.g. Tobo, Celestron, Jiusion, Plugable) or digital eyepiece camera recognized as an optical video input by your operating system.
- **Permissions:** When clicking **"Start Camera"**, approve the browser camera permission prompt.

---

## 6. How to Upload Files to Your Hugging Face Space

### Option 1: Direct Web Upload (Simplest & Fastest)
1. Open your Space in your browser: [https://huggingface.co/spaces/Sumit0123/smart-cell](https://huggingface.co/spaces/Sumit0123/smart-cell).
2. Click **Files and versions** in the top navigation bar.
3. Click **Add file** $\rightarrow$ **Upload files**.
4. Drag and drop all files from `D:\smartcell\deployment\huggingface_static\`:
   - `index.html`
   - `styles.css`
   - `app.js`
   - `config.js`
   - `README.md`
   - `assets/` (if present)
5. Enter commit message: `feat: deploy SmartCell static frontend application`
6. Click **Commit changes to main**. Hugging Face Static Space will build and serve within ~10 seconds.

### Option 2: Using Git CLI
```bash
# Clone your Hugging Face Space repository
git clone https://huggingface.co/spaces/Sumit0123/smart-cell D:\hf_space_smartcell

# Copy the static deployment files into the cloned repository
xcopy /E /I /Y "D:\smartcell\deployment\huggingface_static\*" "D:\hf_space_smartcell\"

# Commit and push
cd D:\hf_space_smartcell
git add index.html styles.css app.js config.js README.md
git commit -m "feat: deploy SmartCell static frontend application"
git push origin main
```

---

## 7. Known Limitations & Research Disclaimer

1. **Experimental Research Prototype:** SmartCell is developed strictly for educational and scientific research. It has not received CE-IVD or FDA clearance and must **never** be used for autonomous clinical diagnosis or patient management.
2. **Offline Behavior:** If the backend is disconnected, clicking **Analyze** will display a clear warning: *"AI backend is not connected. Live analysis is currently unavailable."* It will **never** generate fabricated or simulated prediction results.
3. **CORS (Cross-Origin Resource Sharing):** When connecting to an external backend from `https://huggingface.co`, the backend must include CORS headers (`Access-Control-Allow-Origin: *` or your specific Hugging Face origin). The SmartCell Python server in `scripts/smartcell_demo_app.py` has CORS pre-configured.

---

## 8. Academic Citations

1. **NIH-NLM Malaria Dataset:** Lister Hill National Center for Biomedical Communications, National Library of Medicine, NIH, Bethesda, MD, USA.
2. **ALL-IDB Benchmark:** Acute Lymphoblastic Leukemia Image Database, Università degli Studi di Milano, Dipartimento di Informatica.
