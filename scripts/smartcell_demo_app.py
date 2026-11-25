r"""
smartcell_demo_app.py
---------------------
SmartCell Phase 7: Interactive Local Demonstration Application
Around the Verified Two-Stage YOLOv8n (GPU) + Custom CNN (CPU) Blood-Smear Pipeline.

DISCLAIMER:
SmartCell is an experimental academic research prototype developed for a college
engineering project. It is NOT a clinically validated diagnostic device.
Nor does evaluation on the 25-image NIH-NLM test split establish independent
clinical generalization.

Features:
1. Full-resolution microscope blood-smear image upload (.jpg, .jpeg, .png, .bmp, .tif, .tiff)
   plus quick-selection of the 25 held-out NIH-NLM test-set images for live demonstration.
2. Selection between:
   - Original CNN (`D:\smartcell\models\malaria_cnn_best.keras`)
   - Fine-Tuned Raw CNN (`D:\smartcell\models\malaria_cnn_finetuned_raw_best.keras`)
   - Fine-Tuned Masked CNN (`D:\smartcell\models\malaria_cnn_finetuned_masked_best.keras`)
3. Validated crop mode selection (`raw` rectangular vs `masked` elliptical black-corner)
   and configurable inference thresholds (YOLO conf, YOLO NMS IoU, dedup IoU, CNN threshold,
   crop padding, minimum box size).
4. Runs Stage 1 YOLOv8n detection in `D:\smartcell\venv_yolo_gpu` on NVIDIA GeForce RTX 2050
   (CUDA device 0) with graceful fallback to CPU if GPU is unavailable.
5. Runs Stage 2 CNN classification in `D:\smartcell\venv` (TensorFlow 2.21.0) without modifying
   any environment dependencies.
6. Displays the annotated full-smear image, RBC crop contact sheet, individual cell crop
   thumbnails, and separate counts for:
   - Predicted Parasitized RBCs
   - Predicted Uninfected RBCs
   - Excluded WBC detections
   - Rejected / duplicate detections
7. Displays the experimental parasitemia estimate, model confidence statistics
   (mean/median/min/max for YOLO and CNN), hardware telemetry, and processing time breakdown.
8. Provides one-click CSV, JSON, TXT, and annotated JPG export downloads.
9. Gracefully handles invalid/corrupted images, zero detections, inference errors, and
   unavailable GPU.

Usage:
  D:\smartcell\venv\Scripts\python.exe D:\smartcell\scripts\smartcell_demo_app.py --port 8501
"""

import os
import sys
import csv
import json
import time
import uuid
import shutil
import base64
import mimetypes
import argparse
import traceback
import subprocess
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

BASE_DIR = os.environ.get("SMARTCELL_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
VENV_TF_PYTHON = os.path.join(BASE_DIR, "venv", "Scripts", "python.exe")
VENV_YOLO_GPU_PYTHON = os.path.join(BASE_DIR, "venv_yolo_gpu", "Scripts", "python.exe")
VENV_YOLO_CPU_PYTHON = os.path.join(BASE_DIR, "venv_yolo", "Scripts", "python.exe")


def get_python_exe(candidate_path):
    """Returns candidate_path if it exists; otherwise falls back to sys.executable."""
    if candidate_path and os.path.exists(candidate_path):
        return candidate_path
    return sys.executable

PIPELINE_SCRIPT = os.path.join(BASE_DIR, "scripts", "run_smartcell_pipeline.py")
DEMO_RUNS_ROOT = os.path.join(BASE_DIR, "results", "demo_app_runs")
UPLOADS_DIR = os.path.join(DEMO_RUNS_ROOT, "_uploads")

TEST_IMAGES_DIR = os.path.join(BASE_DIR, "yolo_malaria", "images", "test")
TEST_LABELS_DIR = os.path.join(BASE_DIR, "yolo_malaria", "labels", "test")
POLYGON_SET_DIR = os.path.join(BASE_DIR, "Polygon Set")

MODELS_MAP = {
    "finetuned_raw": {
        "label": "Fine-Tuned Raw CNN (Phase 6 Domain-Adapted — Recommended)",
        "path": os.path.join(BASE_DIR, "models", "malaria_cnn_finetuned_raw_best.keras"),
        "recommended_crop_mode": "raw",
        "description": "Fine-tuned on 24,112 native rectangular full-smear crops (Test Acc: 98.35%, Par F1: 0.9016)."
    },
    "finetuned_masked": {
        "label": "Fine-Tuned Masked CNN (Phase 6 Elliptical-Masked)",
        "path": os.path.join(BASE_DIR, "models", "malaria_cnn_finetuned_masked_best.keras"),
        "recommended_crop_mode": "masked",
        "description": "Validated on elliptical black-corner masked full-smear crops (Test Acc: 98.10%, Par F1: 0.8905)."
    },
    "original": {
        "label": "Original Baseline CNN (Phase 1 Single-Cell Dataset)",
        "path": os.path.join(BASE_DIR, "models", "malaria_cnn_best.keras"),
        "recommended_crop_mode": "masked",
        "description": "Trained on 27,558 black-background NIH single-cell crops (95.04% single-cell Acc; sensitive to raw corners)."
    }
}

YOLO_WEIGHTS_PATH = os.path.join(BASE_DIR, "results", "yolo_malaria_run", "weights", "best.pt")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Cached hardware status so UI loads instantaneously
_CACHED_ENV_STATUS = None


def build_image_to_patient_map():
    mapping = {}
    if os.path.exists(POLYGON_SET_DIR):
        for pat in sorted(os.listdir(POLYGON_SET_DIR)):
            img_dir = os.path.join(POLYGON_SET_DIR, pat, "Img")
            if os.path.isdir(img_dir):
                for fn in os.listdir(img_dir):
                    if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                        mapping[fn] = pat
    return mapping


IMAGE_TO_PATIENT = build_image_to_patient_map()


def inspect_ground_truth_for_stem(stem):
    """Checks if a microscope image stem has a known ground-truth label file in yolo_malaria/labels."""
    for split in ("test", "val", "train"):
        lbl_path = os.path.join(BASE_DIR, "yolo_malaria", "labels", split, f"{stem}.txt")
        if os.path.exists(lbl_path):
            uninf = 0
            par = 0
            wbc = 0
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cid = int(parts[0])
                    if cid == 0:
                        uninf += 1
                    elif cid == 1:
                        par += 1
                    elif cid == 2:
                        wbc += 1
            tot_rbc = uninf + par
            par_pct = round((par / float(tot_rbc)) * 100.0, 2) if tot_rbc > 0 else 0.0
            return {
                "has_ground_truth": True,
                "split": split,
                "patient_id": IMAGE_TO_PATIENT.get(f"{stem}.jpg", "Unknown"),
                "gt_uninfected_rbc": uninf,
                "gt_parasitized_rbc": par,
                "gt_wbc": wbc,
                "gt_total_rbc": tot_rbc,
                "gt_parasitemia_percent": par_pct
            }
    return {"has_ground_truth": False}


SINGLE_CELL_PRESET_DIRS = [
    os.path.join(BASE_DIR, "Testing_images", "cell_images", "Parasitized"),
    os.path.join(BASE_DIR, "Testing_images", "cell_images", "Uninfected"),
    os.path.join(BASE_DIR, "dataset", "test", "Parasitized"),
    os.path.join(BASE_DIR, "dataset", "test", "Uninfected"),
]

LEUKEMIA_PRESET_DIRS = [
    os.path.join(BASE_DIR, "dataset_all", "test", "0_normal"),
    os.path.join(BASE_DIR, "dataset_all", "test", "1_blast"),
    os.path.join(BASE_DIR, "dataset_all", "val", "0_normal"),
    os.path.join(BASE_DIR, "dataset_all", "val", "1_blast"),
]


def get_preset_test_images():
    """Returns metadata for the 25 held-out test set images for quick demo selection."""
    presets = []
    if not os.path.exists(TEST_IMAGES_DIR):
        return presets
    for fn in sorted(os.listdir(TEST_IMAGES_DIR)):
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            stem = os.path.splitext(fn)[0]
            gt = inspect_ground_truth_for_stem(stem)
            presets.append({
                "filename": fn,
                "stem": stem,
                "patient_id": gt.get("patient_id", "Unknown"),
                "gt_uninfected_rbc": gt.get("gt_uninfected_rbc", 0),
                "gt_parasitized_rbc": gt.get("gt_parasitized_rbc", 0),
                "gt_wbc": gt.get("gt_wbc", 0),
                "gt_total_rbc": gt.get("gt_total_rbc", 0),
                "gt_parasitemia_percent": gt.get("gt_parasitemia_percent", 0.0)
            })
    return presets


def get_preset_single_cell_images():
    """Returns curated single-cell test specimens from Testing_images/cell_images and dataset/test."""
    samples = []
    seen = set()
    # Always include the problem-statement specimen first if present
    target_fn = "C100P61ThinF_IMG_20150918_144104_cell_162.png"
    for d in SINGLE_CELL_PRESET_DIRS:
        if not os.path.isdir(d):
            continue
        label = os.path.basename(d)
        cand = os.path.join(d, target_fn)
        if os.path.isfile(cand) and target_fn not in seen:
            seen.add(target_fn)
            samples.append({
                "filename": target_fn,
                "ground_truth_class": label,
                "source_folder": os.path.relpath(d, BASE_DIR),
            })
    for d in SINGLE_CELL_PRESET_DIRS:
        if not os.path.isdir(d):
            continue
        label = os.path.basename(d)
        count_for_dir = 0
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith((".png", ".jpg", ".jpeg")) and fn not in seen:
                seen.add(fn)
                samples.append({
                    "filename": fn,
                    "ground_truth_class": label,
                    "source_folder": os.path.relpath(d, BASE_DIR),
                })
                count_for_dir += 1
                if count_for_dir >= 4:
                    break
    return samples


def resolve_preset_image_path(preset_filename):
    """Resolves a preset filename from full-smear test images or single-cell test directories."""
    preset_fn = os.path.basename(str(preset_filename or ""))
    if not preset_fn:
        return None
    cand_smear = os.path.join(TEST_IMAGES_DIR, preset_fn)
    if os.path.isfile(cand_smear):
        return cand_smear
    for d in SINGLE_CELL_PRESET_DIRS + LEUKEMIA_PRESET_DIRS:
        cand_cell = os.path.join(d, preset_fn)
        if os.path.isfile(cand_cell):
            return cand_cell
    return None


def query_environment_status(force_refresh=False):
    """Queries GPU availability in venv_yolo_gpu and TensorFlow/model readiness in venv."""
    global _CACHED_ENV_STATUS
    if _CACHED_ENV_STATUS is not None and not force_refresh:
        return _CACHED_ENV_STATUS

    gpu_info = {
        "venv_yolo_gpu_exists": os.path.exists(VENV_YOLO_GPU_PYTHON),
        "venv_yolo_cpu_exists": os.path.exists(VENV_YOLO_CPU_PYTHON),
        "venv_tf_exists": os.path.exists(VENV_TF_PYTHON),
        "cuda_available": False,
        "gpu_name": "Unavailable (CPU Fallback Active)",
        "pytorch_version": "Unknown",
        "free_vram_mb": 0.0,
        "total_vram_mb": 0.0,
        "yolo_checkpoint_exists": os.path.exists(YOLO_WEIGHTS_PATH),
        "yolo_checkpoint_path": YOLO_WEIGHTS_PATH
    }

    probe_py = None
    if os.path.exists(VENV_YOLO_GPU_PYTHON):
        probe_py = VENV_YOLO_GPU_PYTHON
    elif os.path.exists(VENV_YOLO_CPU_PYTHON):
        probe_py = VENV_YOLO_CPU_PYTHON
    else:
        probe_py = sys.executable

    if probe_py:
        try:
            probe_code = (
                "import json, torch; "
                "avail = torch.cuda.is_available(); "
                "free_b, tot_b = torch.cuda.mem_get_info(0) if avail else (0, 0); "
                "print(json.dumps({"
                "'cuda_available': avail, "
                "'gpu_name': torch.cuda.get_device_name(0) if avail else 'CPU', "
                "'pytorch_version': torch.__version__, "
                "'free_vram_mb': round(free_b / (1024**2), 1), "
                "'total_vram_mb': round(tot_b / (1024**2), 1)"
                "}))"
            )
            res = subprocess.run(
                [probe_py, "-c", probe_code],
                capture_output=True, text=True, timeout=15
            )
            if res.returncode == 0 and res.stdout.strip():
                parsed = json.loads(res.stdout.strip().splitlines()[-1])
                gpu_info.update(parsed)
        except Exception as e:
            gpu_info["probe_error"] = str(e)

    models_status = {}
    for k, meta in MODELS_MAP.items():
        models_status[k] = {
            "variant": k,
            "label": meta["label"],
            "path": meta["path"],
            "exists": os.path.exists(meta["path"]),
            "recommended_crop_mode": meta["recommended_crop_mode"],
            "description": meta["description"]
        }

    _CACHED_ENV_STATUS = {
        "timestamp": datetime.now().isoformat(),
        "gpu_environment": gpu_info,
        "cnn_models": models_status,
        "preset_test_images": get_preset_test_images(),
        "preset_single_cell_images": get_preset_single_cell_images(),
    }
    return _CACHED_ENV_STATUS


MAX_UPLOAD_BYTES = 35 * 1024 * 1024  # 35 MB max upload payload
MIN_IMAGE_DIMENSION = 32
MAX_IMAGE_DIMENSION = 10000


def sanitize_upload_filename(raw_filename, is_live_capture=False):
    """
    Sanitizes untrusted browser-supplied filenames to prevent path traversal or unsafe characters.
    When `is_live_capture=True`, assigns a unique timestamped `microscope_capture_<ts>_<uid>.jpg` name.
    """
    import re
    base = os.path.basename(str(raw_filename or "")).strip()
    stem, ext = os.path.splitext(base)
    ext = ext.lower()
    if ext not in ALLOWED_EXTENSIONS:
        ext = ".jpg"

    clean_stem = re.sub(r"[^A-Za-z0-9_.-]", "_", stem).strip("._-")
    if is_live_capture or not clean_stem:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:6]
        clean_stem = f"microscope_capture_{ts}_{uid}"
    else:
        clean_stem = clean_stem[:80]

    return f"{clean_stem}{ext}"


def assess_image_quality(rgb_pil_image, capture_source="manual_upload", input_mode="full_smear"):
    """
    Performs non-destructive pre-inference image quality checks on an RGB PIL Image:
      - Mean luminance and contrast (standard deviation) to detect completely black or blank frames
      - 2D discrete Laplacian variance as a focus/sharpness heuristic to flag potentially blurred frames
      - Mode-aware resolution checks (full-smear vs. single-cell) and microscope domain advisory
    Does NOT alter, enhance, or resize the image.
    """
    import numpy as np
    w, h = rgb_pil_image.size
    input_mode = str(input_mode or "full_smear").strip().lower()
    gray = np.asarray(rgb_pil_image.convert("L"), dtype=np.float32)
    mean_lum = round(float(gray.mean()), 2)
    std_lum = round(float(gray.std()), 2)

    # 2D discrete Laplacian variance on interior pixels
    if h >= 4 and w >= 4:
        lap = (
            -4.0 * gray[1:-1, 1:-1]
            + gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
        )
        focus_var = round(float(lap.var()), 2)
    else:
        focus_var = 0.0

    is_black = bool(mean_lum < 8.0 and std_lum < 5.0)
    is_blank = bool((mean_lum > 247.0 and std_lum < 5.0) or std_lum < 3.0)
    is_blurred = bool(not is_black and not is_blank and focus_var < 18.0)
    is_low_res = bool(w < 640 or h < 480)
    looks_like_single_cell = bool(w <= 300 and h <= 300)
    looks_like_full_smear = bool(w >= 640 and h >= 480)
    insufficient_full_smear_res = bool(input_mode == "full_smear" and (w < 640 or h < 480))
    is_low_res_single_cell = bool(input_mode == "single_cell" and (w < 64 or h < 64))

    warnings = []
    if is_black:
        warnings.append(
            f"Quality Warning (Black Frame): Captured/uploaded image is nearly completely black "
            f"(mean luminance={mean_lum}/255, std={std_lum}). Check microscope illumination and camera shutter."
        )
    elif is_blank:
        warnings.append(
            f"Quality Warning (Blank/Uniform Frame): Captured/uploaded image has virtually zero contrast "
            f"(mean luminance={mean_lum}/255, std={std_lum}) and may not contain visible blood-smear cells."
        )

    if is_blurred:
        warnings.append(
            f"Quality Warning (Potential Blur / Out-of-Focus): Frame has a low focus sharpness score "
            f"(Laplacian variance={focus_var} < 18.0). Adjust the microscope fine-focus knob and retake if needed."
        )

    if input_mode == "full_smear":
        if looks_like_single_cell:
            warnings.append(
                f"Insufficient Full-Smear Resolution / Mode Mismatch Warning: Image dimensions ({w}x{h} px) are far below "
                f"full-smear microscopy resolution (>= 640x480 px) and resemble a segmented single-cell patch. "
                f"Stage 1 YOLOv8n full-smear detection is not designed for isolated single-cell crops; switch to Single-Cell Mode if this is a single RBC image."
            )
        elif is_low_res:
            warnings.append(
                f"Resolution Advisory: Image dimensions ({w}x{h} px) are below recommended 640x480 full-smear microscopy resolution; "
                f"small intracellular ring trophozoites and cell boundaries may not be reliably detected."
            )
    elif input_mode == "single_cell":
        if looks_like_full_smear:
            warnings.append(
                f"Input Mode Advisory: Image dimensions ({w}x{h} px) resemble a multi-cell full blood smear, but Single-Cell Mode "
                f"was selected (which classifies the entire image as 1 cell without YOLOv8n localization). Select Full-Smear Mode for multi-cell smears."
            )
        elif is_low_res_single_cell:
            warnings.append(
                f"Low-Resolution Single-Cell Advisory: Single-cell image dimensions ({w}x{h} px) are very small (< 64x64 px); "
                f"intracellular chromatin dots or ring morphology may be degraded."
            )

    if capture_source == "live_microscope":
        warnings.append(
            "Microscope Domain Notice: Model performance depends on objective magnification (100x oil-immersion recommended), "
            "Giemsa staining, illumination, camera resolution, and sensor domain. Heuristic quality checks do not establish clinical suitability."
        )

    return {
        "capture_source": capture_source,
        "input_mode": input_mode,
        "width": w,
        "height": h,
        "mean_luminance": mean_lum,
        "std_luminance": std_lum,
        "focus_laplacian_var": focus_var,
        "is_black_frame": is_black,
        "is_blank_frame": is_blank,
        "is_potentially_blurred": is_blurred,
        "is_low_resolution": is_low_res,
        "looks_like_single_cell": looks_like_single_cell,
        "looks_like_full_smear": looks_like_full_smear,
        "insufficient_full_smear_resolution": insufficient_full_smear_res,
        "is_low_res_single_cell": is_low_res_single_cell,
        "unsuitable_warning": bool(is_black or is_blank or is_blurred or insufficient_full_smear_res),
        "warnings": warnings,
        "clinical_suitability_disclaimer": (
            "Heuristic image quality checks detect gross capture issues (black, blank, or severely defocused frames) "
            "and do NOT certify clinical specimen suitability."
        ),
    }


def validate_image_file(
    filepath,
    max_bytes=MAX_UPLOAD_BYTES,
    min_dim=MIN_IMAGE_DIMENSION,
    max_dim=MAX_IMAGE_DIMENSION,
    capture_source="manual_upload",
    input_mode="full_smear",
    reject_blank_or_black=True,
):
    """Validates that a file is a readable, non-corrupted, non-blank/non-black image within supported size and dimension bounds."""
    from PIL import Image
    if not filepath or not os.path.exists(filepath):
        return False, "Uploaded image file does not exist on disk.", None

    fsize = os.path.getsize(filepath)
    if fsize == 0:
        return False, "Uploaded file is empty (0 bytes).", None
    if max_bytes and fsize > max_bytes:
        mb = round(fsize / (1024.0 * 1024.0), 2)
        max_mb = round(max_bytes / (1024.0 * 1024.0), 1)
        return False, f"Image file size ({mb} MB) exceeds maximum allowed limit ({max_mb} MB).", None

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file extension '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}", None

    try:
        with Image.open(filepath) as im:
            im.verify()
        with Image.open(filepath) as im2:
            rgb = im2.convert("RGB")
            w, h = rgb.size
            if w < min_dim or h < min_dim:
                return (
                    False,
                    f"Image dimensions ({w}x{h} px) are too small for analysis (minimum {min_dim}x{min_dim} px).",
                    None,
                )
            if w > max_dim or h > max_dim:
                return (
                    False,
                    f"Image dimensions ({w}x{h} px) exceed maximum supported resolution ({max_dim}x{max_dim} px).",
                    None,
                )
            quality_info = assess_image_quality(rgb, capture_source=capture_source, input_mode=input_mode)
            dim_payload = {"width": w, "height": h, "byte_size": fsize, "quality": quality_info}
            if reject_blank_or_black:
                if quality_info.get("is_black_frame"):
                    return (
                        False,
                        f"Black image rejected: Uploaded image is completely or nearly black "
                        f"(mean luminance={quality_info['mean_luminance']}/255, std={quality_info['std_luminance']}). "
                        f"Please upload a valid blood-smear or single-cell specimen.",
                        dim_payload,
                    )
                if quality_info.get("is_blank_frame"):
                    return (
                        False,
                        f"Blank/uniform image rejected: Uploaded image has virtually zero contrast "
                        f"(mean luminance={quality_info['mean_luminance']}/255, std={quality_info['std_luminance']}) "
                        f"and contains no discernible cellular content.",
                        dim_payload,
                    )
            return True, "Valid image", dim_payload
    except Exception as e:
        return False, f"Invalid or corrupted image file: {type(e).__name__}: {e}", None


def execute_pipeline_run(
    image_path,
    cnn_variant="finetuned_raw",
    crop_mode="raw",
    input_mode="full_smear",
    yolo_device="0",
    yolo_conf=0.25,
    yolo_iou=0.45,
    dedup_iou=0.50,
    cnn_thresh=0.50,
    pad_px=4,
    min_box_size=16,
    max_det=350,
    test_zero_detections=False,
    inject_test_invalid_boxes=False,
    run_id=None,
    capture_source="manual_upload",
):
    """
    Executes the verified SmartCell pipeline (`run_smartcell_pipeline.py`) in either
    `full_smear` mode (Stage 1 YOLOv8n + Stage 2 CNN) or `single_cell` mode (Stage 2 CNN only)
    and returns the structured payload matching the saved JSON, CSV, TXT, and PDF reports on disk.
    """
    os.makedirs(DEMO_RUNS_ROOT, exist_ok=True)
    input_mode = str(input_mode or "full_smear").strip().lower()
    if input_mode not in ("full_smear", "single_cell"):
        return {
            "success": False,
            "error_type": "InvalidParameterError",
            "error_message": f"Unknown input_mode '{input_mode}'. Valid options: full_smear, single_cell",
        }

    is_valid, err_msg, dim_info = validate_image_file(
        image_path,
        capture_source=capture_source,
        input_mode=input_mode,
        reject_blank_or_black=True,
    )
    if not is_valid:
        return {
            "success": False,
            "error_type": "InvalidImageError",
            "error_message": err_msg
        }

    if cnn_variant not in MODELS_MAP:
        return {
            "success": False,
            "error_type": "InvalidParameterError",
            "error_message": f"Unknown CNN variant '{cnn_variant}'. Valid options: {list(MODELS_MAP.keys())}"
        }

    if crop_mode not in ("raw", "masked", "ellipse_masked"):
        return {
            "success": False,
            "error_type": "InvalidParameterError",
            "error_message": f"Unknown crop mode '{crop_mode}'. Valid options: raw, masked, ellipse_masked"
        }

    # Clamp numeric thresholds safely
    try:
        yolo_conf = max(0.01, min(0.99, float(yolo_conf)))
        yolo_iou = max(0.05, min(0.95, float(yolo_iou)))
        dedup_iou = max(0.05, min(0.99, float(dedup_iou)))
        cnn_thresh = max(0.01, min(0.99, float(cnn_thresh)))
        pad_px = max(0, min(32, int(pad_px)))
        min_box_size = max(4, min(128, int(min_box_size)))
        max_det = max(10, min(1000, int(max_det)))
    except Exception as e:
        return {
            "success": False,
            "error_type": "ThresholdValidationError",
            "error_message": f"Invalid numeric threshold parameter: {e}"
        }

    orig_basename = os.path.basename(image_path)
    stem, orig_ext = os.path.splitext(orig_basename)
    if not orig_ext:
        orig_ext = ".jpg"

    if not run_id:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uid = uuid.uuid4().hex[:6]
        mode_tag = "sc" if input_mode == "single_cell" else "fs"
        run_id = f"run_{ts}_{stem}_{mode_tag}_{cnn_variant}_{crop_mode}_{short_uid}"

    run_dir = os.path.join(DEMO_RUNS_ROOT, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Preserve a copy of the original captured/uploaded image inside run_dir so it can be
    # displayed and downloaded via /api/file without exposing external filesystem paths.
    original_copy_name = f"original_{stem}{orig_ext.lower()}"
    original_copy_path = os.path.join(run_dir, original_copy_name)
    try:
        shutil.copy2(image_path, original_copy_path)
    except Exception:
        original_copy_path = None

    tf_py = get_python_exe(VENV_TF_PYTHON)
    cmd = [
        tf_py,
        "-u",
        PIPELINE_SCRIPT,
        "--image", os.path.abspath(image_path),
        "--output-dir", os.path.abspath(run_dir),
        "--input-mode", input_mode,
        "--cnn-variant", cnn_variant,
        "--crop-mode", crop_mode,
        "--yolo-device", str(yolo_device),
        "--yolo-conf", str(yolo_conf),
        "--yolo-iou", str(yolo_iou),
        "--dedup-iou", str(dedup_iou),
        "--cnn-thresh", str(cnn_thresh),
        "--pad-px", str(pad_px),
        "--min-box-size", str(min_box_size),
        "--max-det", str(max_det)
    ]
    if test_zero_detections:
        cmd.append("--test-zero-detections")
    if inject_test_invalid_boxes:
        cmd.append("--inject-test-invalid-boxes")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error_type": "PipelineTimeoutError",
            "error_message": "Two-stage inference timed out after 300 seconds."
        }
    except Exception as e:
        return {
            "success": False,
            "error_type": "PipelineExecutionError",
            "error_message": f"Failed to launch two-stage pipeline: {e}"
        }

    if proc.returncode != 0:
        return {
            "success": False,
            "error_type": "InferenceSubprocessError",
            "error_message": f"Pipeline exited with code {proc.returncode}.",
            "stdout": proc.stdout[-2000:] if proc.stdout else "",
            "stderr": proc.stderr[-2000:] if proc.stderr else ""
        }

    summary_json_path = os.path.join(run_dir, f"field_summary_{stem}.json")
    csv_path = os.path.join(run_dir, f"cell_predictions_{stem}.csv")
    rejected_json_path = os.path.join(run_dir, f"rejected_and_excluded_{stem}.json")
    summary_txt_path = os.path.join(run_dir, f"field_summary_{stem}.txt")
    annotated_jpg_path = os.path.join(run_dir, f"annotated_{stem}.jpg")
    contact_sheet_path = os.path.join(run_dir, f"contact_sheet_{stem}.jpg")
    pdf_filename = f"SmartCell_Report_{stem}.pdf"
    pdf_report_path = os.path.join(run_dir, pdf_filename)

    if not os.path.exists(summary_json_path):
        return {
            "success": False,
            "error_type": "MissingOutputError",
            "error_message": f"Expected summary report not found at {summary_json_path}."
        }

    with open(summary_json_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    with open(rejected_json_path, "r", encoding="utf-8") as f:
        rejected_data = json.load(f)

    txt_report = ""
    if os.path.exists(summary_txt_path):
        with open(summary_txt_path, "r", encoding="utf-8") as f:
            txt_report = f.read()

    cell_rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                cell_rows.append(r)

    # Attach pre-inference image quality diagnostics and capture source to summary_data
    quality_info = (dim_info or {}).get("quality") or {}
    summary_data["input_mode"] = input_mode
    summary_data["capture_source"] = capture_source
    summary_data["image_quality"] = quality_info
    summary_data["quality_warnings"] = quality_info.get("warnings", [])

    # Check if ground-truth reference exists for this image stem (only applicable in full_smear mode)
    gt_ref = inspect_ground_truth_for_stem(stem) if input_mode == "full_smear" else {"has_ground_truth": False}

    # Generate complete PDF report (with ground-truth comparison if available, and quality warnings)
    pdf_status = {
        "generated": False,
        "filename": pdf_filename,
        "path": None,
        "size_bytes": 0,
        "message": None,
        "error": None
    }
    final_screening = {
        "statement": "",
        "disclaimer": (
            "This AI screening result does not confirm or rule out malaria in a patient. "
            "Clinical assessment and appropriate diagnostic testing are required."
        ),
    }
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import importlib
        import pdf_report_generator
        importlib.reload(pdf_report_generator)

        counts_d = summary_data.get("counts") or {}
        par_est_d = summary_data.get("parasitemia_estimate") or {}
        final_screening["statement"] = pdf_report_generator.format_final_screening_statement(
            valid_rbcs=counts_d.get("valid_candidate_rbc_crops", 0),
            par_rbcs=counts_d.get("cnn_parasitized_rbc_count", 0),
            est_par_pct=par_est_d.get("estimated_parasitemia_percent", 0.0),
            input_mode=input_mode,
            single_cell_result=summary_data.get("single_cell_result"),
        )
        final_screening["disclaimer"] = pdf_report_generator.FINAL_SCREENING_DISCLAIMER
        summary_data["final_ai_screening_result"] = final_screening

        generated_pdf = pdf_report_generator.generate_pdf_report(
            output_pdf_path=pdf_report_path,
            summary_data=summary_data,
            cell_predictions=cell_rows,
            rejected_and_excluded=rejected_data,
            annotated_image_path=annotated_jpg_path if os.path.exists(annotated_jpg_path) else None,
            ground_truth_reference=gt_ref,
            run_id=run_id
        )
        if os.path.isfile(generated_pdf) and os.path.getsize(generated_pdf) > 0:
            pdf_size = os.path.getsize(generated_pdf)
            img_cmp = pdf_report_generator.compare_source_and_pdf_embedded_image(
                source_image_path=image_path,
                pdf_bytes_or_path=generated_pdf,
                annotated_image_path=annotated_jpg_path if os.path.exists(annotated_jpg_path) else None,
            )
            pdf_status = {
                "generated": True,
                "filename": pdf_filename,
                "path": generated_pdf,
                "size_bytes": pdf_size,
                "message": f"Complete PDF analysis report ({pdf_filename}, {round(pdf_size / 1024.0, 1)} KB) generated and ready for download.",
                "source_vs_pdf_image_comparison": img_cmp,
                "error": None
            }
            summary_data.setdefault("output_files", {})["pdf_report"] = generated_pdf
        else:
            pdf_status["error"] = "PDF generator returned an empty or missing file."
    except Exception as pdf_exc:
        pdf_status["error"] = f"PDF report generation failed ({type(pdf_exc).__name__}: {pdf_exc})"

    # Save enriched summary_data back to disk so disk JSON and API payload match 100%
    try:
        with open(summary_json_path, "w", encoding="utf-8") as jf:
            json.dump(summary_data, jf, indent=2)
    except Exception:
        pass

    return {
        "success": True,
        "run_id": run_id,
        "run_directory": run_dir,
        "stem": stem,
        "input_mode": input_mode,
        "capture_source": capture_source,
        "image_quality": quality_info,
        "final_screening_result": final_screening,
        "summary": summary_data,
        "rejected_and_excluded": rejected_data,
        "cell_predictions": cell_rows,
        "text_report": txt_report,
        "ground_truth_reference": gt_ref,
        "pdf_report": pdf_status,
        "urls": {
            "export_pdf": f"/api/file?run_id={run_id}&file={pdf_filename}&download=1" if pdf_status["generated"] else None,
            "original_image": f"/api/file?run_id={run_id}&file={original_copy_name}" if (original_copy_path and os.path.exists(original_copy_path)) else None,
            "annotated_image": f"/api/file?run_id={run_id}&file=annotated_{stem}.jpg" if os.path.exists(annotated_jpg_path) else None,
            "contact_sheet": f"/api/file?run_id={run_id}&file=contact_sheet_{stem}.jpg" if os.path.exists(contact_sheet_path) else None,
            "export_csv": f"/api/file?run_id={run_id}&file=cell_predictions_{stem}.csv&download=1",
            "export_summary_json": f"/api/file?run_id={run_id}&file=field_summary_{stem}.json&download=1",
            "export_rejected_json": f"/api/file?run_id={run_id}&file=rejected_and_excluded_{stem}.json&download=1",
            "export_txt": f"/api/file?run_id={run_id}&file=field_summary_{stem}.txt&download=1"
        }
    }


# ---------------------------------------------------------------------------
# Embedded Single-Page Web Application HTML/CSS/JS
# ---------------------------------------------------------------------------
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SmartCell — AI-Assisted Blood Cell Screening Research Platform</title>
<style>
  :root {
    --bg: #090e17;
    --panel: #111a2e;
    --panel-hover: #15223c;
    --panel-alt: #16233d;
    --border: #1e2f4e;
    --border-light: #2a3f66;
    --text: #e2e8f0;
    --text-bright: #ffffff;
    --muted: #8ea3bf;
    --accent: #0ea5e9;
    --accent-hover: #38bdf8;
    --accent-glow: rgba(14, 165, 233, 0.15);
    --teal: #06b6d4;
    --leukemia: #f43f5e;
    --leukemia-dark: #be123c;
    --leukemia-glow: rgba(244, 63, 94, 0.16);
    --danger: #ef4444;
    --danger-bg: rgba(239, 68, 68, 0.12);
    --success: #10b981;
    --success-bg: rgba(16, 185, 129, 0.12);
    --warn: #f59e0b;
    --warn-bg: rgba(245, 158, 11, 0.12);
    --purple: #a855f7;
    --radius-sm: 8px;
    --radius: 12px;
    --radius-lg: 16px;
    --shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.45);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }

  /* Compact Professional Disclaimer Bar */
  .disclaimer-banner {
    background: #151119;
    border-bottom: 1px solid #4a1d24;
    color: #fecdd3;
    padding: 7px 20px;
    font-size: 0.80rem;
    font-weight: 500;
    text-align: center;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
  }
  .disclaimer-pill {
    background: #9f1239;
    color: #fff;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.6px;
    text-transform: uppercase;
  }

  /* Header */
  .header {
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    padding: 14px 28px;
  }
  .header-inner {
    max-width: 1720px;
    margin: 0 auto;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .brand-icon-box {
    width: 44px;
    height: 44px;
    border-radius: var(--radius-sm);
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.2), rgba(6, 182, 212, 0.1));
    border: 1px solid rgba(14, 165, 233, 0.35);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #38bdf8;
    flex-shrink: 0;
  }
  .brand-title-wrap h1 {
    font-size: 1.35rem;
    font-weight: 800;
    color: var(--text-bright);
    letter-spacing: -0.3px;
    line-height: 1.2;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .brand-tag {
    font-size: 0.70rem;
    font-weight: 700;
    background: rgba(14, 165, 233, 0.15);
    border: 1px solid rgba(14, 165, 233, 0.35);
    color: #38bdf8;
    padding: 2px 7px;
    border-radius: 999px;
    letter-spacing: 0.4px;
    text-transform: uppercase;
  }
  .brand-title-wrap .sub-primary {
    font-size: 0.86rem;
    font-weight: 600;
    color: #cbd5e1;
    margin-top: 1px;
  }
  .brand-title-wrap .sub-secondary {
    font-size: 0.78rem;
    color: var(--muted);
  }

  /* Diagnostics Dropdown in Header */
  .diag-details {
    position: relative;
    font-size: 0.80rem;
  }
  .diag-summary {
    cursor: pointer;
    background: var(--panel-alt);
    border: 1px solid var(--border);
    padding: 6px 14px;
    border-radius: var(--radius-sm);
    color: var(--muted);
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 7px;
    user-select: none;
    transition: all 0.15s ease;
  }
  .diag-summary:hover {
    color: var(--text-bright);
    border-color: var(--accent);
  }
  .diag-body {
    position: absolute;
    right: 0;
    top: calc(100% + 6px);
    background: var(--panel);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-sm);
    padding: 12px;
    width: 320px;
    z-index: 100;
    box-shadow: var(--shadow);
  }
  .hw-badges {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .badge {
    font-size: 0.75rem;
    padding: 5px 10px;
    border-radius: 6px;
    background: var(--panel-alt);
    border: 1px solid var(--border);
    font-weight: 600;
    display: block;
    word-break: break-all;
  }
  .badge.gpu-ok { border-color: var(--success); color: #86efac; }
  .badge.cpu-ok { border-color: var(--accent); color: #7dd3fc; }

  /* Workflow Steps Bar */
  .workflow-stepper-wrap {
    max-width: 1720px;
    margin: 16px auto 0;
    padding: 0 28px;
  }
  .workflow-stepper {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 8px 18px;
    gap: 8px;
    overflow-x: auto;
  }
  .step-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.80rem;
    font-weight: 600;
    color: var(--muted);
    white-space: nowrap;
  }
  .step-item.active {
    color: var(--text-bright);
  }
  .step-num {
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: var(--panel-alt);
    border: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--muted);
  }
  .step-item.active .step-num {
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
  }
  .step-sep {
    color: #334155;
    font-size: 0.85rem;
  }

  /* Disease Mode Selection Cards */
  .disease-selection-wrap {
    max-width: 1720px;
    margin: 16px auto 0;
    padding: 0 28px;
  }
  .disease-cards-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }
  @media (max-width: 860px) {
    .disease-cards-grid { grid-template-columns: 1fr; }
  }
  .disease-tab {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 22px;
    cursor: pointer;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    text-align: left;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    gap: 14px;
    min-height: 125px;
  }
  .disease-tab:hover {
    border-color: var(--border-light);
    transform: translateY(-1px);
    background: var(--panel-hover);
  }
  .disease-tab.active {
    background: linear-gradient(145deg, #101c30, #13243f);
    border-color: var(--accent);
    box-shadow: 0 0 20px var(--accent-glow);
  }
  .disease-tab.active-leukemia {
    background: linear-gradient(145deg, #1b121e, #261628);
    border-color: var(--leukemia);
    box-shadow: 0 0 20px var(--leukemia-glow);
  }
  .disease-card-top {
    display: flex;
    align-items: flex-start;
    gap: 14px;
  }
  .disease-icon {
    width: 44px;
    height: 44px;
    border-radius: var(--radius-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-size: 1.4rem;
  }
  .icon-malaria {
    background: rgba(14, 165, 233, 0.12);
    border: 1px solid rgba(14, 165, 233, 0.3);
    color: #38bdf8;
  }
  .icon-leukemia {
    background: rgba(244, 63, 94, 0.12);
    border: 1px solid rgba(244, 63, 94, 0.3);
    color: #fb7185;
  }
  .disease-tab.active .icon-malaria {
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
  }
  .disease-tab.active-leukemia .icon-leukemia {
    background: var(--leukemia);
    color: #fff;
    border-color: var(--leukemia);
  }
  .disease-info h3 {
    font-size: 1.08rem;
    font-weight: 700;
    color: var(--text-bright);
    margin-bottom: 3px;
  }
  .disease-info p {
    font-size: 0.84rem;
    color: var(--muted);
    line-height: 1.4;
  }
  .disease-tags {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }
  .disease-tag {
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 9px;
    border-radius: 999px;
    background: var(--panel-alt);
    border: 1px solid var(--border);
    color: var(--muted);
  }
  .disease-tab.active .disease-tag {
    background: rgba(14, 165, 233, 0.12);
    border-color: rgba(14, 165, 233, 0.3);
    color: #bae6fd;
  }
  .disease-tab.active-leukemia .disease-tag {
    background: rgba(244, 63, 94, 0.12);
    border-color: rgba(244, 63, 94, 0.3);
    color: #fecdd3;
  }

  /* Mode Subheader Bar */
  .mode-view-header {
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
  }
  .mode-view-header h2 {
    font-size: 1.22rem;
    font-weight: 700;
    color: var(--text-bright);
  }
  .mode-view-header p {
    font-size: 0.84rem;
    color: var(--muted);
  }

  /* Main Two-Column Layout */
  .layout {
    display: grid;
    grid-template-columns: 410px 1fr;
    gap: 20px;
    max-width: 1720px;
    margin: 18px auto 0;
    padding: 0 28px 40px;
  }
  @media (max-width: 1150px) {
    .layout { grid-template-columns: 1fr; }
  }

  /* Cards */
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    margin-bottom: 18px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.2);
  }
  .card-title {
    font-size: 0.94rem;
    font-weight: 700;
    color: var(--text-bright);
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  /* Forms & Inputs */
  label {
    display: block;
    font-size: 0.81rem;
    font-weight: 600;
    color: #cbd5e1;
    margin-bottom: 6px;
    margin-top: 12px;
  }
  select, input[type="number"], input[type="text"], input[type="file"] {
    width: 100%;
    padding: 10px 12px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text);
    font-size: 0.86rem;
    transition: border-color 0.15s;
  }
  select:focus, input[type="text"]:focus, input[type="number"]:focus {
    outline: none;
    border-color: var(--accent);
  }
  .grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }
  .hint {
    font-size: 0.77rem;
    color: var(--muted);
    margin-top: 5px;
    line-height: 1.4;
  }

  /* Buttons */
  .btn-primary {
    width: 100%;
    margin-top: 16px;
    padding: 12px 18px;
    background: linear-gradient(135deg, #0284c7, #2563eb);
    color: #fff;
    border: none;
    border-radius: var(--radius-sm);
    font-size: 0.94rem;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.15s ease;
    box-shadow: 0 2px 10px rgba(2, 132, 199, 0.3);
  }
  .btn-primary:hover:not(:disabled) {
    opacity: 0.94;
    transform: translateY(-1px);
  }
  .btn-primary:disabled { opacity: 0.45; cursor: not-allowed; }

  .btn-primary-leukemia {
    width: 100%;
    margin-top: 16px;
    padding: 12px 18px;
    background: linear-gradient(135deg, #be123c, #e11d48);
    color: #fff;
    border: none;
    border-radius: var(--radius-sm);
    font-size: 0.94rem;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.15s ease;
    box-shadow: 0 2px 10px rgba(225, 29, 72, 0.3);
  }
  .btn-primary-leukemia:hover:not(:disabled) {
    opacity: 0.94;
    transform: translateY(-1px);
  }
  .btn-primary-leukemia:disabled { opacity: 0.45; cursor: not-allowed; }

  .btn-export {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    background: var(--panel-alt);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: var(--radius-sm);
    text-decoration: none;
    font-size: 0.82rem;
    font-weight: 600;
    margin-right: 8px;
    margin-bottom: 8px;
    transition: all 0.15s;
  }
  .btn-export:hover {
    border-color: var(--accent);
    color: #fff;
    background: var(--panel-hover);
  }
  .btn-pdf-primary {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 11px 20px;
    background: linear-gradient(135deg, #059669, #0284c7);
    border: 1px solid #34d399;
    color: #ffffff;
    border-radius: var(--radius-sm);
    text-decoration: none;
    font-size: 0.92rem;
    font-weight: 700;
    margin-right: 12px;
    margin-bottom: 8px;
    box-shadow: 0 2px 12px rgba(16, 185, 129, 0.28);
    transition: all 0.15s;
  }
  .btn-pdf-primary:hover {
    opacity: 0.95;
    border-color: #ffffff;
    color: #ffffff;
    transform: translateY(-1px);
  }

  /* Alerts */
  .alert-success {
    background: var(--success-bg);
    border: 1px solid var(--success);
    color: #a7f3d0;
    padding: 11px 15px;
    border-radius: var(--radius-sm);
    margin-bottom: 14px;
    font-size: 0.84rem;
  }
  .alert-error {
    background: var(--danger-bg);
    border: 1px solid var(--danger);
    color: #fecaca;
    padding: 12px 16px;
    border-radius: var(--radius-sm);
    margin-bottom: 16px;
    font-size: 0.85rem;
  }
  .alert-info {
    background: rgba(14, 165, 233, 0.10);
    border: 1px solid rgba(14, 165, 233, 0.3);
    color: #bae6fd;
    padding: 12px 16px;
    border-radius: var(--radius-sm);
    margin-bottom: 16px;
    font-size: 0.85rem;
  }
  .alert-warn {
    background: var(--warn-bg);
    border: 1px solid var(--warn);
    color: #fde68a;
    padding: 11px 15px;
    border-radius: var(--radius-sm);
    margin-top: 10px;
    margin-bottom: 12px;
    font-size: 0.82rem;
  }

  /* Mode Switcher Buttons (Inside input panel) */
  .mode-switcher {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 14px;
    background: var(--bg);
    padding: 5px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
  }
  .mode-switcher-3 {
    grid-template-columns: repeat(3, 1fr);
  }
  .mode-btn {
    padding: 9px 12px;
    border: none;
    border-radius: 6px;
    background: transparent;
    color: var(--muted);
    font-size: 0.83rem;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.15s;
    text-align: center;
  }
  .mode-btn:hover { color: var(--text-bright); }
  .mode-btn.active {
    background: linear-gradient(135deg, #0284c7, #2563eb);
    color: #fff;
    box-shadow: 0 2px 8px rgba(2, 132, 199, 0.35);
  }
  .mode-btn.active-leukemia {
    background: linear-gradient(135deg, #be123c, #e11d48) !important;
    color: #fff !important;
    box-shadow: 0 2px 8px rgba(225, 29, 72, 0.35) !important;
  }

  /* Camera Status & Stage */
  .cam-status-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    border-radius: var(--radius-sm);
    background: var(--panel-alt);
    border: 1px solid var(--border);
    font-size: 0.80rem;
    margin-bottom: 10px;
    gap: 8px;
  }
  .cam-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    display: inline-block;
    background: var(--muted);
    margin-right: 6px;
  }
  .cam-dot.live { background: var(--success); box-shadow: 0 0 8px var(--success); }
  .cam-dot.captured { background: var(--accent); box-shadow: 0 0 8px var(--accent); }
  .cam-dot.warn { background: var(--warn); }
  .cam-dot.err { background: var(--danger); }

  .video-stage {
    width: 100%;
    background: #05080e;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    overflow: hidden;
    position: relative;
    min-height: 210px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 8px;
    margin-bottom: 10px;
  }
  .video-stage video, .video-stage img {
    width: 100%;
    max-height: 280px;
    object-fit: contain;
    display: block;
  }
  .video-placeholder {
    padding: 24px 16px;
    text-align: center;
    color: var(--muted);
    font-size: 0.82rem;
    line-height: 1.5;
  }
  .btn-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: 8px;
  }
  .btn-secondary {
    padding: 9px 12px;
    background: var(--panel-alt);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    font-size: 0.83rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
  }
  .btn-secondary:hover:not(:disabled) {
    border-color: var(--accent);
    color: #fff;
    background: var(--panel-hover);
  }
  .btn-secondary:disabled { opacity: 0.45; cursor: not-allowed; }
  .btn-danger-outline {
    padding: 9px 12px;
    background: rgba(239, 68, 68, 0.10);
    color: #fca5a5;
    border: 1px solid var(--danger);
    border-radius: var(--radius-sm);
    font-size: 0.83rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
  }
  .btn-danger-outline:disabled { opacity: 0.45; cursor: not-allowed; }
  .btn-capture-analyze {
    width: 100%;
    margin-top: 10px;
    padding: 12px;
    background: linear-gradient(135deg, #059669, #0284c7);
    color: #fff;
    border: 1px solid #34d399;
    border-radius: var(--radius-sm);
    font-size: 0.94rem;
    font-weight: 700;
    cursor: pointer;
    box-shadow: 0 2px 10px rgba(16, 185, 129, 0.25);
    transition: all 0.15s;
  }
  .btn-capture-analyze:hover:not(:disabled) { opacity: 0.94; transform: translateY(-1px); }
  .btn-capture-analyze:disabled { opacity: 0.45; cursor: not-allowed; }
  .btn-link-return {
    background: none;
    border: none;
    color: var(--accent);
    font-size: 0.80rem;
    font-weight: 600;
    cursor: pointer;
    text-decoration: underline;
    margin-top: 10px;
    display: inline-block;
  }

  /* Preset Grid in ALL Mode */
  .preset-btn-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
    margin-bottom: 10px;
  }
  .leukemia-preset-btn {
    text-align: center;
    justify-content: center;
    font-size: 0.80rem;
    padding: 7px 8px;
    margin: 0;
  }
  .leukemia-preset-btn.selected {
    border-color: var(--leukemia) !important;
    background: rgba(244, 63, 94, 0.22) !important;
    color: #fff !important;
    font-weight: 700;
  }

  /* KPI Grid */
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 12px;
    margin-bottom: 18px;
  }
  .kpi-box {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 18px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.25);
    transition: transform 0.15s ease;
  }
  .kpi-box:hover {
    transform: translateY(-1px);
  }
  .kpi-box .kpi-label {
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--muted);
    font-weight: 700;
  }
  .kpi-box .kpi-val {
    font-size: 1.75rem;
    font-weight: 800;
    margin-top: 4px;
    color: var(--text-bright);
    letter-spacing: -0.3px;
  }
  .kpi-box .kpi-sub {
    font-size: 0.77rem;
    color: var(--muted);
    margin-top: 3px;
  }
  .kpi-total { border-left: 4px solid var(--accent); }
  .kpi-total .kpi-val { color: #38bdf8; }
  .kpi-par { border-left: 4px solid var(--danger); }
  .kpi-par .kpi-val { color: #f87171; }
  .kpi-uninf { border-left: 4px solid var(--success); }
  .kpi-uninf .kpi-val { color: #4ade80; }
  .kpi-rate { border-left: 4px solid var(--teal); }
  .kpi-rate .kpi-val { color: #22d3ee; }
  .kpi-wbc { border-left: 4px solid var(--warn); }
  .kpi-wbc .kpi-val { color: #fbbf24; }
  .kpi-rej { border-left: 4px solid var(--purple); }
  .kpi-rej .kpi-val { color: #c084fc; }

  /* Final Screening Result Card */
  .final-screening-card {
    padding: 18px 22px;
    border-radius: var(--radius);
    margin-bottom: 18px;
    border: 1px solid var(--border);
    background: var(--panel);
  }
  .final-screening-card.par-flagged {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.12), rgba(185, 28, 28, 0.16));
    border-color: var(--danger);
  }
  .final-screening-card.par-zero {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.10), rgba(5, 150, 105, 0.14));
    border-color: var(--success);
  }

  /* Leukemia Diagnosis Banner */
  .diagnosis-banner {
    padding: 20px 24px;
    border-radius: var(--radius);
    margin-bottom: 18px;
    border: 1px solid transparent;
  }
  .diagnosis-banner.banner-blast {
    background: linear-gradient(135deg, rgba(244, 63, 94, 0.18), rgba(190, 18, 60, 0.22));
    border-color: #f43f5e !important;
    color: #fecdd3;
  }
  .diagnosis-banner.banner-normal {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.16), rgba(5, 150, 105, 0.20));
    border-color: #10b981 !important;
    color: #a7f3d0;
  }

  /* Leukemia Probability Bars */
  .prob-bar-row {
    margin-bottom: 16px;
  }
  .prob-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.86rem;
    font-weight: 700;
    margin-bottom: 6px;
  }
  .prob-track {
    width: 100%;
    height: 14px;
    background: var(--panel-alt);
    border-radius: 7px;
    overflow: hidden;
    border: 1px solid var(--border);
  }
  .prob-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
  }
  .prob-fill-normal { background: linear-gradient(90deg, #059669, #10b981); }
  .prob-fill-blast { background: linear-gradient(90deg, #e11d48, #f43f5e); }

  /* Tabs for Visual Inspection */
  .tabs {
    display: flex;
    gap: 6px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 16px;
    flex-wrap: wrap;
  }
  .tab-btn {
    padding: 9px 16px;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    color: var(--muted);
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
  }
  .tab-btn:hover { color: var(--text-bright); }
  .tab-btn.active {
    color: var(--accent);
    border-bottom-color: var(--accent);
  }
  .tab-pane { display: none; }
  .tab-pane.active { display: block; }

  .img-viewer {
    width: 100%;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
    background: #060911;
    max-height: 720px;
    object-fit: contain;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.83rem;
  }
  th, td {
    padding: 9px 11px;
    border-bottom: 1px solid var(--border);
    text-align: left;
  }
  th {
    background: var(--panel-alt);
    color: #cbd5e1;
    font-weight: 600;
    position: sticky;
    top: 0;
  }
  .pill {
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 700;
  }
  .pill-par { background: var(--danger-bg); color: #fca5a5; border: 1px solid var(--danger); }
  .pill-uninf { background: var(--success-bg); color: #86efac; border: 1px solid var(--success); }
  .crop-thumb {
    width: 50px;
    height: 50px;
    object-fit: cover;
    border-radius: 6px;
    border: 1px solid var(--border);
  }
  pre.report-box {
    background: #070b13;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 14px;
    font-family: Consolas, Monaco, monospace;
    font-size: 0.80rem;
    overflow-x: auto;
    color: #cbd5e1;
    white-space: pre-wrap;
  }

  /* Collapsible Accordions for Advanced / Technical Info */
  details.clean-accordion {
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--panel-alt);
    margin-top: 14px;
    overflow: hidden;
  }
  details.clean-accordion summary {
    padding: 10px 14px;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--muted);
    cursor: pointer;
    user-select: none;
    transition: color 0.15s;
    list-style: none;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  details.clean-accordion summary::-webkit-details-marker { display: none; }
  details.clean-accordion summary:after {
    content: "▼";
    font-size: 0.65rem;
    transition: transform 0.2s;
  }
  details.clean-accordion[open] summary:after {
    transform: rotate(180deg);
  }
  details.clean-accordion summary:hover {
    color: var(--text-bright);
  }
  details.clean-accordion .accordion-body {
    padding: 14px;
    border-top: 1px solid var(--border);
    background: var(--panel);
  }
  .checkbox-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 8px;
    font-size: 0.81rem;
    color: var(--muted);
  }
  .checkbox-row input { width: auto; }
</style>
</head>
<body>

<!-- Research Prototype Disclaimer Banner -->
<div class="disclaimer-banner">
  <span class="disclaimer-pill">Research Prototype</span>
  <span>Research Prototype — For academic and experimental use only. Not validated for clinical diagnosis or treatment decisions.</span>
</div>

<!-- Main Application Header -->
<header class="header">
  <div class="header-inner">
    <div class="brand">
      <div class="brand-icon-box">
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="10"></circle>
          <path d="m4.93 4.93 4.24 4.24"></path>
          <path d="m14.83 9.17 4.24-4.24"></path>
          <path d="m14.83 14.83 4.24 4.24"></path>
          <path d="m9.17 14.83-4.24 4.24"></path>
          <circle cx="12" cy="12" r="3"></circle>
        </svg>
      </div>
      <div class="brand-title-wrap">
        <h1>
          <span>SmartCell</span>
          <span class="brand-tag">Research Platform</span>
        </h1>
        <p class="sub-primary">AI-Assisted Blood Cell Screening Research Platform</p>
        <p class="sub-secondary">Intelligent microscopy research for malaria and leukemia cell analysis.</p>
      </div>
    </div>
    <div>
      <details class="diag-details">
        <summary class="diag-summary">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20v-6M6 20V10M18 20V4"/></svg>
          System Diagnostics
        </summary>
        <div class="diag-body">
          <div style="font-size:0.75rem; font-weight:700; color:var(--muted); text-transform:uppercase; margin-bottom:6px;">Environment &amp; Hardware</div>
          <div class="hw-badges" id="hwBadges">
            <span class="badge">Checking environment...</span>
          </div>
        </div>
      </details>
    </div>
  </div>
</header>

<!-- Visual Workflow Progress Indicator -->
<div class="workflow-stepper-wrap">
  <div class="workflow-stepper">
    <div class="step-item active">
      <span class="step-num">1</span>
      <span>Screening Mode</span>
    </div>
    <span class="step-sep">&rarr;</span>
    <div class="step-item active">
      <span class="step-num">2</span>
      <span>Image Source</span>
    </div>
    <span class="step-sep">&rarr;</span>
    <div class="step-item">
      <span class="step-num">3</span>
      <span>Preview &amp; Verify</span>
    </div>
    <span class="step-sep">&rarr;</span>
    <div class="step-item">
      <span class="step-num">4</span>
      <span>AI Analysis</span>
    </div>
    <span class="step-sep">&rarr;</span>
    <div class="step-item">
      <span class="step-num">5</span>
      <span>Results &amp; Report</span>
    </div>
  </div>
</div>

<!-- Disease Mode Selection Cards -->
<div class="disease-selection-wrap">
  <div class="disease-cards-grid">
    <!-- Card 1: Malaria Screening -->
    <div class="disease-tab active" id="diseaseTabMalaria" onclick="switchDiseaseScope('malaria')" role="button" tabindex="0">
      <div class="disease-card-top">
        <div class="disease-icon icon-malaria">🔬</div>
        <div class="disease-info">
          <h3>Malaria Screening</h3>
          <p>Analyze blood-smear images to identify red blood cells and screen for malaria-associated patterns.</p>
        </div>
      </div>
      <div class="disease-tags">
        <span class="disease-tag">Blood Smear Analysis</span>
        <span class="disease-tag">RBC Detection</span>
        <span class="disease-tag">Cell Classification</span>
      </div>
    </div>

    <!-- Card 2: ALL Leukemia Screening -->
    <div class="disease-tab" id="diseaseTabLeukemia" onclick="switchDiseaseScope('leukemia')" role="button" tabindex="0">
      <div class="disease-card-top">
        <div class="disease-icon icon-leukemia">🩸</div>
        <div class="disease-info">
          <h3>ALL Leukemia Screening</h3>
          <p>Analyze individual white blood cell images for Normal Lymphocyte vs Lymphoblast classification.</p>
        </div>
      </div>
      <div class="disease-tags">
        <span class="disease-tag">Single-Cell Analysis</span>
        <span class="disease-tag">White Blood Cells</span>
        <span class="disease-tag">AI Classification</span>
      </div>
    </div>
  </div>
</div>

<!-- ========================================================================= -->
<!-- MODE 1: MALARIA WORKFLOW                                                  -->
<!-- ========================================================================= -->
<main class="layout" id="malariaWorkflow">
  <!-- Left Control Panel -->
  <section>
    <div class="card">
      <div class="card-title">
        <span>Image Source</span>
        <span style="font-size:0.75rem; color:var(--accent);">Upload or Live Microscope</span>
      </div>

      <!-- Mode Switcher -->
      <div class="mode-switcher" role="tablist" aria-label="Input Mode Selector">
        <button type="button" class="mode-btn active" id="modeManualBtn" onclick="switchInputMode('manual')">
          Upload Image
        </button>
        <button type="button" class="mode-btn" id="modeMicroscopeBtn" onclick="switchInputMode('microscope')">
          Live Microscope
        </button>
      </div>

      <!-- Mode A: Upload & Preset Workflow -->
      <div id="manualUploadPanel">
        <label for="specimenInputMode">Analysis Mode</label>
        <select id="specimenInputMode" onchange="onSpecimenModeChange()">
          <option value="full_smear" selected>Full-Smear Mode — Multi-Cell Blood Smear</option>
          <option value="single_cell">Single-Cell Mode — Single RBC Patch</option>
        </select>
        <div class="hint" id="specimenModeHint">
          Full-Smear Mode automatically detects erythrocytes, classifies each cell, and calculates estimated parasitemia.
        </div>

        <label for="imageUpload" id="imageUploadLabel">Upload Blood-Smear Image (.jpg, .png, .tif)</label>
        <input type="file" id="imageUpload" accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff">
        <div class="hint" id="presetHintText">Or select a held-out benchmark specimen:</div>

        <label for="presetSelect" id="presetSelectLabel">Benchmark Test Specimen</label>
        <select id="presetSelect">
          <option value="">-- Use Uploaded Image Above --</option>
        </select>

        <button class="btn-primary" id="runBtn" onclick="runAnalysis()">
          Analyze Blood Smear
        </button>
      </div>

      <!-- Mode B: Live Microscope Workflow -->
      <div id="liveMicroscopePanel" style="display:none;">
        <div class="cam-status-bar" id="microscopeStatusBar">
          <div>
            <span class="cam-dot" id="microscopeStatusDot"></span>
            <strong id="microscopeStatusText">Camera Idle — Click Start Camera</strong>
          </div>
          <span id="microscopeResBadge" style="font-size:0.74rem; color:var(--muted);">No stream</span>
        </div>

        <label for="cameraDeviceSelect" style="margin-top:6px;">USB Microscope Device</label>
        <div style="display:flex; gap:6px;">
          <select id="cameraDeviceSelect" onchange="onCameraDeviceChange()" disabled>
            <option value="">-- Start camera to list video devices --</option>
          </select>
          <button type="button" class="btn-secondary" id="refreshDevicesBtn" onclick="refreshCameraDevices()" title="Refresh camera list" style="width:auto; white-space:nowrap;">
            Refresh
          </button>
        </div>
        <div class="hint">Standard UVC USB digital microscopes connect automatically upon browser permission.</div>

        <!-- Start / Stop Camera Controls -->
        <div class="btn-row">
          <button type="button" class="btn-secondary" id="startCameraBtn" onclick="startMicroscopeCamera()">
            Start Camera
          </button>
          <button type="button" class="btn-danger-outline" id="stopCameraBtn" onclick="stopMicroscopeCamera()" disabled>
            Stop Camera
          </button>
        </div>

        <!-- Live Video Preview -->
        <div class="video-stage" id="liveVideoStage">
          <div class="video-placeholder" id="videoPlaceholderText">
            Live USB Microscope preview is inactive.<br>
            Click <strong>Start Camera</strong> to connect your digital microscope.
          </div>
          <video id="microscopeVideo" autoplay playsinline muted style="display:none;"></video>
        </div>

        <!-- Capture / Retake Controls -->
        <div class="btn-row">
          <button type="button" class="btn-secondary" id="captureImageBtn" onclick="captureMicroscopeFrame()" disabled>
            Capture Image
          </button>
          <button type="button" class="btn-secondary" id="retakeImageBtn" onclick="retakeMicroscopeFrame()" disabled>
            Retake
          </button>
        </div>

        <!-- Captured Frame Inspection Preview -->
        <div id="capturedPreviewSection" style="display:none; margin-top:10px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <span style="font-size:0.80rem; font-weight:700; color:#bae6fd;">Captured Frame Preview</span>
            <span id="capturedFrameInfo" style="font-size:0.74rem; color:var(--muted);"></span>
          </div>
          <div class="video-stage" style="margin-top:2px; margin-bottom:6px;">
            <img id="capturedPreviewImg" alt="Captured Microscope Frame Preview">
          </div>
        </div>

        <!-- Hidden Canvas -->
        <canvas id="microscopeCaptureCanvas" style="display:none;"></canvas>

        <!-- Quality Warnings -->
        <div id="microscopeQualityBox"></div>

        <!-- Capture & Analyze Button -->
        <button type="button" class="btn-capture-analyze" id="captureAndAnalyzeBtn" onclick="captureAndAnalyzeMicroscope()" disabled>
          Capture &amp; Analyze Image
        </button>

        <div style="text-align:center;">
          <button type="button" class="btn-link-return" id="returnToManualBtn" onclick="returnToManualUpload()">
            &larr; Return to Upload Workflow
          </button>
        </div>
      </div>

      <!-- Advanced Technical Settings (Collapsed) -->
      <details class="clean-accordion">
        <summary>Advanced Inference Parameters (Optional)</summary>
        <div class="accordion-body">
          <label for="cnnVariant">Classifier Checkpoint</label>
          <select id="cnnVariant" onchange="onVariantChange()">
            <option value="finetuned_raw" selected>Fine-Tuned Raw CNN (Recommended)</option>
            <option value="finetuned_masked">Fine-Tuned Masked CNN</option>
            <option value="original">Original Baseline CNN</option>
          </select>
          <div class="hint" id="variantDesc">Fine-tuned on native rectangular full-smear crops (Test Acc: 98.35%).</div>

          <label for="cropMode">Crop Mode</label>
          <select id="cropMode">
            <option value="raw" selected>raw — Rectangular RGB Crop</option>
            <option value="masked">masked — Elliptical Masked</option>
          </select>

          <label for="yoloDevice">Compute Device</label>
          <select id="yoloDevice">
            <option value="0" selected>Auto / Available GPU (CUDA:0)</option>
            <option value="cpu">Force CPU</option>
          </select>

          <div class="grid-2">
            <div>
              <label for="yoloConf">Detection Conf</label>
              <input type="number" id="yoloConf" value="0.25" min="0.05" max="0.95" step="0.05">
            </div>
            <div>
              <label for="yoloIou">NMS IoU</label>
              <input type="number" id="yoloIou" value="0.45" min="0.10" max="0.90" step="0.05">
            </div>
            <div>
              <label for="dedupIou">Dedup IoU</label>
              <input type="number" id="dedupIou" value="0.50" min="0.10" max="0.95" step="0.05">
            </div>
            <div>
              <label for="cnnThresh">CNN Thresh</label>
              <input type="number" id="cnnThresh" value="0.50" min="0.05" max="0.95" step="0.05">
            </div>
            <div>
              <label for="padPx">Padding (px)</label>
              <input type="number" id="padPx" value="4" min="0" max="16" step="1">
            </div>
            <div>
              <label for="minBoxSize">Min Box (px)</label>
              <input type="number" id="minBoxSize" value="16" min="8" max="64" step="2">
            </div>
          </div>

          <div class="checkbox-row">
            <input type="checkbox" id="testZeroDet">
            <span>Simulate Zero Detections</span>
          </div>
          <div class="checkbox-row">
            <input type="checkbox" id="injectInvalidBoxes">
            <span>Simulate Edge-Case WBC / Invalid Boxes</span>
          </div>
        </div>
      </details>
    </div>
  </section>

  <!-- Right Output Dashboard -->
  <section>
    <div id="errorContainer"></div>

    <div id="welcomeBanner" class="alert-info">
      <strong>Ready for Screening:</strong> Upload a microscope blood-smear image, select a benchmark test image, or use the <strong>Live Microscope</strong> tab to capture and analyze a blood field.
    </div>

    <div id="resultsContainer" style="display:none;">
      <!-- Final Screening Result Banner -->
      <div id="finalScreeningCard" class="final-screening-card"></div>

      <!-- Pre-Inference Quality Warnings -->
      <div id="resultQualityWarnings"></div>

      <!-- Summary KPI Grid -->
      <div class="kpi-grid">
        <div class="kpi-box kpi-total">
          <div class="kpi-label">Total Cells Detected</div>
          <div class="kpi-val" id="kpiTotalCells">0</div>
          <div class="kpi-sub">Candidate Red Blood Cells</div>
        </div>
        <div class="kpi-box kpi-par">
          <div class="kpi-label">Parasitized Cells</div>
          <div class="kpi-val" id="kpiPar">0</div>
          <div class="kpi-sub" id="kpiParSub">AI Flagged Candidates</div>
        </div>
        <div class="kpi-box kpi-uninf">
          <div class="kpi-label">Uninfected Cells</div>
          <div class="kpi-val" id="kpiUninf">0</div>
          <div class="kpi-sub" id="kpiUninfSub">Valid RBC Candidates</div>
        </div>
        <div class="kpi-box kpi-rate">
          <div class="kpi-label">Estimated Parasitemia</div>
          <div class="kpi-val" id="kpiRate">0.00%</div>
          <div class="kpi-sub" id="kpiRateFrac">0 / 0 RBCs</div>
        </div>
      </div>

      <!-- Secondary Metrics Row -->
      <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap:10px; margin-bottom:18px;">
        <div class="kpi-box kpi-wbc" style="padding:10px 14px;">
          <div class="kpi-label" style="font-size:0.70rem;">Excluded WBCs</div>
          <div class="kpi-val" id="kpiWbc" style="font-size:1.25rem; margin-top:2px;">0</div>
        </div>
        <div class="kpi-box kpi-rej" style="padding:10px 14px;">
          <div class="kpi-label" style="font-size:0.70rem;">Suppressed / Duplicates</div>
          <div class="kpi-val" id="kpiRej" style="font-size:1.25rem; margin-top:2px;">0</div>
        </div>
        <div class="kpi-box" style="padding:10px 14px;">
          <div class="kpi-label" style="font-size:0.70rem;">Processing Time</div>
          <div class="kpi-val" id="kpiTime" style="font-size:1.25rem; margin-top:2px;">0.0s</div>
          <div class="kpi-sub" id="kpiTimeSub" style="font-size:0.70rem;">Pipeline Latency</div>
        </div>
      </div>

      <!-- Report Downloads & Export Section -->
      <div class="card">
        <div class="card-title">
          <span>Official Reports &amp; Data Artifacts</span>
          <span id="runIdBadge" style="font-size:0.76rem; color:var(--muted);"></span>
        </div>
        <div id="exportButtons"></div>

        <!-- Optional Collapsed Telemetry Summary -->
        <details class="clean-accordion" style="margin-top:14px;">
          <summary>Technical Execution Details &amp; Confidence Telemetry</summary>
          <div class="accordion-body">
            <div id="execSummaryGrid" style="display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:12px; font-size:0.83rem;"></div>
          </div>
        </details>
      </div>

      <!-- Visual Inspection & Detailed Tables Card -->
      <div class="card">
        <div class="tabs">
          <button class="tab-btn active" onclick="switchTab('tabAnnotated', this)">Annotated Smear</button>
          <button class="tab-btn" onclick="switchTab('tabOriginal', this)">Original Image</button>
          <button class="tab-btn" onclick="switchTab('tabContact', this)">RBC Contact Sheet</button>
          <button class="tab-btn" onclick="switchTab('tabCells', this)">Per-Cell Predictions (<span id="tabCellCount">0</span>)</button>
          <button class="tab-btn" onclick="switchTab('tabExcluded', this)">Excluded &amp; Rejected (<span id="tabExclCount">0</span>)</button>
          <button class="tab-btn" onclick="switchTab('tabTxtReport', this)">Text Summary</button>
        </div>

        <div id="tabAnnotated" class="tab-pane active">
          <img id="annotatedImg" class="img-viewer" alt="Annotated Blood Smear">
        </div>

        <div id="tabOriginal" class="tab-pane">
          <img id="originalInputImg" class="img-viewer" alt="Original Input Image">
        </div>

        <div id="tabContact" class="tab-pane">
          <img id="contactImg" class="img-viewer" alt="RBC Crops Contact Sheet">
        </div>

        <div id="tabCells" class="tab-pane">
          <div style="display:flex; gap:12px; margin-bottom:12px; align-items:center;">
            <label style="margin:0; font-weight:600;">Filter Classification:</label>
            <select id="cellFilter" onchange="renderCellTable()" style="width:220px;">
              <option value="ALL">All Valid Cells</option>
              <option value="Parasitized">Parasitized Cells Only</option>
              <option value="Uninfected">Uninfected Cells Only</option>
            </select>
          </div>
          <div style="max-height:520px; overflow-y:auto;">
            <table>
              <thead>
                <tr>
                  <th>Crop</th>
                  <th>Cell ID</th>
                  <th>Bounding Box</th>
                  <th>Detection Conf</th>
                  <th>Classification</th>
                  <th>Confidence</th>
                  <th>Raw Crop P(Par)</th>
                  <th>Masked Crop P(Par)</th>
                </tr>
              </thead>
              <tbody id="cellTableBody"></tbody>
            </table>
          </div>
        </div>

        <div id="tabExcluded" class="tab-pane">
          <h4 style="margin-bottom:8px; color:#fbbf24; font-size:0.90rem;">Excluded WBC Detections</h4>
          <div id="wbcListContainer" style="margin-bottom:18px;"></div>
          <h4 style="margin-bottom:8px; color:#c084fc; font-size:0.90rem;">Rejected / Duplicate Detections</h4>
          <div id="rejectedListContainer"></div>
        </div>

        <div id="tabTxtReport" class="tab-pane">
          <pre class="report-box" id="txtReportBox"></pre>
        </div>
      </div>
    </div>
  </section>
</main>

<!-- ========================================================================= -->
<!-- MODE 2: ALL LEUKEMIA WORKFLOW                                             -->
<!-- ========================================================================= -->
<main class="layout" id="leukemiaWorkflow" style="display:none;">
  <!-- Left Column: Controls & Input -->
  <section>
    <div class="card">
      <div class="card-title">
        <span>White Blood Cell Specimen Source</span>
        <span style="font-size:0.75rem; color:#fb7185;">Single-Cell Analysis</span>
      </div>

      <!-- 3-way Mode Switcher -->
      <div class="mode-switcher mode-switcher-3" role="tablist" aria-label="ALL Input Option Selector">
        <button type="button" class="mode-btn active" id="leukemiaInputMicroscopeBtn" onclick="switchLeukemiaInputMode('microscope')">
          Live Microscope
        </button>
        <button type="button" class="mode-btn" id="leukemiaInputUploadBtn" onclick="switchLeukemiaInputMode('upload')">
          Upload Image
        </button>
        <button type="button" class="mode-btn" id="leukemiaInputSamplesBtn" onclick="switchLeukemiaInputMode('samples')">
          Dataset Samples
        </button>
      </div>

      <!-- OPTION 1: Live Microscope Panel -->
      <div id="leukemiaLiveMicroscopePanel">
        <div class="cam-status-bar" id="leukemiaMicroscopeStatusBar">
          <div>
            <span class="cam-dot" id="leukemiaMicroscopeStatusDot"></span>
            <strong id="leukemiaMicroscopeStatusText">Camera Idle — Click Start Camera</strong>
          </div>
          <span id="leukemiaMicroscopeResBadge" style="font-size:0.74rem; color:var(--muted);">No stream</span>
        </div>

        <label for="leukemiaCameraDeviceSelect" style="margin-top:6px;">USB Microscope Device</label>
        <div style="display:flex; gap:6px;">
          <select id="leukemiaCameraDeviceSelect" onchange="onLeukemiaCameraDeviceChange()" disabled style="flex:1;">
            <option value="">-- Start camera to list video devices --</option>
          </select>
          <button type="button" class="btn-secondary" id="leukemiaRefreshDevicesBtn" onclick="refreshLeukemiaCameraDevices()" title="Refresh video input list">
            Refresh
          </button>
        </div>
        <div class="hint">Standard UVC USB digital microscopes connect automatically upon browser permission.</div>

        <!-- Start / Stop Camera Controls -->
        <div class="btn-row" style="margin-top:8px;">
          <button type="button" class="btn-secondary" id="leukemiaStartCameraBtn" onclick="startLeukemiaCamera()">
            Start Camera
          </button>
          <button type="button" class="btn-danger-outline" id="leukemiaStopCameraBtn" onclick="stopLeukemiaCamera()" disabled>
            Stop Camera
          </button>
        </div>

        <!-- Live Video Preview Stage -->
        <div class="video-stage" id="leukemiaLiveVideoStage" style="margin-top:10px;">
          <div class="video-placeholder" id="leukemiaVideoPlaceholderText">
            Live USB Microscope preview is inactive.<br>
            Click <strong>Start Camera</strong> to connect your digital microscope.
          </div>
          <video id="leukemiaMicroscopeVideo" autoplay playsinline muted style="display:none;"></video>
        </div>

        <!-- Capture / Retake / Discard Controls -->
        <div class="btn-row" style="margin-top:10px; display:flex; gap:6px;">
          <button type="button" class="btn-secondary" id="leukemiaCaptureImageBtn" onclick="captureLeukemiaFrame()" disabled style="background:#2563eb; color:#fff; border-color:#3b82f6; flex:1.2; font-weight:600;">
            📸 Capture Image
          </button>
          <button type="button" class="btn-secondary" id="leukemiaRetakeImageBtn" onclick="retakeLeukemiaFrame()" disabled style="flex:1;">
            🔄 Retake
          </button>
          <button type="button" class="btn-danger-outline" id="leukemiaDiscardImageBtn" onclick="discardLeukemiaFrame()" disabled style="flex:1;">
            🗑️ Discard
          </button>
        </div>

        <!-- Captured Frame Inspection Preview -->
        <div id="leukemiaCapturedPreviewSection" style="display:none; margin-top:10px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <span style="font-size:0.80rem; font-weight:700; color:#fca5a5;">Captured Frame Preview</span>
            <span id="leukemiaCapturedFrameInfo" style="font-size:0.74rem; color:var(--muted);"></span>
          </div>
          <div class="video-stage" style="margin-top:2px; margin-bottom:6px;">
            <img id="leukemiaCapturedPreviewImg" alt="Captured ALL Microscope Frame Preview" style="max-height:280px; width:auto; margin:0 auto; display:block;">
          </div>
        </div>

        <!-- Client-Side Quality Warnings Box -->
        <div id="leukemiaQualityBox"></div>

        <!-- Direct Analyze Captured Button -->
        <button type="button" class="btn-primary-leukemia" id="leukemiaCaptureAndAnalyzeBtn" onclick="captureAndAnalyzeLeukemiaMicroscope()" disabled>
          🔬 Analyze Captured Cell
        </button>
      </div>

      <!-- OPTION 2: Upload Image Panel -->
      <div id="leukemiaUploadPanel" style="display:none;">
        <label for="leukemiaImageUpload">Upload White Blood Cell Image (.png, .jpg, .tif, .bmp)</label>
        <input type="file" id="leukemiaImageUpload" accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff" onchange="onLeukemiaFileChange()">
        <div class="hint" id="leukemiaSelectedName" style="color:var(--accent); font-weight:600; margin-top:4px;">No file selected.</div>

        <div id="leukemiaUploadPreviewSection" style="display:none; margin-top:10px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <span style="font-size:0.80rem; font-weight:700; color:#93c5fd;">Upload Preview &amp; Details</span>
            <span id="leukemiaUploadDetailsBadge" style="font-size:0.74rem; color:var(--muted);"></span>
          </div>
          <div class="video-stage" style="margin-top:2px; max-height:240px;">
            <img id="leukemiaUploadPreviewImg" alt="Uploaded Cell Preview" style="max-height:220px; width:auto; margin:0 auto; display:block;">
          </div>
          <div id="leukemiaUploadMetaBox" style="font-size:0.78rem; color:#cbd5e1; margin-top:6px; background:var(--panel-alt); padding:8px 12px; border-radius:6px; border:1px solid var(--border);"></div>
        </div>

        <button type="button" class="btn-primary-leukemia" id="btnRunLeukemiaUpload" onclick="runLeukemiaAnalysis('upload')">
          Classify White Blood Cell
        </button>
      </div>

      <!-- OPTION 3: Dataset Samples Panel -->
      <div id="leukemiaSamplesPanel" style="display:none;">
        <label>Benchmark Test Cell Presets (ALL-IDB2)</label>
        <div style="font-size:0.78rem; font-weight:700; color:#86efac; margin:8px 0 4px;">Normal Lymphocytes:</div>
        <div class="preset-btn-grid">
          <button type="button" class="btn-export leukemia-preset-btn" onclick="selectLeukemiaPreset(this, 'Im133_0.png')">Im133_0</button>
          <button type="button" class="btn-export leukemia-preset-btn" onclick="selectLeukemiaPreset(this, 'Im145_0.png')">Im145_0</button>
          <button type="button" class="btn-export leukemia-preset-btn" onclick="selectLeukemiaPreset(this, 'Im223_0.png')">Im223_0</button>
          <button type="button" class="btn-export leukemia-preset-btn" onclick="selectLeukemiaPreset(this, 'Im260_0.png')">Im260_0</button>
          <button type="button" class="btn-export leukemia-preset-btn" onclick="selectLeukemiaPreset(this, 'Im151_0.png')">Im151_0</button>
          <button type="button" class="btn-export leukemia-preset-btn" onclick="selectLeukemiaPreset(this, 'Im152_0.png')">Im152_0</button>
        </div>

        <div style="font-size:0.78rem; font-weight:700; color:#fca5a5; margin:8px 0 4px;">Lymphoblasts (ALL Leukemia):</div>
        <div class="preset-btn-grid">
          <button type="button" class="btn-export leukemia-preset-btn" onclick="selectLeukemiaPreset(this, 'Im001_1.png')">Im001_1</button>
          <button type="button" class="btn-export leukemia-preset-btn" onclick="selectLeukemiaPreset(this, 'Im003_1.png')">Im003_1</button>
          <button type="button" class="btn-export leukemia-preset-btn" onclick="selectLeukemiaPreset(this, 'Im005_1.png')">Im005_1</button>
          <button type="button" class="btn-export leukemia-preset-btn" onclick="selectLeukemiaPreset(this, 'Im065_1.png')">Im065_1</button>
          <button type="button" class="btn-export leukemia-preset-btn" onclick="selectLeukemiaPreset(this, 'Im090_1.png')">Im090_1</button>
          <button type="button" class="btn-export leukemia-preset-btn" onclick="selectLeukemiaPreset(this, 'Im128_1.png')">Im128_1</button>
        </div>

        <div class="hint" id="leukemiaPresetSelectedHint" style="color:#86efac; font-weight:600; margin-bottom:10px;">Selected: None</div>

        <div id="leukemiaSamplePreviewSection" style="display:none; margin-top:8px; margin-bottom:10px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <span style="font-size:0.80rem; font-weight:700; color:#bae6fd;">Selected Benchmark Cell</span>
            <span id="leukemiaSampleDetailsBadge" style="font-size:0.74rem; color:var(--muted);"></span>
          </div>
          <div class="video-stage" style="margin-top:2px; max-height:220px;">
            <img id="leukemiaSamplePreviewImg" alt="Selected Benchmark Cell Preview" style="max-height:200px; width:auto; margin:0 auto; display:block;">
          </div>
        </div>

        <button type="button" class="btn-primary-leukemia" id="btnRunLeukemiaSample" onclick="runLeukemiaAnalysis('sample')" style="margin-top:4px;">
          Classify Benchmark Cell
        </button>
      </div>

      <!-- Specimen Case Reference -->
      <div style="margin-top:14px; border-top:1px solid var(--border); padding-top:12px;">
        <label for="leukemiaSampleId">Case / Patient Reference (Optional)</label>
        <input type="text" id="leukemiaSampleId" placeholder="e.g. CASE-2026-001" style="background:var(--panel-alt);">
      </div>

      <div id="leukemiaErrorAlert" style="display:none; margin-top:14px; padding:12px; background:rgba(239,68,68,0.15); border:1px solid #ef4444; border-radius:6px; color:#fca5a5; font-size:0.85rem;"></div>

      <!-- Model Specifications (Collapsed) -->
      <details class="clean-accordion">
        <summary>Model Architecture &amp; Benchmark Specifications</summary>
        <div class="accordion-body" style="font-size:0.81rem; line-height:1.7;">
          <div><strong>Architecture:</strong> EfficientNetB0 Transfer Learning</div>
          <div><strong>Pre-training:</strong> ImageNet &rarr; ALL-IDB2 Fine-Tuned</div>
          <div><strong>Input Resolution:</strong> 224 × 224 × 3 RGB</div>
          <div><strong>Test Accuracy:</strong> <span style="color:#86efac; font-weight:bold;">100.0% (39/39 correct)</span></div>
          <div><strong>Test ROC-AUC:</strong> <span style="color:#86efac; font-weight:bold;">1.0000</span></div>
          <div><strong>Decision Boundary:</strong> 0.50 (Standard sigmoid threshold)</div>
        </div>
      </details>
    </div>
  </section>

  <!-- Right Column: Results & Diagnosis -->
  <section>
    <div class="card" id="leukemiaEmptyState" style="text-align:center; padding:54px 24px; color:var(--muted);">
      <div style="font-size:3.2rem; margin-bottom:14px;">🩸</div>
      <h3 style="color:#fff; margin-bottom:8px; font-size:1.2rem;">ALL Leukemia Screening Ready</h3>
      <p style="font-size:0.9rem; max-width:540px; margin:0 auto 20px; line-height:1.5;">Connect a USB microscope, upload a white blood cell image, or select a dataset sample from the left panel to begin classification.</p>
    </div>

    <div id="leukemiaResultsCard" style="display:none;">
      <!-- Diagnosis Banner -->
      <div id="leukemiaDiagnosisBanner" class="diagnosis-banner banner-normal"></div>

      <!-- Main Visual & Probability Card -->
      <div class="card">
        <div class="card-title">
          <span>Morphology &amp; Classification Confidence</span>
        </div>
        <div style="display:grid; grid-template-columns: 240px 1fr; gap:22px; align-items:start;">
          <div>
            <img id="leukemiaPreviewImg" src="" alt="Analyzed Cell" style="width:100%; border-radius:8px; border:1px solid var(--border); box-shadow:0 4px 14px rgba(0,0,0,0.45);">
            <div style="font-size:0.75rem; color:var(--muted); text-align:center; margin-top:6px;">Input Cell (224×224 px)</div>
          </div>
          <div>
            <!-- Probability Bar 1: Normal -->
            <div class="prob-bar-row">
              <div class="prob-bar-label">
                <span style="color:#86efac;">Normal Lymphocyte</span>
                <span id="pNormVal" style="color:#86efac; font-weight:bold;">0%</span>
              </div>
              <div class="prob-track">
                <div id="pNormBar" class="prob-fill prob-fill-normal" style="width:0%;"></div>
              </div>
            </div>

            <!-- Probability Bar 2: Blast -->
            <div class="prob-bar-row">
              <div class="prob-bar-label">
                <span style="color:#fca5a5;">Lymphoblast (ALL-associated cell category)</span>
                <span id="pBlastVal" style="color:#fca5a5; font-weight:bold;">0%</span>
              </div>
              <div class="prob-track">
                <div id="pBlastBar" class="prob-fill prob-fill-blast" style="width:0%;"></div>
              </div>
            </div>

            <div style="margin-top:16px; padding:12px 14px; background:var(--panel-alt); border-radius:8px; border:1px solid var(--border);">
              <div style="font-size:0.80rem; font-weight:700; color:var(--accent); margin-bottom:4px;">Morphological Assessment:</div>
              <p id="leukemiaMorphSummary" style="font-size:0.84rem; color:var(--text); line-height:1.45;"></p>
            </div>
          </div>
        </div>
      </div>

      <!-- Downloads Card -->
      <div class="card">
        <div class="card-title">
          <span>Official Reports &amp; Artifact Exports</span>
        </div>
        <div style="display:flex; gap:12px; flex-wrap:wrap;">
          <a id="dlLeukemiaPdf" href="#" class="btn-export" style="background:#be123c; border-color:#e11d48; color:#fff; font-weight:700;" target="_blank">
            📄 Download Leukemia PDF Report
          </a>
          <a id="dlLeukemiaJson" href="#" class="btn-export" target="_blank">
            💾 Download JSON Result
          </a>
          <a id="dlLeukemiaTxt" href="#" class="btn-export" target="_blank">
            📝 Download TXT Summary
          </a>
        </div>
      </div>

      <!-- Regulatory Disclaimer Card -->
      <div class="card" style="border-color:#be123c; background:rgba(190,18,60,0.06); padding:14px 18px;">
        <div style="color:#fca5a5; font-size:0.80rem; font-weight:700; margin-bottom:4px;">RESEARCH &amp; EDUCATIONAL PROTOTYPE ONLY</div>
        <p style="font-size:0.77rem; color:#fecdd3; line-height:1.4;">
          This acute lymphoblastic leukemia screening tool is an experimental research prototype for scientific evaluation. It is NOT a clinically certified diagnostic device. Definitive medical diagnosis requires certified pathologist hematomorphology, bone marrow biopsy, and flow cytometry.
        </p>
      </div>
    </div>
  </section>
</main>

<script>
/**
 * Safely parse an HTTP response from the SmartCell backend.
 * Guarantees that empty bodies, HTML error pages, and network/proxy disconnects
 * never throw unhandled "Unexpected end of JSON input" SyntaxErrors.
 */
async function parseApiResponse(resp) {
  const status = resp ? resp.status : 0;
  let text = '';
  try {
    text = await resp.text();
  } catch (err) {
    text = '';
  }

  let data = null;
  if (text && text.trim().length > 0) {
    try {
      data = JSON.parse(text);
    } catch (e) {
      data = null;
    }
  }

  if (resp && resp.ok && data) {
    return { ok: true, status, data, errorType: null, errorMessage: null };
  }

  let errorType = 'ServerError';
  let errorMessage = '';

  if (data && (data.error_message || data.error || data.detail)) {
    errorMessage = data.error_message || data.error || data.detail;
    errorType = data.error_type || `HTTP ${status}`;
  } else if (status === 502) {
    errorType = 'BackendUnavailable';
    errorMessage = 'The backend container restarted or encountered an out-of-memory event (HTTP 502 Bad Gateway). Please wait ~30 seconds for the container to reload and try again.';
  } else if (status === 503) {
    errorType = 'ServiceUnavailable';
    errorMessage = 'The backend is currently spinning up from cold sleep (HTTP 503 Service Unavailable). Please wait 30–45 seconds and retry.';
  } else if (status === 504) {
    errorType = 'GatewayTimeout';
    errorMessage = 'Inference request timed out on the backend proxy (HTTP 504 Gateway Timeout). The server may be under heavy load; please retry in a moment.';
  } else if (status === 413) {
    errorType = 'PayloadTooLarge';
    errorMessage = 'The uploaded image payload exceeds the maximum allowed upload limit (35 MB).';
  } else if (status === 404) {
    errorType = 'NotFound';
    errorMessage = 'API route not found (HTTP 404). Please verify your endpoint.';
  } else if (resp && !resp.ok) {
    errorType = `HTTP ${status}`;
    errorMessage = (text && text.length < 250 && !text.includes('<html'))
      ? text.trim()
      : `Backend server returned HTTP status ${status}.`;
  } else if (!data) {
    errorType = 'InvalidResponse';
    errorMessage = 'Backend server returned an empty or unparseable response body.';
  }

  return {
    ok: false,
    status,
    data: data || { success: false, error_type: errorType, error_message: errorMessage },
    errorType,
    errorMessage
  };
}

let CURRENT_RESULT = null;
let ENV_STATUS = null;
let ACTIVE_INPUT_MODE = 'manual';

// Live Microscope State (Shared across Malaria & ALL Leukemia Modes)
const SmartCellMicroscope = {
  stream: null,
  activeDeviceId: '',
  activeScope: 'malaria',
  capturedBlob: null,
  capturedObjectUrl: null,
  capturedWidth: 0,
  capturedHeight: 0,
  capturedQuality: null,
  isAnalyzing: false
};
window.SmartCellMicroscope = SmartCellMicroscope;

function getCameraElements(scope) {
  const s = scope || SmartCellMicroscope.activeScope || (typeof currentDiseaseScope !== 'undefined' ? currentDiseaseScope : 'malaria');
  if (s === 'leukemia') {
    return {
      scope: 'leukemia',
      video: document.getElementById('leukemiaMicroscopeVideo'),
      placeholder: document.getElementById('leukemiaVideoPlaceholderText'),
      deviceSelect: document.getElementById('leukemiaCameraDeviceSelect'),
      startBtn: document.getElementById('leukemiaStartCameraBtn'),
      stopBtn: document.getElementById('leukemiaStopCameraBtn'),
      captureBtn: document.getElementById('leukemiaCaptureImageBtn'),
      retakeBtn: document.getElementById('leukemiaRetakeImageBtn'),
      discardBtn: document.getElementById('leukemiaDiscardImageBtn'),
      previewSection: document.getElementById('leukemiaCapturedPreviewSection'),
      previewImg: document.getElementById('leukemiaCapturedPreviewImg'),
      frameInfo: document.getElementById('leukemiaCapturedFrameInfo'),
      canvas: document.getElementById('microscopeCaptureCanvas'),
      qualityBox: document.getElementById('leukemiaQualityBox'),
      analyzeBtn: document.getElementById('leukemiaCaptureAndAnalyzeBtn'),
      dot: document.getElementById('leukemiaMicroscopeStatusDot'),
      text: document.getElementById('leukemiaMicroscopeStatusText'),
      resBadge: document.getElementById('leukemiaMicroscopeResBadge'),
      errBox: document.getElementById('leukemiaErrorAlert')
    };
  } else {
    return {
      scope: 'malaria',
      video: document.getElementById('microscopeVideo'),
      placeholder: document.getElementById('videoPlaceholderText'),
      deviceSelect: document.getElementById('cameraDeviceSelect'),
      startBtn: document.getElementById('startCameraBtn'),
      stopBtn: document.getElementById('stopCameraBtn'),
      captureBtn: document.getElementById('captureImageBtn'),
      retakeBtn: document.getElementById('retakeImageBtn'),
      previewSection: document.getElementById('capturedPreviewSection'),
      previewImg: document.getElementById('capturedPreviewImg'),
      frameInfo: document.getElementById('capturedFrameInfo'),
      canvas: document.getElementById('microscopeCaptureCanvas'),
      qualityBox: document.getElementById('microscopeQualityBox'),
      analyzeBtn: document.getElementById('captureAndAnalyzeBtn'),
      dot: document.getElementById('microscopeStatusDot'),
      text: document.getElementById('microscopeStatusText'),
      resBadge: document.getElementById('microscopeResBadge'),
      errBox: document.getElementById('errorContainer')
    };
  }
}

function setMicroscopeStatus(state, message, resText, scope) {
  const els = getCameraElements(scope);
  if (!els || !els.dot || !els.text) return;
  els.dot.className = 'cam-dot';
  if (state === 'live') els.dot.classList.add('live');
  else if (state === 'captured') els.dot.classList.add('captured');
  else if (state === 'warn') els.dot.classList.add('warn');
  else if (state === 'err') els.dot.classList.add('err');
  els.text.textContent = message;
  if (resText !== undefined && els.resBadge) els.resBadge.textContent = resText;
}

function switchInputMode(mode) {
  ACTIVE_INPUT_MODE = mode;
  const manualBtn = document.getElementById('modeManualBtn');
  const microBtn = document.getElementById('modeMicroscopeBtn');
  const manualPanel = document.getElementById('manualUploadPanel');
  const microPanel = document.getElementById('liveMicroscopePanel');

  if (mode === 'microscope') {
    if (manualBtn) manualBtn.classList.remove('active');
    if (microBtn) microBtn.classList.add('active');
    if (manualPanel) manualPanel.style.display = 'none';
    if (microPanel) microPanel.style.display = 'block';
  } else {
    stopMicroscopeCamera(true, 'malaria');
    if (microBtn) microBtn.classList.remove('active');
    if (manualBtn) manualBtn.classList.add('active');
    if (microPanel) microPanel.style.display = 'none';
    if (manualPanel) manualPanel.style.display = 'block';
  }
}

function returnToManualUpload() {
  switchInputMode('manual');
}

function formatCameraError(err) {
  if (!err) return 'Unknown camera initialization error.';
  const name = err.name || '';
  if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
    return 'Camera Permission Denied: Browser access to the USB microscope/camera was blocked. Please allow camera access in your browser settings (look for the camera icon in the URL bar) and retry.';
  }
  if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
    return 'No Camera Found: No USB microscope or webcam video input device was detected. Verify that your USB microscope is plugged in and recognized by Windows.';
  }
  if (name === 'NotReadableError' || name === 'TrackStartError') {
    return 'Camera In Use / Hardware Error: The USB microscope or camera is currently locked by another application (e.g. Windows Camera app, Zoom, or another browser tab). Close competing apps and retry.';
  }
  if (name === 'OverconstrainedError') {
    return 'Unsupported Camera Resolution: Requested camera constraints were not supported by the device.';
  }
  return `Camera Error (${name || 'Error'}): ${err.message || String(err)}`;
}

async function refreshCameraDevices(preferredDeviceId, scope) {
  const els = getCameraElements(scope);
  const sel = els.deviceSelect;
  if (!sel) return [];
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    sel.innerHTML = `<option value="">-- Browser does not support device enumeration --</option>`;
    sel.disabled = true;
    return [];
  }
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoInputs = devices.filter(d => d.kind === 'videoinput');
    sel.innerHTML = '';
    if (!videoInputs.length) {
      sel.innerHTML = `<option value="">-- No video input devices detected --</option>`;
      sel.disabled = true;
      return [];
    }
    videoInputs.forEach((dev, idx) => {
      const opt = document.createElement('option');
      opt.value = dev.deviceId;
      opt.textContent = dev.label || `USB Camera / Microscope #${idx + 1}`;
      sel.appendChild(opt);
    });
    sel.disabled = false;
    const targetId = preferredDeviceId || SmartCellMicroscope.activeDeviceId;
    if (targetId && videoInputs.some(d => d.deviceId === targetId)) {
      sel.value = targetId;
    }
    const otherScope = (els.scope === 'leukemia') ? 'malaria' : 'leukemia';
    const otherEls = getCameraElements(otherScope);
    if (otherEls.deviceSelect && otherEls.deviceSelect !== sel) {
      otherEls.deviceSelect.innerHTML = sel.innerHTML;
      otherEls.deviceSelect.disabled = sel.disabled;
      otherEls.deviceSelect.value = sel.value;
    }
    return videoInputs;
  } catch (err) {
    sel.innerHTML = `<option value="">-- Error enumerating devices: ${err.message} --</option>`;
    sel.disabled = true;
    return [];
  }
}

async function startMicroscopeCamera(requestedDeviceId, scope) {
  const targetScope = scope || (typeof currentDiseaseScope !== 'undefined' ? currentDiseaseScope : 'malaria');
  SmartCellMicroscope.activeScope = targetScope;
  const els = getCameraElements(targetScope);
  const errBox = els.errBox;
  if (errBox) {
    errBox.innerHTML = '';
    errBox.style.display = 'none';
  }

  if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== 'function') {
    const msg = 'Unsupported Browser: navigator.mediaDevices.getUserMedia() is unavailable. Open http://127.0.0.1:8501 in Chrome, Edge, or Firefox.';
    setMicroscopeStatus('err', 'Browser Camera API Unavailable', 'Unsupported', targetScope);
    if (errBox) {
      errBox.innerHTML = `<div class="alert-error"><strong>Camera API Unavailable:</strong> ${msg}</div>`;
      errBox.style.display = 'block';
    }
    return false;
  }

  releaseMicroscopeStreamOnly(targetScope);
  setMicroscopeStatus('warn', 'Requesting Camera Permission...', 'Initializing...', targetScope);

  const deviceId = requestedDeviceId !== undefined
    ? requestedDeviceId
    : (els.deviceSelect ? els.deviceSelect.value : '');

  const preferredConstraints = {
    audio: false,
    video: deviceId
      ? { deviceId: { exact: deviceId }, width: { ideal: 1920 }, height: { ideal: 1080 } }
      : { width: { ideal: 1920 }, height: { ideal: 1080 } }
  };

  let stream = null;
  try {
    stream = await navigator.mediaDevices.getUserMedia(preferredConstraints);
  } catch (firstErr) {
    if (firstErr && (firstErr.name === 'OverconstrainedError' || firstErr.name === 'ConstraintNotSatisfiedError')) {
      try {
        const fallbackConstraints = {
          audio: false,
          video: deviceId ? { deviceId: { exact: deviceId } } : true
        };
        stream = await navigator.mediaDevices.getUserMedia(fallbackConstraints);
      } catch (secondErr) {
        const msg2 = formatCameraError(secondErr);
        setMicroscopeStatus('err', 'Camera Initialization Failed', 'Error', targetScope);
        if (errBox) {
          errBox.innerHTML = `<div class="alert-error">${msg2}</div>`;
          errBox.style.display = 'block';
        }
        return false;
      }
    } else {
      const msg1 = formatCameraError(firstErr);
      setMicroscopeStatus('err', 'Camera Access Failed', 'Error', targetScope);
      if (errBox) {
        errBox.innerHTML = `<div class="alert-error">${msg1}</div>`;
        errBox.style.display = 'block';
      }
      return false;
    }
  }

  SmartCellMicroscope.stream = stream;
  const videoTracks = stream.getVideoTracks();
  const activeTrack = videoTracks[0] || null;
  const trackSettings = (activeTrack && activeTrack.getSettings) ? activeTrack.getSettings() : {};
  SmartCellMicroscope.activeDeviceId = trackSettings.deviceId || deviceId || '';

  if (activeTrack) {
    activeTrack.onended = () => {
      handleCameraDisconnected(targetScope);
    };
  }

  const videoEl = els.video;
  const placeholderEl = els.placeholder;
  if (videoEl) {
    videoEl.srcObject = stream;
    videoEl.style.display = 'block';
  }
  if (placeholderEl) {
    placeholderEl.style.display = 'none';
  }

  await new Promise(resolve => {
    let resolved = false;
    const finish = () => {
      if (!resolved) {
        resolved = true;
        resolve();
      }
    };
    if (videoEl) videoEl.onloadedmetadata = finish;
    setTimeout(finish, 600);
  });

  try {
    if (videoEl) await videoEl.play();
  } catch (_) {}

  await refreshCameraDevices(SmartCellMicroscope.activeDeviceId, targetScope);

  const vw = (videoEl && videoEl.videoWidth) || trackSettings.width || 640;
  const vh = (videoEl && videoEl.videoHeight) || trackSettings.height || 480;
  setMicroscopeStatus('live', 'Live Microscope Stream Active', `${vw}\u00d7${vh} px`, targetScope);

  if (els.startBtn) els.startBtn.disabled = false;
  if (els.stopBtn) els.stopBtn.disabled = false;
  if (els.captureBtn) els.captureBtn.disabled = false;
  if (els.analyzeBtn) els.analyzeBtn.disabled = false;
  return true;
}

function handleCameraDisconnected(scope) {
  const targetScope = scope || SmartCellMicroscope.activeScope || (typeof currentDiseaseScope !== 'undefined' ? currentDiseaseScope : 'malaria');
  const els = getCameraElements(targetScope);
  releaseMicroscopeStreamOnly(targetScope);
  setMicroscopeStatus('err', 'USB Microscope Disconnected', 'Disconnected', targetScope);
  if (els.errBox) {
    els.errBox.innerHTML = `
      <div class="alert-error">
        <strong>Camera Disconnected:</strong> The USB microscope or camera video stream ended unexpectedly. Check the USB cable and reconnect.
      </div>
    `;
    els.errBox.style.display = 'block';
  }
}

function releaseMicroscopeStreamOnly(scope) {
  const targetScope = scope || SmartCellMicroscope.activeScope || (typeof currentDiseaseScope !== 'undefined' ? currentDiseaseScope : 'malaria');
  const els = getCameraElements(targetScope);
  if (SmartCellMicroscope.stream) {
    try {
      SmartCellMicroscope.stream.getTracks().forEach(track => {
        track.onended = null;
        track.stop();
      });
    } catch (_) {}
    SmartCellMicroscope.stream = null;
  }
  if (els.video) {
    els.video.srcObject = null;
    els.video.style.display = 'none';
  }
  if (els.placeholder) {
    els.placeholder.style.display = 'block';
  }
  if (els.stopBtn) els.stopBtn.disabled = true;
  if (els.captureBtn) els.captureBtn.disabled = true;
}

function stopMicroscopeCamera(silent, scope) {
  const targetScope = scope || SmartCellMicroscope.activeScope || (typeof currentDiseaseScope !== 'undefined' ? currentDiseaseScope : 'malaria');
  const els = getCameraElements(targetScope);
  releaseMicroscopeStreamOnly(targetScope);
  if (!SmartCellMicroscope.capturedBlob) {
    if (els.analyzeBtn) els.analyzeBtn.disabled = true;
    if (!silent) {
      setMicroscopeStatus('idle', 'Camera Stopped — Tracks Released', 'No stream', targetScope);
    }
  } else if (!silent) {
    setMicroscopeStatus(
      'captured',
      'Camera Stopped — Captured Frame Retained',
      `${SmartCellMicroscope.capturedWidth}\u00d7${SmartCellMicroscope.capturedHeight} px`,
      targetScope
    );
  }
}

async function onCameraDeviceChange(scope) {
  const targetScope = scope || SmartCellMicroscope.activeScope || (typeof currentDiseaseScope !== 'undefined' ? currentDiseaseScope : 'malaria');
  const els = getCameraElements(targetScope);
  const newDeviceId = els.deviceSelect ? els.deviceSelect.value : '';
  if (!newDeviceId) return;
  SmartCellMicroscope.activeDeviceId = newDeviceId;
  if (SmartCellMicroscope.stream) {
    await startMicroscopeCamera(newDeviceId, targetScope);
  }
}

function computeCanvasQualityHeuristics(ctx, width, height) {
  const imgData = ctx.getImageData(0, 0, width, height).data;
  const totalPixels = width * height;
  if (totalPixels === 0) {
    return { meanLuminance: 0, stdLuminance: 0, edgeSharpness: 0, warnings: ['Captured frame has 0 pixels.'] };
  }
  const stride = Math.max(1, Math.floor(Math.sqrt(totalPixels / 40000)));
  const grayVals = [];
  let sum = 0;
  for (let y = 0; y < height; y += stride) {
    for (let x = 0; x < width; x += stride) {
      const idx = (y * width + x) * 4;
      const lum = 0.299 * imgData[idx] + 0.587 * imgData[idx + 1] + 0.114 * imgData[idx + 2];
      grayVals.push(lum);
      sum += lum;
    }
  }
  const n = grayVals.length || 1;
  const meanLum = sum / n;
  let sqDiff = 0;
  for (let i = 0; i < n; i++) {
    const d = grayVals[i] - meanLum;
    sqDiff += d * d;
  }
  const stdLum = Math.sqrt(sqDiff / n);

  let gradSum = 0;
  let gradCount = 0;
  const step = Math.max(1, stride);
  for (let y = step; y < height - step; y += step * 2) {
    for (let x = step; x < width - step; x += step * 2) {
      const cIdx = (y * width + x) * 4;
      const rIdx = (y * width + (x + step)) * 4;
      const dIdx = ((y + step) * width + x) * 4;
      const cLum = 0.299 * imgData[cIdx] + 0.587 * imgData[cIdx + 1] + 0.114 * imgData[cIdx + 2];
      const rLum = 0.299 * imgData[rIdx] + 0.587 * imgData[rIdx + 1] + 0.114 * imgData[rIdx + 2];
      const dLum = 0.299 * imgData[dIdx] + 0.587 * imgData[dIdx + 1] + 0.114 * imgData[dIdx + 2];
      const gx = rLum - cLum;
      const gy = dLum - cLum;
      gradSum += (gx * gx + gy * gy);
      gradCount++;
    }
  }
  const edgeSharpness = gradCount > 0 ? (gradSum / gradCount) : 0;

  const warnings = [];
  if (meanLum < 8.0 && stdLum < 5.0) {
    warnings.push(
      `Black Frame Warning: Captured image is nearly completely black (mean luminance ${meanLum.toFixed(1)}/255). Verify USB microscope illumination or lens cap.`
    );
  } else if ((meanLum > 247.0 && stdLum < 5.0) || stdLum < 3.0) {
    warnings.push(
      `Blank/Uniform Frame Warning: Captured image has extremely low contrast (std ${stdLum.toFixed(1)}) and may not contain visible cells.`
    );
  } else if (edgeSharpness < 12.0) {
    warnings.push(
      `Focus Advisory: Captured frame appears potentially blurred or out of focus (sharpness score ${edgeSharpness.toFixed(1)}). Adjust fine focus knob on microscope.`
    );
  }
  if (width < 640 || height < 480) {
    warnings.push(
      `Resolution Advisory: Captured frame (${width}\u00d7${height} px) is below standard 640\u00d7480 microscopy resolution.`
    );
  }

  return {
    meanLuminance: Number(meanLum.toFixed(2)),
    stdLuminance: Number(stdLum.toFixed(2)),
    edgeSharpness: Number(edgeSharpness.toFixed(2)),
    warnings
  };
}

async function captureMicroscopeFrame(scope) {
  const targetScope = scope || SmartCellMicroscope.activeScope || (typeof currentDiseaseScope !== 'undefined' ? currentDiseaseScope : 'malaria');
  const els = getCameraElements(targetScope);
  const errBox = els.errBox;
  if (errBox) {
    errBox.innerHTML = '';
    errBox.style.display = 'none';
  }

  const videoEl = els.video;
  if (!SmartCellMicroscope.stream || !videoEl || !videoEl.srcObject) {
    const msg = '<div class="alert-error"><strong>Camera Not Active:</strong> Start the USB microscope camera before capturing a frame.</div>';
    if (errBox) {
      errBox.innerHTML = msg;
      errBox.style.display = 'block';
    }
    return null;
  }

  const tracks = SmartCellMicroscope.stream.getVideoTracks();
  const trackSettings = (tracks[0] && tracks[0].getSettings) ? tracks[0].getSettings() : {};
  const vw = videoEl.videoWidth || trackSettings.width || 640;
  const vh = videoEl.videoHeight || trackSettings.height || 480;

  if (vw < 32 || vh < 32 || vw > 10000 || vh > 10000) {
    const msg = `<div class="alert-error"><strong>Unsupported Frame Dimensions:</strong> Captured frame dimensions (${vw}\u00d7${vh} px) are invalid.</div>`;
    if (errBox) {
      errBox.innerHTML = msg;
      errBox.style.display = 'block';
    }
    return null;
  }

  const canvas = document.getElementById('microscopeCaptureCanvas');
  canvas.width = vw;
  canvas.height = vh;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoEl, 0, 0, vw, vh);

  const quality = computeCanvasQualityHeuristics(ctx, vw, vh);
  SmartCellMicroscope.capturedQuality = quality;

  const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.96));
  if (!blob || blob.size === 0) {
    const msg = '<div class="alert-error"><strong>Frame Capture Error:</strong> Failed to encode video frame as JPEG.</div>';
    if (errBox) {
      errBox.innerHTML = msg;
      errBox.style.display = 'block';
    }
    return null;
  }
  if (blob.size > 35 * 1024 * 1024) {
    const msg = '<div class="alert-error"><strong>Frame Too Large:</strong> Captured image exceeds the 35 MB size limit.</div>';
    if (errBox) {
      errBox.innerHTML = msg;
      errBox.style.display = 'block';
    }
    return null;
  }

  if (SmartCellMicroscope.capturedObjectUrl) {
    URL.revokeObjectURL(SmartCellMicroscope.capturedObjectUrl);
  }
  const objUrl = URL.createObjectURL(blob);
  SmartCellMicroscope.capturedBlob = blob;
  SmartCellMicroscope.capturedObjectUrl = objUrl;
  SmartCellMicroscope.capturedWidth = vw;
  SmartCellMicroscope.capturedHeight = vh;

  if (els.previewImg) els.previewImg.src = objUrl;
  if (els.previewSection) els.previewSection.style.display = 'block';
  if (els.frameInfo) {
    els.frameInfo.textContent = `JPEG • ${vw}\u00d7${vh} px • ${(blob.size / 1024).toFixed(1)} KB`;
  }
  if (els.retakeBtn) els.retakeBtn.disabled = false;
  if (els.discardBtn) els.discardBtn.disabled = false;
  if (els.analyzeBtn) els.analyzeBtn.disabled = false;

  const qBox = els.qualityBox;
  if (qBox) {
    if (quality.warnings.length > 0) {
      qBox.innerHTML = `<div class="alert-warn"><strong>Pre-Inference Image Quality Notice:</strong><br>${quality.warnings.map(w => '• ' + w).join('<br>')}</div>`;
    } else {
      qBox.innerHTML = `
        <div class="hint" style="margin-top:6px; color:#86efac;">
          Frame validated (${vw}\u00d7${vh} px, luminance=${quality.meanLuminance}, sharpness=${quality.edgeSharpness}).
          Basic optical checks passed.
        </div>
      `;
    }
  }

  setMicroscopeStatus('captured', 'Frame Captured — Ready to Analyze', `${vw}\u00d7${vh} px`, targetScope);
  return blob;
}

function retakeMicroscopeFrame(scope) {
  const targetScope = scope || SmartCellMicroscope.activeScope || (typeof currentDiseaseScope !== 'undefined' ? currentDiseaseScope : 'malaria');
  const els = getCameraElements(targetScope);

  if (SmartCellMicroscope.capturedObjectUrl) {
    URL.revokeObjectURL(SmartCellMicroscope.capturedObjectUrl);
  }
  SmartCellMicroscope.capturedBlob = null;
  SmartCellMicroscope.capturedObjectUrl = null;
  SmartCellMicroscope.capturedWidth = 0;
  SmartCellMicroscope.capturedHeight = 0;
  SmartCellMicroscope.capturedQuality = null;

  if (els.previewSection) els.previewSection.style.display = 'none';
  if (els.previewImg) els.previewImg.src = '';
  if (els.qualityBox) els.qualityBox.innerHTML = '';
  if (els.retakeBtn) els.retakeBtn.disabled = true;
  if (els.discardBtn) els.discardBtn.disabled = true;

  if (SmartCellMicroscope.stream) {
    const videoEl = els.video;
    const vw = (videoEl && videoEl.videoWidth) || 640;
    const vh = (videoEl && videoEl.videoHeight) || 480;
    setMicroscopeStatus('live', 'Live Microscope Stream Active', `${vw}\u00d7${vh} px`, targetScope);
    if (els.analyzeBtn) els.analyzeBtn.disabled = false;
  } else {
    setMicroscopeStatus('idle', 'Camera Idle — Click Start Camera', 'No stream', targetScope);
    if (els.analyzeBtn) els.analyzeBtn.disabled = true;
  }
}

// Mode 2 Leukemia Camera Aliases
function startLeukemiaCamera() {
  return startMicroscopeCamera(undefined, 'leukemia');
}

function stopLeukemiaCamera() {
  return stopMicroscopeCamera(false, 'leukemia');
}

function captureLeukemiaFrame() {
  return captureMicroscopeFrame('leukemia');
}

function retakeLeukemiaFrame() {
  return retakeMicroscopeFrame('leukemia');
}

function discardLeukemiaFrame() {
  return retakeMicroscopeFrame('leukemia');
}

function refreshLeukemiaCameraDevices() {
  return refreshCameraDevices(undefined, 'leukemia');
}

function onLeukemiaCameraDeviceChange() {
  return onCameraDeviceChange('leukemia');
}

async function captureAndAnalyzeLeukemiaMicroscope() {
  if (SmartCellMicroscope.isAnalyzing) return;
  const els = getCameraElements('leukemia');
  const errBox = els.errBox;
  if (errBox) {
    errBox.innerHTML = '';
    errBox.style.display = 'none';
  }

  let blobToAnalyze = SmartCellMicroscope.capturedBlob;
  if (!blobToAnalyze || SmartCellMicroscope.stream) {
    if (SmartCellMicroscope.stream && !SmartCellMicroscope.capturedBlob) {
      blobToAnalyze = await captureMicroscopeFrame('leukemia');
    }
  }

  if (!blobToAnalyze || blobToAnalyze.size === 0) {
    if (errBox) {
      errBox.innerHTML = '<div class="alert-error"><strong>No Captured Frame:</strong> Please start the USB microscope camera and capture a cell frame before analyzing.</div>';
      errBox.style.display = 'block';
    }
    return;
  }

  const caBtn = els.analyzeBtn;
  SmartCellMicroscope.isAnalyzing = true;
  if (caBtn) {
    caBtn.disabled = true;
    caBtn.textContent = 'Classifying Captured Cell...';
  }
  setMicroscopeStatus('captured', 'Classifying Captured Frame...', `${SmartCellMicroscope.capturedWidth}\u00d7${SmartCellMicroscope.capturedHeight} px`, 'leukemia');

  try {
    const ts = new Date().toISOString().replace(/[-:T]/g, '_').split('.')[0];
    const filename = `microscope_capture_${ts}.jpg`;
    const fd = new FormData();
    fd.append('disease_mode', 'leukemia');
    fd.append('capture_source', 'live_microscope');
    fd.append('image_file', blobToAnalyze, filename);

    const sampleIdInput = document.getElementById('leukemiaSampleId');
    if (sampleIdInput && sampleIdInput.value.trim()) {
      fd.append('patient_sample_id', sampleIdInput.value.trim());
    }

    const resp = await fetch('/api/analyze', { method: 'POST', body: fd });
    const parsed = await parseApiResponse(resp);

    if (!parsed.ok || !parsed.data || !parsed.data.success) {
      const errType = (parsed.data && parsed.data.error_type) || parsed.errorType || 'Analysis Error';
      const errMsg = (parsed.data && parsed.data.error_message) || parsed.errorMessage || 'Inference failed.';
      if (errBox) {
        errBox.innerHTML = `<strong>${errType}:</strong> ${errMsg}`;
        errBox.style.display = 'block';
      }
      setMicroscopeStatus('err', 'Analysis Error — See Details Below', 'Error', 'leukemia');
      return;
    }

    const data = parsed.data;

    renderLeukemiaResults(data);
    const emptyState = document.getElementById('leukemiaEmptyState');
    const resultsCard = document.getElementById('leukemiaResultsCard');
    if (emptyState) emptyState.style.display = 'none';
    if (resultsCard) {
      resultsCard.style.display = 'block';
      resultsCard.scrollIntoView({ behavior: 'smooth' });
    }

    setMicroscopeStatus('captured', 'Analysis Complete — Results Displayed', `${SmartCellMicroscope.capturedWidth}\u00d7${SmartCellMicroscope.capturedHeight} px`, 'leukemia');
  } catch (err) {
    if (errBox) {
      errBox.innerHTML = `<div class="alert-error"><strong>Live Capture Analysis Failed:</strong> ${err.message}</div>`;
      errBox.style.display = 'block';
    }
    setMicroscopeStatus('err', 'Network / Analysis Error', 'Error', 'leukemia');
  } finally {
    SmartCellMicroscope.isAnalyzing = false;
    if (caBtn) {
      caBtn.disabled = !(SmartCellMicroscope.stream || SmartCellMicroscope.capturedBlob);
      caBtn.textContent = '🔬 Analyze Captured Cell';
    }
  }
}

async function captureAndAnalyzeMicroscope() {
  if (SmartCellMicroscope.isAnalyzing) return;
  const errBox = document.getElementById('errorContainer');
  errBox.innerHTML = '';

  let blobToAnalyze = SmartCellMicroscope.capturedBlob;
  if (!blobToAnalyze || SmartCellMicroscope.stream) {
    if (SmartCellMicroscope.stream && !SmartCellMicroscope.capturedBlob) {
      blobToAnalyze = await captureMicroscopeFrame();
    }
  }

  if (!blobToAnalyze || blobToAnalyze.size === 0) {
    errBox.innerHTML = `<div class="alert-error"><strong>No Captured Frame:</strong> Please start the camera and capture a frame before analyzing.</div>`;
    return;
  }

  const caBtn = document.getElementById('captureAndAnalyzeBtn');
  const runBtn = document.getElementById('runBtn');
  SmartCellMicroscope.isAnalyzing = true;
  caBtn.disabled = true;
  runBtn.disabled = true;
  caBtn.textContent = 'Analyzing Blood Smear...';
  setMicroscopeStatus(
    'captured',
    'Running Two-Stage AI Analysis...',
    `${SmartCellMicroscope.capturedWidth}\u00d7${SmartCellMicroscope.capturedHeight} px`
  );

  try {
    const ts = new Date().toISOString().replace(/[-:T.Z]/g, '').slice(0, 14);
    const filename = `microscope_capture_${ts}.jpg`;

    const fd = new FormData();
    fd.append('image_file', blobToAnalyze, filename);
    fd.append('capture_source', 'live_microscope');
    fd.append('cnn_variant', document.getElementById('cnnVariant').value);
    fd.append('crop_mode', document.getElementById('cropMode').value);
    fd.append('yolo_device', document.getElementById('yoloDevice').value);
    fd.append('yolo_conf', document.getElementById('yoloConf').value);
    fd.append('yolo_iou', document.getElementById('yoloIou').value);
    fd.append('dedup_iou', document.getElementById('dedupIou').value);
    fd.append('cnn_thresh', document.getElementById('cnnThresh').value);
    fd.append('pad_px', document.getElementById('padPx').value);
    fd.append('min_box_size', document.getElementById('minBoxSize').value);
    fd.append('test_zero_detections', document.getElementById('testZeroDet').checked ? '1' : '0');
    fd.append('inject_test_invalid_boxes', document.getElementById('injectInvalidBoxes').checked ? '1' : '0');

    const resp = await fetch('/api/analyze', { method: 'POST', body: fd });
    const parsed = await parseApiResponse(resp);

    if (!parsed.ok || !parsed.data || !parsed.data.success) {
      setMicroscopeStatus('err', 'Analysis Failed', 'Error');
      const errType = (parsed.data && parsed.data.error_type) || parsed.errorType || 'Analysis Error';
      const errMsg = (parsed.data && parsed.data.error_message) || parsed.errorMessage || 'Unknown error occurred.';
      errBox.innerHTML = `<div class="alert-error"><strong>${errType}:</strong> ${errMsg}</div>`;
      return;
    }

    const data = parsed.data;

    CURRENT_RESULT = data;
    renderResults(data);
    setMicroscopeStatus(
      'live',
      `Analysis Complete (${data.run_id})`,
      `${SmartCellMicroscope.capturedWidth}\u00d7${SmartCellMicroscope.capturedHeight} px`
    );
  } catch (err) {
    setMicroscopeStatus('err', 'Request Error', 'Error');
    errBox.innerHTML = `<div class="alert-error"><strong>Live Capture Analysis Failed:</strong> ${err.message}</div>`;
  } finally {
    SmartCellMicroscope.isAnalyzing = false;
    caBtn.disabled = !(SmartCellMicroscope.stream || SmartCellMicroscope.capturedBlob);
    runBtn.disabled = false;
    caBtn.textContent = 'Capture & Analyze Image';
  }
}

function populatePresetDropdown() {
  if (!ENV_STATUS) return;
  const modeEl = document.getElementById('specimenInputMode');
  const modeVal = modeEl ? modeEl.value : 'full_smear';
  const sel = document.getElementById('presetSelect');
  const labelEl = document.getElementById('presetSelectLabel');
  if (!sel) return;
  sel.innerHTML = '<option value="">-- Use Uploaded Image Above --</option>';

  if (modeVal === 'single_cell') {
    if (labelEl) labelEl.textContent = 'Benchmark Single-Cell Specimen';
    (ENV_STATUS.preset_single_cell_images || []).forEach(item => {
      const opt = document.createElement('option');
      opt.value = item.filename;
      opt.textContent = `${item.filename} [GT: ${item.ground_truth_class} | ${item.source_folder}]`;
      sel.appendChild(opt);
    });
  } else {
    if (labelEl) labelEl.textContent = 'Benchmark Test Smear (25 Held-Out Specimens)';
    (ENV_STATUS.preset_test_images || []).forEach(item => {
      const opt = document.createElement('option');
      opt.value = item.filename;
      opt.textContent = `${item.filename} [Patient ${item.patient_id} | GT: ${item.gt_parasitized_rbc}/${item.gt_total_rbc} Par (${item.gt_parasitemia_percent}%)]`;
      sel.appendChild(opt);
    });
  }
}

function onSpecimenModeChange() {
  const modeEl = document.getElementById('specimenInputMode');
  const modeVal = modeEl ? modeEl.value : 'full_smear';
  const hintEl = document.getElementById('specimenModeHint');
  const runBtn = document.getElementById('runBtn');
  if (modeVal === 'single_cell') {
    if (hintEl) {
      hintEl.textContent = 'Single-Cell Mode classifies 1 pre-cropped erythrocyte directly (whole-smear parasitemia is not calculated).';
    }
    if (runBtn) runBtn.textContent = 'Classify Single Cell';
  } else {
    if (hintEl) {
      hintEl.textContent = 'Full-Smear Mode automatically detects erythrocytes, classifies each cell, and calculates estimated parasitemia.';
    }
    if (runBtn) runBtn.textContent = 'Analyze Blood Smear';
  }
  populatePresetDropdown();
}

function inspectUploadedImageDimensions(fileObj) {
  return new Promise(resolve => {
    if (!fileObj) {
      resolve(null);
      return;
    }
    const url = URL.createObjectURL(fileObj);
    const img = new Image();
    img.onload = () => {
      const dims = { width: img.naturalWidth || img.width, height: img.naturalHeight || img.height };
      URL.revokeObjectURL(url);
      resolve(dims);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      resolve(null);
    };
    img.src = url;
  });
}

async function loadEnvStatus() {
  try {
    const resp = await fetch('/api/status');
    const parsed = await parseApiResponse(resp);
    if (!parsed.ok || !parsed.data) return;
    ENV_STATUS = parsed.data;
    const g = ENV_STATUS.gpu_environment;
    const badges = document.getElementById('hwBadges');
    if (badges) {
      const gpuClass = g.cuda_available ? 'gpu-ok' : 'cpu-ok';
      const gpuText = g.cuda_available
        ? `Stage 1 GPU: ${g.gpu_name}`
        : `Stage 1: ${g.gpu_name} (CPU)`;
      badges.innerHTML = `
        <span class="badge ${gpuClass}">${gpuText}</span>
        <span class="badge cpu-ok">Stage 2: Classifier Engine Active</span>
      `;
    }
    populatePresetDropdown();
  } catch (err) {
    console.error('Status check failed:', err);
  }
}

function onVariantChange() {
  const v = document.getElementById('cnnVariant').value;
  const cropSel = document.getElementById('cropMode');
  const desc = document.getElementById('variantDesc');
  if (v === 'finetuned_raw') {
    cropSel.value = 'raw';
    desc.textContent = 'Fine-tuned on native rectangular full-smear crops (Test Acc: 98.35%). Recommended.';
  } else if (v === 'finetuned_masked') {
    cropSel.value = 'masked';
    desc.textContent = 'Validated on elliptical black-corner masked crops (Test Acc: 98.10%).';
  } else {
    cropSel.value = 'masked';
    desc.textContent = 'Original single-cell CNN (95.04% test accuracy).';
  }
}

function switchTab(paneId, btnEl) {
  document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById(paneId).classList.add('active');
  btnEl.classList.add('active');
}

async function runAnalysis() {
  if (ACTIVE_INPUT_MODE === 'microscope' && (SmartCellMicroscope.capturedBlob || SmartCellMicroscope.stream)) {
    await captureAndAnalyzeMicroscope();
    return;
  }

  const errBox = document.getElementById('errorContainer');
  errBox.innerHTML = '';
  const fileInput = document.getElementById('imageUpload');
  const presetVal = document.getElementById('presetSelect').value;
  const modeSel = document.getElementById('specimenInputMode');
  let selectedSpecimenMode = modeSel ? modeSel.value : 'full_smear';

  if (!fileInput.files.length && !presetVal) {
    errBox.innerHTML = `<div class="alert-error"><strong>Input Required:</strong> Please upload a blood-smear image file, select a benchmark test image, or capture a frame in Live Microscope mode.</div>`;
    return;
  }

  if (fileInput.files.length > 0) {
    const dims = await inspectUploadedImageDimensions(fileInput.files[0]);
    if (dims && selectedSpecimenMode === 'full_smear' && dims.width <= 300 && dims.height <= 300) {
      const userConfirmedSingleCell = window.confirm(
        `The uploaded image (${dims.width}\u00d7${dims.height} px) has dimensions characteristic of a single-cell RBC patch rather than a full blood smear.\n\n` +
        `Click OK to switch to Single-Cell Mode, or Cancel to continue in Full-Smear Mode.`
      );
      if (userConfirmedSingleCell && modeSel) {
        modeSel.value = 'single_cell';
        selectedSpecimenMode = 'single_cell';
        onSpecimenModeChange();
      }
    } else if (dims && selectedSpecimenMode === 'single_cell' && dims.width >= 640 && dims.height >= 480) {
      const userConfirmedFullSmear = window.confirm(
        `The uploaded image (${dims.width}\u00d7${dims.height} px) has full-smear dimensions, but Single-Cell Mode is selected.\n\n` +
        `Click OK to switch to Full-Smear Mode, or Cancel to keep Single-Cell Mode.`
      );
      if (userConfirmedFullSmear && modeSel) {
        modeSel.value = 'full_smear';
        selectedSpecimenMode = 'full_smear';
        onSpecimenModeChange();
      }
    }
  }

  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  btn.textContent = selectedSpecimenMode === 'single_cell'
    ? 'Classifying Single Cell...'
    : 'Analyzing Blood Smear...';

  try {
    const fd = new FormData();
    if (fileInput.files.length > 0) {
      fd.append('image_file', fileInput.files[0]);
      fd.append('capture_source', 'manual_upload');
    } else {
      fd.append('preset_filename', presetVal);
      fd.append('capture_source', 'preset_test_image');
    }
    fd.append('input_mode', selectedSpecimenMode);
    fd.append('cnn_variant', document.getElementById('cnnVariant').value);
    fd.append('crop_mode', document.getElementById('cropMode').value);
    fd.append('yolo_device', document.getElementById('yoloDevice').value);
    fd.append('yolo_conf', document.getElementById('yoloConf').value);
    fd.append('yolo_iou', document.getElementById('yoloIou').value);
    fd.append('dedup_iou', document.getElementById('dedupIou').value);
    fd.append('cnn_thresh', document.getElementById('cnnThresh').value);
    fd.append('pad_px', document.getElementById('padPx').value);
    fd.append('min_box_size', document.getElementById('minBoxSize').value);
    fd.append('test_zero_detections', document.getElementById('testZeroDet').checked ? '1' : '0');
    fd.append('inject_test_invalid_boxes', document.getElementById('injectInvalidBoxes').checked ? '1' : '0');

    const resp = await fetch('/api/analyze', { method: 'POST', body: fd });
    const parsed = await parseApiResponse(resp);

    if (!parsed.ok || !parsed.data || !parsed.data.success) {
      const errType = (parsed.data && parsed.data.error_type) || parsed.errorType || 'Analysis Error';
      const errMsg = (parsed.data && parsed.data.error_message) || parsed.errorMessage || 'Unknown error occurred.';
      errBox.innerHTML = `<div class="alert-error"><strong>${errType}:</strong> ${errMsg}</div>`;
      return;
    }

    const data = parsed.data;
    CURRENT_RESULT = data;
    renderResults(data);
  } catch (err) {
    errBox.innerHTML = `<div class="alert-error"><strong>Request Failed:</strong> ${err.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = selectedSpecimenMode === 'single_cell'
      ? 'Classify Single Cell'
      : 'Analyze Blood Smear';
  }
}

function renderResults(data) {
  document.getElementById('welcomeBanner').style.display = 'none';
  document.getElementById('resultsContainer').style.display = 'block';

  const s = data.summary;
  const inputMode = data.input_mode || s.input_mode || 'full_smear';
  const isSingleCell = (inputMode === 'single_cell');
  const scRes = s.single_cell_result || null;
  const cnt = s.counts;
  const parEst = s.parasitemia_estimate || {};
  const times = s.processing_time_seconds;
  const conf = s.confidence_statistics || {};
  const hw = s.hardware_execution || {};

  // Render Final AI Screening Result Card
  const finalCard = document.getElementById('finalScreeningCard');
  const fs = data.final_screening_result || s.final_ai_screening_result || {};
  let stmt = fs.statement;
  if (!stmt) {
    if (isSingleCell && scRes) {
      stmt = `Single-Cell Classification Result: Predicted ${scRes.predicted_class} (Confidence: ${(scRes.confidence * 100).toFixed(2)}%). Whole-smear parasitemia is not applicable to single-cell input.`;
    } else if (cnt.valid_candidate_rbc_crops <= 0) {
      stmt = 'No reliable result is available for this image.';
    } else if (cnt.cnn_parasitized_rbc_count >= 1) {
      stmt = `The AI flagged ${cnt.cnn_parasitized_rbc_count} potentially parasitized red blood cells. Estimated parasitemia: ${(parEst.estimated_parasitemia_percent || 0).toFixed(2)}%.`;
    } else {
      stmt = 'The AI did not flag any potentially parasitized red blood cells in this image.';
    }
  }
  const disc = fs.disclaimer || 'This AI screening result does not confirm or rule out malaria in a patient. Clinical assessment and appropriate diagnostic testing are required.';
  finalCard.className = 'final-screening-card';
  if (cnt.valid_candidate_rbc_crops > 0 && cnt.cnn_parasitized_rbc_count >= 1) {
    finalCard.classList.add('par-flagged');
  } else if (cnt.valid_candidate_rbc_crops > 0) {
    finalCard.classList.add('par-zero');
  }
  const modeHeaderLabel = isSingleCell
    ? 'Single-Cell Screening'
    : (data.capture_source === 'live_microscope' ? 'Live USB Microscope Capture' : 'Blood-Smear Analysis');
  finalCard.innerHTML = `
    <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.6px; color:var(--muted); font-weight:700; margin-bottom:3px;">
      AI Screening Assessment (${modeHeaderLabel})
    </div>
    <div style="font-size:1.10rem; font-weight:800; color:#fff; margin-bottom:5px;" id="finalScreeningStatementText">${stmt}</div>
    <div style="font-size:0.80rem; color:#cbd5e1; line-height:1.4;" id="finalScreeningDisclaimerText">${disc}</div>
  `;

  // Render Pre-Inference Image Quality Warnings if present
  const qWarnEl = document.getElementById('resultQualityWarnings');
  const qWarnings = (data.image_quality && data.image_quality.warnings) || s.quality_warnings || [];
  if (qWarnings.length > 0) {
    qWarnEl.innerHTML = `
      <div class="alert-warn" id="analysisQualityWarningBanner">
        <strong>Image Quality &amp; Advisory Notice:</strong><br>
        ${qWarnings.map(w => `&bull; ${w}`).join('<br>')}
      </div>
    `;
  } else {
    qWarnEl.innerHTML = '';
  }

  // Update Total Cells Detected KPI
  const kpiTotalEl = document.getElementById('kpiTotalCells');
  if (kpiTotalEl) {
    kpiTotalEl.textContent = isSingleCell ? '1' : (cnt.valid_candidate_rbc_crops || 0);
  }

  if (isSingleCell && scRes) {
    const sigVal = scRes.parasitized_prob !== undefined ? scRes.parasitized_prob : (scRes.sigmoid_score || 0);
    document.getElementById('kpiPar').textContent = scRes.predicted_class;
    document.getElementById('kpiParSub').textContent = `Single Cell (${(scRes.confidence * 100).toFixed(1)}% conf)`;
    document.getElementById('kpiUninf').textContent = `${(scRes.confidence * 100).toFixed(1)}%`;
    document.getElementById('kpiUninfSub').textContent = `Score: ${Number(sigVal).toFixed(4)}`;
    document.getElementById('kpiRate').textContent = 'N/A';
    document.getElementById('kpiRateFrac').textContent = 'Single-cell input';
  } else if (cnt.valid_candidate_rbc_crops <= 0) {
    document.getElementById('kpiPar').textContent = '0';
    document.getElementById('kpiParSub').textContent = 'No cells identified';
    document.getElementById('kpiUninf').textContent = '0';
    document.getElementById('kpiUninfSub').textContent = '0 valid crops';
    document.getElementById('kpiRate').textContent = 'N/A';
    document.getElementById('kpiRateFrac').textContent = 'No reliable cells found';
  } else {
    document.getElementById('kpiPar').textContent = cnt.cnn_parasitized_rbc_count;
    document.getElementById('kpiParSub').textContent = `Parasitized RBCs Flagged`;
    document.getElementById('kpiUninf').textContent = cnt.cnn_uninfected_rbc_count;
    document.getElementById('kpiUninfSub').textContent = `Out of ${cnt.valid_candidate_rbc_crops} cells`;
    document.getElementById('kpiRate').textContent = `${Number(parEst.estimated_parasitemia_percent || 0).toFixed(2)}%`;
    document.getElementById('kpiRateFrac').textContent = parEst.calculation_fraction;
  }
  document.getElementById('kpiWbc').textContent = cnt.excluded_wbc_detections;
  document.getElementById('kpiRej').textContent = cnt.rejected_or_duplicate_boxes;
  document.getElementById('kpiTime').textContent = `${times.total_pipeline_seconds.toFixed(2)}s`;
  document.getElementById('kpiTimeSub').textContent = `Inference time`;

  document.getElementById('runIdBadge').textContent = `Run ID: ${data.run_id}`;

  const yStat = conf.yolo_detection_confidence || {mean:0, median:0, min:0, max:0};
  const cStat = conf.cnn_overall_confidence || {mean:0, median:0, min:0, max:0};
  const pStat = conf.cnn_parasitized_confidence || {mean:0, median:0, min:0, max:0};

  let gtHtml = isSingleCell
    ? '<em>Single-cell specimen mode (whole-smear labels not applicable)</em>'
    : '<em>Uploaded / live image</em>';
  if (data.ground_truth_reference && data.ground_truth_reference.has_ground_truth) {
    const gt = data.ground_truth_reference;
    gtHtml = `Patient <code>${gt.patient_id}</code> | GT Parasitized: <strong>${gt.gt_parasitized_rbc} / ${gt.gt_total_rbc} (${gt.gt_parasitemia_percent}%)</strong> | GT WBC: ${gt.gt_wbc}`;
  }

  const rawCls = scRes ? (scRes.raw_predicted_class || scRes.raw_crop_predicted_class || 'N/A') : 'N/A';
  const rawCf = scRes ? Number(scRes.raw_confidence !== undefined ? scRes.raw_confidence : (scRes.raw_crop_confidence || 0)) : 0;
  const maskCls = scRes ? (scRes.masked_predicted_class || scRes.ellipse_masked_predicted_class || 'N/A') : 'N/A';
  const maskCf = scRes ? Number(scRes.masked_confidence !== undefined ? scRes.masked_confidence : (scRes.ellipse_masked_confidence || 0)) : 0;

  const auditComparisonText = isSingleCell && scRes
    ? `Raw=${rawCls} (${(rawCf * 100).toFixed(1)}%) vs Masked=${maskCls} (${(maskCf * 100).toFixed(1)}%)`
    : (cnt.valid_candidate_rbc_crops > 0
        ? `Raw=${Number(parEst.raw_crop_parasitemia_percent || 0).toFixed(2)}% vs Masked=${Number(parEst.ellipse_masked_parasitemia_percent || 0).toFixed(2)}%`
        : 'N/A');

  const yoloConfText = isSingleCell
    ? 'Bypassed (Single Cell)'
    : `${(yStat.mean*100).toFixed(1)}% (Range: ${(yStat.min*100).toFixed(1)}%–${(yStat.max*100).toFixed(1)}%)`;

  document.getElementById('execSummaryGrid').innerHTML = `
    <div>
      <div><strong>Image:</strong> <code>${s.image_filename}</code> (${s.image_dimensions.width}&times;${s.image_dimensions.height} px)</div>
      <div><strong>Analysis Mode:</strong> <code>${inputMode}</code> (${data.capture_source || s.capture_source || 'manual_upload'})</div>
      <div><strong>Compute Hardware:</strong> ${hw.stage1_device} / ${hw.stage2_device}</div>
    </div>
    <div>
      <div><strong>CNN Variant:</strong> <code>${s.models.stage2_cnn_variant}</code></div>
      <div><strong>Primary Crop Mode:</strong> <code>${s.models.primary_crop_preprocessing_mode}</code></div>
      <div><strong>Dual-Crop Comparison:</strong> ${auditComparisonText}</div>
    </div>
    <div>
      <div><strong>YOLO Detection Conf:</strong> ${yoloConfText}</div>
      <div><strong>CNN Confidence (Mean):</strong> ${(cStat.mean*100).toFixed(1)}%</div>
      <div><strong>Ground-Truth Reference:</strong> ${gtHtml}</div>
    </div>
  `;

  const pdfInfo = data.pdf_report || {};
  const pdfFilename = pdfInfo.filename || `SmartCell_Report_${data.stem}.pdf`;
  const pdfDownloadUrl = (data.urls && data.urls.export_pdf)
    ? data.urls.export_pdf
    : `/api/file?run_id=${encodeURIComponent(data.run_id)}&file=${encodeURIComponent(pdfFilename)}&download=1`;
  const pdfReady = (pdfInfo.generated !== false) && Boolean(pdfDownloadUrl);

  let pdfStatusBanner = '';
  let pdfPrimaryBtn = '';
  if (pdfReady) {
    pdfStatusBanner = `<div class="alert-success" id="pdfStatusBanner"><strong>PDF Report Generated:</strong> ${pdfInfo.message || ('Ready to download ' + pdfFilename)}</div>`;
    pdfPrimaryBtn = `<a class="btn-pdf-primary" id="downloadPdfBtn" href="${pdfDownloadUrl}" download="${pdfFilename}">📄 Download Complete PDF Report</a>`;
  } else {
    const errMsg = pdfInfo.error || 'PDF report could not be generated for this run.';
    pdfStatusBanner = `<div class="alert-error" id="pdfStatusBanner"><strong>PDF Report Error:</strong> ${errMsg}</div>`;
  }

  document.getElementById('exportButtons').innerHTML = `
    ${pdfStatusBanner}
    <div style="margin-bottom:12px;">
      ${pdfPrimaryBtn}
    </div>
    <div class="hint" style="margin-bottom:8px; font-weight:600; color:#cbd5e1;">Export Data Artifacts:</div>
    <div style="display:flex; flex-wrap:wrap; gap:8px;">
      <a class="btn-export" href="${data.urls.export_csv}" download>📊 Per-Cell CSV</a>
      <a class="btn-export" href="${data.urls.export_summary_json}" download>💾 Summary JSON</a>
      <a class="btn-export" href="${data.urls.export_rejected_json}" download>📋 Excluded JSON</a>
      <a class="btn-export" href="${data.urls.export_txt}" download>📝 Text Summary</a>
      ${data.urls.annotated_image ? `<a class="btn-export" href="${data.urls.annotated_image}&download=1" download>🖼️ Annotated Image</a>` : ''}
    </div>
  `;

  const annEl = document.getElementById('annotatedImg');
  if (data.urls.annotated_image) {
    annEl.src = data.urls.annotated_image + '&t=' + Date.now();
    annEl.style.display = 'block';
  } else {
    annEl.style.display = 'none';
  }

  const origEl = document.getElementById('originalInputImg');
  if (data.urls.original_image) {
    origEl.src = data.urls.original_image + '&t=' + Date.now();
    origEl.style.display = 'block';
  } else {
    origEl.style.display = 'none';
  }

  const csEl = document.getElementById('contactImg');
  if (data.urls.contact_sheet) {
    csEl.src = data.urls.contact_sheet + '&t=' + Date.now();
    csEl.style.display = 'block';
  } else {
    csEl.style.display = 'none';
  }

  document.getElementById('tabCellCount').textContent = data.cell_predictions.length;
  document.getElementById('tabExclCount').textContent =
    (data.rejected_and_excluded.excluded_wbc_count || 0) + (data.rejected_and_excluded.rejected_or_duplicate_count || 0);
  document.getElementById('txtReportBox').textContent = data.text_report || '';

  renderCellTable();
  renderExcludedAndRejected(data.rejected_and_excluded);
}

function renderCellTable() {
  if (!CURRENT_RESULT) return;
  const filterVal = document.getElementById('cellFilter').value;
  const rows = CURRENT_RESULT.cell_predictions || [];
  const filtered = rows.filter(r => filterVal === 'ALL' || r.cnn_predicted_class === filterVal);

  const tbody = document.getElementById('cellTableBody');
  if (!filtered.length) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; color:var(--muted); padding:20px;">No candidate RBC cells match the current filter.</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered.map(r => {
    const isPar = r.cnn_predicted_class === 'Parasitized';
    const pillClass = isPar ? 'pill-par' : 'pill-uninf';
    const cropUrl = `/api/file?run_id=${CURRENT_RESULT.run_id}&file=${encodeURIComponent(r.crop_rel_path.replace(/\\/g, '/'))}`;
    const yoloConfCell = (r.yolo_class_name === 'SINGLE_CELL_INPUT')
      ? '<span style="color:var(--muted);">N/A (Single Cell)</span>'
      : `${(parseFloat(r.yolo_confidence)*100).toFixed(1)}%`;
    return `
      <tr>
        <td><img class="crop-thumb" src="${cropUrl}" loading="lazy" alt="${r.crop_id}"></td>
        <td><code>${r.crop_id}</code></td>
        <td>[${r.bbox_x1}, ${r.bbox_y1}, ${r.bbox_x2}, ${r.bbox_y2}]</td>
        <td>${yoloConfCell}</td>
        <td><span class="pill ${pillClass}">${r.cnn_predicted_class}</span></td>
        <td><strong>${(parseFloat(r.cnn_confidence)*100).toFixed(1)}%</strong></td>
        <td>${(parseFloat(r.cnn_raw_parasitized_prob)*100).toFixed(1)}%</td>
        <td>${(parseFloat(r.cnn_masked_parasitized_prob)*100).toFixed(1)}%</td>
      </tr>
    `;
  }).join('');
}

function renderExcludedAndRejected(rejData) {
  const wbcs = rejData.excluded_wbcs || [];
  const rejs = rejData.rejected_boxes || [];

  const wbcBox = document.getElementById('wbcListContainer');
  if (!wbcs.length) {
    wbcBox.innerHTML = `<div class="hint">0 WBC detections excluded in this field.</div>`;
  } else {
    wbcBox.innerHTML = `<table><thead><tr><th>Class</th><th>Confidence</th><th>Clipped BBox</th><th>Exclusion Reason</th></tr></thead><tbody>` +
      wbcs.map(w => `<tr><td>${w.yolo_class_name} (ID ${w.yolo_class_id})</td><td>${(w.yolo_confidence*100).toFixed(1)}%</td><td>[${w.clipped_bbox.join(', ')}]</td><td>${w.reason}</td></tr>`).join('') +
      `</tbody></table>`;
  }

  const rejBox = document.getElementById('rejectedListContainer');
  if (!rejs.length) {
    rejBox.innerHTML = `<div class="hint">0 rejected or duplicate detections in this field.</div>`;
  } else {
    rejBox.innerHTML = `<table><thead><tr><th>Reason</th><th>Details</th></tr></thead><tbody>` +
      rejs.map(r => `<tr><td>${r.reason}</td><td><code>${JSON.stringify(r.clipped_bbox || (r.suppressed_detection && r.suppressed_detection.bbox) || {})}</code></td></tr>`).join('') +
      `</tbody></table>`;
  }
}

// Disease Scope Switcher
let currentDiseaseScope = 'malaria';

function switchDiseaseScope(scope) {
  if (SmartCellMicroscope.stream) {
    stopMicroscopeCamera(true);
  }

  currentDiseaseScope = scope;
  const tabMalaria = document.getElementById('diseaseTabMalaria');
  const tabLeukemia = document.getElementById('diseaseTabLeukemia');
  const viewMalaria = document.getElementById('malariaWorkflow');
  const viewLeukemia = document.getElementById('leukemiaWorkflow');

  if (scope === 'malaria') {
    tabMalaria.className = 'disease-tab active';
    tabLeukemia.className = 'disease-tab';
    viewMalaria.style.display = 'grid';
    viewLeukemia.style.display = 'none';
  } else {
    tabMalaria.className = 'disease-tab';
    tabLeukemia.className = 'disease-tab active-leukemia';
    viewMalaria.style.display = 'none';
    viewLeukemia.style.display = 'grid';
  }
}

// Mode 2 Input Option Switcher (Live Microscope, Upload Image, Dataset Samples)
let activeLeukemiaInputMode = 'microscope';

function switchLeukemiaInputMode(mode) {
  activeLeukemiaInputMode = mode;
  const btnMicro = document.getElementById('leukemiaInputMicroscopeBtn');
  const btnUpload = document.getElementById('leukemiaInputUploadBtn');
  const btnSamples = document.getElementById('leukemiaInputSamplesBtn');
  const panelMicro = document.getElementById('leukemiaLiveMicroscopePanel');
  const panelUpload = document.getElementById('leukemiaUploadPanel');
  const panelSamples = document.getElementById('leukemiaSamplesPanel');

  if (btnMicro) {
    btnMicro.classList.toggle('active', mode === 'microscope');
    btnMicro.classList.toggle('active-leukemia', mode === 'microscope');
  }
  if (btnUpload) {
    btnUpload.classList.toggle('active', mode === 'upload');
    btnUpload.classList.toggle('active-leukemia', mode === 'upload');
  }
  if (btnSamples) {
    btnSamples.classList.toggle('active', mode === 'samples');
    btnSamples.classList.toggle('active-leukemia', mode === 'samples');
  }

  if (panelMicro) panelMicro.style.display = (mode === 'microscope') ? 'block' : 'none';
  if (panelUpload) panelUpload.style.display = (mode === 'upload') ? 'block' : 'none';
  if (panelSamples) panelSamples.style.display = (mode === 'samples') ? 'block' : 'none';

  if (mode !== 'microscope' && SmartCellMicroscope.stream && currentDiseaseScope === 'leukemia') {
    stopMicroscopeCamera(true, 'leukemia');
  }
}

let selectedLeukemiaPreset = '';

function selectLeukemiaPreset(btn, filename) {
  selectedLeukemiaPreset = filename;
  document.querySelectorAll('.leukemia-preset-btn').forEach(b => b.classList.remove('selected'));
  btn.classList.add('selected');
  const fileInput = document.getElementById('leukemiaImageUpload');
  if (fileInput) fileInput.value = '';
  const isBlast = filename.includes('_1');
  const label = isBlast ? 'Lymphoblast (ALL-associated category)' : 'Normal Lymphocyte';
  const hintEl = document.getElementById('leukemiaPresetSelectedHint');
  if (hintEl) hintEl.textContent = 'Selected: ' + filename + ' [' + label + ']';

  const prevSection = document.getElementById('leukemiaSamplePreviewSection');
  const prevImg = document.getElementById('leukemiaSamplePreviewImg');
  const badgeEl = document.getElementById('leukemiaSampleDetailsBadge');
  if (prevSection && prevImg) {
    prevImg.src = `/api/preset_image?filename=${encodeURIComponent(filename)}&disease=leukemia`;
    if (badgeEl) badgeEl.textContent = isBlast ? 'Blast Candidate' : 'Normal Reference';
    prevSection.style.display = 'block';
  }
}

function onLeukemiaFileChange() {
  const fileInput = document.getElementById('leukemiaImageUpload');
  if (fileInput && fileInput.files.length > 0) {
    const file = fileInput.files[0];
    selectedLeukemiaPreset = '';
    document.querySelectorAll('.leukemia-preset-btn').forEach(b => b.classList.remove('selected'));
    document.getElementById('leukemiaSelectedName').textContent = 'Selected: ' + file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)';

    const prevSection = document.getElementById('leukemiaUploadPreviewSection');
    const prevImg = document.getElementById('leukemiaUploadPreviewImg');
    const badgeEl = document.getElementById('leukemiaUploadDetailsBadge');
    const metaBox = document.getElementById('leukemiaUploadMetaBox');

    if (prevSection && prevImg) {
      const objUrl = URL.createObjectURL(file);
      prevImg.src = objUrl;
      const testImg = new Image();
      testImg.onload = function() {
        if (badgeEl) badgeEl.textContent = `${testImg.naturalWidth} × ${testImg.naturalHeight} px`;
        if (metaBox) {
          metaBox.innerHTML = `<strong>File:</strong> ${file.name} &bull; <strong>Size:</strong> ${(file.size / 1024).toFixed(1)} KB &bull; <strong>Resolution:</strong> ${testImg.naturalWidth} × ${testImg.naturalHeight} px`;
        }
      };
      testImg.src = objUrl;
      prevSection.style.display = 'block';
    }
  }
}

async function runLeukemiaAnalysis(sourceType) {
  const fileInput = document.getElementById('leukemiaImageUpload');
  const hasFile = fileInput && fileInput.files.length > 0;
  const isUpload = (sourceType === 'upload');
  const isSample = (sourceType === 'sample');

  const fd = new FormData();
  fd.append('disease_mode', 'leukemia');

  if (isUpload || (activeLeukemiaInputMode === 'upload' && hasFile)) {
    if (!hasFile) {
      alert('Please select an image file to upload.');
      return;
    }
    fd.append('image_file', fileInput.files[0]);
    fd.append('capture_source', 'manual_upload');
  } else if (isSample || (activeLeukemiaInputMode === 'samples' && selectedLeukemiaPreset)) {
    if (!selectedLeukemiaPreset) {
      alert('Please select a benchmark test cell preset.');
      return;
    }
    fd.append('preset_filename', selectedLeukemiaPreset);
    fd.append('capture_source', 'preset_test_image');
  } else if (activeLeukemiaInputMode === 'microscope') {
    await captureAndAnalyzeLeukemiaMicroscope();
    return;
  } else {
    if (hasFile) {
      fd.append('image_file', fileInput.files[0]);
      fd.append('capture_source', 'manual_upload');
    } else if (selectedLeukemiaPreset) {
      fd.append('preset_filename', selectedLeukemiaPreset);
      fd.append('capture_source', 'preset_test_image');
    } else {
      alert('Please select a specimen: capture via live microscope, upload an image, or select a dataset sample.');
      return;
    }
  }

  const sampleIdInput = document.getElementById('leukemiaSampleId');
  if (sampleIdInput && sampleIdInput.value.trim()) {
    fd.append('patient_sample_id', sampleIdInput.value.trim());
  }

  const errBox = document.getElementById('leukemiaErrorAlert');
  const resultsCard = document.getElementById('leukemiaResultsCard');
  const emptyState = document.getElementById('leukemiaEmptyState');
  if (errBox) errBox.style.display = 'none';

  const btnUpload = document.getElementById('btnRunLeukemiaUpload');
  const btnSample = document.getElementById('btnRunLeukemiaSample');
  if (btnUpload) btnUpload.disabled = true;
  if (btnSample) btnSample.disabled = true;

  try {
    const resp = await fetch('/api/analyze', { method: 'POST', body: fd });
    const parsed = await parseApiResponse(resp);

    if (!parsed.ok || !parsed.data || !parsed.data.success) {
      const errType = (parsed.data && parsed.data.error_type) || parsed.errorType || 'Analysis Error';
      const errMsg = (parsed.data && parsed.data.error_message) || parsed.errorMessage || 'Inference failed.';
      if (errBox) {
        errBox.innerHTML = `<strong>${errType}:</strong> ${errMsg}`;
        errBox.style.display = 'block';
      }
      return;
    }

    const data = parsed.data;

    renderLeukemiaResults(data);
    if (emptyState) emptyState.style.display = 'none';
    if (resultsCard) {
      resultsCard.style.display = 'block';
      resultsCard.scrollIntoView({ behavior: 'smooth' });
    }
  } catch (err) {
    if (errBox) {
      errBox.innerHTML = `<strong>Network / Server Error:</strong> ${err.message}`;
      errBox.style.display = 'block';
    }
  } finally {
    if (btnUpload) btnUpload.disabled = false;
    if (btnSample) btnSample.disabled = false;
  }
}

function renderLeukemiaResults(data) {
  const c = data.classification;
  const isBlast = c.is_blast;
  const banner = document.getElementById('leukemiaDiagnosisBanner');

  const sourceLabel = data.capture_source || 'Manual Upload';
  const dimStr = data.original_dimensions ? `${data.original_dimensions[0]}×${data.original_dimensions[1]} px` : '224×224 px';

  banner.className = isBlast ? 'diagnosis-banner banner-blast' : 'diagnosis-banner banner-normal';
  
  const cellTypeTitle = isBlast 
    ? 'Lymphoblast (ALL-associated cell category)' 
    : 'Normal Lymphocyte';
  const iconEmoji = isBlast ? '⚠️' : '✅';
  const cellSubtitle = isBlast
    ? 'Suspected Acute Lymphoblastic Leukemia Pattern'
    : 'Normal Mature White Blood Cell Morphology';

  banner.innerHTML = `
    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">
      <div>
        <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.8px; font-weight:700; opacity:0.85; margin-bottom:4px;">
          AI Classification Result
        </div>
        <div style="font-size:1.35rem; font-weight:800; letter-spacing:-0.2px;">
          ${iconEmoji} ${cellTypeTitle}
        </div>
        <div style="font-size:0.84rem; opacity:0.9; margin-top:4px;">
          ${cellSubtitle}
        </div>
      </div>
      <div style="text-align:right;">
        <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.8px; font-weight:700; opacity:0.85; margin-bottom:2px;">
          Confidence Score
        </div>
        <div style="font-size:1.6rem; font-weight:800;">
          ${c.confidence_percent.toFixed(2)}%
        </div>
      </div>
    </div>
    <div style="margin-top:12px; padding-top:10px; border-top:1px solid rgba(255,255,255,0.15); display:flex; gap:16px; flex-wrap:wrap; font-size:0.80rem; opacity:0.88;">
      <span>Specimen Source: <strong>${sourceLabel}</strong></span>
      <span>Image Size: <strong>${dimStr}</strong></span>
      <span>Inference Latency: <strong>${data.performance.inference_time_ms.toFixed(1)} ms</strong></span>
      <span>Model: <strong>${data.model_metadata.model_architecture}</strong></span>
    </div>
  `;

  // Update Image Preview
  const imgElem = document.getElementById('leukemiaPreviewImg');
  imgElem.src = data.download_urls.image + '&t=' + Date.now();

  // Update Probabilities
  const pNorm = (c.probability_normal_lymphocyte * 100).toFixed(2);
  const pBlast = (c.probability_lymphoblast * 100).toFixed(2);

  document.getElementById('pNormVal').textContent = pNorm + '%';
  document.getElementById('pBlastVal').textContent = pBlast + '%';
  document.getElementById('pNormBar').style.width = pNorm + '%';
  document.getElementById('pBlastBar').style.width = pBlast + '%';

  // Morphological summary
  document.getElementById('leukemiaMorphSummary').textContent = data.morphological_assessment;

  // Downloads
  document.getElementById('dlLeukemiaPdf').href = data.download_urls.pdf;
  document.getElementById('dlLeukemiaJson').href = data.download_urls.json;
  document.getElementById('dlLeukemiaTxt').href = data.download_urls.txt;
}

window.addEventListener('beforeunload', () => {
  releaseMicroscopeStreamOnly();
});

window.addEventListener('DOMContentLoaded', () => {
  loadEnvStatus();
  const upEl = document.getElementById('imageUpload');
  if (upEl) {
    upEl.addEventListener('change', () => {
      if (upEl.files.length > 0) {
        const ps = document.getElementById('presetSelect');
        if (ps) ps.value = '';
      }
    });
  }
});
</script>
</body>
</html>
"""


def parse_multipart_form_data(headers, body_bytes):
    """
    Lightweight, zero-dependency multipart/form-data parser compatible with Python 3.11+.
    Returns (fields_dict, files_dict).
    """
    content_type = headers.get("Content-Type", "")
    if "multipart/form-data" not in content_type or "boundary=" not in content_type:
        return {}, {}

    boundary = content_type.split("boundary=")[1].split(";")[0].strip()
    if boundary.startswith('"') and boundary.endswith('"'):
        boundary = boundary[1:-1]

    delimiter = ("--" + boundary).encode("utf-8")
    parts = body_bytes.split(delimiter)

    fields = {}
    files = {}

    for part in parts[1:]:
        if part.startswith(b"--") or len(part.strip()) == 0:
            continue
        if part.startswith(b"\r\n"):
            part = part[2:]
        if part.endswith(b"\r\n"):
            part = part[:-2]

        header_end = part.find(b"\r\n\r\n")
        if header_end == -1:
            continue

        raw_headers = part[:header_end].decode("utf-8", errors="replace")
        content = part[header_end + 4:]

        disp_name = None
        filename = None
        for hline in raw_headers.split("\r\n"):
            if hline.lower().startswith("content-disposition:"):
                for token in hline.split(";"):
                    token = token.strip()
                    if token.startswith("name="):
                        disp_name = token.split("=", 1)[1].strip('"')
                    elif token.startswith("filename="):
                        filename = token.split("=", 1)[1].strip('"')

        if not disp_name:
            continue

        if filename is not None and filename != "":
            files[disp_name] = {
                "filename": os.path.basename(filename),
                "content": content
            }
        else:
            fields[disp_name] = content.decode("utf-8", errors="replace")

    return fields, files


class SmartCellRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stdout.write(f"[SmartCell WebApp] {self.address_string()} - {fmt % args}\n")

    def _is_allowed_origin(self, origin):
        if not origin:
            return False
        origin_lower = origin.lower()
        if origin_lower == "https://huggingface.co":
            return True
        if origin_lower.endswith(".hf.space") and (origin_lower.startswith("https://") or origin_lower.startswith("http://")):
            return True
        if origin_lower.startswith("http://localhost:") or origin_lower.startswith("http://127.0.0.1:"):
            return True
        if origin_lower == "http://localhost" or origin_lower == "http://127.0.0.1":
            return True
        if "onrender.com" in origin_lower:
            return True
        return False

    def _add_cors_headers(self):
        origin = self.headers.get("Origin")
        if origin and self._is_allowed_origin(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With, Accept, Origin")
            self.send_header("Access-Control-Max-Age", "86400")
            self.send_header("Vary", "Origin")
        elif not origin:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With, Accept, Origin")

    def _send_json(self, status_code, payload):
        raw = json.dumps(payload, indent=2).encode("utf-8")
        try:
            self.send_response(status_code)
            self._add_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass

    def do_OPTIONS(self):
        self.send_response(204)
        self._add_cors_headers()
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def do_HEAD(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html", "/api/health", "/health", "/healthz"):
            self.send_response(200)
            self._add_cors_headers()
            self.send_header("Content-Type", "application/json" if "health" in path else "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
        else:
            self.send_response(404)
            self._add_cors_headers()
            self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            raw = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self._add_cors_headers()
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            self.wfile.write(raw)
            return

        if path in ("/api/health", "/health", "/healthz"):
            self._send_json(200, {
                "status": "healthy",
                "service": "SmartCell Research Platform",
                "timestamp": datetime.now().isoformat(),
                "modes": ["malaria_screening", "all_leukemia"]
            })
            return

        if path in ("/api/status", "/api/models/status"):
            refresh = qs.get("refresh", ["0"])[0] == "1"
            status = query_environment_status(force_refresh=refresh)
            self._send_json(200, status)
            return

        if path == "/api/preset_image":
            fn = qs.get("filename", [""])[0]
            cand = resolve_preset_image_path(fn)
            if cand and os.path.isfile(cand):
                ctype, _ = mimetypes.guess_type(cand)
                with open(cand, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self._add_cors_headers()
                self.send_header("Content-Type", ctype or "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                self.wfile.write(data)
                return
            self._send_json(404, {"error": "Preset image not found."})
            return

        if path == "/api/file":
            run_id = qs.get("run_id", [""])[0]
            rel_file = qs.get("file", [""])[0]
            download = qs.get("download", ["0"])[0] == "1"

            # Prevent path traversal outside DEMO_RUNS_ROOT
            safe_run = os.path.basename(run_id)
            run_dir = os.path.abspath(os.path.join(DEMO_RUNS_ROOT, safe_run))
            target_path = os.path.abspath(os.path.join(run_dir, rel_file))

            if not target_path.startswith(run_dir) or not os.path.isfile(target_path):
                self._send_json(404, {"error": "Requested artifact file not found."})
                return

            if target_path.lower().endswith(".pdf"):
                ctype = "application/pdf"
            else:
                ctype, _ = mimetypes.guess_type(target_path)
                if not ctype:
                    ctype = "application/octet-stream"

            with open(target_path, "rb") as f:
                data = f.read()

            self.send_response(200)
            self._add_cors_headers()
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            if download:
                fname = os.path.basename(target_path)
                self.send_header("Content-Disposition", f'attachment; filename="{fname}"')
            self.end_headers()
            self.wfile.write(data)
            return

        self._send_json(404, {"error": f"Route '{path}' not found."})

    def handle_leukemia_request(self, fields, files):
        run_id = f"run_leukemia_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        run_dir = os.path.join(DEMO_RUNS_ROOT, run_id)
        os.makedirs(run_dir, exist_ok=True)

        capture_source_raw = fields.get("capture_source", "manual_upload")
        if capture_source_raw == "live_microscope":
            capture_source_label = "USB Microscope Capture"
        elif capture_source_raw == "preset_test_image":
            capture_source_label = "Hold-Out Benchmark Test Specimen"
        else:
            capture_source_label = "Manual File Upload"

        image_path = None
        if "image_file" in files and len(files["image_file"]["content"]) > 0:
            file_bytes = files["image_file"]["content"]
            if len(file_bytes) > MAX_UPLOAD_BYTES:
                self._send_json(400, {
                    "success": False,
                    "error_type": "InvalidImageError",
                    "error_message": f"Uploaded file exceeds maximum allowed limit (35.0 MB)."
                })
                return
            raw_fn = files["image_file"]["filename"]
            is_live = (capture_source_raw == "live_microscope")
            safe_fn = sanitize_upload_filename(raw_fn, is_live_capture=is_live)
            upload_path = os.path.join(run_dir, safe_fn)
            with open(upload_path, "wb") as uf:
                uf.write(file_bytes)
            image_path = upload_path
        elif fields.get("preset_filename"):
            cand_path = resolve_preset_image_path(fields["preset_filename"])
            if not cand_path or not os.path.exists(cand_path):
                self._send_json(400, {
                    "success": False,
                    "error_type": "PresetNotFoundError",
                    "error_message": f"Preset leukemia image '{fields['preset_filename']}' not found."
                })
                return
            image_path = cand_path
            capture_source_label = "Hold-Out Benchmark Test Specimen"
        else:
            self._send_json(400, {
                "success": False,
                "error_type": "MissingImageError",
                "error_message": "No image file uploaded, captured from microscope, or preset test cell selected for leukemia analysis."
            })
            return

        try:
            from leukemia_analyzer import run_leukemia_analysis
            result = run_leukemia_analysis(
                image_path=image_path,
                run_dir=run_dir,
                patient_sample_id=fields.get("patient_sample_id"),
                capture_source=capture_source_label
            )
            result["download_urls"] = {
                "pdf": f"/api/file?run_id={run_id}&file={result['files']['pdf']}&download=1",
                "json": f"/api/file?run_id={run_id}&file={result['files']['json']}&download=1",
                "txt": f"/api/file?run_id={run_id}&file={result['files']['txt']}&download=1",
                "image": f"/api/file?run_id={run_id}&file={result['files']['image']}&download=0"
            }
            self._send_json(200, result)
        except Exception as e:
            self._send_json(400, {
                "success": False,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            })

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/analyze":
            self._send_json(404, {"error": "Unknown POST endpoint."})
            return

        try:
            content_len = int(self.headers.get("Content-Length", "0"))
            if content_len <= 0:
                self._send_json(400, {
                    "success": False,
                    "error_type": "MissingImageError",
                    "error_message": "Empty request body (0 bytes)."
                })
                return

            # Drain up to a reasonable ceiling so socket closes cleanly on Windows
            if content_len > MAX_UPLOAD_BYTES + 10 * 1024 * 1024:
                self.close_connection = True
                self._send_json(400, {
                    "success": False,
                    "error_type": "InvalidImageError",
                    "error_message": f"Uploaded payload ({round(content_len / (1024*1024), 2)} MB) exceeds maximum allowed limit (35.0 MB)."
                })
                return

            body_bytes = self.rfile.read(content_len)
            if content_len > MAX_UPLOAD_BYTES:
                self._send_json(400, {
                    "success": False,
                    "error_type": "InvalidImageError",
                    "error_message": f"Uploaded payload ({round(content_len / (1024*1024), 2)} MB) exceeds maximum allowed limit (35.0 MB)."
                })
                return

            fields, files = parse_multipart_form_data(self.headers, body_bytes)

            if fields.get("disease_mode") == "leukemia":
                self.handle_leukemia_request(fields, files)
                return

            os.makedirs(UPLOADS_DIR, exist_ok=True)
            image_path = None
            capture_source = fields.get("capture_source", "manual_upload")
            if capture_source not in ("manual_upload", "live_microscope", "preset_test_image"):
                capture_source = "manual_upload"

            if "image_file" in files:
                file_bytes = files["image_file"]["content"]
                if len(file_bytes) == 0:
                    self._send_json(400, {
                        "success": False,
                        "error_type": "InvalidImageError",
                        "error_message": "Uploaded image file is empty (0 bytes)."
                    })
                    return
                if len(file_bytes) > MAX_UPLOAD_BYTES:
                    self._send_json(400, {
                        "success": False,
                        "error_type": "InvalidImageError",
                        "error_message": "Uploaded image file exceeds maximum allowed limit (35.0 MB)."
                    })
                    return

                raw_fn = files["image_file"]["filename"]
                is_live = (capture_source == "live_microscope")
                safe_fn = sanitize_upload_filename(raw_fn, is_live_capture=is_live)
                upload_sub = os.path.join(UPLOADS_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}")
                os.makedirs(upload_sub, exist_ok=True)
                upload_path = os.path.join(upload_sub, safe_fn)
                with open(upload_path, "wb") as uf:
                    uf.write(file_bytes)
                image_path = upload_path
            elif fields.get("preset_filename"):
                preset_fn = os.path.basename(fields["preset_filename"])
                cand_path = resolve_preset_image_path(preset_fn)
                if not cand_path or not os.path.exists(cand_path):
                    self._send_json(400, {
                        "success": False,
                        "error_type": "PresetNotFoundError",
                        "error_message": f"Preset test image '{preset_fn}' not found."
                    })
                    return
                image_path = cand_path
                capture_source = "preset_test_image"
            else:
                self._send_json(400, {
                    "success": False,
                    "error_type": "MissingImageError",
                    "error_message": "No image file uploaded or preset selected."
                })
                return

            result = execute_pipeline_run(
                image_path=image_path,
                cnn_variant=fields.get("cnn_variant", "finetuned_raw"),
                crop_mode=fields.get("crop_mode", "raw"),
                input_mode=fields.get("input_mode", "full_smear"),
                yolo_device=fields.get("yolo_device", "0"),
                yolo_conf=fields.get("yolo_conf", 0.25),
                yolo_iou=fields.get("yolo_iou", 0.45),
                dedup_iou=fields.get("dedup_iou", 0.50),
                cnn_thresh=fields.get("cnn_thresh", 0.50),
                pad_px=fields.get("pad_px", 4),
                min_box_size=fields.get("min_box_size", 16),
                test_zero_detections=(fields.get("test_zero_detections") == "1"),
                inject_test_invalid_boxes=(fields.get("inject_test_invalid_boxes") == "1"),
                capture_source=capture_source,
            )
            status_code = 200 if result.get("success") else 400
            self._send_json(status_code, result)
        except Exception as e:
            sys.stderr.write(f"[SmartCell Error] {traceback.format_exc()}\n")
            self._send_json(500, {
                "success": False,
                "error_type": type(e).__name__,
                "error_message": "An internal server error occurred while processing the request."
            })


def main():
    default_host = os.environ.get("HOST", "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")
    default_port = int(os.environ.get("PORT", "8501"))
    parser = argparse.ArgumentParser(description="SmartCell Phase 7 Interactive Demonstration Web Application")
    parser.add_argument("--host", type=str, default=default_host, help=f"Host interface to bind (default: {default_host})")
    parser.add_argument("--port", type=int, default=default_port, help=f"Port to serve the web application (default: {default_port})")
    args = parser.parse_args()

    os.makedirs(DEMO_RUNS_ROOT, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    print("=" * 78)
    print("SMARTCELL PHASE 7: INTERACTIVE LOCAL DEMONSTRATION APPLICATION")
    print("=" * 78)
    print("DISCLAIMER: Experimental academic research prototype. Not for clinical use.")
    status = query_environment_status(force_refresh=True)
    g = status["gpu_environment"]
    print(f"Stage 1 GPU Env:  {g['gpu_name']} (CUDA Available: {g['cuda_available']}, PyTorch: {g['pytorch_version']})")
    print(f"Stage 2 CNN Env:  {VENV_TF_PYTHON}")
    print(f"Output Root Dir:  {DEMO_RUNS_ROOT}")
    print(f"Web UI URL:       http://{args.host}:{args.port}")
    print("=" * 78)

    server = ThreadingHTTPServer((args.host, args.port), SmartCellRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down SmartCell demonstration server.")
        server.server_close()


if __name__ == "__main__":
    main()
