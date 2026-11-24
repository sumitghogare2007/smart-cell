r"""
render_app.py
-------------
SmartCell Cloud AI Inference Backend for Render Free Tier.
Provides high-performance, memory-optimized REST API endpoints for:
  - Mode 1: Malaria Blood Smear Screening (YOLOv8n RBC Localization + Custom/Fine-Tuned CNN)
  - Mode 2: ALL Leukemia Screening (ALL-IDB Single-Cell EfficientNetB0)

Key Architecture & Optimization Features for Render Free Tier (512 MB RAM ceiling):
1. Subprocess Isolation: Heavy deep-learning frameworks (PyTorch, Ultralytics, TensorFlow)
   are executed within dedicated, sequential worker subprocesses. The main web server
   process remains lightweight (< 35 MB RSS), preventing Out-Of-Memory (OOM) kills.
2. Full CORS Support: Supports cross-origin requests from Hugging Face Static Spaces
   (https://sumit0123-smart-cell.hf.space, https://*.hf.space) and local testing.
3. Preflight OPTIONS: Immediate 204 No-Content handling for CORS preflights.
4. Absolute Artifact URLs: Automatically resolves download and preview URLs against
   RENDER_EXTERNAL_URL or request Host/Proto headers for seamless static-frontend integration.
5. Strict Validation: Path traversal protection, payload limits (35 MB), and image sanity checks.
6. Real Inference Only: Zero mock predictions, zero simulated outputs.

Usage:
  python render_app.py --host 0.0.0.0 --port 8501
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

# Setup base paths
DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.environ.get("SMARTCELL_ROOT", os.path.abspath(os.path.join(DEPLOY_DIR, "..", "..")))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

PIPELINE_SCRIPT = os.path.join(SCRIPTS_DIR, "run_smartcell_pipeline.py")
LEUKEMIA_WORKER_SCRIPT = os.path.join(DEPLOY_DIR, "leukemia_worker.py")
DEMO_RUNS_ROOT = os.path.join(PROJECT_ROOT, "results", "demo_app_runs")
UPLOADS_DIR = os.path.join(DEMO_RUNS_ROOT, "_uploads")

TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "yolo_malaria", "images", "test")
TEST_LABELS_DIR = os.path.join(PROJECT_ROOT, "yolo_malaria", "labels", "test")
POLYGON_SET_DIR = os.path.join(PROJECT_ROOT, "Polygon Set")

SINGLE_CELL_PRESET_DIRS = [
    os.path.join(PROJECT_ROOT, "Testing_images", "cell_images", "Parasitized"),
    os.path.join(PROJECT_ROOT, "Testing_images", "cell_images", "Uninfected"),
    os.path.join(PROJECT_ROOT, "dataset", "test", "Parasitized"),
    os.path.join(PROJECT_ROOT, "dataset", "test", "Uninfected"),
]

LEUKEMIA_PRESET_DIRS = [
    os.path.join(PROJECT_ROOT, "dataset_all", "test", "0_normal"),
    os.path.join(PROJECT_ROOT, "dataset_all", "test", "1_blast"),
    os.path.join(PROJECT_ROOT, "dataset_all", "val", "0_normal"),
    os.path.join(PROJECT_ROOT, "dataset_all", "val", "1_blast"),
]

MODELS_MAP = {
    "finetuned_raw": {
        "label": "Fine-Tuned Raw CNN (Phase 6 Domain-Adapted — Recommended)",
        "path": os.path.join(PROJECT_ROOT, "models", "malaria_cnn_finetuned_raw_best.keras"),
        "recommended_crop_mode": "raw",
        "description": "Fine-tuned on 24,112 native rectangular full-smear crops (Test Acc: 98.35%, Par F1: 0.9016)."
    },
    "finetuned_masked": {
        "label": "Fine-Tuned Masked CNN (Phase 6 Elliptical-Masked)",
        "path": os.path.join(PROJECT_ROOT, "models", "malaria_cnn_finetuned_masked_best.keras"),
        "recommended_crop_mode": "masked",
        "description": "Validated on elliptical black-corner masked full-smear crops (Test Acc: 98.10%, Par F1: 0.8905)."
    },
    "original": {
        "label": "Original Baseline CNN (Phase 1 Single-Cell Dataset)",
        "path": os.path.join(PROJECT_ROOT, "models", "malaria_cnn_best.keras"),
        "recommended_crop_mode": "masked",
        "description": "Trained on 27,558 black-background NIH single-cell crops (95.04% single-cell Acc; sensitive to raw corners)."
    }
}

YOLO_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "results", "yolo_malaria_run", "weights", "best.pt")
LEUKEMIA_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "all_idb_leukemia_best.keras")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_UPLOAD_BYTES = 35 * 1024 * 1024  # 35 MB max upload
MIN_IMAGE_DIMENSION = 32
MAX_IMAGE_DIMENSION = 10000

_CACHED_ENV_STATUS = None


def get_python_exe():
    """Returns sys.executable or falls back to standard python on path."""
    return sys.executable if sys.executable else "python"


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
        lbl_path = os.path.join(PROJECT_ROOT, "yolo_malaria", "labels", split, f"{stem}.txt")
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
                "source_folder": os.path.relpath(d, PROJECT_ROOT),
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
                    "source_folder": os.path.relpath(d, PROJECT_ROOT),
                })
                count_for_dir += 1
                if count_for_dir >= 4:
                    break
    return samples


def resolve_preset_image_path(preset_filename):
    """Resolves a preset filename from full-smear test images, single-cell test dirs, or leukemia dirs."""
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
    """Queries hardware telemetry, model file availability, and test presets."""
    global _CACHED_ENV_STATUS
    if _CACHED_ENV_STATUS is not None and not force_refresh:
        return _CACHED_ENV_STATUS

    gpu_info = {
        "cuda_available": False,
        "gpu_name": "Cloud CPU Environment (Render Free Tier)",
        "pytorch_version": "CPU",
        "free_vram_mb": 0.0,
        "total_vram_mb": 0.0,
        "yolo_checkpoint_exists": os.path.exists(YOLO_WEIGHTS_PATH),
        "yolo_checkpoint_path": YOLO_WEIGHTS_PATH
    }

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

    leukemia_status = {
        "architecture": "EfficientNetB0",
        "path": LEUKEMIA_MODEL_PATH,
        "exists": os.path.exists(LEUKEMIA_MODEL_PATH),
        "target_diseases": ["Acute Lymphoblastic Leukemia (ALL)"]
    }

    _CACHED_ENV_STATUS = {
        "timestamp": datetime.now().isoformat(),
        "deployment_tier": "Render Free Web Service (512 MB RAM)",
        "gpu_environment": gpu_info,
        "cnn_models": models_status,
        "leukemia_model": leukemia_status,
        "preset_test_images": get_preset_test_images(),
        "preset_single_cell_images": get_preset_single_cell_images(),
    }
    return _CACHED_ENV_STATUS


def sanitize_upload_filename(raw_filename, is_live_capture=False):
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
    import numpy as np
    w, h = rgb_pil_image.size
    input_mode = str(input_mode or "full_smear").strip().lower()
    gray = np.asarray(rgb_pil_image.convert("L"), dtype=np.float32)
    mean_lum = round(float(gray.mean()), 2)
    std_lum = round(float(gray.std()), 2)

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
            f"(mean luminance={mean_lum}/255, std={std_lum}). Check illumination and sensor."
        )
    elif is_blank:
        warnings.append(
            f"Quality Warning (Blank/Uniform Frame): Captured/uploaded image has virtually zero contrast "
            f"(mean luminance={mean_lum}/255, std={std_lum}) and may not contain blood-smear cells."
        )

    if is_blurred:
        warnings.append(
            f"Quality Warning (Potential Blur / Out-of-Focus): Frame has a low focus sharpness score "
            f"(Laplacian variance={focus_var} < 18.0). Check fine-focus adjustment."
        )

    if input_mode == "full_smear":
        if looks_like_single_cell:
            warnings.append(
                f"Mode Advisory: Image dimensions ({w}x{h} px) resemble a single-cell crop rather than a full blood smear. "
                f"Switch to Single-Cell Mode if analyzing an isolated RBC crop."
            )
        elif is_low_res:
            warnings.append(
                f"Resolution Advisory: Image dimensions ({w}x{h} px) are below recommended 640x480 full-smear resolution."
            )

    if capture_source == "live_microscope":
        warnings.append(
            "Microscope Domain Notice: Prototype performance depends on 100x oil-immersion optics, Giemsa stain quality, and illumination."
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
        "clinical_suitability_disclaimer": "Heuristic quality checks do NOT certify clinical specimen suitability."
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
                return False, f"Image dimensions ({w}x{h} px) are too small (minimum {min_dim}x{min_dim} px).", None
            if w > max_dim or h > max_dim:
                return False, f"Image dimensions ({w}x{h} px) exceed maximum supported ({max_dim}x{max_dim} px).", None
            quality_info = assess_image_quality(rgb, capture_source=capture_source, input_mode=input_mode)
            dim_payload = {"width": w, "height": h, "byte_size": fsize, "quality": quality_info}
            if reject_blank_or_black:
                if quality_info.get("is_black_frame"):
                    return False, "Black image rejected: frame is nearly completely black.", dim_payload
                if quality_info.get("is_blank_frame"):
                    return False, "Blank image rejected: frame has zero contrast and no cellular content.", dim_payload
            return True, "Valid image", dim_payload
    except Exception as e:
        return False, f"Corrupted or invalid image file: {e}", None


def parse_multipart_form_data(headers, body_bytes):
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


def execute_pipeline_run(
    image_path,
    cnn_variant="finetuned_raw",
    crop_mode="raw",
    input_mode="full_smear",
    yolo_device="cpu",
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
    base_url="",
):
    os.makedirs(DEMO_RUNS_ROOT, exist_ok=True)
    input_mode = str(input_mode or "full_smear").strip().lower()
    if input_mode not in ("full_smear", "single_cell"):
        return {
            "success": False,
            "error_type": "InvalidParameterError",
            "error_message": f"Unknown input_mode '{input_mode}'. Valid: full_smear, single_cell",
        }

    is_valid, err_msg, dim_info = validate_image_file(
        image_path,
        capture_source=capture_source,
        input_mode=input_mode,
        reject_blank_or_black=True,
    )
    if not is_valid:
        return {"success": False, "error_type": "InvalidImageError", "error_message": err_msg}

    if cnn_variant not in MODELS_MAP:
        return {
            "success": False,
            "error_type": "InvalidParameterError",
            "error_message": f"Unknown CNN variant '{cnn_variant}'. Valid: {list(MODELS_MAP.keys())}"
        }

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

    original_copy_name = f"original_{stem}{orig_ext.lower()}"
    original_copy_path = os.path.join(run_dir, original_copy_name)
    try:
        shutil.copy2(image_path, original_copy_path)
    except Exception:
        original_copy_path = None

    py_exe = get_python_exe()
    cmd = [
        py_exe,
        "-u",
        PIPELINE_SCRIPT,
        "--image", os.path.abspath(image_path),
        "--output-dir", os.path.abspath(run_dir),
        "--input-mode", input_mode,
        "--cnn-variant", cnn_variant,
        "--crop-mode", crop_mode,
        "--yolo-device", "cpu",
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
            "stdout": proc.stdout[-1500:] if proc.stdout else "",
            "stderr": proc.stderr[-1500:] if proc.stderr else ""
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

    quality_info = (dim_info or {}).get("quality") or {}
    summary_data["input_mode"] = input_mode
    summary_data["capture_source"] = capture_source
    summary_data["image_quality"] = quality_info
    summary_data["quality_warnings"] = quality_info.get("warnings", [])

    gt_ref = inspect_ground_truth_for_stem(stem) if input_mode == "full_smear" else {"has_ground_truth": False}

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
        "disclaimer": "This AI screening result does not confirm or rule out malaria in a patient. Clinical assessment and appropriate diagnostic testing are required.",
    }
    try:
        import pdf_report_generator
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
            pdf_status = {
                "generated": True,
                "filename": pdf_filename,
                "path": generated_pdf,
                "size_bytes": pdf_size,
                "message": f"Complete PDF report ({pdf_filename}, {round(pdf_size / 1024.0, 1)} KB) ready.",
                "error": None
            }
            summary_data.setdefault("output_files", {})["pdf_report"] = generated_pdf
        else:
            pdf_status["error"] = "PDF generator returned an empty or missing file."
    except Exception as pdf_exc:
        pdf_status["error"] = f"PDF generation failed ({type(pdf_exc).__name__}: {pdf_exc})"

    try:
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
    except Exception:
        pass

    # Build absolute URLs using base_url
    prefix = f"{base_url}" if base_url else ""
    return {
        "success": True,
        "run_id": run_id,
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
            "export_pdf": f"{prefix}/api/file?run_id={run_id}&file={pdf_filename}&download=1" if pdf_status["generated"] else None,
            "original_image": f"{prefix}/api/file?run_id={run_id}&file={original_copy_name}" if (original_copy_path and os.path.exists(original_copy_path)) else None,
            "annotated_image": f"{prefix}/api/file?run_id={run_id}&file=annotated_{stem}.jpg" if os.path.exists(annotated_jpg_path) else None,
            "contact_sheet": f"{prefix}/api/file?run_id={run_id}&file=contact_sheet_{stem}.jpg" if os.path.exists(contact_sheet_path) else None,
            "export_csv": f"{prefix}/api/file?run_id={run_id}&file=cell_predictions_{stem}.csv&download=1",
            "export_summary_json": f"{prefix}/api/file?run_id={run_id}&file=field_summary_{stem}.json&download=1",
            "export_rejected_json": f"{prefix}/api/file?run_id={run_id}&file=rejected_and_excluded_{stem}.json&download=1",
            "export_txt": f"{prefix}/api/file?run_id={run_id}&file=field_summary_{stem}.txt&download=1"
        }
    }


class RenderRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stdout.write(f"[Render SmartCell] {self.address_string()} - {fmt % args}\n")

    def _add_cors_headers(self):
        origin = self.headers.get("Origin", "*")
        self.send_header("Access-Control-Allow-Origin", origin if origin else "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With, Accept, Origin")
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Vary", "Origin")

    def _get_base_url(self):
        ext_url = os.environ.get("RENDER_EXTERNAL_URL") or os.environ.get("PUBLIC_BACKEND_URL")
        if ext_url:
            return ext_url.rstrip("/")
        host = self.headers.get("Host", "localhost:8501")
        proto = self.headers.get("X-Forwarded-Proto", "https" if "render.com" in host else "http")
        return f"{proto}://{host}".rstrip("/")

    def _send_json(self, status_code, payload):
        raw = json.dumps(payload, indent=2).encode("utf-8")
        try:
            self.send_response(status_code)
            self._add_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
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

        # Root Status / Welcome Dashboard
        if path in ("/", "/index.html"):
            base_url = self._get_base_url()
            html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SmartCell AI Inference Backend — Render Web Service</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #0b1120; color: #f1f5f9; padding: 40px 20px; line-height: 1.6; margin: 0; }}
    .container {{ max-width: 720px; margin: 0 auto; background: #1e293b; padding: 32px; border-radius: 12px; border: 1px solid #334155; box-shadow: 0 10px 25px rgba(0,0,0,0.4); }}
    h1 {{ font-size: 1.6rem; color: #38bdf8; margin-top: 0; display: flex; align-items: center; gap: 10px; }}
    .badge {{ display: inline-block; padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; background: #065f46; color: #6ee7b7; text-transform: uppercase; }}
    .info-card {{ background: #0f172a; padding: 16px; border-radius: 8px; margin: 20px 0; border: 1px solid #1e293b; font-family: monospace; font-size: 0.88rem; }}
    a {{ color: #38bdf8; text-decoration: none; font-weight: 600; }}
    a:hover {{ text-decoration: underline; }}
    ul {{ padding-left: 20px; }}
    li {{ margin-bottom: 8px; }}
    .btn {{ display: inline-block; background: #0284c7; color: white; padding: 10px 18px; border-radius: 6px; font-weight: 600; text-decoration: none; margin-top: 10px; }}
    .btn:hover {{ background: #0369a1; text-decoration: none; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>SmartCell AI Inference Engine <span class="badge">Online</span></h1>
    <p>This is the production AI inference backend for the SmartCell automated blood-cell screening platform on Render Free Tier.</p>
    
    <div class="info-card">
      <div>Backend URL: <strong>{base_url}</strong></div>
      <div>Platform: Render Web Service (Free Tier, 512 MB RAM Ceiling)</div>
      <div>Architecture: Subprocess Memory-Isolated CPU Inference</div>
      <div>Modes: Mode 1 (Malaria YOLOv8+CNN) &amp; Mode 2 (ALL Leukemia)</div>
    </div>

    <h3>Active Frontend</h3>
    <p>Access the web application interface hosted on Hugging Face Spaces:</p>
    <a href="https://huggingface.co/spaces/Sumit0123/smart-cell" target="_blank" class="btn">🚀 Open SmartCell Frontend Space</a>

    <h3 style="margin-top:28px;">API Endpoints</h3>
    <ul>
      <li><a href="/api/health"><code>GET /api/health</code></a> — Health status and diagnostic verification</li>
      <li><a href="/api/status"><code>GET /api/status</code></a> — Model availability &amp; hardware telemetry</li>
      <li><code>POST /api/analyze</code> — Multipart form-data inference (Malaria &amp; ALL Leukemia)</li>
      <li><code>GET /api/file?run_id=...&amp;file=...</code> — Download PDF reports, CSVs, and inspect images</li>
    </ul>
  </div>
</body>
</html>"""
            raw = html.encode("utf-8")
            self.send_response(200)
            self._add_cors_headers()
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)
            return

        # Health endpoint
        if path in ("/api/health", "/health", "/healthz"):
            self._send_json(200, {
                "status": "healthy",
                "service": "SmartCell Research Platform",
                "environment": "render_free_tier",
                "timestamp": datetime.now().isoformat(),
                "modes": ["malaria_screening", "all_leukemia"]
            })
            return

        # Status / Telemetry endpoint
        if path in ("/api/status", "/api/models/status"):
            refresh = qs.get("refresh", ["0"])[0] == "1"
            status = query_environment_status(force_refresh=refresh)
            self._send_json(200, status)
            return

        # Artifact File Download endpoint
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
                    "error_message": "Uploaded file exceeds maximum allowed limit (35.0 MB)."
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
                "error_message": "No image file uploaded or preset test cell selected for leukemia analysis."
            })
            return

        # Execute leukemia analysis via worker subprocess to protect 512 MB RAM ceiling
        py_exe = get_python_exe()
        worker_json_out = os.path.join(run_dir, "leukemia_worker_result.json")
        cmd = [
            py_exe,
            "-u",
            LEUKEMIA_WORKER_SCRIPT,
            "--image", os.path.abspath(image_path),
            "--run-dir", os.path.abspath(run_dir),
            "--capture-source", capture_source_label,
            "--output-json", os.path.abspath(worker_json_out)
        ]
        if fields.get("patient_sample_id"):
            cmd.extend(["--patient-sample-id", str(fields["patient_sample_id"])])

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if proc.returncode != 0:
                self._send_json(400, {
                    "success": False,
                    "error_type": "LeukemiaInferenceError",
                    "error_message": f"Leukemia worker failed with code {proc.returncode}.",
                    "stderr": proc.stderr[-1000:] if proc.stderr else ""
                })
                return

            if os.path.exists(worker_json_out):
                with open(worker_json_out, "r", encoding="utf-8") as f:
                    result = json.load(f)
            else:
                self._send_json(500, {
                    "success": False,
                    "error_type": "MissingOutputError",
                    "error_message": "Leukemia worker exited without generating result payload."
                })
                return

            base_url = self._get_base_url()
            result["download_urls"] = {
                "pdf": f"{base_url}/api/file?run_id={run_id}&file={result['files']['pdf']}&download=1",
                "json": f"{base_url}/api/file?run_id={run_id}&file={result['files']['json']}&download=1",
                "txt": f"{base_url}/api/file?run_id={run_id}&file={result['files']['txt']}&download=1",
                "image": f"{base_url}/api/file?run_id={run_id}&file={result['files']['image']}&download=0"
            }
            self._send_json(200, result)

        except subprocess.TimeoutExpired:
            self._send_json(400, {
                "success": False,
                "error_type": "TimeoutError",
                "error_message": "Leukemia analysis exceeded maximum time limit (120s)."
            })
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

            if content_len > MAX_UPLOAD_BYTES:
                self._send_json(400, {
                    "success": False,
                    "error_type": "InvalidImageError",
                    "error_message": f"Uploaded payload exceeds maximum allowed limit (35.0 MB)."
                })
                return

            body_bytes = self.rfile.read(content_len)
            fields, files = parse_multipart_form_data(self.headers, body_bytes)

            # Disease routing: ALL Leukemia vs. Malaria
            if fields.get("disease_mode") == "leukemia":
                self.handle_leukemia_request(fields, files)
                return

            # Malaria Pipeline execution
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

            base_url = self._get_base_url()
            result = execute_pipeline_run(
                image_path=image_path,
                cnn_variant=fields.get("cnn_variant", "finetuned_raw"),
                crop_mode=fields.get("crop_mode", "raw"),
                input_mode=fields.get("input_mode", "full_smear"),
                yolo_device="cpu",
                yolo_conf=fields.get("yolo_conf", 0.25),
                yolo_iou=fields.get("yolo_iou", 0.45),
                dedup_iou=fields.get("dedup_iou", 0.50),
                cnn_thresh=fields.get("cnn_thresh", 0.50),
                pad_px=fields.get("pad_px", 4),
                min_box_size=fields.get("min_box_size", 16),
                test_zero_detections=(fields.get("test_zero_detections") == "1"),
                inject_test_invalid_boxes=(fields.get("inject_test_invalid_boxes") == "1"),
                capture_source=capture_source,
                base_url=base_url
            )
            status_code = 200 if result.get("success") else 400
            self._send_json(status_code, result)

        except Exception as e:
            sys.stderr.write(f"[Render SmartCell Error] {traceback.format_exc()}\n")
            self._send_json(500, {
                "success": False,
                "error_type": type(e).__name__,
                "error_message": "An internal server error occurred while processing the request."
            })


def main():
    default_host = os.environ.get("HOST", "0.0.0.0")
    default_port = int(os.environ.get("PORT", "8501"))
    parser = argparse.ArgumentParser(description="SmartCell Render Free AI Inference Backend Server")
    parser.add_argument("--host", type=str, default=default_host, help=f"Host interface (default: {default_host})")
    parser.add_argument("--port", type=int, default=default_port, help=f"Port (default: {default_port})")
    args = parser.parse_args()

    os.makedirs(DEMO_RUNS_ROOT, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    print("=" * 78)
    print("SMARTCELL AI INFERENCE ENGINE — RENDER FREE CLOUD BACKEND")
    print("=" * 78)
    print("Academic research prototype. Verified Stage 1 YOLOv8n + Stage 2 CNN & ALL-IDB EfficientNetB0.")
    print(f"Server Host:      {args.host}")
    print(f"Server Port:      {args.port}")
    print(f"Project Root:     {PROJECT_ROOT}")
    print(f"Runs Directory:   {DEMO_RUNS_ROOT}")
    print(f"Python Runtime:   {get_python_exe()}")
    print("Subprocess Memory Isolation Active (Safe for 512 MB RAM Free Tier)")
    print("=" * 78)

    server = ThreadingHTTPServer((args.host, args.port), RenderRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down SmartCell Render server.")
        server.server_close()


if __name__ == "__main__":
    main()
