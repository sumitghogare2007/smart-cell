r"""
run_smartcell_pipeline.py
-------------------------
SmartCell Phase 5: Experimental Two-Stage Blood-Cell Analysis Pipeline
(Stage 1: YOLOv8n Candidate Cell Localization -> Stage 2: Custom CNN Classification)

DISCLAIMER:
SmartCell is an experimental research prototype. Outputs represent experimental
AI screening estimates and must NEVER be interpreted as a confirmed clinical diagnosis.

Architecture & Environment Bridge:
- Stage 1 (YOLOv8n): Runs in D:\smartcell\venv_yolo (Python 3.13, Ultralytics 8.4.160, PyTorch 2.14.0+cpu)
  using checkpoint D:\smartcell\results\yolo_malaria_run\weights\best.pt.
- Stage 2 (CNN): Runs in D:\smartcell\venv (Python 3.11, TensorFlow 2.21.0, Keras 3.15.1)
  using checkpoint D:\smartcell\models\malaria_cnn_best.keras.
- This script automatically dispatches Stage 1 and Stage 2 to their respective isolated
  virtual environments via JSON/file intermediate artifacts, so it can be invoked from
  either environment without modifying dependencies.

Usage:
  python scripts/run_smartcell_pipeline.py <image_path> [output_dir]
  python scripts/run_smartcell_pipeline.py --image <image_path> --output-dir <output_dir>
"""

import os
import sys
import csv
import json
import time
import math
import argparse
import subprocess
import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Paths & Default Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.environ.get("SMARTCELL_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
VENV_TF_PYTHON = os.path.join(BASE_DIR, "venv", "Scripts", "python.exe")
VENV_YOLO_GPU_PYTHON = os.path.join(BASE_DIR, "venv_yolo_gpu", "Scripts", "python.exe")
VENV_YOLO_CPU_PYTHON = os.path.join(BASE_DIR, "venv_yolo", "Scripts", "python.exe")


def get_python_exe(candidate_path):
    """Returns candidate_path if it exists; otherwise falls back to sys.executable."""
    if candidate_path and os.path.exists(candidate_path):
        return candidate_path
    return sys.executable


VENV_YOLO_PYTHON = VENV_YOLO_GPU_PYTHON if os.path.exists(VENV_YOLO_GPU_PYTHON) else (
    VENV_YOLO_CPU_PYTHON if os.path.exists(VENV_YOLO_CPU_PYTHON) else sys.executable
)

DEFAULT_YOLO_WEIGHTS = os.path.join(BASE_DIR, "results", "yolo_malaria_run", "weights", "best.pt")
DEFAULT_CNN_MODEL = os.path.join(BASE_DIR, "models", "malaria_cnn_best.keras")
FINETUNED_RAW_CNN_MODEL = os.path.join(BASE_DIR, "models", "malaria_cnn_finetuned_raw_best.keras")
FINETUNED_MASKED_CNN_MODEL = os.path.join(BASE_DIR, "models", "malaria_cnn_finetuned_masked_best.keras")

CNN_VARIANT_PATHS = {
    "original": DEFAULT_CNN_MODEL,
    "finetuned_raw": FINETUNED_RAW_CNN_MODEL,
    "finetuned_masked": FINETUNED_MASKED_CNN_MODEL
}

DEFAULT_CLASS_NAMES_JSON = os.path.join(BASE_DIR, "models", "malaria_class_names.json")
DEFAULT_PIPELINE_OUT_ROOT = os.path.join(BASE_DIR, "results", "smartcell_pipeline")

YOLO_CLASS_MAP = {
    0: "Uninfected RBC",
    1: "Parasitized RBC",
    2: "WBC"
}

CNN_CLASS_MAP = {
    0: "Uninfected",
    1: "Parasitized"
}


def compute_iou(box_a, box_b):
    """Computes Intersection over Union (IoU) between two boxes [x1, y1, x2, y2]."""
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    inter_area = float((x_right - x_left) * (y_bottom - y_top))
    area_a = float((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = float((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def apply_elliptical_mask(crop_rgb_np):
    """
    Zeroes out pixels outside the inscribed ellipse of a rectangular cell crop
    to match the black [0,0,0] background convention of the NIH single-cell dataset.
    Used for domain-mismatch comparison auditing.
    Guards against memory exhaustion if supplied with large whole-smear images.
    """
    ch, cw = crop_rgb_np.shape[:2]
    if ch > 512 or cw > 512:
        from PIL import Image as _PILImg
        pil_tmp = _PILImg.fromarray(crop_rgb_np)
        pil_tmp.thumbnail((512, 512), _PILImg.Resampling.BILINEAR)
        crop_rgb_np = np.array(pil_tmp)
        ch, cw = crop_rgb_np.shape[:2]

    yy, xx = np.ogrid[:ch, :cw]
    cy, cx = ch / 2.0, cw / 2.0
    ry, rx = ch / 2.0, cw / 2.0
    mask = ((xx - cx) ** 2 / (rx ** 2 + 1e-6) + (yy - cy) ** 2 / (ry ** 2 + 1e-6)) <= 1.0
    masked = crop_rgb_np.copy()
    masked[~mask] = 0
    return masked


# ---------------------------------------------------------------------------
# STAGE 1: YOLOv8n Candidate Cell Detection & Native-Resolution Cropping
# ---------------------------------------------------------------------------
def run_stage1_yolo(args):
    """
    Executes in venv_yolo_gpu (or venv_yolo fallback):
    1. Loads YOLOv8n checkpoint (best.pt) and checks CUDA/VRAM on RTX 2050 GPU.
    2. Runs inference on the full-resolution image (or simulates 0 detections if --test-zero-detections).
    3. Clips boxes, validates dimensions, separates WBCs, and deduplicates overlapping RBC boxes.
    4. Crops valid candidate RBCs from the native-resolution RGB image with modest padding.
    5. Saves crops and stage1_detections.json.
    """
    import torch
    from ultralytics import YOLO

    t0 = time.time()
    image_path = os.path.abspath(args.image)
    output_dir = os.path.abspath(args.output_dir)
    crops_dir = os.path.join(output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input microscope image not found: {image_path}")
    if not os.path.exists(args.yolo_weights):
        raise FileNotFoundError(f"YOLO checkpoint not found: {args.yolo_weights}")

    # Load original full-resolution RGB image via PIL (never modified on disk)
    with Image.open(image_path) as pil_img:
        orig_rgb = pil_img.convert("RGB")
        img_w, img_h = orig_rgb.size
        orig_np = np.array(orig_rgb)

    stem = os.path.splitext(os.path.basename(image_path))[0]

    # Resolve GPU vs CPU device safely with graceful fallback
    req_dev = str(getattr(args, "yolo_device", "auto")).lower()
    gpu_fallback_triggered = False
    gpu_fallback_reason = None
    if req_dev in ("auto", "0", "cuda", "gpu"):
        if torch.cuda.is_available():
            yolo_dev = 0
            dev_name = f"{torch.cuda.get_device_name(0)} (CUDA device=0, PyTorch {torch.__version__})"
            torch.cuda.reset_peak_memory_stats(0)
        else:
            yolo_dev = "cpu"
            gpu_fallback_triggered = True
            gpu_fallback_reason = "CUDA requested but torch.cuda.is_available() returned False; fell back to CPU."
            dev_name = f"CPU Fallback (PyTorch {torch.__version__} - CUDA unavailable)"
    else:
        yolo_dev = "cpu"
        dev_name = f"CPU (PyTorch {torch.__version__})"

    # Load YOLO model
    model = YOLO(args.yolo_weights)

    raw_detections = []
    if getattr(args, "test_zero_detections", False):
        # Synthetic zero-detection test mode
        pass
    else:
        try:
            results = model.predict(
                source=image_path,
                imgsz=args.yolo_imgsz,
                conf=args.yolo_conf,
                iou=args.yolo_iou,
                max_det=args.max_det,
                rect=True,
                device=yolo_dev,
                verbose=False
            )[0]
        except Exception as gpu_err:
            if yolo_dev != "cpu":
                yolo_dev = "cpu"
                gpu_fallback_triggered = True
                gpu_fallback_reason = f"GPU inference error ({type(gpu_err).__name__}: {gpu_err}); fell back to CPU."
                dev_name = f"CPU Fallback (PyTorch {torch.__version__})"
                results = model.predict(
                    source=image_path,
                    imgsz=args.yolo_imgsz,
                    conf=args.yolo_conf,
                    iou=args.yolo_iou,
                    max_det=args.max_det,
                    rect=True,
                    device="cpu",
                    verbose=False
                )[0]
            else:
                raise

        if results.boxes is not None and len(results.boxes) > 0:
            xyxy = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)
            for idx, ((x1, y1, x2, y2), cf, cl) in enumerate(zip(xyxy, confs, clss)):
                raw_detections.append({
                    "raw_index": idx,
                    "yolo_class_id": int(cl),
                    "yolo_class_name": YOLO_CLASS_MAP.get(int(cl), f"Unknown_{cl}"),
                    "yolo_confidence": round(float(cf), 6),
                    "raw_bbox": [float(x1), float(y1), float(x2), float(y2)]
                })

    # Optional synthetic invalid box injection for unit testing
    if getattr(args, "inject_test_invalid_boxes", False):
        raw_detections.append({
            "raw_index": 999001,
            "yolo_class_id": 0,
            "yolo_class_name": "Uninfected RBC",
            "yolo_confidence": 0.99,
            "raw_bbox": [100.0, 100.0, 105.0, 104.0]  # Tiny 5x4 invalid box (< min_box_size)
        })
        raw_detections.append({
            "raw_index": 999002,
            "yolo_class_id": 0,
            "yolo_class_name": "Uninfected RBC",
            "yolo_confidence": 0.98,
            "raw_bbox": [-50.0, -50.0, -10.0, -10.0]  # Completely outside image bounds
        })
        raw_detections.append({
            "raw_index": 999003,
            "yolo_class_id": 2,
            "yolo_class_name": "WBC",
            "yolo_confidence": 0.85,
            "raw_bbox": [500.0, 500.0, 720.0, 720.0]  # Excluded WBC detection
        })

    excluded_wbcs = []
    rejected_boxes = []
    valid_rbc_candidates = []

    for det in raw_detections:
        rx1, ry1, rx2, ry2 = det["raw_bbox"]
        # Clip to image boundaries
        cx1 = max(0, min(img_w, int(round(rx1))))
        cy1 = max(0, min(img_h, int(round(ry1))))
        cx2 = max(0, min(img_w, int(round(rx2))))
        cy2 = max(0, min(img_h, int(round(ry2))))

        bw = cx2 - cx1
        bh = cy2 - cy1

        # Reject invalid / degenerate / tiny boxes
        if bw < args.min_box_size or bh < args.min_box_size:
            rejected_boxes.append({
                "reason": f"Invalid or too-small bounding box after clipping ({bw}x{bh} px < {args.min_box_size} px)",
                "detection": det,
                "clipped_bbox": [cx1, cy1, cx2, cy2]
            })
            continue

        # Exclude WBC detections (class 2) from malaria RBC classification
        if det["yolo_class_id"] == 2:
            excluded_wbcs.append({
                "reason": "WBC detection excluded from RBC malaria classification stage",
                "yolo_class_id": 2,
                "yolo_class_name": "WBC",
                "yolo_confidence": det["yolo_confidence"],
                "clipped_bbox": [cx1, cy1, cx2, cy2]
            })
            continue

        if det["yolo_class_id"] not in (0, 1):
            rejected_boxes.append({
                "reason": f"Unsupported class ID {det['yolo_class_id']}",
                "detection": det,
                "clipped_bbox": [cx1, cy1, cx2, cy2]
            })
            continue

        valid_rbc_candidates.append({
            "raw_index": det["raw_index"],
            "yolo_class_id": det["yolo_class_id"],
            "yolo_class_name": det["yolo_class_name"],
            "yolo_confidence": det["yolo_confidence"],
            "bbox": [cx1, cy1, cx2, cy2],
            "bbox_width": bw,
            "bbox_height": bh
        })

    # Class-agnostic deduplication across candidate RBCs (sorted by descending YOLO confidence)
    valid_rbc_candidates.sort(key=lambda d: d["yolo_confidence"], reverse=True)
    deduplicated_rbcs = []
    for cand in valid_rbc_candidates:
        is_duplicate = False
        for kept in deduplicated_rbcs:
            iou_val = compute_iou(cand["bbox"], kept["bbox"])
            if iou_val >= args.dedup_iou:
                is_duplicate = True
                rejected_boxes.append({
                    "reason": f"Duplicate overlapping detection suppressed (IoU={iou_val:.3f} >= {args.dedup_iou})",
                    "suppressed_detection": cand,
                    "kept_detection_raw_index": kept["raw_index"]
                })
                break
        if not is_duplicate:
            deduplicated_rbcs.append(cand)

    # Sort deduplicated RBCs spatially (top-to-bottom, left-to-right) for deterministic IDs
    deduplicated_rbcs.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))

    # Crop candidate RBCs from full-resolution RGB image with modest padding
    pad = int(args.pad_px)
    for idx, cand in enumerate(deduplicated_rbcs, start=1):
        crop_id = f"{stem}_cell_{idx:04d}"
        x1, y1, x2, y2 = cand["bbox"]
        px1 = max(0, x1 - pad)
        py1 = max(0, y1 - pad)
        px2 = min(img_w, x2 + pad)
        py2 = min(img_h, y2 + pad)

        crop_arr = orig_np[py1:py2, px1:px2]
        crop_filename = f"{crop_id}.png"
        crop_rel_path = os.path.join("crops", crop_filename)
        crop_abs_path = os.path.join(crops_dir, crop_filename)

        Image.fromarray(crop_arr).save(crop_abs_path)

        cand["crop_id"] = crop_id
        cand["padded_bbox"] = [px1, py1, px2, py2]
        cand["crop_width"] = int(px2 - px1)
        cand["crop_height"] = int(py2 - py1)
        cand["crop_path"] = crop_abs_path
        cand["crop_rel_path"] = crop_rel_path

    stage1_elapsed = round(time.time() - t0, 4)
    peak_vram_mb = round(torch.cuda.max_memory_allocated(0) / (1024 ** 2), 2) if yolo_dev == 0 else 0.0

    stage1_payload = {
        "image_path": image_path,
        "image_filename": os.path.basename(image_path),
        "image_stem": stem,
        "image_width": img_w,
        "image_height": img_h,
        "yolo_weights": os.path.abspath(args.yolo_weights),
        "stage1_device": dev_name,
        "stage1_peak_vram_mb": peak_vram_mb,
        "gpu_fallback_triggered": gpu_fallback_triggered,
        "gpu_fallback_reason": gpu_fallback_reason,
        "yolo_conf_threshold": float(args.yolo_conf),
        "yolo_iou_threshold": float(args.yolo_iou),
        "dedup_iou_threshold": float(args.dedup_iou),
        "max_det": int(args.max_det),
        "pad_px": pad,
        "min_box_size": int(args.min_box_size),
        "stage1_processing_time_seconds": stage1_elapsed,
        "counts": {
            "raw_yolo_detections": len(raw_detections),
            "valid_candidate_rbc_crops": len(deduplicated_rbcs),
            "excluded_wbc_detections": len(excluded_wbcs),
            "rejected_or_duplicate_boxes": len(rejected_boxes)
        },
        "candidate_rbcs": deduplicated_rbcs,
        "excluded_wbcs": excluded_wbcs,
        "rejected_boxes": rejected_boxes
    }

    stage1_json_path = os.path.join(output_dir, f"stage1_detections_{stem}.json")
    with open(stage1_json_path, "w", encoding="utf-8") as f:
        json.dump(stage1_payload, f, indent=2)

    print(f"[Stage 1 Complete on {dev_name}] Raw detections: {len(raw_detections)} | "
          f"Valid RBC crops: {len(deduplicated_rbcs)} | "
          f"Excluded WBCs: {len(excluded_wbcs)} | "
          f"Rejected/Duplicates: {len(rejected_boxes)} | "
          f"Time: {stage1_elapsed:.2f}s")
    return stage1_json_path


# ---------------------------------------------------------------------------
# STAGE 2: CNN Classification, Visualization, and Field-Level Reporting
# ---------------------------------------------------------------------------
def create_contact_sheet(crops_metadata, output_path, cols=5, max_items=25):
    """Creates a clean visual contact sheet of representative candidate RBC crops."""
    if not crops_metadata:
        return None

    # Select up to max_items crops (prioritizing any Parasitized predictions + uniform sample)
    parasitized_items = [c for c in crops_metadata if c["cnn_predicted_class"] == "Parasitized"]
    uninfected_items = [c for c in crops_metadata if c["cnn_predicted_class"] == "Uninfected"]

    selected = parasitized_items[:12] + uninfected_items[:max(0, max_items - len(parasitized_items[:12]))]
    selected = selected[:max_items]

    rows = math.ceil(len(selected) / float(cols))
    tile_w, tile_h = 220, 260
    header_h = 60

    sheet_w = cols * tile_w + 20
    sheet_h = rows * tile_h + header_h + 20

    sheet = Image.new("RGB", (sheet_w, sheet_h), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)

    draw.text((20, 18),
              f"SmartCell Candidate RBC Crops Contact Sheet ({len(selected)} shown of {len(crops_metadata)} valid crops)",
              fill=(255, 255, 255))

    for idx, item in enumerate(selected):
        r = idx // cols
        c = idx % cols
        x0 = 10 + c * tile_w
        y0 = header_h + r * tile_h

        # Border color based on CNN prediction
        is_par = item["cnn_predicted_class"] == "Parasitized"
        border_color = (231, 76, 60) if is_par else (46, 204, 113)

        draw.rectangle([x0 + 4, y0 + 4, x0 + tile_w - 4, y0 + tile_h - 4], outline=border_color, width=2)

        if os.path.exists(item["crop_path"]):
            with Image.open(item["crop_path"]) as cim:
                cim_disp = cim.convert("RGB").resize((160, 160), Image.Resampling.BILINEAR)
                sheet.paste(cim_disp, (x0 + 30, y0 + 12))

        short_id = item["crop_id"].split("_cell_")[-1]
        draw.text((x0 + 12, y0 + 178), f"ID: cell_{short_id} ({item['bbox_width']}x{item['bbox_height']}px)",
                  fill=(220, 220, 220))
        draw.text((x0 + 12, y0 + 198), f"YOLO: {item['yolo_class_name'][:10]} ({item['yolo_confidence']:.2f})",
                  fill=(180, 200, 255))
        draw.text((x0 + 12, y0 + 218),
                  f"CNN:  {item['cnn_predicted_class']} ({item['cnn_confidence']*100:.1f}%)",
                  fill=border_color)
        draw.text((x0 + 12, y0 + 236),
                  f"Masked P(Par): {item['cnn_masked_parasitized_prob']*100:.1f}%",
                  fill=(200, 200, 150))

    sheet.save(output_path, quality=95)
    return output_path


def create_annotated_full_image(image_path, candidates, excluded_wbcs, summary, output_path):
    """
    Draws two-stage bounding boxes on the actual source microscope image for the current run.
    Preserves exact source aspect ratio, never upscales low-resolution images, and never
    covers small or medium-resolution images with a fixed-4K black header rectangle.
    """
    import cv2

    if not image_path or not os.path.exists(image_path):
        return None
    try:
        with Image.open(image_path) as pil_im:
            rgb_np = np.array(pil_im.convert("RGB"))
            img = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
    except Exception:
        img = cv2.imread(image_path)
        if img is None:
            return None

    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return None

    scale_factor = max(0.35, min(1.0, min(w, h) / 1200.0))
    box_thick_uninf = max(1, int(round(3 * scale_factor)))
    box_thick_par = max(2, int(round(5 * scale_factor)))
    font_scale_lbl = max(0.35, round(0.85 * scale_factor, 2))
    font_thick_lbl = max(1, int(round(3 * scale_factor)))

    # Draw Uninfected RBCs first (Green)
    for c in candidates:
        if c["cnn_predicted_class"] == "Uninfected":
            x1, y1, x2, y2 = c["bbox"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (113, 204, 46), box_thick_uninf)

    # Draw Parasitized RBCs on top (Red/Crimson)
    for c in candidates:
        if c["cnn_predicted_class"] == "Parasitized":
            x1, y1, x2, y2 = c["bbox"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (60, 60, 235), box_thick_par)
            lbl = f"Par {c['cnn_confidence']*100:.0f}%"
            cv2.putText(
                img,
                lbl,
                (x1, max(int(20 * scale_factor), y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale_lbl,
                (60, 60, 235),
                font_thick_lbl,
                cv2.LINE_AA,
            )

    # Draw Excluded WBCs (Blue/Orange)
    for wbc in excluded_wbcs:
        x1, y1, x2, y2 = wbc["clipped_bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (235, 140, 40), box_thick_par)
        cv2.putText(
            img,
            f"WBC (Excl {wbc['yolo_confidence']:.2f})",
            (x1, max(int(20 * scale_factor), y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale_lbl,
            (235, 140, 40),
            font_thick_lbl,
            cv2.LINE_AA,
        )

    # Only draw the in-frame top-left header banner on high-resolution full-smear fields
    # (>= 1600x900 px) where the banner occupies < 9% of frame height. Never draw a large
    # black banner over smaller images or single-cell crops.
    if w >= 1600 and h >= 900:
        banner_x2 = min(2450, w - 20)
        banner_y2 = min(210, int(h * 0.09))
        cv2.rectangle(img, (20, 20), (banner_x2, banner_y2), (20, 20, 20), -1)
        cv2.rectangle(img, (20, 20), (banner_x2, banner_y2), (200, 200, 200), 2)

        cnt = summary["counts"]
        valid_n = cnt.get("valid_candidate_rbc_crops", 0)
        par_est = summary.get("parasitemia_estimate") or {}
        if valid_n > 0 and par_est.get("raw_crop_parasitemia_percent") is not None:
            par_str = (
                f"Est. Parasitemia (Raw Crop): {par_est['raw_crop_parasitemia_percent']:.2f}% "
                f"(Masked Crop: {par_est['ellipse_masked_parasitemia_percent']:.2f}%)"
            )
        else:
            par_str = "Est. Parasitemia: N/A (0 valid RBC crops)"

        cv2.putText(
            img,
            f"SMARTCELL TWO-STAGE SCREENING PROTOTYPE | {summary['image_filename']} ({w}x{h})",
            (45, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.35,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"Valid RBCs: {valid_n} | "
            f"CNN Parasitized: {cnt['cnn_parasitized_rbc_count']} | "
            f"CNN Uninfected: {cnt['cnn_uninfected_rbc_count']} | "
            f"{par_str}",
            (45, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.15,
            (100, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "EXPERIMENTAL RESEARCH PROTOTYPE ONLY - NOT A CLINICAL MEDICAL DIAGNOSIS",
            (45, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (120, 180, 255),
            2,
            cv2.LINE_AA,
        )

    # Preserve exact aspect ratio and NEVER upscale low-resolution images
    if w > 1920:
        out_w = 1920
        out_h = max(1, int(round(out_w * h / float(w))))
        out_img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
    else:
        out_img = img

    cv2.imwrite(output_path, out_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return output_path


def create_annotated_single_cell_image(image_path, single_cell_result, output_path):
    """
    Creates the annotated image for Single-Cell Mode directly from the current run's
    source image, preserving exact native resolution and aspect ratio (no upscaling
    and no black overlay banner covering the cell). Draws only a thin 2px perimeter
    border indicating the predicted class (Red = Parasitized, Green = Uninfected).
    """
    import cv2

    if not image_path or not os.path.exists(image_path):
        return None
    try:
        with Image.open(image_path) as pil_im:
            rgb_np = np.array(pil_im.convert("RGB"))
            img = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
    except Exception:
        img = cv2.imread(image_path)
        if img is None:
            return None

    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return None

    is_par = (single_cell_result or {}).get("predicted_class") == "Parasitized"
    border_bgr = (60, 60, 235) if is_par else (113, 204, 46)
    border_px = max(1, min(3, int(round(min(w, h) * 0.015))))
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), border_bgr, border_px)

    if w > 1920:
        out_w = 1920
        out_h = max(1, int(round(out_w * h / float(w))))
        img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)

    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return output_path


def run_stage2_cnn(args, stage1_json_path):
    """
    Executes in venv (TensorFlow 2.21.0):
    1. Loads malaria_cnn_best.keras and malaria_class_names.json.
    2. Preprocesses each native-resolution crop to (224, 224, 3) float32 RGB.
    3. Runs CNN inference on both unmodified rectangular crops and elliptical-masked crops.
    4. Saves CSV, JSON summary, TXT report, contact sheet, and annotated image.
    """
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf

    t0 = time.time()
    output_dir = os.path.abspath(args.output_dir)

    with open(stage1_json_path, "r", encoding="utf-8") as f:
        s1 = json.load(f)

    if not os.path.exists(args.cnn_model):
        raise FileNotFoundError(f"CNN model checkpoint not found: {args.cnn_model}")

    # Verify class mapping from malaria_class_names.json
    class_map = {"0": "Uninfected", "1": "Parasitized"}
    if os.path.exists(DEFAULT_CLASS_NAMES_JSON):
        with open(DEFAULT_CLASS_NAMES_JSON, "r", encoding="utf-8") as f:
            class_map = json.load(f)

    # Load CNN model
    cnn_model = tf.keras.models.load_model(args.cnn_model)

    candidates = s1["candidate_rbcs"]
    raw_batch = []
    masked_batch = []

    for cand in candidates:
        with Image.open(cand["crop_path"]) as cim:
            rgb_im = cim.convert("RGB")
            # Exact preprocessing from original CNN training pipeline:
            # Resize to (224, 224) RGB float32 in [0, 255] (internal Rescaling(1/255) layer inside model)
            resized_raw = rgb_im.resize((224, 224))
            raw_arr = np.array(resized_raw, dtype=np.float32)
            raw_batch.append(raw_arr)

            # Domain-aligned elliptical black-masked crop for domain shift comparison
            crop_np = np.array(rgb_im)
            masked_np = apply_elliptical_mask(crop_np)
            resized_masked = Image.fromarray(masked_np).resize((224, 224))
            masked_arr = np.array(resized_masked, dtype=np.float32)
            masked_batch.append(masked_arr)

    if candidates:
        raw_tensor = np.stack(raw_batch, axis=0)
        masked_tensor = np.stack(masked_batch, axis=0)

        raw_probs = cnn_model.predict(raw_tensor, batch_size=64, verbose=0).flatten()
        masked_probs = cnn_model.predict(masked_tensor, batch_size=64, verbose=0).flatten()
    else:
        raw_probs = np.array([])
        masked_probs = np.array([])

    thresh = float(args.cnn_thresh)
    use_masked = (args.crop_mode in ("masked", "ellipse_masked"))

    par_count_primary = 0
    uninf_count_primary = 0
    par_count_raw = 0
    par_count_masked = 0

    for idx, cand in enumerate(candidates):
        p_raw = float(raw_probs[idx])
        p_mask = float(masked_probs[idx])

        cls_raw = class_map["1"] if p_raw >= thresh else class_map["0"]
        conf_raw = p_raw if p_raw >= thresh else (1.0 - p_raw)

        cls_mask = class_map["1"] if p_mask >= thresh else class_map["0"]
        conf_mask = p_mask if p_mask >= thresh else (1.0 - p_mask)

        if cls_raw == "Parasitized":
            par_count_raw += 1
        if cls_mask == "Parasitized":
            par_count_masked += 1

        # Primary prediction depends on --crop-mode (default: raw rectangular as trained, or masked/ellipse_masked)
        p_primary = p_mask if use_masked else p_raw
        cls_primary = cls_mask if use_masked else cls_raw
        conf_primary = conf_mask if use_masked else conf_raw

        if cls_primary == "Parasitized":
            par_count_primary += 1
        else:
            uninf_count_primary += 1

        cand["cnn_parasitized_prob"] = round(p_primary, 6)
        cand["cnn_uninfected_prob"] = round(1.0 - p_primary, 6)
        cand["cnn_predicted_class"] = cls_primary
        cand["cnn_confidence"] = round(conf_primary, 6)

        cand["cnn_raw_parasitized_prob"] = round(p_raw, 6)
        cand["cnn_raw_predicted_class"] = cls_raw
        cand["cnn_raw_confidence"] = round(conf_raw, 6)

        cand["cnn_masked_parasitized_prob"] = round(p_mask, 6)
        cand["cnn_masked_predicted_class"] = cls_mask
        cand["cnn_masked_confidence"] = round(conf_mask, 6)

    stage2_elapsed = round(time.time() - t0, 4)
    total_elapsed = round(s1["stage1_processing_time_seconds"] + stage2_elapsed, 4)

    total_rbcs = len(candidates)
    if total_rbcs > 0:
        primary_parasitemia = round((par_count_primary / float(total_rbcs)) * 100.0, 4)
        raw_parasitemia = round((par_count_raw / float(total_rbcs)) * 100.0, 4)
        masked_parasitemia = round((par_count_masked / float(total_rbcs)) * 100.0, 4)
        denom_status = f"{par_count_primary} / {total_rbcs} classified RBCs"
    else:
        primary_parasitemia = None
        raw_parasitemia = None
        masked_parasitemia = None
        denom_status = "No reliable result is available for this image (0 valid candidate RBCs detected)"

    stem = s1["image_stem"]

    # 1. Save CSV (one row per candidate cell)
    csv_path = os.path.join(output_dir, f"cell_predictions_{stem}.csv")
    csv_fields = [
        "crop_id", "image_filename",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "bbox_width", "bbox_height",
        "padded_x1", "padded_y1", "padded_x2", "padded_y2",
        "yolo_class_id", "yolo_class_name", "yolo_confidence",
        "cnn_predicted_class", "cnn_confidence", "cnn_parasitized_prob", "cnn_uninfected_prob",
        "cnn_raw_predicted_class", "cnn_raw_parasitized_prob",
        "cnn_masked_predicted_class", "cnn_masked_parasitized_prob",
        "crop_rel_path"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=csv_fields)
        writer.writeheader()
        for c in candidates:
            writer.writerow({
                "crop_id": c["crop_id"],
                "image_filename": s1["image_filename"],
                "bbox_x1": c["bbox"][0],
                "bbox_y1": c["bbox"][1],
                "bbox_x2": c["bbox"][2],
                "bbox_y2": c["bbox"][3],
                "bbox_width": c["bbox_width"],
                "bbox_height": c["bbox_height"],
                "padded_x1": c["padded_bbox"][0],
                "padded_y1": c["padded_bbox"][1],
                "padded_x2": c["padded_bbox"][2],
                "padded_y2": c["padded_bbox"][3],
                "yolo_class_id": c["yolo_class_id"],
                "yolo_class_name": c["yolo_class_name"],
                "yolo_confidence": c["yolo_confidence"],
                "cnn_predicted_class": c["cnn_predicted_class"],
                "cnn_confidence": c["cnn_confidence"],
                "cnn_parasitized_prob": c["cnn_parasitized_prob"],
                "cnn_uninfected_prob": c["cnn_uninfected_prob"],
                "cnn_raw_predicted_class": c["cnn_raw_predicted_class"],
                "cnn_raw_parasitized_prob": c["cnn_raw_parasitized_prob"],
                "cnn_masked_predicted_class": c["cnn_masked_predicted_class"],
                "cnn_masked_parasitized_prob": c["cnn_masked_parasitized_prob"],
                "crop_rel_path": c["crop_rel_path"]
            })

    # 2. Save Rejected & Excluded Detections Log
    rejected_json_path = os.path.join(output_dir, f"rejected_and_excluded_{stem}.json")
    with open(rejected_json_path, "w", encoding="utf-8") as rf:
        json.dump({
            "image_filename": s1["image_filename"],
            "excluded_wbc_count": len(s1["excluded_wbcs"]),
            "rejected_or_duplicate_count": len(s1["rejected_boxes"]),
            "excluded_wbcs": s1["excluded_wbcs"],
            "rejected_boxes": s1["rejected_boxes"]
        }, rf, indent=2)

    # Compute aggregate confidence statistics for valid RBC crops
    yolo_confs = [c["yolo_confidence"] for c in candidates]
    cnn_confs = [c["cnn_confidence"] for c in candidates]
    par_confs = [c["cnn_confidence"] for c in candidates if c["cnn_predicted_class"] == "Parasitized"]
    uninf_confs = [c["cnn_confidence"] for c in candidates if c["cnn_predicted_class"] == "Uninfected"]

    def _stat_dict(vals):
        if not vals:
            return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
        arr = np.asarray(vals, dtype=float)
        return {
            "count": int(len(arr)),
            "mean": round(float(np.mean(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4)
        }

    confidence_stats = {
        "yolo_detection_confidence": _stat_dict(yolo_confs),
        "cnn_overall_confidence": _stat_dict(cnn_confs),
        "cnn_parasitized_confidence": _stat_dict(par_confs),
        "cnn_uninfected_confidence": _stat_dict(uninf_confs)
    }

    # 3. Build Field-Level Summary JSON
    summary = {
        "disclaimer": "EXPERIMENTAL RESEARCH PROTOTYPE ONLY — NOT A CLINICALLY VALIDATED DIAGNOSTIC SYSTEM",
        "input_mode": "full_smear",
        "single_cell_result": None,
        "image_filename": s1["image_filename"],
        "image_path": s1["image_path"],
        "image_dimensions": {"width": s1["image_width"], "height": s1["image_height"]},
        "hardware_execution": {
            "stage1_device": s1.get("stage1_device", "Unknown"),
            "stage1_peak_vram_mb": s1.get("stage1_peak_vram_mb", 0.0),
            "gpu_fallback_triggered": s1.get("gpu_fallback_triggered", False),
            "gpu_fallback_reason": s1.get("gpu_fallback_reason", None),
            "stage2_device": f"CPU (TensorFlow {tf.__version__} native Windows)"
        },
        "models": {
            "stage1_yolo_checkpoint": s1["yolo_weights"],
            "stage2_cnn_variant": args.cnn_variant,
            "stage2_cnn_checkpoint": os.path.abspath(args.cnn_model),
            "cnn_input_shape": [224, 224, 3],
            "cnn_internal_rescaling": "1.0 / 255.0",
            "cnn_class_mapping": class_map,
            "primary_crop_preprocessing_mode": args.crop_mode
        },
        "thresholds_and_parameters": {
            "yolo_confidence_threshold": s1["yolo_conf_threshold"],
            "yolo_nms_iou_threshold": s1["yolo_iou_threshold"],
            "deduplication_iou_threshold": s1["dedup_iou_threshold"],
            "yolo_max_det": s1["max_det"],
            "crop_padding_pixels": s1["pad_px"],
            "min_box_size_pixels": s1["min_box_size"],
            "cnn_parasitized_threshold": thresh
        },
        "counts": {
            "raw_yolo_detections": s1["counts"]["raw_yolo_detections"],
            "valid_candidate_rbc_crops": total_rbcs,
            "cnn_parasitized_rbc_count": par_count_primary,
            "cnn_uninfected_rbc_count": uninf_count_primary,
            "cnn_raw_crop_parasitized_count": par_count_raw,
            "cnn_raw_crop_uninfected_count": total_rbcs - par_count_raw,
            "cnn_ellipse_masked_parasitized_count": par_count_masked,
            "cnn_ellipse_masked_uninfected_count": total_rbcs - par_count_masked,
            "excluded_wbc_detections": s1["counts"]["excluded_wbc_detections"],
            "rejected_or_duplicate_boxes": s1["counts"]["rejected_or_duplicate_boxes"]
        },
        "confidence_statistics": confidence_stats,
        "parasitemia_estimate": {
            "applicable": bool(total_rbcs > 0),
            "note": "Experimental image-level estimate (CNN Parasitized RBCs / Total CNN-classified RBCs * 100). Not a clinical measurement.",
            "primary_mode": args.crop_mode,
            "estimated_parasitemia_percent": primary_parasitemia,
            "calculation_fraction": denom_status,
            "raw_crop_parasitemia_percent": raw_parasitemia,
            "ellipse_masked_parasitemia_percent": masked_parasitemia
        },
        "Domain_Mismatch_Analysis": {
            "observation": "The NIH single-cell CNN dataset (D:\\smartcell\\dataset) contains segmented cells on a black [0,0,0] background, whereas raw rectangular crops from full smears contain bright pink plasma corners and neighboring cell edges.",
            "raw_rectangular_crops_parasitized_flagged": par_count_raw,
            "ellipse_masked_crops_parasitized_flagged": par_count_masked
        },
        "processing_time_seconds": {
            "stage1_yolo_detection_and_cropping": s1["stage1_processing_time_seconds"],
            "stage2_cnn_classification": stage2_elapsed,
            "total_pipeline_seconds": total_elapsed
        },
        "output_files": {}
    }

    # 4. Generate Visual Outputs
    annotated_path = os.path.join(output_dir, f"annotated_{stem}.jpg")
    contact_sheet_path = os.path.join(output_dir, f"contact_sheet_{stem}.jpg")

    ann_saved = create_annotated_full_image(s1["image_path"], candidates, s1["excluded_wbcs"], summary, annotated_path)
    cs_saved = create_contact_sheet(candidates, contact_sheet_path)

    summary_json_path = os.path.join(output_dir, f"field_summary_{stem}.json")
    summary_txt_path = os.path.join(output_dir, f"field_summary_{stem}.txt")
    pdf_report_path = os.path.join(output_dir, f"SmartCell_Report_{stem}.pdf")

    summary["output_files"] = {
        "output_directory": output_dir,
        "annotated_full_image": ann_saved,
        "contact_sheet_image": cs_saved,
        "crops_directory": os.path.join(output_dir, "crops"),
        "cell_predictions_csv": csv_path,
        "rejected_and_excluded_json": rejected_json_path,
        "field_summary_json": summary_json_path,
        "field_summary_txt": summary_txt_path,
        "pdf_report": pdf_report_path
    }

    # Generate PDF Analysis Report using scripts/pdf_report_generator.py
    pdf_saved = None
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import pdf_report_generator
        with open(csv_path, "r", encoding="utf-8") as cf_read:
            csv_rows_for_pdf = list(csv.DictReader(cf_read))
        with open(rejected_json_path, "r", encoding="utf-8") as rf_read:
            rej_for_pdf = json.load(rf_read)
        pdf_saved = pdf_report_generator.generate_pdf_report(
            output_pdf_path=pdf_report_path,
            summary_data=summary,
            cell_predictions=csv_rows_for_pdf,
            rejected_and_excluded=rej_for_pdf,
            annotated_image_path=ann_saved,
            run_id=os.path.basename(output_dir.rstrip("\\/"))
        )
        summary["output_files"]["pdf_report"] = pdf_saved
    except Exception as pdf_err:
        summary["output_files"]["pdf_report"] = None
        summary["pdf_generation_warning"] = f"{type(pdf_err).__name__}: {pdf_err}"

    with open(summary_json_path, "w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)

    # 5. Save Human-Readable Text Report
    par_disp_primary = f"{primary_parasitemia:.2f}%" if primary_parasitemia is not None else "N/A (No reliable result — 0 valid RBCs)"
    par_disp_raw = f"{raw_parasitemia:.2f}%" if raw_parasitemia is not None else "N/A (0 valid RBCs)"
    par_disp_masked = f"{masked_parasitemia:.2f}%" if masked_parasitemia is not None else "N/A (0 valid RBCs)"
    txt_lines = [
        "=" * 78,
        "SMARTCELL TWO-STAGE BLOOD SMEAR ANALYSIS REPORT (EXPERIMENTAL)",
        "=" * 78,
        "DISCLAIMER: Experimental research prototype only. Not a clinical diagnosis.",
        "",
        f"Input Mode:                   full_smear (Stage 1 YOLOv8n + Stage 2 CNN)",
        f"Image Filename:               {s1['image_filename']}",
        f"Image Path:                   {s1['image_path']}",
        f"Original Dimensions:          {s1['image_width']} x {s1['image_height']} px",
        f"Stage 1 Detector (YOLOv8n):   {s1['yolo_weights']}",
        f"Stage 1 Hardware Device:      {s1.get('stage1_device', 'Unknown')} (Peak VRAM: {s1.get('stage1_peak_vram_mb', 0.0)} MB)",
        f"Stage 2 Classifier (CNN):     {os.path.abspath(args.cnn_model)} (Variant: {args.cnn_variant})",
        f"Stage 2 Hardware Device:      CPU (TensorFlow {tf.__version__} native Windows)",
        f"Primary Crop Mode:            {args.crop_mode}",
        "",
        "-" * 78,
        "1. DETECTION & CLASSIFICATION COUNTS",
        "-" * 78,
        f"  Total Raw YOLO Detections:          {s1['counts']['raw_yolo_detections']}",
        f"  Valid Candidate RBC Crops:          {total_rbcs}",
        f"  Excluded WBC Detections:            {s1['counts']['excluded_wbc_detections']}",
        f"  Rejected / Duplicate Boxes:         {s1['counts']['rejected_or_duplicate_boxes']}",
        "",
        f"  CNN Classified Parasitized RBCs:    {par_count_primary} (Primary mode: {args.crop_mode})",
        f"  CNN Classified Uninfected RBCs:     {uninf_count_primary}",
        f"  [Audit] Raw Unmasked Crop Par:      {par_count_raw} / {total_rbcs} ({par_disp_raw})",
        f"  [Audit] Ellipse-Masked Crop Par:    {par_count_masked} / {total_rbcs} ({par_disp_masked})",
        "",
        "-" * 78,
        "2. EXPERIMENTAL IMAGE-LEVEL PARASITEMIA ESTIMATE",
        "-" * 78,
        f"  Formula:                            CNN Parasitized RBCs / Total Classified RBCs * 100",
        f"  Fraction:                           {denom_status}",
        f"  Estimated Parasitemia (Primary):    {par_disp_primary}",
        f"  Estimated Parasitemia (Raw Crop):   {par_disp_raw}",
        f"  Estimated Parasitemia (Masked):     {par_disp_masked}",
        "  *NOTE: Experimental image-level AI screening estimate, NOT a clinical measurement.*",
        "",
        "-" * 78,
        "3. THRESHOLDS & TIMING",
        "-" * 78,
        f"  YOLO Confidence Threshold:          {s1['yolo_conf_threshold']}",
        f"  YOLO NMS IoU Threshold:             {s1['yolo_iou_threshold']}",
        f"  Deduplication IoU Threshold:        {s1['dedup_iou_threshold']}",
        f"  CNN Sigmoid Threshold:              {thresh}",
        f"  Crop Padding:                       {s1['pad_px']} px",
        f"  Stage 1 (YOLO + Cropping) Time:     {s1['stage1_processing_time_seconds']:.3f} s",
        f"  Stage 2 (CNN + Reporting) Time:     {stage2_elapsed:.3f} s",
        f"  Total End-to-End Processing Time:   {total_elapsed:.3f} s",
        "",
        "-" * 78,
        "4. GENERATED OUTPUT FILES",
        "-" * 78,
        f"  PDF Analysis Report: {pdf_saved}",
        f"  Annotated Image:     {ann_saved}",
        f"  Contact Sheet:       {cs_saved}",
        f"  Crops Directory:     {os.path.join(output_dir, 'crops')}",
        f"  Predictions CSV:     {csv_path}",
        f"  Rejected/Excl JSON:  {rejected_json_path}",
        f"  Summary JSON:        {summary_json_path}",
        f"  Summary TXT:         {summary_txt_path}",
        "=" * 78
    ]
    txt_report = "\n".join(txt_lines) + "\n"
    with open(summary_txt_path, "w", encoding="utf-8") as tf_out:
        tf_out.write(txt_report)

    print("\n" + txt_report)
    return summary_json_path


# ---------------------------------------------------------------------------
# SINGLE-CELL MODE: Direct Stage 2 CNN Classification (Bypassing YOLOv8n)
# ---------------------------------------------------------------------------
def run_single_cell_cnn(args):
    """
    Executes directly in venv (TensorFlow 2.21.0) when --input-mode single_cell is selected:
    1. Bypasses Stage 1 YOLOv8n completely.
    2. Loads the already-cropped single-cell image at native resolution.
    3. Reuses the exact existing CNN preprocessing ((224, 224, 3) float32 RGB in [0, 255]
       with internal Rescaling(1.0/255.0) and optional elliptical masking) and threshold.
    4. Reports one cell's classification only and does NOT calculate or display whole-smear parasitemia.
    """
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf

    t0 = time.time()
    image_path = os.path.abspath(args.image)
    output_dir = os.path.abspath(args.output_dir)
    crops_dir = os.path.join(output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input single-cell image not found: {image_path}")
    if not os.path.exists(args.cnn_model):
        raise FileNotFoundError(f"CNN model checkpoint not found: {args.cnn_model}")

    class_map = {"0": "Uninfected", "1": "Parasitized"}
    if os.path.exists(DEFAULT_CLASS_NAMES_JSON):
        with open(DEFAULT_CLASS_NAMES_JSON, "r", encoding="utf-8") as f:
            class_map = json.load(f)

    stem = os.path.splitext(os.path.basename(image_path))[0]
    img_filename = os.path.basename(image_path)

    with Image.open(image_path) as pil_img:
        rgb_im = pil_img.convert("RGB")
        img_w, img_h = rgb_im.size

    crop_id = f"{stem}_cell_0001"
    crop_filename = f"{crop_id}.png"
    crop_rel_path = os.path.join("crops", crop_filename)
    crop_abs_path = os.path.join(crops_dir, crop_filename)
    
    # Save crop for visualization (downscale huge non-cropped frames to avoid runaway RAM)
    save_im = rgb_im
    if img_w > 512 or img_h > 512:
        save_im = rgb_im.copy()
        save_im.thumbnail((512, 512), Image.Resampling.BILINEAR)
    save_im.save(crop_abs_path, format="PNG")

    cnn_model = tf.keras.models.load_model(args.cnn_model)

    resized_raw = rgb_im.resize((224, 224), Image.Resampling.BILINEAR)
    raw_np = np.array(resized_raw, dtype=np.float32)
    raw_arr = np.expand_dims(raw_np, axis=0)

    # Apply elliptical mask directly to the normalized 224x224 patch
    masked_np = apply_elliptical_mask(np.array(resized_raw))
    masked_arr = np.expand_dims(masked_np.astype(np.float32), axis=0)

    p_raw = float(cnn_model.predict(raw_arr, verbose=0).flatten()[0])
    p_mask = float(cnn_model.predict(masked_arr, verbose=0).flatten()[0])

    thresh = float(args.cnn_thresh)
    use_masked = (args.crop_mode in ("masked", "ellipse_masked"))

    cls_raw = class_map["1"] if p_raw >= thresh else class_map["0"]
    conf_raw = p_raw if p_raw >= thresh else (1.0 - p_raw)

    cls_mask = class_map["1"] if p_mask >= thresh else class_map["0"]
    conf_mask = p_mask if p_mask >= thresh else (1.0 - p_mask)

    p_primary = p_mask if use_masked else p_raw
    cls_primary = cls_mask if use_masked else cls_raw
    conf_primary = conf_mask if use_masked else conf_raw

    par_count_primary = 1 if cls_primary == "Parasitized" else 0
    uninf_count_primary = 1 if cls_primary == "Uninfected" else 0
    par_count_raw = 1 if cls_raw == "Parasitized" else 0
    par_count_masked = 1 if cls_mask == "Parasitized" else 0

    stage2_elapsed = round(time.time() - t0, 4)

    single_cell_result = {
        "is_single_cell_mode": True,
        "crop_id": crop_id,
        "predicted_class": cls_primary,
        "confidence": round(conf_primary, 6),
        "confidence_percent": round(conf_primary * 100.0, 2),
        "parasitized_prob": round(p_primary, 6),
        "uninfected_prob": round(1.0 - p_primary, 6),
        "raw_predicted_class": cls_raw,
        "raw_confidence": round(conf_raw, 6),
        "raw_parasitized_prob": round(p_raw, 6),
        "masked_predicted_class": cls_mask,
        "masked_confidence": round(conf_mask, 6),
        "masked_parasitized_prob": round(p_mask, 6),
        "threshold_used": thresh,
        "cnn_variant": args.cnn_variant,
        "crop_mode": args.crop_mode,
    }

    cand = {
        "crop_id": crop_id,
        "bbox": [0, 0, img_w, img_h],
        "bbox_width": img_w,
        "bbox_height": img_h,
        "padded_bbox": [0, 0, img_w, img_h],
        "yolo_class_id": -1,
        "yolo_class_name": "Single-Cell Input (YOLO Bypassed)",
        "yolo_confidence": 1.0,
        "cnn_predicted_class": cls_primary,
        "cnn_confidence": round(conf_primary, 6),
        "cnn_parasitized_prob": round(p_primary, 6),
        "cnn_uninfected_prob": round(1.0 - p_primary, 6),
        "cnn_raw_predicted_class": cls_raw,
        "cnn_raw_parasitized_prob": round(p_raw, 6),
        "cnn_masked_predicted_class": cls_mask,
        "cnn_masked_parasitized_prob": round(p_mask, 6),
        "crop_path": crop_abs_path,
        "crop_rel_path": crop_rel_path,
    }
    candidates = [cand]

    csv_path = os.path.join(output_dir, f"cell_predictions_{stem}.csv")
    csv_fields = [
        "crop_id", "image_filename",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "bbox_width", "bbox_height",
        "padded_x1", "padded_y1", "padded_x2", "padded_y2",
        "yolo_class_id", "yolo_class_name", "yolo_confidence",
        "cnn_predicted_class", "cnn_confidence", "cnn_parasitized_prob", "cnn_uninfected_prob",
        "cnn_raw_predicted_class", "cnn_raw_parasitized_prob",
        "cnn_masked_predicted_class", "cnn_masked_parasitized_prob",
        "crop_rel_path"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerow({
            "crop_id": cand["crop_id"],
            "image_filename": img_filename,
            "bbox_x1": 0,
            "bbox_y1": 0,
            "bbox_x2": img_w,
            "bbox_y2": img_h,
            "bbox_width": img_w,
            "bbox_height": img_h,
            "padded_x1": 0,
            "padded_y1": 0,
            "padded_x2": img_w,
            "padded_y2": img_h,
            "yolo_class_id": -1,
            "yolo_class_name": "Single-Cell Input (YOLO Bypassed)",
            "yolo_confidence": 1.0,
            "cnn_predicted_class": cand["cnn_predicted_class"],
            "cnn_confidence": cand["cnn_confidence"],
            "cnn_parasitized_prob": cand["cnn_parasitized_prob"],
            "cnn_uninfected_prob": cand["cnn_uninfected_prob"],
            "cnn_raw_predicted_class": cand["cnn_raw_predicted_class"],
            "cnn_raw_parasitized_prob": cand["cnn_raw_parasitized_prob"],
            "cnn_masked_predicted_class": cand["cnn_masked_predicted_class"],
            "cnn_masked_parasitized_prob": cand["cnn_masked_parasitized_prob"],
            "crop_rel_path": cand["crop_rel_path"],
        })

    rejected_json_path = os.path.join(output_dir, f"rejected_and_excluded_{stem}.json")
    with open(rejected_json_path, "w", encoding="utf-8") as rf:
        json.dump({
            "image_filename": img_filename,
            "input_mode": "single_cell",
            "note": "Stage 1 YOLOv8n detection was bypassed because Single-Cell Mode was explicitly selected.",
            "excluded_wbc_count": 0,
            "rejected_or_duplicate_count": 0,
            "excluded_wbcs": [],
            "rejected_boxes": []
        }, rf, indent=2)

    conf_val = round(conf_primary, 4)
    single_stat = {"count": 1, "mean": conf_val, "median": conf_val, "min": conf_val, "max": conf_val}
    empty_stat = {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    confidence_stats = {
        "yolo_detection_confidence": empty_stat,
        "cnn_overall_confidence": single_stat,
        "cnn_parasitized_confidence": single_stat if cls_primary == "Parasitized" else empty_stat,
        "cnn_uninfected_confidence": single_stat if cls_primary == "Uninfected" else empty_stat,
    }

    summary = {
        "disclaimer": "EXPERIMENTAL RESEARCH PROTOTYPE ONLY — NOT A CLINICALLY VALIDATED DIAGNOSTIC SYSTEM",
        "input_mode": "single_cell",
        "single_cell_result": single_cell_result,
        "image_filename": img_filename,
        "image_path": image_path,
        "image_dimensions": {"width": img_w, "height": img_h},
        "hardware_execution": {
            "stage1_device": "Bypassed (Single-Cell Mode — Direct Stage 2 CNN)",
            "stage1_peak_vram_mb": 0.0,
            "gpu_fallback_triggered": False,
            "gpu_fallback_reason": None,
            "stage2_device": f"CPU (TensorFlow {tf.__version__} native Windows)"
        },
        "models": {
            "stage1_yolo_checkpoint": "Bypassed (Single-Cell Mode)",
            "stage2_cnn_variant": args.cnn_variant,
            "stage2_cnn_checkpoint": os.path.abspath(args.cnn_model),
            "cnn_input_shape": [224, 224, 3],
            "cnn_internal_rescaling": "1.0 / 255.0",
            "cnn_class_mapping": class_map,
            "primary_crop_preprocessing_mode": args.crop_mode
        },
        "thresholds_and_parameters": {
            "yolo_confidence_threshold": "N/A (Bypassed)",
            "yolo_nms_iou_threshold": "N/A (Bypassed)",
            "deduplication_iou_threshold": "N/A (Bypassed)",
            "yolo_max_det": "N/A (Bypassed)",
            "crop_padding_pixels": 0,
            "min_box_size_pixels": "N/A (Bypassed)",
            "cnn_parasitized_threshold": thresh
        },
        "counts": {
            "raw_yolo_detections": 0,
            "valid_candidate_rbc_crops": 1,
            "cnn_parasitized_rbc_count": par_count_primary,
            "cnn_uninfected_rbc_count": uninf_count_primary,
            "cnn_raw_crop_parasitized_count": par_count_raw,
            "cnn_raw_crop_uninfected_count": 1 - par_count_raw,
            "cnn_ellipse_masked_parasitized_count": par_count_masked,
            "cnn_ellipse_masked_uninfected_count": 1 - par_count_masked,
            "excluded_wbc_detections": 0,
            "rejected_or_duplicate_boxes": 0
        },
        "confidence_statistics": confidence_stats,
        "parasitemia_estimate": {
            "applicable": False,
            "note": "Whole-smear parasitemia is NOT calculated or displayed for a single-cell input. Only the individual cell's classification is reported.",
            "primary_mode": args.crop_mode,
            "estimated_parasitemia_percent": None,
            "calculation_fraction": "N/A (Single-Cell Mode — whole-smear parasitemia not applicable)",
            "raw_crop_parasitemia_percent": None,
            "ellipse_masked_parasitemia_percent": None
        },
        "Domain_Mismatch_Analysis": {
            "observation": "Single-cell mode classifies an already-cropped cell directly via Stage 2 CNN without Stage 1 YOLO localization.",
            "raw_rectangular_crops_parasitized_flagged": par_count_raw,
            "ellipse_masked_crops_parasitized_flagged": par_count_masked
        },
        "processing_time_seconds": {
            "stage1_yolo_detection_and_cropping": 0.0,
            "stage2_cnn_classification": stage2_elapsed,
            "total_pipeline_seconds": stage2_elapsed
        },
        "output_files": {}
    }

    annotated_path = os.path.join(output_dir, f"annotated_{stem}.jpg")
    contact_sheet_path = os.path.join(output_dir, f"contact_sheet_{stem}.jpg")

    ann_saved = create_annotated_single_cell_image(image_path, single_cell_result, annotated_path)
    cs_saved = create_contact_sheet(candidates, contact_sheet_path, cols=1, max_items=1)

    summary_json_path = os.path.join(output_dir, f"field_summary_{stem}.json")
    summary_txt_path = os.path.join(output_dir, f"field_summary_{stem}.txt")
    pdf_report_path = os.path.join(output_dir, f"SmartCell_Report_{stem}.pdf")

    summary["output_files"] = {
        "output_directory": output_dir,
        "annotated_full_image": ann_saved,
        "contact_sheet_image": cs_saved,
        "crops_directory": crops_dir,
        "cell_predictions_csv": csv_path,
        "rejected_and_excluded_json": rejected_json_path,
        "field_summary_json": summary_json_path,
        "field_summary_txt": summary_txt_path,
        "pdf_report": pdf_report_path
    }

    pdf_saved = None
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import pdf_report_generator
        with open(csv_path, "r", encoding="utf-8") as cf_read:
            csv_rows_for_pdf = list(csv.DictReader(cf_read))
        with open(rejected_json_path, "r", encoding="utf-8") as rf_read:
            rej_for_pdf = json.load(rf_read)
        pdf_saved = pdf_report_generator.generate_pdf_report(
            output_pdf_path=pdf_report_path,
            summary_data=summary,
            cell_predictions=csv_rows_for_pdf,
            rejected_and_excluded=rej_for_pdf,
            annotated_image_path=ann_saved,
            run_id=os.path.basename(output_dir.rstrip("\\/"))
        )
        summary["output_files"]["pdf_report"] = pdf_saved
    except Exception as pdf_err:
        summary["output_files"]["pdf_report"] = None
        summary["pdf_generation_warning"] = f"{type(pdf_err).__name__}: {pdf_err}"

    with open(summary_json_path, "w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)

    txt_lines = [
        "=" * 78,
        "SMARTCELL SINGLE-CELL AI CLASSIFICATION REPORT (EXPERIMENTAL)",
        "=" * 78,
        "DISCLAIMER: Experimental research prototype only. Not a clinical diagnosis.",
        "",
        f"Input Mode:                   single_cell (Direct Stage 2 CNN — Stage 1 YOLO Bypassed)",
        f"Image Filename:               {img_filename}",
        f"Image Path:                   {image_path}",
        f"Original Dimensions:          {img_w} x {img_h} px",
        f"Stage 1 Detector (YOLOv8n):   Bypassed (Single-Cell Mode)",
        f"Stage 2 Classifier (CNN):     {os.path.abspath(args.cnn_model)} (Variant: {args.cnn_variant})",
        f"Stage 2 Hardware Device:      CPU (TensorFlow {tf.__version__} native Windows)",
        f"Primary Crop Mode:            {args.crop_mode}",
        "",
        "-" * 78,
        "1. SINGLE-CELL CLASSIFICATION RESULT",
        "-" * 78,
        f"  Predicted Class (Primary):          {cls_primary}",
        f"  Prediction Confidence:              {conf_primary * 100.0:.2f}% (Threshold: {thresh})",
        f"  Parasitized Probability:            {p_primary * 100.0:.2f}% ({p_primary:.6f})",
        f"  Uninfected Probability:             {(1.0 - p_primary) * 100.0:.2f}% ({1.0 - p_primary:.6f})",
        f"  [Audit] Raw Crop Prediction:        {cls_raw} (P(Par)={p_raw * 100.0:.2f}%, Conf={conf_raw * 100.0:.2f}%)",
        f"  [Audit] Ellipse-Masked Prediction:  {cls_mask} (P(Par)={p_mask * 100.0:.2f}%, Conf={conf_mask * 100.0:.2f}%)",
        f"  Whole-Smear Parasitemia:            N/A (Single-Cell Mode — not applicable)",
        "",
        "-" * 78,
        "2. PROCESSING TIMING & OUTPUT FILES",
        "-" * 78,
        f"  Stage 2 (CNN + Reporting) Time:     {stage2_elapsed:.3f} s",
        f"  PDF Analysis Report:                {pdf_saved}",
        f"  Annotated Cell Image:               {ann_saved}",
        f"  Contact Sheet:                      {cs_saved}",
        f"  Predictions CSV:                    {csv_path}",
        f"  Summary JSON:                       {summary_json_path}",
        f"  Summary TXT:                        {summary_txt_path}",
        "=" * 78
    ]
    txt_report = "\n".join(txt_lines) + "\n"
    with open(summary_txt_path, "w", encoding="utf-8") as tf_out:
        tf_out.write(txt_report)

    print("\n" + txt_report)
    return summary_json_path


# ---------------------------------------------------------------------------
# CLI Orchestrator
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="SmartCell Two-Stage YOLOv8n + CNN Malaria Screening Pipeline")
    parser.add_argument("pos_image", nargs="?", default=None, help="Path to microscope image")
    parser.add_argument("pos_output_dir", nargs="?", default=None, help="Output directory path")
    parser.add_argument("--image", type=str, default=None, help="Path to microscope image")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for run artifacts")
    parser.add_argument("--input-mode", type=str, choices=["full_smear", "single_cell"], default="full_smear",
                        help="Specimen input mode: 'full_smear' (Stage 1 YOLO + Stage 2 CNN) or 'single_cell' (Direct Stage 2 CNN, bypass YOLO)")
    parser.add_argument("--yolo-weights", type=str, default=DEFAULT_YOLO_WEIGHTS, help="Path to YOLOv8n best.pt")
    parser.add_argument("--yolo-device", type=str, default="0", help="YOLO inference device ('0' for RTX 2050 GPU or 'cpu')")
    parser.add_argument("--cnn-variant", type=str, choices=["original", "finetuned_raw", "finetuned_masked"], default="original",
                        help="Select CNN model variant ('original' fallback, 'finetuned_raw', or 'finetuned_masked')")
    parser.add_argument("--cnn-model", type=str, default=None, help="Explicit path to CNN .keras model (overrides --cnn-variant)")
    parser.add_argument("--yolo-imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--yolo-iou", type=float, default=0.45, help="YOLO NMS IoU threshold")
    parser.add_argument("--dedup-iou", type=float, default=0.50, help="Class-agnostic deduplication IoU threshold")
    parser.add_argument("--max-det", type=int, default=350, help="Maximum detections per field (>=350)")
    parser.add_argument("--pad-px", type=int, default=4, help="Modest padding around candidate RBC crops in pixels")
    parser.add_argument("--min-box-size", type=int, default=16, help="Minimum valid bounding box width/height in pixels")
    parser.add_argument("--cnn-thresh", type=float, default=0.50, help="CNN Parasitized classification threshold")
    parser.add_argument("--crop-mode", type=str, choices=["raw", "masked", "ellipse_masked"], default="raw",
                        help="Primary CNN input crop mode ('raw' = unmodified rectangular crop; 'masked'/'ellipse_masked' = NIH-matched black corners)")
    parser.add_argument("--stage1-only", action="store_true", help="Internal worker flag: run Stage 1 only")
    parser.add_argument("--stage2-only", action="store_true", help="Internal worker flag: run Stage 2 only")
    parser.add_argument("--single-cell-worker", action="store_true", help="Internal worker flag: run Single-Cell Stage 2 CNN directly")
    parser.add_argument("--stage1-json", type=str, default=None, help="Path to stage1_detections.json for Stage 2 worker")
    parser.add_argument("--test-zero-detections", action="store_true", help="Simulate zero YOLO detections for edge-case validation")
    parser.add_argument("--inject-test-invalid-boxes", action="store_true", help="Inject synthetic invalid/out-of-bounds/WBC boxes for validation")
    args = parser.parse_args()

    # Resolve positional vs named arguments
    if args.image is None and args.pos_image is not None:
        args.image = args.pos_image
    if args.output_dir is None and args.pos_output_dir is not None:
        args.output_dir = args.pos_output_dir

    if args.cnn_model is None:
        args.cnn_model = CNN_VARIANT_PATHS.get(args.cnn_variant, DEFAULT_CNN_MODEL)

    if not args.stage2_only and not args.image:
        parser.error("An input microscope image must be provided via positional arg or --image.")

    if args.image and not args.output_dir:
        stem = os.path.splitext(os.path.basename(args.image))[0]
        args.output_dir = os.path.join(DEFAULT_PIPELINE_OUT_ROOT, stem)

    return args


def build_worker_cmd(python_exe, args, stage_flag, stage1_json_path=None):
    cmd = [
        python_exe,
        "-u",
        os.path.abspath(__file__),
        "--image", os.path.abspath(args.image),
        "--output-dir", os.path.abspath(args.output_dir),
        "--input-mode", str(args.input_mode),
        "--yolo-weights", os.path.abspath(args.yolo_weights),
        "--yolo-device", str(args.yolo_device),
        "--cnn-variant", str(args.cnn_variant),
        "--cnn-model", os.path.abspath(args.cnn_model),
        "--yolo-imgsz", str(args.yolo_imgsz),
        "--yolo-conf", str(args.yolo_conf),
        "--yolo-iou", str(args.yolo_iou),
        "--dedup-iou", str(args.dedup_iou),
        "--max-det", str(args.max_det),
        "--pad-px", str(args.pad_px),
        "--min-box-size", str(args.min_box_size),
        "--cnn-thresh", str(args.cnn_thresh),
        "--crop-mode", str(args.crop_mode),
        stage_flag
    ]
    if stage1_json_path:
        cmd.extend(["--stage1-json", os.path.abspath(stage1_json_path)])
    if args.test_zero_detections:
        cmd.append("--test-zero-detections")
    if args.inject_test_invalid_boxes:
        cmd.append("--inject-test-invalid-boxes")
    return cmd


def main():
    args = parse_args()

    if args.single_cell_worker:
        run_single_cell_cnn(args)
        return

    if args.input_mode == "single_cell":
        print("=" * 78)
        print("SMARTCELL SINGLE-CELL MODE ORCHESTRATOR (BYPASSING STAGE 1 YOLO)")
        print("=" * 78)
        os.makedirs(args.output_dir, exist_ok=True)
        tf_py = get_python_exe(VENV_TF_PYTHON)
        # If already inside an environment where TensorFlow is available, run directly
        if os.path.normcase(os.path.abspath(sys.executable)) == os.path.normcase(os.path.abspath(tf_py)):
            run_single_cell_cnn(args)
        else:
            cmd_sc = build_worker_cmd(tf_py, args, "--single-cell-worker")
            subprocess.run(cmd_sc, check=True)
        return

    if args.stage1_only:
        run_stage1_yolo(args)
        return

    if args.stage2_only:
        if not args.stage1_json:
            stem = os.path.splitext(os.path.basename(args.image))[0]
            args.stage1_json = os.path.join(args.output_dir, f"stage1_detections_{stem}.json")
        run_stage2_cnn(args, args.stage1_json)
        return

    # Full Two-Stage Orchestration across isolated environments
    print("=" * 78)
    print("SMARTCELL TWO-STAGE PIPELINE ORCHESTRATOR")
    print("=" * 78)
    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    stage1_json = os.path.join(args.output_dir, f"stage1_detections_{stem}.json")

    yolo_cpu_py = get_python_exe(VENV_YOLO_CPU_PYTHON)
    yolo_py = yolo_cpu_py if str(args.yolo_device).lower() == "cpu" else get_python_exe(VENV_YOLO_PYTHON)
    tf_py = get_python_exe(VENV_TF_PYTHON)

    # Dispatch Stage 1 to venv_yolo_gpu (or fallback to sys.executable)
    print(f"[1/2] Dispatching Stage 1 (YOLOv8n Detection & Native Cropping) to: {yolo_py}")
    cmd_s1 = build_worker_cmd(yolo_py, args, "--stage1-only")
    subprocess.run(cmd_s1, check=True)

    # Dispatch Stage 2 to venv (TensorFlow) or fallback to sys.executable
    print(f"[2/2] Dispatching Stage 2 (CNN Classification & Reporting) to: {tf_py}")
    cmd_s2 = build_worker_cmd(tf_py, args, "--stage2-only", stage1_json_path=stage1_json)
    subprocess.run(cmd_s2, check=True)


if __name__ == "__main__":
    main()
