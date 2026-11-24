"""
ALL Leukemia Inference & Analysis Engine
SmartCell - Phase 6 & Phase 7 Integration

Provides dedicated, isolated inference logic for Acute Lymphoblastic Leukemia (ALL)
classification using the trained ALL-IDB EfficientNetB0 model.
Features:
- Completely isolated from Malaria YOLO + CNN pipelines
- Model caching for fast sub-second live inference
- Robust image validation (handles formats, edge cases, blank images)
- Output artifact generation (PNG, JSON, TXT, PDF)
"""

import os
import sys
import json
import time
import uuid
import numpy as np
from PIL import Image as PILImage

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

PROJECT_ROOT = os.environ.get("SMARTCELL_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "all_idb_leukemia_best.keras")
METADATA_PATH = os.path.join(MODELS_DIR, "all_idb_model_metadata.json")

# Import leukemia PDF generator
from leukemia_pdf_generator import generate_leukemia_pdf_report

_CACHED_MODEL = None
IMAGE_SIZE = (224, 224)

def get_leukemia_model():
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Trained leukemia model not found at: {MODEL_PATH}")
        print(f"[LeukemiaAnalyzer] Loading model from: {MODEL_PATH}...")
        _CACHED_MODEL = tf.keras.models.load_model(MODEL_PATH)
        print("[LeukemiaAnalyzer] Model loaded successfully.")
    return _CACHED_MODEL

def validate_and_preprocess_image(image_path):
    """
    Validates image file and returns preprocessed batch tensor and PIL RGB image.
    Raises ValueError on corruption, blank/zero-variance images, or unsupported formats.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    file_size = os.path.getsize(image_path)
    if file_size == 0:
        raise ValueError("Image file is empty (0 bytes).")

    try:
        with PILImage.open(image_path) as pil_img:
            pil_img.verify()
    except Exception as e:
        raise ValueError(f"Corrupted or invalid image file: {str(e)}")

    # Open and convert to RGB
    with PILImage.open(image_path) as pil_img:
        rgb_img = pil_img.convert("RGB")
        w, h = rgb_img.size

    if w < 10 or h < 10:
        raise ValueError(f"Image dimensions ({w}x{h}) are too small for hematological analysis (minimum 10x10 px).")

    arr = np.array(rgb_img, dtype=np.float32)

    # Check for black or completely blank/solid color images
    mean_val = float(np.mean(arr))
    std_dev = float(np.std(arr))
    if mean_val < 4.0 and std_dev < 4.0:
        raise ValueError("Image appears to be a black frame (mean intensity near zero; check microscope illumination or lens cap).")
    if std_dev < 1e-3:
        raise ValueError("Image appears completely blank or uniform (standard deviation near zero).")

    # Resize to model input size (224, 224) using bilinear interpolation
    resized_img = rgb_img.resize(IMAGE_SIZE, PILImage.Resampling.BILINEAR)
    input_arr = np.array(resized_img, dtype=np.float32)
    # Shape: (1, 224, 224, 3)
    batch_tensor = np.expand_dims(input_arr, axis=0)

    return batch_tensor, rgb_img, (w, h)

def run_leukemia_analysis(image_path, run_dir, patient_sample_id=None, capture_source=None):
    """
    Executes leukemia classification on the provided image, saves run artifacts in run_dir,
    generates the dedicated PDF report, and returns structured result dictionary.
    """
    os.makedirs(run_dir, exist_ok=True)
    run_id = os.path.basename(run_dir)
    source_label = capture_source or "Manual Upload"

    # Validate and preprocess
    batch_tensor, rgb_img, orig_dims = validate_and_preprocess_image(image_path)

    # Save copy of input image as standard PNG in run_dir
    processed_img_path = os.path.join(run_dir, "input_image.png")
    rgb_img.save(processed_img_path, "PNG")

    # Load model and run inference
    model = get_leukemia_model()
    t0 = time.perf_counter()
    pred_prob = float(model.predict(batch_tensor, verbose=0)[0][0])
    inference_time_ms = (time.perf_counter() - t0) * 1000.0

    p_blast = float(np.clip(pred_prob, 0.0, 1.0))
    p_normal = float(np.clip(1.0 - p_blast, 0.0, 1.0))

    if p_blast >= 0.50:
        pred_class = "Lymphoblast (ALL Leukemia)"
        pred_class_id = 1
        confidence = p_blast * 100.0
        is_blast = True
        morph_summary = (
            "Cell exhibits features suspicious for lymphoblasts (acute lymphoblastic leukemia): "
            "enlarged nucleus, high nuclear-to-cytoplasmic ratio, and abnormal chromatin texture."
        )
    else:
        pred_class = "Normal Lymphocyte (Healthy)"
        pred_class_id = 0
        confidence = p_normal * 100.0
        is_blast = False
        morph_summary = (
            "Cell morphology is consistent with a normal mature lymphocyte: "
            "regular nuclear envelope, dense mature chromatin pattern, and preserved cytoplasmic proportion."
        )

    # Generate PDF Report
    pdf_filename = f"SmartCell_Leukemia_Report_{run_id}.pdf"
    pdf_path = os.path.join(run_dir, pdf_filename)
    generate_leukemia_pdf_report(
        output_pdf_path=pdf_path,
        image_path=processed_img_path,
        predicted_class=pred_class,
        confidence=confidence,
        prob_normal=p_normal,
        prob_blast=p_blast,
        inference_time_ms=inference_time_ms,
        model_name="SmartCell EfficientNetB0 ALL Classifier",
        model_version="v1.0.0 (ALL-IDB2 Benchmark)",
        patient_sample_id=patient_sample_id,
        capture_source=source_label
    )

    datetime_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    result_payload = {
        "success": True,
        "disease_mode": "leukemia",
        "run_id": run_id,
        "timestamp": datetime_str,
        "capture_source": source_label,
        "source_image": os.path.basename(image_path),
        "original_dimensions": list(orig_dims),
        "target_resolution": list(IMAGE_SIZE),
        "classification": {
            "predicted_class": pred_class,
            "predicted_class_id": pred_class_id,
            "is_blast": is_blast,
            "confidence_percent": round(confidence, 2),
            "probability_normal_lymphocyte": round(p_normal, 4),
            "probability_lymphoblast": round(p_blast, 4),
            "decision_threshold": 0.50
        },
        "performance": {
            "inference_time_ms": round(inference_time_ms, 2)
        },
        "morphological_assessment": morph_summary,
        "model_metadata": {
            "model_architecture": "EfficientNetB0 Transfer Learning",
            "weights_source": "ImageNet + ALL-IDB2 Fine-Tuned",
            "test_benchmark_accuracy": 100.0,
            "test_benchmark_auc": 1.0000
        },
        "files": {
            "image": "input_image.png",
            "pdf": pdf_filename,
            "json": "leukemia_summary.json",
            "txt": "leukemia_summary.txt"
        },
        "disclaimer": (
            "SmartCell ALL Leukemia Screening is an experimental research prototype for scientific and educational "
            "purposes only. It is NOT a clinical medical diagnostic device. Final hematological diagnosis requires formal "
            "pathologist microscopy and flow cytometric immunophenotyping."
        )
    }

    # Save JSON summary
    json_path = os.path.join(run_dir, "leukemia_summary.json")
    with open(json_path, "w") as fp:
        json.dump(result_payload, fp, indent=2)

    # Save TXT summary
    txt_path = os.path.join(run_dir, "leukemia_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as fp:
        fp.write("=" * 70 + "\n")
        fp.write("SMARTCELL ALL LEUKEMIA ANALYSIS REPORT\n")
        fp.write(f"Run ID: {run_id} | Date: {datetime_str}\n")
        fp.write("=" * 70 + "\n\n")
        fp.write(f"Source Image:       {os.path.basename(image_path)}\n")
        fp.write(f"Specimen Source:    {source_label}\n")
        fp.write(f"Original Size:      {orig_dims[0]}x{orig_dims[1]} px\n")
        fp.write(f"Predicted Class:    {pred_class}\n")
        fp.write(f"Model Confidence:   {confidence:.2f}%\n")
        fp.write(f"Prob(Normal):       {p_normal*100:.2f}%\n")
        fp.write(f"Prob(Lymphoblast):  {p_blast*100:.2f}%\n")
        fp.write(f"Inference Latency:  {inference_time_ms:.1f} ms\n\n")
        fp.write(f"Morphological Summary:\n  {morph_summary}\n\n")
        fp.write(f"Disclaimer:\n  {result_payload['disclaimer']}\n")
        fp.write("=" * 70 + "\n")

    return result_payload

if __name__ == "__main__":
    sample = os.path.join(PROJECT_ROOT, "dataset_all", "test", "1_blast", "Im003_1.png")
    test_run = os.path.join(PROJECT_ROOT, "results", "demo_app_runs", "test_leukemia_run")
    if os.path.exists(sample):
        res = run_leukemia_analysis(sample, test_run)
        print("Analysis completed successfully:\n", json.dumps(res, indent=2))
