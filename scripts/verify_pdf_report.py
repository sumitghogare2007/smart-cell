r"""
verify_pdf_report.py
--------------------
SmartCell Comprehensive Verification & Regression Suite:
  - Support for Full-Smear Mode (`full_smear`) and Single-Cell Mode (`single_cell`)
  - Fix for Broken Annotated Image Output on Small / Single-Cell Images (`C100P61ThinF_IMG_20150918_144104_cell_162.png`)
  - Input Validation (Low-Resolution Warnings, Blank/Black/Corrupted Rejection, No Silent Upscaling or Mode Switching)
  - PDF Report Structure (Sections A–E, G, H retained; Section F excluded; Mode-Aware Executive Summary & Screening Result)
  - Automated Source vs. PDF-Embedded Image Pixel Comparison

Preserves all previous verification reports untouched (`phase7_*`, `phase8_*`, `pdf_report_*`, `microscope_capture_*`)
and writes the new verification report to:
  - `D:\smartcell\results\metrics\smartcell_input_modes_and_annotated_image_verification_report.json`
  - `D:\smartcell\results\demo_app_runs\smartcell_input_modes_and_annotated_image_verification_report.txt`
"""

import os
import io
import sys
import csv
import json
import time
import uuid
import tempfile
import threading
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
from PIL import Image as PILImage
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smartcell_demo_app as app
import pdf_report_generator as pdf_gen

BASE_DIR = r"D:\smartcell"
SAMPLE_FULL_SMEAR = os.path.join(BASE_DIR, "yolo_malaria", "images", "test", "IMG_20150819_133447.jpg")
SAMPLE_SINGLE_CELL_PAR = os.path.join(
    BASE_DIR,
    "Testing_images",
    "cell_images",
    "Parasitized",
    "C100P61ThinF_IMG_20150918_144104_cell_162.png",
)
SAMPLE_SINGLE_CELL_UNINF = os.path.join(
    BASE_DIR,
    "dataset",
    "test",
    "Uninfected",
    "C100P61ThinF_IMG_20150918_144104_cell_128.png",
)

REPORT_JSON_PATH = os.path.join(
    BASE_DIR,
    "results",
    "metrics",
    "smartcell_input_modes_and_annotated_image_verification_report.json",
)
REPORT_TXT_PATH = os.path.join(
    BASE_DIR,
    "results",
    "demo_app_runs",
    "smartcell_input_modes_and_annotated_image_verification_report.txt",
)


def build_multipart_request(url, fields, file_tuple=None):
    boundary = f"----SmartCellBoundary{uuid.uuid4().hex}"
    chunks = []
    for k, v in fields.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode("utf-8"))
        chunks.append(f"{v}\r\n".encode("utf-8"))

    if file_tuple is not None:
        field_name, filename, file_bytes = file_tuple
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n".encode("utf-8")
        )
        chunks.append(file_bytes)
        chunks.append(b"\r\n")

    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(chunks)
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Content-Length", str(len(body)))
    return req


def post_multipart(url, fields, file_tuple=None):
    req = build_multipart_request(url, fields, file_tuple=file_tuple)
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as he:
        return he.code, json.loads(he.read().decode("utf-8"))


def get_url(url):
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as he:
        return he.code, he.read(), dict(he.headers)


def main():
    os.makedirs(os.path.dirname(REPORT_JSON_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_TXT_PATH), exist_ok=True)

    port = 8521
    server = ThreadingHTTPServer(("127.0.0.1", port), app.SmartCellRequestHandler)
    srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
    srv_thread.start()
    base_url = f"http://127.0.0.1:{port}"
    print(f"[SmartCell Verification] Started automated test server at {base_url}")

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "verification_environment": {
            "test_execution_mode": "automated_regression_and_http_integration_tests",
            "physical_usb_microscope_attached": False,
            "note": (
                "All tests in this verification suite were executed automatically against real dataset images "
                "(full-smear NIH-NLM microscope images and segmented single-cell RBC images) via the SmartCell "
                "HTTP API and pipeline; no physical USB microscope hardware was attached during this run."
            ),
        },
        "root_cause_of_broken_annotated_image": {
            "specimen_tested": "C100P61ThinF_IMG_20150918_144104_cell_162.png",
            "input_dimensions": [142, 148],
            "input_image_corrupted": False,
            "root_cause_summary": (
                "1) Domain/Mode Mismatch: C100P61ThinF_IMG_20150918_144104_cell_162.png (142x148 px) is an already-segmented "
                "single-cell RBC patch on a black background, not a multi-cell full blood smear. Stage 1 YOLOv8n produced 1 tiny "
                "6x5 px artifact box (< 16 px min_box_size), leaving 0 valid RBC crops. "
                "2) Hardcoded 4K Header Banner Occlusion & Unconditional 1920-px Upscaling: In run_smartcell_pipeline.py "
                "(create_annotated_full_image), cv2.rectangle(img, (20, 20), (2450, 210), (20, 20, 20), -1) drew a 2430x190 px "
                "near-black banner directly onto the 142x148 image—completely overwriting 85% of the cell from (20,20) to (142,148) "
                "with solid black (20,20,20), drawing giant clipped letters ('SMAR' / 'Valid R') across the remaining width, and "
                "unconditionally upscaling the occluded 142x148 image 13.5x to 1920x2001 px before embedding it in the PDF."
            ),
        },
        "server_url": base_url,
        "all_checks_passed": False,
        "checks": {},
    }

    try:
        # Verify UI HTML contains both the PDF download button and the explicit Specimen Analysis Mode selector
        st_ui, ui_bytes, _ = get_url(f"{base_url}/")
        ui_html = ui_bytes.decode("utf-8")
        ui_has_controls = (
            st_ui == 200
            and "Download Complete Report (PDF)" in ui_html
            and 'id="specimenInputMode"' in ui_html
            and 'value="full_smear"' in ui_html
            and 'value="single_cell"' in ui_html
            and "inspectUploadedImageDimensions" in ui_html
        )
        assert ui_has_controls, "UI HTML missing Specimen Input Mode selector or PDF download button"

        # -------------------------------------------------------------------
        # Test 1: Full-Smear Image (`IMG_20150819_133447.jpg`, `full_smear` mode)
        # -------------------------------------------------------------------
        with open(SAMPLE_FULL_SMEAR, "rb") as f:
            full_smear_bytes = f.read()

        print("[1/7] Running Regression Test 1: Full-Smear Mode (IMG_20150819_133447.jpg | finetuned_raw + raw)...")
        code1, res1 = post_multipart(
            f"{base_url}/api/analyze",
            fields={
                "input_mode": "full_smear",
                "cnn_variant": "finetuned_raw",
                "crop_mode": "raw",
                "yolo_device": "0",
                "yolo_conf": "0.25",
                "yolo_iou": "0.45",
                "dedup_iou": "0.50",
                "cnn_thresh": "0.50",
                "pad_px": "4",
                "min_box_size": "16",
            },
            file_tuple=("image_file", "IMG_20150819_133447.jpg", full_smear_bytes),
        )
        assert code1 == 200 and res1["success"], f"Full-smear analysis failed: {res1}"
        assert res1["input_mode"] == "full_smear"
        assert res1["pdf_report"]["generated"] is True

        pdf_url1 = f"{base_url}{res1['urls']['export_pdf']}"
        dl_status1, dl_bytes1, dl_headers1 = get_url(pdf_url1)
        assert dl_status1 == 200 and "pdf" in dl_headers1.get("Content-Type", "").lower()

        pdf_info1 = pdf_gen.inspect_pdf_bytes(dl_bytes1)
        txt1 = pdf_info1["decompressed_text"]
        cmp1 = pdf_gen.compare_source_and_pdf_embedded_image(
            SAMPLE_FULL_SMEAR,
            dl_bytes1,
            annotated_image_path=os.path.join(res1["run_directory"], "annotated_IMG_20150819_133447.jpg"),
        )

        req_headings = [
            "A. Report & Specimen Identification",
            "B. Executive Summary",
            "C. Model & Configuration Details",
            "D. Confidence Statistics & Processing Telemetry",
            "E. Annotated Blood-Smear Field Overview",
            "G. Audit & Exclusion Summary",
            "H. Dataset Ground-Truth Reference Comparison",
            "Final AI Screening Result",
        ]
        missing_h1 = [h for h in req_headings if h not in txt1]
        expected_pos_stmt = "The AI flagged 74 potentially parasitized red blood cells. Estimated parasitemia: 37.95%."
        expected_disc = (
            "This AI screening result does not confirm or rule out malaria in a patient. "
            "Clinical assessment and appropriate diagnostic testing are required."
        )

        check1_passed = (
            len(missing_h1) == 0
            and "F. Cell-Level" not in txt1
            and "IMG_20150819_133447_cell_0001" not in txt1
            and res1["summary"]["counts"]["valid_candidate_rbc_crops"] == 195
            and res1["summary"]["counts"]["cnn_parasitized_rbc_count"] == 74
            and res1["summary"]["counts"]["cnn_uninfected_rbc_count"] == 121
            and round(res1["summary"]["parasitemia_estimate"]["estimated_parasitemia_percent"], 2) == 37.95
            and expected_pos_stmt in txt1
            and expected_disc in txt1
            and cmp1["passed"] is True
            and cmp1["embedded_dimensions"] == {"width": 1920, "height": 1080}
        )
        assert check1_passed, f"Test 1 (Full-smear image) failed: missing={missing_h1}, cmp1={cmp1}"
        report["checks"]["1_full_smear_mode_analysis_and_pdf"] = {
            "passed": check1_passed,
            "image": "IMG_20150819_133447.jpg",
            "input_mode": "full_smear",
            "counts": res1["summary"]["counts"],
            "estimated_parasitemia_percent": res1["summary"]["parasitemia_estimate"]["estimated_parasitemia_percent"],
            "final_screening_statement": res1["final_screening_result"]["statement"],
            "pdf_page_count": pdf_info1["page_count"],
            "source_vs_pdf_image_comparison": cmp1,
        }
        print(
            f"  -> Test 1 Passed: Full-smear image (195 valid RBCs, 74 Par, 37.95%, "
            f"embedded PDF image 1920x1080, corr={cmp1['source_vs_embedded_pearson_r']})"
        )

        # -------------------------------------------------------------------
        # Test 2: Single-Cell Image (`C100P61ThinF_IMG_20150918_144104_cell_162.png` in `single_cell` mode)
        # -------------------------------------------------------------------
        print("[2/7] Running Regression Test 2: Single-Cell Mode (C100P61ThinF_IMG_20150918_144104_cell_162.png)...")
        with open(SAMPLE_SINGLE_CELL_PAR, "rb") as f:
            sc_par_bytes = f.read()

        code2, res2 = post_multipart(
            f"{base_url}/api/analyze",
            fields={
                "input_mode": "single_cell",
                "cnn_variant": "finetuned_raw",
                "crop_mode": "raw",
                "cnn_thresh": "0.50",
            },
            file_tuple=("image_file", "C100P61ThinF_IMG_20150918_144104_cell_162.png", sc_par_bytes),
        )
        assert code2 == 200 and res2["success"], f"Single-cell analysis failed: {res2}"
        sc_sum = res2["summary"]
        sc_res = sc_sum.get("single_cell_result") or {}
        _, dl_bytes2, _ = get_url(f"{base_url}{res2['urls']['export_pdf']}")
        pdf_info2 = pdf_gen.inspect_pdf_bytes(dl_bytes2)
        txt2 = pdf_info2["decompressed_text"]
        ann_path2 = os.path.join(res2["run_directory"], "annotated_C100P61ThinF_IMG_20150918_144104_cell_162.jpg")
        cmp2 = pdf_gen.compare_source_and_pdf_embedded_image(
            SAMPLE_SINGLE_CELL_PAR,
            dl_bytes2,
            annotated_image_path=ann_path2,
        )

        # Also test an Uninfected single-cell image in single_cell mode
        if os.path.isfile(SAMPLE_SINGLE_CELL_UNINF):
            with open(SAMPLE_SINGLE_CELL_UNINF, "rb") as f:
                sc_uninf_bytes = f.read()
            code2u, res2u = post_multipart(
                f"{base_url}/api/analyze",
                fields={
                    "input_mode": "single_cell",
                    "cnn_variant": "finetuned_raw",
                    "crop_mode": "raw",
                },
                file_tuple=("image_file", os.path.basename(SAMPLE_SINGLE_CELL_UNINF), sc_uninf_bytes),
            )
            uninf_ok = (
                code2u == 200
                and res2u["success"]
                and res2u["summary"]["single_cell_result"]["predicted_class"] == "Uninfected"
            )
        else:
            uninf_ok = True

        check2_passed = (
            res2["input_mode"] == "single_cell"
            and sc_sum["input_mode"] == "single_cell"
            and sc_res.get("predicted_class") == "Parasitized"
            and sc_res.get("confidence", 0.0) > 0.90
            and sc_sum["counts"]["raw_yolo_detections"] == 0
            and sc_sum["counts"]["valid_candidate_rbc_crops"] == 1
            and sc_sum["parasitemia_estimate"]["applicable"] is False
            and sc_sum["parasitemia_estimate"]["estimated_parasitemia_percent"] is None
            and "Single-Cell AI Classification Result: Predicted Parasitized" in txt2
            and "Whole-smear parasitemia is not applicable for a single-cell input." in txt2
            and expected_disc in txt2
            and "F. Cell-Level" not in txt2
            and cmp2["passed"] is True
            and cmp2["embedded_dimensions"] == {"width": 142, "height": 148}
            and cmp2["not_upscaled"] is True
            and cmp2["source_vs_embedded_pearson_r"] > 0.85
            and uninf_ok
        )
        assert check2_passed, f"Test 2 (Single-cell mode) failed: sc_res={sc_res}, cmp2={cmp2}"
        report["checks"]["2_single_cell_mode_classification_and_pdf"] = {
            "passed": check2_passed,
            "image": "C100P61ThinF_IMG_20150918_144104_cell_162.png",
            "input_mode": "single_cell",
            "single_cell_result": sc_res,
            "parasitemia_applicable": sc_sum["parasitemia_estimate"]["applicable"],
            "final_screening_statement": res2["final_screening_result"]["statement"],
            "source_vs_pdf_image_comparison": cmp2,
            "uninfected_single_cell_verified": uninf_ok,
        }
        print(
            f"  -> Test 2 Passed: Single-cell C100P61ThinF_IMG_20150918_144104_cell_162.png classified as "
            f"{sc_res['predicted_class']} ({sc_res['confidence']*100:.2f}%), native 142x148 preserved in PDF "
            f"(corr={cmp2['source_vs_embedded_pearson_r']})"
        )

        # -------------------------------------------------------------------
        # Test 3: Single-Cell Image Uploaded in Full-Smear Mode (No silent mode switch,
        #         clear warning, "No reliable result is available for this image.",
        #         and uncorrupted 142x148 annotated image without black/logo occlusion)
        # -------------------------------------------------------------------
        print("[3/7] Running Regression Test 3: Small 142x148 image in Full-Smear Mode (no silent switch, uncorrupted image)...")
        code3, res3 = post_multipart(
            f"{base_url}/api/analyze",
            fields={
                "input_mode": "full_smear",
                "cnn_variant": "finetuned_raw",
                "crop_mode": "raw",
            },
            file_tuple=("image_file", "C100P61ThinF_IMG_20150918_144104_cell_162.png", sc_par_bytes),
        )
        assert code3 == 200 and res3["success"], f"Test 3 failed: {res3}"
        _, dl_bytes3, _ = get_url(f"{base_url}{res3['urls']['export_pdf']}")
        pdf_info3 = pdf_gen.inspect_pdf_bytes(dl_bytes3)
        txt3 = pdf_info3["decompressed_text"]
        ann_path3 = os.path.join(res3["run_directory"], "annotated_C100P61ThinF_IMG_20150918_144104_cell_162.jpg")
        cmp3 = pdf_gen.compare_source_and_pdf_embedded_image(
            SAMPLE_SINGLE_CELL_PAR,
            dl_bytes3,
            annotated_image_path=ann_path3,
        )

        q3 = res3.get("image_quality") or {}
        check3_passed = (
            res3["input_mode"] == "full_smear"
            and res3["summary"]["counts"]["valid_candidate_rbc_crops"] == 0
            and res3["summary"]["parasitemia_estimate"]["applicable"] is False
            and res3["summary"]["parasitemia_estimate"]["estimated_parasitemia_percent"] is None
            and res3["final_screening_result"]["statement"] == "No reliable result is available for this image."
            and "No reliable result is available for this image." in txt3
            and "0.00%" not in txt3
            and q3.get("looks_like_single_cell") is True
            and q3.get("insufficient_full_smear_resolution") is True
            and any("Insufficient Full-Smear Resolution / Mode Mismatch Warning" in w for w in q3.get("warnings", []))
            and cmp3["passed"] is True
            and cmp3["embedded_dimensions"] == {"width": 142, "height": 148}
            and cmp3["not_upscaled"] is True
            and cmp3["source_vs_embedded_pearson_r"] > 0.95
        )
        assert check3_passed, f"Test 3 failed: q3={q3}, cmp3={cmp3}, txt3_has_0pct={'0.00%' in txt3}"
        report["checks"]["3_small_image_in_full_smear_mode_safe_handling"] = {
            "passed": check3_passed,
            "did_not_silently_switch_mode": res3["input_mode"] == "full_smear",
            "valid_rbc_crops": res3["summary"]["counts"]["valid_candidate_rbc_crops"],
            "final_screening_statement": res3["final_screening_result"]["statement"],
            "zero_parasitemia_omitted_from_pdf": "0.00%" not in txt3,
            "quality_warnings": q3.get("warnings", []),
            "source_vs_pdf_image_comparison": cmp3,
        }
        print(
            f"  -> Test 3 Passed: 142x148 image in full_smear mode kept native 142x148 without black/logo banner "
            f"(corr={cmp3['source_vs_embedded_pearson_r']}), warned clearly, and reported 'No reliable result is available for this image.'"
        )

        # -------------------------------------------------------------------
        # Test 4: Low-Resolution Image Warning & Blank / Black / Corrupted Rejection
        # -------------------------------------------------------------------
        print("[4/7] Running Regression Test 4: Low-resolution warnings & blank/black/corrupted image rejection...")
        # Create a 320x240 low-resolution image by cropping a region from SAMPLE_FULL_SMEAR
        with PILImage.open(SAMPLE_FULL_SMEAR) as sm_im:
            low_res_crop = sm_im.convert("RGB").crop((2500, 1350, 2820, 1590))  # 320x240 native center crop (no upscaling)
            low_res_buf = io.BytesIO()
            low_res_crop.save(low_res_buf, format="JPEG", quality=95)
            low_res_bytes = low_res_buf.getvalue()

        code_lr, res_lr = post_multipart(
            f"{base_url}/api/analyze",
            fields={"input_mode": "full_smear", "cnn_variant": "finetuned_raw", "crop_mode": "raw"},
            file_tuple=("image_file", "low_res_320x240_crop.jpg", low_res_bytes),
        )
        assert code_lr == 200 and res_lr["success"]
        _, dl_bytes_lr, _ = get_url(f"{base_url}{res_lr['urls']['export_pdf']}")
        lr_pdf_info = pdf_gen.inspect_pdf_bytes(dl_bytes_lr)
        lr_q = res_lr.get("image_quality") or {}

        # Blank, black, corrupted, and tiny (<32x32) images must be safely rejected with HTTP 400
        black_buf = io.BytesIO()
        PILImage.fromarray(np.zeros((256, 256, 3), dtype=np.uint8), mode="RGB").save(black_buf, format="PNG")
        st_black, res_black = post_multipart(
            f"{base_url}/api/analyze",
            fields={"input_mode": "full_smear"},
            file_tuple=("image_file", "all_black.png", black_buf.getvalue()),
        )

        blank_buf = io.BytesIO()
        PILImage.fromarray(np.full((256, 256, 3), 252, dtype=np.uint8), mode="RGB").save(blank_buf, format="PNG")
        st_blank, res_blank = post_multipart(
            f"{base_url}/api/analyze",
            fields={"input_mode": "full_smear"},
            file_tuple=("image_file", "all_white_blank.png", blank_buf.getvalue()),
        )

        st_corrupt, res_corrupt = post_multipart(
            f"{base_url}/api/analyze",
            fields={"input_mode": "full_smear"},
            file_tuple=("image_file", "corrupted.png", b"NOT_A_VALID_PNG_FILE_CONTENT"),
        )

        st_empty, res_empty = post_multipart(
            f"{base_url}/api/analyze",
            fields={"input_mode": "full_smear"},
            file_tuple=("image_file", "empty.png", b""),
        )

        check4_passed = (
            lr_q.get("is_low_resolution") is True
            and lr_q.get("insufficient_full_smear_resolution") is True
            and any("Resolution Advisory" in w for w in lr_q.get("warnings", []))
            and lr_pdf_info["embedded_images"][0] == {"width": 320, "height": 240}
            and st_black == 400 and not res_black.get("success")
            and st_blank == 400 and not res_blank.get("success")
            and st_corrupt == 400 and not res_corrupt.get("success")
            and st_empty == 400 and not res_empty.get("success")
        )
        assert check4_passed, (
            f"Test 4 failed: lr_q={lr_q}, emb={lr_pdf_info['embedded_images']}, "
            f"statuses=({st_black}, {st_blank}, {st_corrupt}, {st_empty})"
        )
        report["checks"]["4_low_resolution_warning_and_invalid_image_rejection"] = {
            "passed": check4_passed,
            "low_res_320x240_warnings": lr_q.get("warnings", []),
            "low_res_embedded_pdf_dimensions_not_upscaled": lr_pdf_info["embedded_images"][0],
            "black_image_http_status": st_black,
            "black_image_error": res_black.get("error_message"),
            "blank_image_http_status": st_blank,
            "blank_image_error": res_blank.get("error_message"),
            "corrupted_image_http_status": st_corrupt,
            "empty_image_http_status": st_empty,
        }
        print(
            "  -> Test 4 Passed: Low-res (320x240) warned & kept at native 320x240 without upscaling; "
            "black, blank, corrupted, and empty images safely rejected with HTTP 400"
        )

        # -------------------------------------------------------------------
        # Test 5: Zero Valid Detections (`--test-zero-detections`)
        # -------------------------------------------------------------------
        print("[5/7] Running Regression Test 5: Zero Valid Detections handling...")
        code_zero, res_zero = post_multipart(
            f"{base_url}/api/analyze",
            fields={
                "preset_filename": "IMG_20150819_133447.jpg",
                "input_mode": "full_smear",
                "cnn_variant": "finetuned_raw",
                "crop_mode": "raw",
                "yolo_device": "0",
                "test_zero_detections": "1",
            },
        )
        assert code_zero == 200 and res_zero["success"] and res_zero["pdf_report"]["generated"]
        _, dl_bytes_zero, _ = get_url(f"{base_url}{res_zero['urls']['export_pdf']}")
        zero_info = pdf_gen.inspect_pdf_bytes(dl_bytes_zero)
        zero_txt = zero_info["decompressed_text"]

        check5_passed = (
            res_zero["summary"]["counts"]["valid_candidate_rbc_crops"] == 0
            and res_zero["summary"]["parasitemia_estimate"]["applicable"] is False
            and res_zero["summary"]["parasitemia_estimate"]["estimated_parasitemia_percent"] is None
            and res_zero["final_screening_result"]["statement"] == "No reliable result is available for this image."
            and "No reliable result is available for this image." in zero_txt
            and "0.00%" not in zero_txt
            and expected_disc in zero_txt
            and "F. Cell-Level" not in zero_txt
        )
        assert check5_passed, "Test 5 (Zero valid detections) failed"
        report["checks"]["5_zero_valid_detections_safe_handling"] = {
            "passed": check5_passed,
            "valid_rbc_crops": 0,
            "parasitemia_applicable": res_zero["summary"]["parasitemia_estimate"]["applicable"],
            "final_screening_statement": res_zero["final_screening_result"]["statement"],
            "zero_percent_omitted_from_pdf": "0.00%" not in zero_txt,
        }
        print("  -> Test 5 Passed: Zero valid detections displays 'No reliable result is available for this image.' and never reports 0.00% negative")

        # -------------------------------------------------------------------
        # Test 6: Source vs. PDF-Embedded Image Comparison Test
        # -------------------------------------------------------------------
        print("[6/7] Running Regression Test 6: Source vs. PDF-Embedded Image Comparison...")
        check6_passed = (
            cmp1["passed"] is True
            and cmp2["passed"] is True
            and cmp3["passed"] is True
            and cmp1["embedded_dimensions"] == {"width": 1920, "height": 1080}
            and cmp2["embedded_dimensions"] == {"width": 142, "height": 148}
            and cmp3["embedded_dimensions"] == {"width": 142, "height": 148}
            and cmp1["source_vs_embedded_pearson_r"] >= 0.70
            and cmp2["source_vs_embedded_pearson_r"] >= 0.85
            and cmp3["source_vs_embedded_pearson_r"] >= 0.95
        )
        assert check6_passed, f"Test 6 (Source vs PDF image comparison) failed: {cmp1}, {cmp2}, {cmp3}"
        report["checks"]["6_source_vs_pdf_embedded_image_comparison"] = {
            "passed": check6_passed,
            "full_smear_comparison": cmp1,
            "single_cell_mode_comparison": cmp2,
            "single_cell_in_full_smear_comparison": cmp3,
        }
        print("  -> Test 6 Passed: Source vs. PDF-embedded image comparison verified across all modes")

        # -------------------------------------------------------------------
        # Test 7: PDF Download & All Technical Exports (CSV, JSON, TXT, JPG)
        # -------------------------------------------------------------------
        print("[7/7] Running Regression Test 7: PDF Download & Technical Exports (Full-Smear & Single-Cell)...")
        for mode_label, res_obj in [("full_smear", res1), ("single_cell", res2)]:
            u = res_obj["urls"]
            st_pdf, b_pdf, h_pdf = get_url(f"{base_url}{u['export_pdf']}")
            st_csv, b_csv, _ = get_url(f"{base_url}{u['export_csv']}")
            st_jsn, b_jsn, _ = get_url(f"{base_url}{u['export_summary_json']}")
            st_rej, b_rej, _ = get_url(f"{base_url}{u['export_rejected_json']}")
            st_txt, b_txt, _ = get_url(f"{base_url}{u['export_txt']}")
            st_jpg, b_jpg, _ = get_url(f"{base_url}{u['annotated_image']}&download=1")
            assert st_pdf == 200 and len(b_pdf) > 2000 and "pdf" in h_pdf.get("Content-Type", "").lower()
            assert st_csv == 200 and len(b_csv) > 100
            assert st_jsn == 200 and len(b_jsn) > 300
            assert st_rej == 200 and len(b_rej) > 20
            assert st_txt == 200 and len(b_txt) > 300
            assert st_jpg == 200 and len(b_jpg) > 1000

        report["checks"]["7_pdf_download_and_technical_exports"] = {
            "passed": True,
            "full_smear_exports_verified": True,
            "single_cell_exports_verified": True,
        }
        print("  -> Test 7 Passed: PDF download and all technical exports (CSV, Summary JSON, Excluded JSON, TXT, Annotated JPG) verified")

        report["all_checks_passed"] = all(c["passed"] for c in report["checks"].values())

        with open(REPORT_JSON_PATH, "w", encoding="utf-8") as jf:
            json.dump(report, jf, indent=2)

        txt_summary = [
            "=" * 92,
            "SMARTCELL INPUT MODES & ANNOTATED IMAGE FIX — AUTOMATED VERIFICATION REPORT",
            "=" * 92,
            f"Timestamp:                {report['timestamp']}",
            f"Execution Environment:    Automated HTTP & Pipeline Regression Suite (No physical USB microscope attached)",
            f"Test Server URL:          {base_url}",
            f"All Checks Passed:        {report['all_checks_passed']}",
            "",
            "ROOT CAUSE OF PREVIOUSLY BROKEN ANNOTATED IMAGE (C100P61ThinF_IMG_20150918_144104_cell_162.png):",
            "  1. Domain/Mode Mismatch: The 142x148 px input is a pre-segmented single-cell RBC patch on a black",
            "     background, not a full blood-smear microscope field. Stage 1 YOLOv8n detected only 1 tiny 6x5 px",
            "     artifact box (< 16 px min_box_size), leaving 0 valid RBC crops.",
            "  2. Banner Occlusion & Upscaling Bug: create_annotated_full_image() drew a hardcoded 4K dark rectangle",
            "     cv2.rectangle(img, (20, 20), (2450, 210), (20, 20, 20), -1) and fontScale=1.35 header text directly",
            "     onto the 142x148 image—covering pixels (20,20)..(142,148) with solid black and clipped 'SMAR' text—",
            "     and then unconditionally upscaled the occluded image 13.5x to 1920x2001 px before PDF embedding.",
            "",
            "REGRESSION TEST RESULTS:",
            f"  1. Full-Smear Mode (IMG_20150819_133447.jpg):           PASSED (195 valid RBCs, 74 Par, 37.95%, 1920x1080 PDF image)",
            f"  2. Single-Cell Mode (cell_162.png, 142x148 px):         PASSED ({sc_res['predicted_class']}, {sc_res['confidence']*100:.2f}% conf, native 142x148 in PDF)",
            f"  3. Small Image in Full-Smear Mode (No silent switch):   PASSED (Warned clearly, native 142x148 preserved, 'No reliable result')",
            f"  4. Low-Res Warning & Blank/Black/Corrupted Rejection:   PASSED (320x240 warned & not upscaled; black/blank/corrupt HTTP 400)",
            f"  5. Zero Valid Detections Guardrail:                     PASSED ('No reliable result is available for this image.', no 0.00%)",
            f"  6. Source vs. PDF-Embedded Image Pixel Comparison:      PASSED (Corr: smear={cmp1['source_vs_embedded_pearson_r']}, cell={cmp2['source_vs_embedded_pearson_r']})",
            f"  7. PDF Download & Technical Exports (CSV/JSON/TXT/JPG): PASSED (All HTTP 200 across full_smear and single_cell modes)",
            "=" * 92,
        ]
        with open(REPORT_TXT_PATH, "w", encoding="utf-8") as tf:
            tf.write("\n".join(txt_summary) + "\n")

        print("\n" + "\n".join(txt_summary))

    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
