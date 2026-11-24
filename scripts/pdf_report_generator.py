r"""
pdf_report_generator.py
-----------------------
SmartCell Enhancement: Complete, Professional, Human-Readable PDF Analysis Report Generator
(Sections A–E, G, H, and Final AI Screening Result — Section F Removed).

Generates a concise 2-page PDF report (`SmartCell_Report_<stem>.pdf`) using ReportLab
for every completed blood-smear analysis run. Includes:
  - A. Report & Specimen Identification
  - B. Executive Summary
  - C. Model & Configuration Details
  - D. Confidence Statistics & Processing Telemetry
  - E. Annotated Blood-Smear Field Overview
  - G. Audit & Exclusion Summary
  - H. Dataset Ground-Truth Reference Comparison (only when valid associated annotations exist)
  - Final AI Screening Result (with mandatory clinical disclaimer)

Section F (Cell-Level Detection & Classification Results table) is intentionally omitted
from the PDF so the report remains concise (2–3 pages); per-cell rows remain available via
the Per-Cell CSV export.
"""

import os
import csv
import json
import argparse
from datetime import datetime
from xml.sax.saxutils import escape as xml_escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
)
from reportlab.pdfgen import canvas


PROTOTYPE_NOTICE_TEXT = (
    "SmartCell is an experimental AI research prototype for blood-smear erythrocyte localization "
    "and malaria screening. Algorithmic cell counts and parasitemia estimates are for research "
    "evaluation only and do not constitute a confirmed clinical diagnosis."
)

FINAL_SCREENING_DISCLAIMER = (
    "This AI screening result does not confirm or rule out malaria in a patient. "
    "Clinical assessment and appropriate diagnostic testing are required."
)

FINAL_RESULT_ZERO_PARASITIZED = (
    "The AI did not flag any potentially parasitized red blood cells in this image."
)

FINAL_RESULT_NO_RELIABLE = (
    "No reliable result is available for this image."
)


def _safe_str(val, default="N/A"):
    """Safely converts any value to an XML-escaped string for ReportLab Paragraphs."""
    if val is None or val == "":
        return xml_escape(str(default))
    return xml_escape(str(val))


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val, default=0):
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def format_final_screening_statement(
    valid_rbcs,
    par_rbcs,
    est_par_pct,
    analysis_failed=False,
    input_mode="full_smear",
    single_cell_result=None,
):
    """
    Returns the exact Final AI Screening Result statement required by the specification:
      - In single_cell mode:
        'Single-Cell AI Classification Result: Predicted [Class] (Confidence: [C]%). Whole-smear parasitemia is not applicable for a single-cell input.'
      - In full_smear mode:
        - If >= 1 RBC is predicted as parasitized (and valid_rbcs > 0):
          'The AI flagged [N] potentially parasitized red blood cells. Estimated parasitemia: [P]%.'
        - If 0 RBCs are predicted as parasitized (and valid_rbcs > 0):
          'The AI did not flag any potentially parasitized red blood cells in this image.'
        - If 0 valid RBCs were classified or analysis failed:
          'No reliable result is available for this image.'
    """
    if analysis_failed:
        return FINAL_RESULT_NO_RELIABLE

    if str(input_mode).lower() == "single_cell":
        if not isinstance(single_cell_result, dict) or not single_cell_result.get("predicted_class"):
            return FINAL_RESULT_NO_RELIABLE
        pred_cls = str(single_cell_result.get("predicted_class"))
        conf_pct = _safe_float(
            single_cell_result.get("confidence_percent"),
            _safe_float(single_cell_result.get("confidence"), 0.0) * 100.0,
        )
        return (
            f"Single-Cell AI Classification Result: Predicted {pred_cls} "
            f"(Confidence: {conf_pct:.2f}%). Whole-smear parasitemia is not applicable for a single-cell input."
        )

    valid_n = _safe_int(valid_rbcs, 0)
    par_n = _safe_int(par_rbcs, 0)
    pct_val = _safe_float(est_par_pct, 0.0)

    if valid_n <= 0:
        return FINAL_RESULT_NO_RELIABLE
    if par_n >= 1:
        return f"The AI flagged {par_n} potentially parasitized red blood cells. Estimated parasitemia: {pct_val:.2f}%."
    return FINAL_RESULT_ZERO_PARASITIZED


def extract_embedded_images_from_pdf(pdf_bytes_or_path):
    """
    Extracts and decodes all `/Subtype /Image` XObject streams from a ReportLab-generated PDF.
    Returns a list of dicts containing `width`, `height`, `pil_image`, and `raw_bytes`.
    """
    import io
    import re
    import zlib
    from reportlab.pdfbase.pdfutils import asciiBase85Decode

    if isinstance(pdf_bytes_or_path, (bytes, bytearray)):
        raw = bytes(pdf_bytes_or_path)
    else:
        with open(pdf_bytes_or_path, "rb") as f:
            raw = f.read()

    extracted = []
    for m in re.finditer(rb"<<([^>]*?/Subtype\s*/Image[^>]*?)>>\s*stream\r?\n", raw, flags=re.DOTALL):
        hdr = m.group(1)
        s_start = m.end()
        s_end = raw.find(b"endstream", s_start)
        if s_end == -1:
            continue
        body = raw[s_start:s_end].strip()

        w_m = re.search(rb"/Width\s+(\d+)", hdr)
        h_m = re.search(rb"/Height\s+(\d+)", hdr)
        ew = int(w_m.group(1)) if w_m else 0
        eh = int(h_m.group(1)) if h_m else 0

        decoded_bytes = body
        if b"ASCII85Decode" in hdr:
            try:
                decoded_bytes = asciiBase85Decode(decoded_bytes)
            except Exception:
                pass

        pil_im = None
        if b"DCTDecode" in hdr:
            try:
                with PILImage.open(io.BytesIO(decoded_bytes)) as im:
                    pil_im = im.convert("RGB")
            except Exception:
                pil_im = None
        elif b"FlateDecode" in hdr and ew > 0 and eh > 0:
            try:
                raw_pixels = zlib.decompress(decoded_bytes)
                pil_im = PILImage.frombytes("RGB", (ew, eh), raw_pixels)
            except Exception:
                pil_im = None

        extracted.append({
            "width": ew,
            "height": eh,
            "pil_image": pil_im,
            "byte_length": len(decoded_bytes),
        })
    return extracted


def compare_source_and_pdf_embedded_image(source_image_path, pdf_bytes_or_path, annotated_image_path=None):
    """
    Verifies that the image embedded in the generated PDF corresponds directly to the
    current run's source image (and annotated image), preserves aspect ratio, and is
    not a black/occluded logo, stale image, or unrelated preview.
    """
    import numpy as np

    if not source_image_path or not os.path.isfile(source_image_path):
        raise FileNotFoundError(f"Source image not found: {source_image_path}")

    with PILImage.open(source_image_path) as sim:
        src_rgb = sim.convert("RGB")
        sw, sh = src_rgb.size

    embedded_list = extract_embedded_images_from_pdf(pdf_bytes_or_path)
    if not embedded_list or embedded_list[0].get("pil_image") is None:
        return {
            "passed": False,
            "reason": "No decodable embedded image found in PDF.",
            "embedded_count": len(embedded_list),
        }

    emb_info = embedded_list[0]
    emb_rgb = emb_info["pil_image"]
    ew, eh = emb_rgb.size

    src_aspect = sw / float(sh)
    emb_aspect = ew / float(eh)
    aspect_diff = abs(src_aspect - emb_aspect)
    aspect_preserved = aspect_diff <= 0.015

    not_upscaled = (ew <= sw and eh <= sh)

    # Compare against annotated_image_path if provided (should be 100% identical to embedded XObject)
    matches_annotated_file = True
    annotated_mae = 0.0
    if annotated_image_path and os.path.isfile(annotated_image_path):
        with PILImage.open(annotated_image_path) as aim:
            ann_rgb = aim.convert("RGB")
            if ann_rgb.size != emb_rgb.size:
                matches_annotated_file = False
            else:
                ann_np = np.asarray(ann_rgb, dtype=np.float32)
                emb_np_full = np.asarray(emb_rgb, dtype=np.float32)
                annotated_mae = round(float(np.mean(np.abs(ann_np - emb_np_full))), 4)
                matches_annotated_file = annotated_mae < 1.0

    # Compare interior region of source image resized to (ew, eh) vs PDF embedded image
    src_resized = src_rgb.resize((ew, eh), PILImage.Resampling.BILINEAR) if (sw, sh) != (ew, eh) else src_rgb
    src_np = np.asarray(src_resized, dtype=np.float32)
    emb_np = np.asarray(emb_rgb, dtype=np.float32)

    # Exclude thin outer 3px perimeter border (or top 9% banner on >=1600x900 full-smear frames)
    y_start = int(eh * 0.10) if (sw >= 1600 and sh >= 900) else min(3, eh // 6)
    margin = min(3, min(ew, eh) // 6)
    src_roi = src_np[y_start : eh - margin, margin : ew - margin]
    emb_roi = emb_np[y_start : eh - margin, margin : ew - margin]

    roi_mae = round(float(np.mean(np.abs(src_roi - emb_roi))), 3)
    src_flat = src_roi.flatten() - float(np.mean(src_roi))
    emb_flat = emb_roi.flatten() - float(np.mean(emb_roi))
    denom = float(np.linalg.norm(src_flat) * np.linalg.norm(emb_flat))
    pearson_r = round(float(np.dot(src_flat, emb_flat) / denom), 5) if denom > 1e-6 else 1.0

    # Verify the embedded image is NOT covered by a black box where source has non-black cell pixels
    src_nonblack_mask = np.mean(src_roi, axis=2) > 25.0
    if np.any(src_nonblack_mask):
        emb_on_nonblack = np.mean(emb_roi, axis=2)[src_nonblack_mask]
        black_occluded_fraction = round(float(np.mean(emb_on_nonblack < 22.0)), 4)
    else:
        black_occluded_fraction = 0.0

    passed = bool(
        aspect_preserved
        and not_upscaled
        and matches_annotated_file
        and pearson_r >= 0.85
        and black_occluded_fraction < 0.05
    )

    return {
        "passed": passed,
        "source_dimensions": {"width": sw, "height": sh},
        "embedded_dimensions": {"width": ew, "height": eh},
        "aspect_ratio_source": round(src_aspect, 4),
        "aspect_ratio_embedded": round(emb_aspect, 4),
        "aspect_ratio_preserved": aspect_preserved,
        "not_upscaled": not_upscaled,
        "matches_current_run_annotated_image": matches_annotated_file,
        "annotated_vs_embedded_mae": annotated_mae,
        "source_vs_embedded_roi_mae": roi_mae,
        "source_vs_embedded_pearson_r": pearson_r,
        "black_occluded_fraction_on_cell": black_occluded_fraction,
    }


def inspect_pdf_bytes(pdf_bytes_or_path):
    """
    Decompresses and inspects a ReportLab-generated PDF file or byte buffer.
    Returns a dictionary with page_count, has_embedded_image, image_dimensions,
    and the concatenated decompressed content streams for verification testing.
    """
    import re
    import zlib
    from reportlab.pdfbase.pdfutils import asciiBase85Decode

    if isinstance(pdf_bytes_or_path, (bytes, bytearray)):
        raw = bytes(pdf_bytes_or_path)
    else:
        with open(pdf_bytes_or_path, "rb") as f:
            raw = f.read()

    is_valid_pdf = raw.startswith(b"%PDF-") and b"%%EOF" in raw[-1024:]
    page_count = len(re.findall(rb"/Type\s*/Page\b", raw))
    has_image = b"/Subtype /Image" in raw

    img_dims = []
    for m in re.finditer(
        rb"/Height\s+(\d+)\s+/Length\s+\d+\s+/Subtype\s*/Image\s*/Type\s*/XObject\s*/Width\s+(\d+)",
        raw,
    ):
        img_dims.append({"width": int(m.group(2)), "height": int(m.group(1))})

    decompressed_streams = []
    pos = 0
    while True:
        idx = raw.find(b"FlateDecode", pos)
        if idx == -1:
            break
        s_kw = raw.find(b"stream", idx)
        e_kw = raw.find(b"endstream", s_kw)
        if s_kw == -1 or e_kw == -1:
            break
        body = raw[s_kw + 6 : e_kw].strip()
        try:
            b85 = asciiBase85Decode(body)
            dec = zlib.decompress(b85).decode("latin1", errors="ignore")
            decompressed_streams.append(dec)
        except Exception:
            try:
                dec = zlib.decompress(body).decode("latin1", errors="ignore")
                decompressed_streams.append(dec)
            except Exception:
                pass
        pos = e_kw + 9

    raw_decompressed = "\n".join(decompressed_streams)
    literals = []
    for m in re.finditer(r"\(((?:\\.|[^\\()])*)\)", raw_decompressed):
        tok = m.group(1).replace(r"\(", "(").replace(r"\)", ")").replace(r"\\", "\\")
        literals.append(tok)
    plain_text = " ".join(literals)
    unwrapped_text = "".join(literals)

    return {
        "is_valid_pdf": is_valid_pdf,
        "byte_size": len(raw),
        "page_count": page_count,
        "has_embedded_image": has_image,
        "embedded_images": img_dims,
        "decoded_stream_count": len(decompressed_streams),
        "decompressed_text": raw_decompressed + "\n" + plain_text + "\n" + unwrapped_text,
    }


class NumberedReportCanvas(canvas.Canvas):
    """
    Custom ReportLab canvas that draws a consistent header bar and 'Page X of Y'
    footer with research disclaimer on every page of the generated PDF.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def _draw_page_decorations(self, page_count):
        page_w, page_h = A4
        margin_x = 14 * mm

        self.saveState()
        # Top subtle accent bar
        self.setStrokeColor(colors.HexColor("#1e293b"))
        self.setLineWidth(0.6)
        self.line(margin_x, page_h - 12 * mm, page_w - margin_x, page_h - 12 * mm)

        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#475569"))
        self.drawString(
            margin_x,
            page_h - 10.5 * mm,
            "SMARTCELL — AI-ASSISTED BLOOD SMEAR SCREENING PROTOTYPE (RESEARCH REPORT)"
        )
        self.setFont("Helvetica", 7.5)
        self.drawRightString(
            page_w - margin_x,
            page_h - 10.5 * mm,
            "NOT A CLINICAL DIAGNOSTIC DEVICE"
        )

        # Bottom footer line and page numbering
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(margin_x, 14 * mm, page_w - margin_x, 14 * mm)

        self.setFont("Helvetica-Oblique", 7.0)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawString(
            margin_x,
            10 * mm,
            "SmartCell Research Prototype — Algorithmic Screening Output Only (Requires Clinical Validation)"
        )

        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#334155"))
        self.drawRightString(
            page_w - margin_x,
            10 * mm,
            f"Page {self._pageNumber} of {page_count}"
        )
        self.restoreState()


def _build_styles():
    base = getSampleStyleSheet()
    styles = {}

    styles["Title"] = ParagraphStyle(
        "SmartCellTitle",
        parent=base["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=2,
    )
    styles["Subtitle"] = ParagraphStyle(
        "SmartCellSubtitle",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=9.0,
        leading=12,
        textColor=colors.HexColor("#475569"),
        spaceAfter=6,
    )
    styles["BannerDisclaimer"] = ParagraphStyle(
        "BannerDisclaimer",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.0,
        leading=11,
        textColor=colors.HexColor("#7f1d1d"),
        alignment=TA_LEFT,
    )
    styles["SectionHeading"] = ParagraphStyle(
        "SectionHeading",
        parent=base["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11.0,
        leading=14,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=9,
        spaceAfter=4,
    )
    styles["Body"] = ParagraphStyle(
        "SmartCellBody",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=4,
    )
    styles["BodySmall"] = ParagraphStyle(
        "SmartCellBodySmall",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=10.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=3,
    )
    styles["Caption"] = ParagraphStyle(
        "SmartCellCaption",
        parent=base["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.0,
        leading=11,
        textColor=colors.HexColor("#475569"),
        alignment=TA_CENTER,
        spaceBefore=4,
        spaceAfter=5,
    )
    styles["TableHeader"] = ParagraphStyle(
        "SmartCellTableHeader",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.0,
        leading=10,
        textColor=colors.white,
    )
    styles["TableCell"] = ParagraphStyle(
        "SmartCellTableCell",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=10,
        textColor=colors.HexColor("#1e293b"),
    )
    styles["TableCellBold"] = ParagraphStyle(
        "SmartCellTableCellBold",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.8,
        leading=10,
        textColor=colors.HexColor("#0f172a"),
    )
    styles["TableCellPar"] = ParagraphStyle(
        "SmartCellTableCellPar",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.8,
        leading=10,
        textColor=colors.HexColor("#b91c1c"),
    )
    styles["TableCellUninf"] = ParagraphStyle(
        "SmartCellTableCellUninf",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.8,
        leading=10,
        textColor=colors.HexColor("#15803d"),
    )
    styles["FinalScreeningStatement"] = ParagraphStyle(
        "FinalScreeningStatement",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.2,
        leading=14,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4,
    )
    styles["FinalScreeningDisclaimer"] = ParagraphStyle(
        "FinalScreeningDisclaimer",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=8.2,
        leading=11.5,
        textColor=colors.HexColor("#334155"),
    )
    return styles


def _make_kv_table(rows_data, styles, col_widths):
    """Builds a clean 2-column or 4-column key-value metadata table."""
    formatted_rows = []
    for row in rows_data:
        f_row = []
        for idx, cell_text in enumerate(row):
            st = styles["TableCellBold"] if idx % 2 == 0 else styles["TableCell"]
            f_row.append(Paragraph(str(cell_text), st))
        formatted_rows.append(f_row)

    tbl = Table(formatted_rows, colWidths=col_widths)
    ts = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
    ]
    for r_idx in range(len(formatted_rows)):
        bg = colors.HexColor("#f8fafc") if r_idx % 2 == 0 else colors.white
        ts.append(("BACKGROUND", (0, r_idx), (-1, r_idx), bg))
        ts.append(("BACKGROUND", (0, r_idx), (0, r_idx), colors.HexColor("#f1f5f9")))
        if len(rows_data[r_idx]) == 4:
            ts.append(("BACKGROUND", (2, r_idx), (2, r_idx), colors.HexColor("#f1f5f9")))

    tbl.setStyle(TableStyle(ts))
    return tbl


def generate_pdf_report(
    output_pdf_path,
    summary_data,
    cell_predictions=None,
    rejected_and_excluded=None,
    annotated_image_path=None,
    ground_truth_reference=None,
    run_id=None,
    generated_at=None,
):
    """
    Generates a concise PDF analysis report (`output_pdf_path`) from the current
    SmartCell blood-smear analysis results containing:
      - A. Report & Specimen Identification
      - B. Executive Summary
      - C. Model & Configuration Details
      - D. Confidence Statistics & Processing Telemetry
      - E. Annotated Blood-Smear Field Overview
      - G. Audit & Exclusion Summary
      - H. Dataset Ground-Truth Reference Comparison (only when valid annotations exist)
      - Final AI Screening Result

    Section F (Cell-Level Detection & Classification Results) and the individual per-cell
    rows table are completely omitted from the PDF.
    """
    if not isinstance(summary_data, dict) or not summary_data:
        raise ValueError("summary_data must be a non-empty dictionary.")

    if not output_pdf_path:
        raise ValueError("output_pdf_path must be specified.")

    abs_pdf_path = os.path.abspath(output_pdf_path)
    out_dir = os.path.dirname(abs_pdf_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cell_predictions = cell_predictions if isinstance(cell_predictions, list) else []
    rejected_and_excluded = rejected_and_excluded if isinstance(rejected_and_excluded, dict) else {}
    ground_truth_reference = ground_truth_reference if isinstance(ground_truth_reference, dict) else {}

    if not generated_at:
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    page_w, page_h = A4
    margin_lr = 14 * mm
    margin_tb = 16 * mm
    usable_w = page_w - 2 * margin_lr

    styles = _build_styles()
    story = []

    # Extract sub-dictionaries safely with defaults for missing optional metadata
    img_fn = summary_data.get("image_filename", "Unknown_Image")
    dims = summary_data.get("image_dimensions") or {}
    img_w = dims.get("width", "N/A")
    img_h = dims.get("height", "N/A")
    dims_str = f"{img_w} x {img_h} px" if (img_w != "N/A" and img_h != "N/A") else "N/A"

    if not run_id:
        out_files = summary_data.get("output_files") or {}
        out_dir_meta = out_files.get("output_directory", "")
        run_id = os.path.basename(out_dir_meta.rstrip("\\/")) if out_dir_meta else "N/A"

    input_mode = str(summary_data.get("input_mode", "full_smear")).strip().lower()
    is_single_cell = input_mode == "single_cell"
    single_cell_result = summary_data.get("single_cell_result") or {}
    analysis_failed = bool(summary_data.get("analysis_failed", False))

    counts = summary_data.get("counts") or {}
    raw_yolo_det = _safe_int(counts.get("raw_yolo_detections"), 0)
    valid_rbcs = _safe_int(counts.get("valid_candidate_rbc_crops"), len(cell_predictions))
    par_rbcs = _safe_int(counts.get("cnn_parasitized_rbc_count"), 0)
    uninf_rbcs = _safe_int(counts.get("cnn_uninfected_rbc_count"), 0)
    excl_wbcs = _safe_int(counts.get("excluded_wbc_detections"), 0)
    rej_boxes = _safe_int(counts.get("rejected_or_duplicate_boxes"), 0)

    par_est = summary_data.get("parasitemia_estimate") or {}
    par_applicable = bool(par_est.get("applicable", not is_single_cell and valid_rbcs > 0))
    est_par_pct = _safe_float(par_est.get("estimated_parasitemia_percent"), 0.0) if par_applicable else None
    calc_frac = par_est.get(
        "calculation_fraction",
        (
            "Single-cell classification only — whole-smear parasitemia not applicable"
            if is_single_cell
            else (
                f"{par_rbcs} / {valid_rbcs} classified RBCs"
                if valid_rbcs > 0
                else "No reliable result is available for this image (0 valid RBC crops)"
            )
        ),
    )
    raw_par_pct = _safe_float(par_est.get("raw_crop_parasitemia_percent"), est_par_pct or 0.0)
    mask_par_pct = _safe_float(par_est.get("ellipse_masked_parasitemia_percent"), est_par_pct or 0.0)

    times = summary_data.get("processing_time_seconds") or {}
    s1_time = _safe_float(times.get("stage1_yolo_detection_and_cropping"), 0.0)
    s2_time = _safe_float(times.get("stage2_cnn_classification"), 0.0)
    tot_time = _safe_float(times.get("total_pipeline_seconds"), s1_time + s2_time)

    models_meta = summary_data.get("models") or {}
    cnn_variant = models_meta.get("stage2_cnn_variant", "N/A")
    cnn_ckpt_full = models_meta.get("stage2_cnn_checkpoint", "N/A")
    cnn_ckpt_name = os.path.basename(cnn_ckpt_full) if cnn_ckpt_full and cnn_ckpt_full != "N/A" else "N/A"
    yolo_ckpt_full = models_meta.get("stage1_yolo_checkpoint", "N/A")
    yolo_ckpt_name = (
        "Bypassed (Single-Cell Mode)"
        if is_single_cell
        else (os.path.basename(yolo_ckpt_full) if yolo_ckpt_full and yolo_ckpt_full != "N/A" else "N/A")
    )
    crop_mode = models_meta.get("primary_crop_preprocessing_mode", par_est.get("primary_mode", "N/A"))
    cnn_input_shape = models_meta.get("cnn_input_shape", [224, 224, 3])

    thresh_meta = summary_data.get("thresholds_and_parameters") or {}
    yolo_conf_th = thresh_meta.get("yolo_confidence_threshold", "N/A")
    yolo_nms_th = thresh_meta.get("yolo_nms_iou_threshold", "N/A")
    dedup_iou_th = thresh_meta.get("deduplication_iou_threshold", "N/A")
    cnn_sig_th = thresh_meta.get("cnn_parasitized_threshold", "N/A")
    pad_px = thresh_meta.get("crop_padding_pixels", "N/A")
    min_box_px = thresh_meta.get("min_box_size_pixels", "N/A")

    hw_meta = summary_data.get("hardware_execution") or {}
    s1_dev = hw_meta.get("stage1_device", "N/A")
    s1_vram = hw_meta.get("stage1_peak_vram_mb", "N/A")
    gpu_fb = bool(hw_meta.get("gpu_fallback_triggered", False))
    gpu_fb_reason = hw_meta.get("gpu_fallback_reason", None)
    s2_dev = hw_meta.get("stage2_device", "N/A")

    conf_meta = summary_data.get("confidence_statistics") or {}
    y_stat = conf_meta.get("yolo_detection_confidence") or {}
    c_stat = conf_meta.get("cnn_overall_confidence") or {}
    p_stat = conf_meta.get("cnn_parasitized_confidence") or {}
    u_stat = conf_meta.get("cnn_uninfected_confidence") or {}

    # -----------------------------------------------------------------------
    # REPORT TITLE & BRIEF RESEARCH PROTOTYPE NOTICE
    # -----------------------------------------------------------------------
    story.append(Paragraph("SmartCell — AI-Assisted Blood-Smear Analysis Report", styles["Title"]))
    story.append(
        Paragraph(
            "Single-Cell Stage 2 CNN Malaria Classification Prototype (Direct Segmented RBC Evaluation)"
            if is_single_cell
            else "Two-Stage YOLOv8n Erythrocyte Localization &amp; Domain-Adapted CNN Malaria Screening Prototype",
            styles["Subtitle"],
        )
    )

    notice_tbl = Table(
        [[Paragraph(f"<b>RESEARCH PROTOTYPE NOTICE:</b> {_safe_str(PROTOTYPE_NOTICE_TEXT)}", styles["BannerDisclaimer"])]],
        colWidths=[usable_w],
    )
    notice_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef2f2")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#dc2626")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(notice_tbl)
    story.append(Spacer(1, 4))

    # -----------------------------------------------------------------------
    # SECTION A: REPORT & SPECIMEN IDENTIFICATION
    # -----------------------------------------------------------------------
    story.append(Paragraph("A. Report &amp; Specimen Identification", styles["SectionHeading"]))
    mode_label_str = (
        "Single-Cell Mode (Stage 2 CNN Only)"
        if is_single_cell
        else "Full-Smear Mode (Stage 1 YOLOv8n + Stage 2 CNN)"
    )
    header_rows = [
        [
            "Project System:",
            "SmartCell Blood-Smear Analyzer",
            "Report Generated:",
            _safe_str(generated_at),
        ],
        [
            "Original Image Filename:",
            _safe_str(img_fn),
            "Image Dimensions:",
            _safe_str(dims_str),
        ],
        [
            "Analysis / Run ID:",
            _safe_str(run_id),
            "Input Mode / Crop Mode:",
            f"{_safe_str(mode_label_str)} | {_safe_str(crop_mode)}",
        ],
    ]
    story.append(
        _make_kv_table(
            header_rows,
            styles,
            col_widths=[usable_w * 0.22, usable_w * 0.28, usable_w * 0.22, usable_w * 0.28],
        )
    )

    # -----------------------------------------------------------------------
    # SECTION B: EXECUTIVE SUMMARY
    # -----------------------------------------------------------------------
    story.append(Paragraph("B. Executive Summary", styles["SectionHeading"]))

    if is_single_cell:
        sc_pred = single_cell_result.get("predicted_class") or (
            cell_predictions[0].get("cnn_predicted_class", "Unknown") if cell_predictions else "Unknown"
        )
        sc_conf = _safe_float(
            single_cell_result.get(
                "confidence",
                cell_predictions[0].get("cnn_confidence", 0.0) if cell_predictions else 0.0,
            ),
            0.0,
        ) * 100.0
        sc_sig = _safe_float(
            single_cell_result.get(
                "parasitized_prob",
                single_cell_result.get(
                    "sigmoid_score",
                    cell_predictions[0].get("cnn_parasitized_prob", 0.0) if cell_predictions else 0.0,
                ),
            ),
            0.0,
        )
        sc_raw_cls = single_cell_result.get(
            "raw_predicted_class",
            single_cell_result.get(
                "raw_crop_predicted_class",
                cell_predictions[0].get("cnn_raw_predicted_class", sc_pred) if cell_predictions else sc_pred,
            ),
        )
        sc_raw_cf = _safe_float(
            single_cell_result.get(
                "raw_confidence",
                single_cell_result.get(
                    "raw_crop_confidence",
                    cell_predictions[0].get("cnn_raw_confidence", sc_conf / 100.0) if cell_predictions else 0.0,
                ),
            ),
            0.0,
        ) * 100.0
        sc_mask_cls = single_cell_result.get(
            "masked_predicted_class",
            single_cell_result.get(
                "ellipse_masked_predicted_class",
                cell_predictions[0].get("cnn_masked_predicted_class", sc_pred) if cell_predictions else sc_pred,
            ),
        )
        sc_mask_cf = _safe_float(
            single_cell_result.get(
                "masked_confidence",
                single_cell_result.get(
                    "ellipse_masked_confidence",
                    cell_predictions[0].get("cnn_masked_confidence", sc_conf / 100.0) if cell_predictions else 0.0,
                ),
            ),
            0.0,
        ) * 100.0

        pred_cell_style = styles["TableCellPar"] if sc_pred == "Parasitized" else styles["TableCellUninf"]
        kpi_header = [
            Paragraph("<b>Evaluation Mode</b>", styles["TableHeader"]),
            Paragraph("<b>Single-Cell Predicted Class</b>", styles["TableHeader"]),
            Paragraph("<b>Classification Confidence</b>", styles["TableHeader"]),
            Paragraph("<b>Raw vs. Masked Mode Audit</b>", styles["TableHeader"]),
            Paragraph("<b>Whole-Smear Parasitemia</b>", styles["TableHeader"]),
            Paragraph("<b>Processing Duration</b>", styles["TableHeader"]),
        ]
        kpi_values = [
            Paragraph("<b>Single-Cell Classification</b><br/><font size=6.8>(1 Segmented RBC Patch)</font>", styles["TableCellBold"]),
            Paragraph(f"<b>{_safe_str(sc_pred)}</b>", pred_cell_style),
            Paragraph(f"<b>{sc_conf:.2f}%</b><br/><font size=6.8>(Sigmoid: {sc_sig:.4f})</font>", styles["TableCellBold"]),
            Paragraph(
                f"Raw: <b>{_safe_str(sc_raw_cls)}</b> ({sc_raw_cf:.1f}%)<br/>"
                f"Masked: <b>{_safe_str(sc_mask_cls)}</b> ({sc_mask_cf:.1f}%)",
                styles["TableCell"],
            ),
            Paragraph("<b>N/A</b><br/><font size=6.8>(Single-cell input only)</font>", styles["TableCell"]),
            Paragraph(f"<b>{tot_time:.2f} s</b><br/><font size=6.8>(Stage 2 CNN Only)</font>", styles["TableCell"]),
        ]
        col_w6 = usable_w / 6.0
        kpi_tbl = Table([kpi_header, kpi_values], colWidths=[col_w6] * 6)
        kpi_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                    ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#f8fafc")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
                ]
            )
        )
        story.append(kpi_tbl)
    elif valid_rbcs > 0 and not analysis_failed:
        kpi_header = [
            Paragraph("<b>Total Valid RBC Crops Classified</b>", styles["TableHeader"]),
            Paragraph("<b>Predicted Parasitized RBCs</b>", styles["TableHeader"]),
            Paragraph("<b>Predicted Uninfected RBCs</b>", styles["TableHeader"]),
            Paragraph("<b>Estimated Parasitemia (%)</b>", styles["TableHeader"]),
            Paragraph("<b>Excluded WBC Count</b>", styles["TableHeader"]),
            Paragraph("<b>Rejected / Duplicate Boxes</b>", styles["TableHeader"]),
            Paragraph("<b>Processing Duration</b>", styles["TableHeader"]),
        ]
        kpi_values = [
            Paragraph(f"<b>{valid_rbcs}</b> (of {raw_yolo_det} raw)", styles["TableCellBold"]),
            Paragraph(f"<b>{par_rbcs}</b>", styles["TableCellPar"]),
            Paragraph(f"<b>{uninf_rbcs}</b>", styles["TableCellUninf"]),
            Paragraph(f"<b>{(est_par_pct or 0.0):.2f}%</b><br/><font size=6.8>({_safe_str(calc_frac)})</font>", styles["TableCellBold"]),
            Paragraph(f"<b>{excl_wbcs}</b>", styles["TableCellBold"]),
            Paragraph(f"<b>{rej_boxes}</b>", styles["TableCellBold"]),
            Paragraph(f"<b>{tot_time:.2f} s</b><br/><font size=6.8>(S1: {s1_time:.2f}s | S2: {s2_time:.2f}s)</font>", styles["TableCell"]),
        ]
        col_w7 = usable_w / 7.0
        kpi_tbl = Table([kpi_header, kpi_values], colWidths=[col_w7] * 7)
        kpi_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                    ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#f8fafc")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
                ]
            )
        )
        story.append(kpi_tbl)
    else:
        # Zero valid RBC detections in full-smear mode: do not show misleading cell counts or 0.00% parasitemia
        zero_kpi_rows = [
            [
                Paragraph("<b>Executive Status</b>", styles["TableHeader"]),
                Paragraph("<b>Stage 1 Detection Summary</b>", styles["TableHeader"]),
                Paragraph("<b>Excluded / Rejected Boxes</b>", styles["TableHeader"]),
                Paragraph("<b>Processing Duration</b>", styles["TableHeader"]),
            ],
            [
                Paragraph("<b>No reliable result is available for this image.</b>", styles["TableCellBold"]),
                Paragraph(
                    f"<b>0 valid RBC crops</b> (of {raw_yolo_det} raw detections)<br/>"
                    "<font size=6.8>Cell counts &amp; parasitemia omitted (zero valid RBCs)</font>",
                    styles["TableCell"],
                ),
                Paragraph(f"Excluded WBCs: <b>{excl_wbcs}</b> | Rejected Boxes: <b>{rej_boxes}</b>", styles["TableCell"]),
                Paragraph(f"<b>{tot_time:.2f} s</b><br/><font size=6.8>(S1: {s1_time:.2f}s | S2: {s2_time:.2f}s)</font>", styles["TableCell"]),
            ],
        ]
        kpi_tbl = Table(
            zero_kpi_rows,
            colWidths=[usable_w * 0.34, usable_w * 0.28, usable_w * 0.22, usable_w * 0.16],
        )
        kpi_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
                    ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#f8fafc")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
                ]
            )
        )
        story.append(kpi_tbl)
    story.append(Spacer(1, 3))

    # -----------------------------------------------------------------------
    # SECTION C: MODEL & CONFIGURATION DETAILS
    # -----------------------------------------------------------------------
    story.append(Paragraph("C. Model &amp; Configuration Details", styles["SectionHeading"]))
    vram_str = f"{s1_vram} MB Peak VRAM" if s1_vram not in ("N/A", None) else "N/A"
    config_rows = [
        [
            "Stage 2 CNN Variant:",
            _safe_str(cnn_variant),
            "CNN Checkpoint File:",
            _safe_str(cnn_ckpt_name),
        ],
        [
            "Crop Preprocessing Mode:",
            _safe_str(crop_mode),
            "CNN Classification Threshold:",
            _safe_str(cnn_sig_th),
        ],
        [
            "Stage 1 YOLO Checkpoint:",
            _safe_str(yolo_ckpt_name),
            "YOLO Confidence Threshold:",
            _safe_str(yolo_conf_th),
        ],
        [
            "YOLO NMS / Dedup IoU:",
            f"NMS: {_safe_str(yolo_nms_th)} | Dedup: {_safe_str(dedup_iou_th)}",
            "Crop Padding / Min Box Size:",
            f"+{_safe_str(pad_px)} px | Min: {_safe_str(min_box_px)} px",
        ],
        [
            "Stage 1 Inference Device:",
            _safe_str(s1_dev) if is_single_cell else f"{_safe_str(s1_dev)} ({_safe_str(vram_str)})",
            "Stage 2 Inference Device:",
            f"{_safe_str(s2_dev)} | Input: {_safe_str(cnn_input_shape)}",
        ],
    ]
    story.append(
        _make_kv_table(
            config_rows,
            styles,
            col_widths=[usable_w * 0.22, usable_w * 0.28, usable_w * 0.22, usable_w * 0.28],
        )
    )

    # -----------------------------------------------------------------------
    # SECTION D: CONFIDENCE STATISTICS & PROCESSING TELEMETRY
    # -----------------------------------------------------------------------
    story.append(Paragraph("D. Confidence Statistics &amp; Processing Telemetry", styles["SectionHeading"]))

    def _fmt_conf(stat_d, include_range=True):
        cnt = _safe_int(stat_d.get("count"), 0)
        if cnt <= 0:
            return "N/A (0 cells)"
        mean_v = _safe_float(stat_d.get("mean"), 0.0) * 100.0
        med_v = _safe_float(stat_d.get("median"), 0.0) * 100.0
        min_v = _safe_float(stat_d.get("min"), 0.0) * 100.0
        max_v = _safe_float(stat_d.get("max"), 0.0) * 100.0
        if include_range:
            return f"Mean: {mean_v:.2f}% | Median: {med_v:.2f}% | Range: [{min_v:.2f}% - {max_v:.2f}%] (n={cnt})"
        return f"Mean: {mean_v:.2f}% | Median: {med_v:.2f}% (n={cnt})"

    warnings_list = []
    if is_single_cell:
        warnings_list.append(
            "Single-Cell Mode: Stage 1 YOLOv8n full-smear localization was bypassed. Whole-smear parasitemia is not calculated from a single segmented cell patch."
        )
    elif valid_rbcs == 0:
        warnings_list.append(
            "No reliable result is available for this image: Zero valid candidate RBC crops were detected; zero detections are never treated as a negative malaria result."
        )
    if gpu_fb:
        warnings_list.append(
            f"GPU Fallback Triggered: {_safe_str(gpu_fb_reason or 'Stage 1 executed on CPU fallback.')}"
        )
    raw_par_c = _safe_int(counts.get("cnn_raw_crop_parasitized_count"), par_rbcs)
    mask_par_c = _safe_int(counts.get("cnn_ellipse_masked_parasitized_count"), par_rbcs)
    if not is_single_cell and valid_rbcs > 0 and abs(raw_par_c - mask_par_c) > 0:
        warnings_list.append(
            f"Cross-mode audit note: Raw rectangular crops flagged {raw_par_c} Parasitized RBCs ({raw_par_pct:.2f}%) "
            f"vs. Elliptical-masked crops flagging {mask_par_c} Parasitized RBCs ({mask_par_pct:.2f}%)."
        )
    extra_q_warnings = summary_data.get("quality_warnings") or (summary_data.get("image_quality") or {}).get("warnings") or []
    for qw in extra_q_warnings:
        if qw:
            warnings_list.append(_safe_str(qw))
    if not warnings_list:
        warnings_list.append("None — Two-stage pipeline completed normally without fallback or denominator warnings.")

    raw_vs_masked_str = (
        f"Raw: {single_cell_result.get('raw_predicted_class', single_cell_result.get('raw_crop_predicted_class', 'N/A'))} | "
        f"Masked: {single_cell_result.get('masked_predicted_class', single_cell_result.get('ellipse_masked_predicted_class', 'N/A'))}"
        if is_single_cell
        else (
            f"Raw: {raw_par_c}/{valid_rbcs} ({raw_par_pct:.2f}%) | Masked: {mask_par_c}/{valid_rbcs} ({mask_par_pct:.2f}%)"
            if valid_rbcs > 0
            else "N/A (0 valid RBC crops)"
        )
    )

    conf_rows = [
        [
            "CNN Overall Confidence:",
            _fmt_conf(c_stat, include_range=True),
            "YOLO Detection Confidence:",
            "Bypassed (Single-Cell Mode)" if is_single_cell else _fmt_conf(y_stat, include_range=True),
        ],
        [
            "CNN Parasitized Confidence:",
            _fmt_conf(p_stat, include_range=True),
            "CNN Uninfected Confidence:",
            _fmt_conf(u_stat, include_range=True),
        ],
        [
            "Stage 1 / Stage 2 Timing:",
            f"Stage 1 (YOLO + Cropping): {s1_time:.3f} s | Stage 2 (CNN): {s2_time:.3f} s",
            "Total Processing Time:",
            f"{tot_time:.3f} s",
        ],
        [
            "Processing Warnings / Notes:",
            " ".join(warnings_list),
            "Raw vs Masked Audit:",
            _safe_str(raw_vs_masked_str),
        ],
    ]
    story.append(
        _make_kv_table(
            conf_rows,
            styles,
            col_widths=[usable_w * 0.22, usable_w * 0.28, usable_w * 0.22, usable_w * 0.28],
        )
    )

    # -----------------------------------------------------------------------
    # SECTION E: ANNOTATED BLOOD-SMEAR FIELD OVERVIEW
    # -----------------------------------------------------------------------
    story.append(Paragraph("E. Annotated Blood-Smear Field Overview", styles["SectionHeading"]))

    if not annotated_image_path:
        out_files = summary_data.get("output_files") or {}
        annotated_image_path = out_files.get("annotated_full_image")

    if annotated_image_path and os.path.isfile(annotated_image_path):
        try:
            with PILImage.open(annotated_image_path) as pim:
                iw, ih = pim.size
            max_img_w = usable_w
            max_img_h = 58 * mm if is_single_cell else 84 * mm
            # Never upscale small native images excessively on the page; cap display scale at 1.25x for tiny images
            scale = min(max_img_w / float(iw), max_img_h / float(ih), 1.25)
            disp_w = iw * scale
            disp_h = ih * scale

            rl_img = RLImage(annotated_image_path, width=disp_w, height=disp_h)
            rl_img.hAlign = "CENTER"
            story.append(rl_img)
            if is_single_cell:
                story.append(
                    Paragraph(
                        f"Figure 1: Annotated blood-smear overview (single-cell specimen <b>{_safe_str(img_fn)}</b>, "
                        f"native resolution {iw}x{ih} px). Perimeter border color indicates direct Stage 2 CNN classification "
                        "(<b>Red</b> = CNN Predicted Parasitized RBC, <b>Green</b> = CNN Predicted Uninfected RBC).",
                        styles["Caption"],
                    )
                )
            else:
                story.append(
                    Paragraph(
                        f"Figure 1: Annotated blood-smear overview for <b>{_safe_str(img_fn)}</b>. "
                        "All bounding boxes and labels represent two-stage algorithmic model predictions "
                        "(<b>Red</b> = CNN Predicted Parasitized RBC with confidence %, "
                        "<b>Green</b> = CNN Predicted Uninfected RBC, "
                        "<b>Orange</b> = Excluded WBC detection).",
                        styles["Caption"],
                    )
                )
        except Exception as img_err:
            story.append(
                Paragraph(
                    f"<i>Annotated image could not be embedded ({_safe_str(img_err)}).</i>",
                    styles["BodySmall"],
                )
            )
    else:
        story.append(
            Paragraph(
                "<i>Annotated blood-smear overview image is not available for this run.</i>",
                styles["BodySmall"],
            )
        )

    # -----------------------------------------------------------------------
    # SECTION G: AUDIT & EXCLUSION SUMMARY
    # (Note: Section F is intentionally omitted from the PDF report)
    # -----------------------------------------------------------------------
    story.append(Paragraph("G. Audit &amp; Exclusion Summary", styles["SectionHeading"]))
    excl_wbc_list = rejected_and_excluded.get("excluded_wbcs") or []
    rej_box_list = rejected_and_excluded.get("rejected_boxes") or []
    dom_analysis = summary_data.get("Domain_Mismatch_Analysis") or {}
    dom_obs = dom_analysis.get(
        "observation",
        "Stage 1 excludes class 2 (WBC) detections from the RBC denominator and suppresses duplicate overlapping boxes (IoU >= dedup_iou).",
    )

    story.append(
        Paragraph(
            f"<b>Excluded WBC Detections (Class 2):</b> {excl_wbcs} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Rejected / Duplicate Bounding Boxes:</b> {rej_boxes}<br/>"
            f"<b>Pipeline Domain Audit Note:</b> {_safe_str(dom_obs)}",
            styles["BodySmall"],
        )
    )

    if excl_wbc_list or rej_box_list:
        audit_rows = [
            [
                Paragraph("<b>Category</b>", styles["TableHeader"]),
                Paragraph("<b>Class / Confidence</b>", styles["TableHeader"]),
                Paragraph("<b>Bounding Box [x1, y1, x2, y2]</b>", styles["TableHeader"]),
                Paragraph("<b>Audit Exclusion / Suppression Reason</b>", styles["TableHeader"]),
            ]
        ]
        for w in excl_wbc_list[:20]:
            bbox_s = str(w.get("clipped_bbox", "N/A"))
            w_conf = _safe_float(w.get("yolo_confidence"), 0.0) * 100.0
            audit_rows.append(
                [
                    Paragraph("Excluded WBC", styles["TableCellBold"]),
                    Paragraph(f"{_safe_str(w.get('yolo_class_name', 'WBC'))} ({w_conf:.1f}%)", styles["TableCell"]),
                    Paragraph(_safe_str(bbox_s), styles["TableCell"]),
                    Paragraph(_safe_str(w.get("reason", "WBC excluded from RBC count")), styles["TableCell"]),
                ]
            )
        for r in rej_box_list[:20]:
            sup_det = r.get("suppressed_detection") or r.get("detection") or {}
            bbox_s = str(r.get("clipped_bbox") or sup_det.get("bbox") or sup_det.get("raw_bbox") or "N/A")
            r_conf = _safe_float(sup_det.get("yolo_confidence"), 0.0) * 100.0
            audit_rows.append(
                [
                    Paragraph("Rejected / Duplicate", styles["TableCellBold"]),
                    Paragraph(f"{_safe_str(sup_det.get('yolo_class_name', 'Box'))} ({r_conf:.1f}%)", styles["TableCell"]),
                    Paragraph(_safe_str(bbox_s), styles["TableCell"]),
                    Paragraph(_safe_str(r.get("reason", "Rejected or duplicate detection")), styles["TableCell"]),
                ]
            )
        audit_tbl = Table(
            audit_rows,
            colWidths=[usable_w * 0.18, usable_w * 0.18, usable_w * 0.24, usable_w * 0.40],
            repeatRows=1,
        )
        audit_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(audit_tbl)

    # -----------------------------------------------------------------------
    # SECTION H: DATASET GROUND-TRUTH REFERENCE COMPARISON
    # (Included ONLY when valid associated annotations are available)
    # -----------------------------------------------------------------------
    if ground_truth_reference and ground_truth_reference.get("has_ground_truth"):
        story.append(Paragraph("H. Dataset Ground-Truth Reference Comparison", styles["SectionHeading"]))
        gt_split = ground_truth_reference.get("split", "unknown")
        gt_pat = ground_truth_reference.get("patient_id", "Unknown")
        gt_par = _safe_int(ground_truth_reference.get("gt_parasitized_rbc"), 0)
        gt_uninf = _safe_int(ground_truth_reference.get("gt_uninfected_rbc"), 0)
        gt_wbc = _safe_int(ground_truth_reference.get("gt_wbc"), 0)
        gt_tot = _safe_int(ground_truth_reference.get("gt_total_rbc"), gt_par + gt_uninf)
        gt_pct = _safe_float(ground_truth_reference.get("gt_parasitemia_percent"), 0.0)
        abs_diff_pct = abs((est_par_pct or 0.0) - gt_pct)

        split_notice = (
            "Held-out Test Partition (unseen during model training and hyperparameter tuning)."
            if gt_split == "test"
            else f"Dataset '{gt_split.upper()}' Partition (Reference dataset annotation only — NOT independent clinical confirmation; "
                 f"{'used during training' if gt_split == 'train' else 'used for validation selection'})."
        )

        story.append(
            Paragraph(
                f"<b>Ground-Truth Source Notice:</b> Associated NIH-NLM Thin Blood Smears polygon annotations were found "
                f"for <code>{_safe_str(img_fn)}</code> (Patient ID: <b>{_safe_str(gt_pat)}</b>, Partition: <b>{_safe_str(gt_split)}</b>). "
                f"{_safe_str(split_notice)} Ground-truth values are shown strictly for experimental benchmarking and are "
                f"separate from algorithmic model predictions.",
                styles["BodySmall"],
            )
        )

        gt_rows = [
            [
                "Dataset Partition / Patient ID:",
                f"{_safe_str(gt_split)} / {_safe_str(gt_pat)}",
                "Total Annotated GT RBCs (GT WBCs):",
                f"{gt_tot} RBCs ({gt_wbc} WBCs)",
            ],
            [
                "Ground-Truth Parasitized RBCs:",
                f"{gt_par} ({gt_pct:.2f}% GT Parasitemia)",
                "Model Predicted Parasitized RBCs:",
                f"{par_rbcs} ({(est_par_pct or 0.0):.2f}% Est. Parasitemia)" if valid_rbcs > 0 else "N/A (0 valid RBCs)",
            ],
            [
                "Ground-Truth Uninfected RBCs:",
                f"{gt_uninf}",
                "Model Predicted Uninfected RBCs:",
                f"{uninf_rbcs} (Abs. Parasitemia Diff: {abs_diff_pct:.2f}%)" if valid_rbcs > 0 else "N/A (0 valid RBCs)",
            ],
        ]
        story.append(
            _make_kv_table(
                gt_rows,
                styles,
                col_widths=[usable_w * 0.24, usable_w * 0.26, usable_w * 0.24, usable_w * 0.26],
            )
        )

    # -----------------------------------------------------------------------
    # FINAL AI SCREENING RESULT SECTION
    # -----------------------------------------------------------------------
    story.append(Spacer(1, 6))
    story.append(Paragraph("Final AI Screening Result", styles["SectionHeading"]))

    final_statement = format_final_screening_statement(
        valid_rbcs=valid_rbcs,
        par_rbcs=par_rbcs,
        est_par_pct=est_par_pct or 0.0,
        analysis_failed=analysis_failed,
        input_mode=input_mode,
        single_cell_result=single_cell_result,
    )

    if analysis_failed or valid_rbcs <= 0:
        box_bg = colors.HexColor("#f8fafc")
        box_border = colors.HexColor("#64748b")
        stmt_color = "#334155"
    elif par_rbcs >= 1:
        box_bg = colors.HexColor("#fef2f2")
        box_border = colors.HexColor("#dc2626")
        stmt_color = "#991b1b"
    else:
        box_bg = colors.HexColor("#f0fdf4")
        box_border = colors.HexColor("#16a34a")
        stmt_color = "#166534"

    final_box_content = [
        Paragraph(
            f"<font color='{stmt_color}'><b>{_safe_str(final_statement)}</b></font>",
            styles["FinalScreeningStatement"],
        ),
        Paragraph(
            _safe_str(FINAL_SCREENING_DISCLAIMER),
            styles["FinalScreeningDisclaimer"],
        ),
    ]

    final_box = Table([[final_box_content]], colWidths=[usable_w])
    final_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), box_bg),
                ("BOX", (0, 0), (-1, -1), 1.0, box_border),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story.append(KeepTogether([final_box]))

    # Build multi-page PDF document
    doc = SimpleDocTemplate(
        abs_pdf_path,
        pagesize=A4,
        leftMargin=margin_lr,
        rightMargin=margin_lr,
        topMargin=margin_tb,
        bottomMargin=margin_tb,
        title=f"SmartCell Analysis Report - {img_fn}",
        author="SmartCell Research Prototype",
        subject="AI-Assisted Blood-Smear Analysis Report",
    )
    doc.build(story, canvasmaker=NumberedReportCanvas)

    if not os.path.isfile(abs_pdf_path) or os.path.getsize(abs_pdf_path) == 0:
        raise RuntimeError(f"PDF generation failed to produce a valid non-empty file at {abs_pdf_path}")

    return abs_pdf_path


def generate_pdf_from_run_dir(run_dir, stem=None, output_pdf_path=None, ground_truth_reference=None):
    """
    Convenience helper that loads `field_summary_<stem>.json`, `cell_predictions_<stem>.csv`,
    `rejected_and_excluded_<stem>.json`, and `annotated_<stem>.jpg` from a run directory
    and generates `SmartCell_Report_<stem>.pdf`.
    """
    run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    if not stem:
        for fn in os.listdir(run_dir):
            if fn.startswith("field_summary_") and fn.endswith(".json"):
                stem = fn[len("field_summary_") : -len(".json")]
                break
    if not stem:
        raise FileNotFoundError(f"Could not locate field_summary_<stem>.json in {run_dir}")

    summary_json_path = os.path.join(run_dir, f"field_summary_{stem}.json")
    csv_path = os.path.join(run_dir, f"cell_predictions_{stem}.csv")
    rejected_json_path = os.path.join(run_dir, f"rejected_and_excluded_{stem}.json")
    annotated_jpg_path = os.path.join(run_dir, f"annotated_{stem}.jpg")

    if not os.path.isfile(summary_json_path):
        raise FileNotFoundError(f"Summary JSON not found: {summary_json_path}")

    with open(summary_json_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    cell_rows = []
    if os.path.isfile(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            cell_rows = list(csv.DictReader(f))

    rej_data = {}
    if os.path.isfile(rejected_json_path):
        with open(rejected_json_path, "r", encoding="utf-8") as f:
            rej_data = json.load(f)

    if not output_pdf_path:
        output_pdf_path = os.path.join(run_dir, f"SmartCell_Report_{stem}.pdf")

    run_id = os.path.basename(run_dir.rstrip("\\/"))
    return generate_pdf_report(
        output_pdf_path=output_pdf_path,
        summary_data=summary_data,
        cell_predictions=cell_rows,
        rejected_and_excluded=rej_data,
        annotated_image_path=annotated_jpg_path if os.path.isfile(annotated_jpg_path) else None,
        ground_truth_reference=ground_truth_reference,
        run_id=run_id,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate SmartCell PDF Analysis Report from a pipeline run directory.")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to the completed pipeline output directory")
    parser.add_argument("--stem", type=str, default=None, help="Optional image stem")
    parser.add_argument("--output-pdf", type=str, default=None, help="Optional explicit output PDF path")
    args = parser.parse_args()

    pdf_path = generate_pdf_from_run_dir(args.run_dir, stem=args.stem, output_pdf_path=args.output_pdf)
    print(f"[PDF Report Generated] {pdf_path} ({os.path.getsize(pdf_path)} bytes)")


if __name__ == "__main__":
    main()
