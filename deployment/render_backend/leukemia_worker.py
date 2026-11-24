r"""
leukemia_worker.py
------------------
Subprocess worker script for Acute Lymphoblastic Leukemia (ALL) inference.
Designed for memory-constrained cloud environments (e.g., Render Free Tier 512 MB RAM).

By executing the EfficientNetB0 TensorFlow model in an isolated worker subprocess,
the parent web server never permanently retains TensorFlow memory (~360 MB),
guaranteeing zero memory leak and preventing Out-Of-Memory (OOM) SIGKILL termination.
"""

import os
import sys
import json
import argparse
import traceback

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Ensure project root and scripts directory are in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.environ.get("SMARTCELL_ROOT", os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..")))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)


def main():
    parser = argparse.ArgumentParser(description="SmartCell Leukemia Subprocess Worker")
    parser.add_argument("--image", required=True, help="Path to input blood smear cell image")
    parser.add_argument("--run-dir", required=True, help="Directory to store inference run outputs")
    parser.add_argument("--patient-sample-id", default=None, help="Optional patient or sample identifier")
    parser.add_argument("--capture-source", default="Manual File Upload", help="Label describing source of specimen")
    parser.add_argument("--output-json", default=None, help="Optional output JSON path for result payload")
    args = parser.parse_args()

    try:
        from leukemia_analyzer import run_leukemia_analysis
        result = run_leukemia_analysis(
            image_path=os.path.abspath(args.image),
            run_dir=os.path.abspath(args.run_dir),
            patient_sample_id=args.patient_sample_id,
            capture_source=args.capture_source
        )

        out_json_path = args.output_json
        if not out_json_path:
            out_json_path = os.path.join(args.run_dir, "leukemia_worker_result.json")

        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        # Print structured delimiter for parent process communication
        print("__SMARTCELL_LEUKEMIA_JSON_START__")
        print(json.dumps(result))
        print("__SMARTCELL_LEUKEMIA_JSON_END__")
        sys.exit(0)

    except Exception as e:
        err_payload = {
            "success": False,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
        print("__SMARTCELL_LEUKEMIA_JSON_START__", file=sys.stderr)
        print(json.dumps(err_payload), file=sys.stderr)
        print("__SMARTCELL_LEUKEMIA_JSON_END__", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
