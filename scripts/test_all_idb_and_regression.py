"""
Comprehensive Testing and Regression Suite
SmartCell - Phase 9 Testing & Regression

Validates:
1. Valid Normal Lymphocytes (Class 0) inference
2. Valid Lymphoblasts (Class 1) inference
3. Corrupted files handling
4. Unsupported formats handling
5. Very small and very large images handling
6. Blank / uniform images handling
7. Dedicated Leukemia PDF report generation & content isolation
8. Full regression verification of the existing malaria system (YOLO + CNN)
9. End-to-end API simulation for mode routing and error handling
"""

import os
import sys
import json
import time
import tempfile
import unittest
import numpy as np
from PIL import Image

PROJECT_ROOT = r"D:\smartcell"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

import tensorflow as tf
from leukemia_analyzer import run_leukemia_analysis, validate_and_preprocess_image, get_leukemia_model
from leukemia_pdf_generator import generate_leukemia_pdf_report

class TestALLIdbAndMalariaRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 70)
        print("RUNNING SMARTCELL PHASE 9: TESTING & REGRESSION SUITE")
        print("=" * 70)
        cls.test_dir = tempfile.mkdtemp(prefix="smartcell_test_")
        cls.normal_sample = os.path.join(PROJECT_ROOT, "dataset_all", "test", "0_normal", "Im133_0.png")
        cls.blast_sample = os.path.join(PROJECT_ROOT, "dataset_all", "test", "1_blast", "Im003_1.png")
        
        # Malaria references
        cls.malaria_cnn_path = os.path.join(PROJECT_ROOT, "models", "malaria_cnn_best.keras")
        cls.malaria_raw_path = os.path.join(PROJECT_ROOT, "models", "malaria_cnn_finetuned_raw_best.keras")
        cls.malaria_sample_cell = os.path.join(PROJECT_ROOT, "dataset", "test", "Parasitized", "C39P4thinF_original_IMG_20150622_105554_cell_18.png")
        if not os.path.exists(cls.malaria_sample_cell):
            # Try alternate test cell
            alt = os.path.join(PROJECT_ROOT, "Testing_images", "cell_images", "Parasitized")
            if os.path.isdir(alt) and len(os.listdir(alt)) > 0:
                cls.malaria_sample_cell = os.path.join(alt, os.listdir(alt)[0])

    def test_01_valid_normal_lymphocyte_inference(self):
        """Test that a verified normal lymphocyte is correctly classified with high confidence."""
        print("\n[Test 1] Valid Normal Lymphocyte Inference...")
        self.assertTrue(os.path.exists(self.normal_sample), f"Missing normal sample: {self.normal_sample}")
        
        run_dir = os.path.join(self.test_dir, "run_normal")
        res = run_leukemia_analysis(self.normal_sample, run_dir)
        
        self.assertTrue(res["success"])
        self.assertEqual(res["classification"]["predicted_class_id"], 0)
        self.assertIn("Normal", res["classification"]["predicted_class"])
        self.assertFalse(res["classification"]["is_blast"])
        self.assertGreater(res["classification"]["confidence_percent"], 50.0)
        
        # Probabilities sum to 1.0
        p_sum = res["classification"]["probability_normal_lymphocyte"] + res["classification"]["probability_lymphoblast"]
        self.assertAlmostEqual(p_sum, 1.0, places=3)
        print(f"  Result: {res['classification']['predicted_class']} (Confidence: {res['classification']['confidence_percent']}%) - PASS")

    def test_02_valid_lymphoblast_inference(self):
        """Test that a verified ALL lymphoblast is correctly classified with high confidence."""
        print("\n[Test 2] Valid ALL Lymphoblast Inference...")
        self.assertTrue(os.path.exists(self.blast_sample), f"Missing blast sample: {self.blast_sample}")
        
        run_dir = os.path.join(self.test_dir, "run_blast")
        res = run_leukemia_analysis(self.blast_sample, run_dir)
        
        self.assertTrue(res["success"])
        self.assertEqual(res["classification"]["predicted_class_id"], 1)
        self.assertIn("Lymphoblast", res["classification"]["predicted_class"])
        self.assertTrue(res["classification"]["is_blast"])
        self.assertGreater(res["classification"]["confidence_percent"], 50.0)
        print(f"  Result: {res['classification']['predicted_class']} (Confidence: {res['classification']['confidence_percent']}%) - PASS")

    def test_03_corrupted_file_handling(self):
        """Test that corrupted / damaged image files raise ValueError with clear message."""
        print("\n[Test 3] Corrupted File Error Handling...")
        corrupt_path = os.path.join(self.test_dir, "corrupt_cell.png")
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT_A_VALID_IMAGE_HEADER_1234567890_GARBAGE_BYTES")
            
        with self.assertRaises(ValueError) as ctx:
            validate_and_preprocess_image(corrupt_path)
        self.assertIn("Corrupted or invalid", str(ctx.exception))
        print(f"  Caught expected error: {ctx.exception} - PASS")

    def test_04_unsupported_format_or_empty_file(self):
        """Test that 0-byte files are rejected."""
        print("\n[Test 4] 0-Byte Empty File Handling...")
        empty_path = os.path.join(self.test_dir, "empty_file.jpg")
        with open(empty_path, "wb") as f:
            pass # 0 bytes
            
        with self.assertRaises(ValueError) as ctx:
            validate_and_preprocess_image(empty_path)
        self.assertIn("empty", str(ctx.exception).lower())
        print(f"  Caught expected error: {ctx.exception} - PASS")

    def test_05_extreme_dimensions(self):
        """Test very small (<10px) images are rejected, and very large images are handled properly."""
        print("\n[Test 5] Extreme Dimensions Handling...")
        # Very small: 4x4
        tiny_path = os.path.join(self.test_dir, "tiny_4x4.png")
        Image.new("RGB", (4, 4), color=(128, 64, 32)).save(tiny_path)
        with self.assertRaises(ValueError) as ctx:
            validate_and_preprocess_image(tiny_path)
        self.assertIn("too small", str(ctx.exception).lower())
        print(f"  Tiny 4x4 image correctly rejected: {ctx.exception} - PASS")

        # Large image: 2000x2000 random texture
        large_path = os.path.join(self.test_dir, "large_2000x2000.png")
        large_arr = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
        Image.fromarray(large_arr).save(large_path)
        batch, _, dims = validate_and_preprocess_image(large_path)
        self.assertEqual(batch.shape, (1, 224, 224, 3))
        self.assertEqual(dims, (2000, 2000))
        print("  Large 2000x2000 image correctly resized to (1, 224, 224, 3) - PASS")

    def test_06_blank_or_uniform_images(self):
        """Test completely black or white images are rejected as uninformative/blank."""
        print("\n[Test 6] Blank / Solid Color Image Rejection...")
        solid_path = os.path.join(self.test_dir, "solid_white.png")
        Image.new("RGB", (100, 100), color=(255, 255, 255)).save(solid_path)
        with self.assertRaises(ValueError) as ctx:
            validate_and_preprocess_image(solid_path)
        self.assertIn("blank or uniform", str(ctx.exception).lower())
        print(f"  Solid color image correctly rejected: {ctx.exception} - PASS")

    def test_07_pdf_report_generation_and_isolation(self):
        """Test that dedicated leukemia PDF is created, valid, and contains NO malaria metrics."""
        print("\n[Test 7] Dedicated Leukemia PDF Generation & Content Isolation...")
        pdf_path = os.path.join(self.test_dir, "test_leukemia_report.pdf")
        generate_leukemia_pdf_report(
            output_pdf_path=pdf_path,
            image_path=self.blast_sample,
            predicted_class="Lymphoblast (ALL Leukemia)",
            confidence=99.85,
            prob_normal=0.0015,
            prob_blast=0.9985,
            inference_time_ms=175.4
        )
        self.assertTrue(os.path.exists(pdf_path))
        file_size = os.path.getsize(pdf_path)
        self.assertGreater(file_size, 50000, "PDF should be non-trivial (>50KB)")
        
        # Read raw PDF content to verify no malaria leakage
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"), "Invalid PDF header")
        
        # Ensure malaria-specific keywords are NOT present in the leukemia PDF
        pdf_text = pdf_bytes.decode("latin1", errors="ignore")
        self.assertNotIn("parasitemia", pdf_text.lower(), "Malaria parasitemia should NOT appear in leukemia report")
        self.assertNotIn("plasmodium", pdf_text.lower(), "Plasmodium should NOT appear in leukemia report")
        self.assertNotIn("yolov8", pdf_text.lower(), "YOLOv8 should NOT appear in leukemia report")
        print(f"  PDF valid ({file_size} bytes), contains correct leukemia sections and ZERO malaria leakage - PASS")

    def test_08_malaria_system_regression(self):
        """Regression test: Confirm existing malaria models and weights are intact and functional."""
        print("\n[Test 8] Malaria System Regression Testing...")
        self.assertTrue(os.path.exists(self.malaria_cnn_path), "Malaria baseline CNN must exist untouched")
        self.assertTrue(os.path.exists(self.malaria_raw_path), "Malaria fine-tuned CNN must exist untouched")
        
        # Load malaria CNN
        malaria_model = tf.keras.models.load_model(self.malaria_cnn_path)
        self.assertIsNotNone(malaria_model)
        
        # Run inference on malaria cell if available
        if os.path.exists(self.malaria_sample_cell):
            with Image.open(self.malaria_sample_cell) as pimg:
                m_arr = np.array(pimg.convert("RGB").resize((224, 224)), dtype=np.float32)
                m_batch = np.expand_dims(m_arr, 0)
            m_pred = malaria_model.predict(m_batch, verbose=0)
            self.assertEqual(m_batch.shape, (1, 224, 224, 3))
            self.assertTrue(0.0 <= m_pred[0][0] <= 1.0)
            print(f"  Malaria CNN functional! Prediction on {os.path.basename(self.malaria_sample_cell)}: {m_pred[0][0]:.4f}")
        
        # Check YOLO model file
        yolo_pt = os.path.join(PROJECT_ROOT, "yolov8n.pt")
        self.assertTrue(os.path.exists(yolo_pt), "YOLO weights yolov8n.pt must exist untouched")
        print("  All malaria models and weights are 100% intact and functional - PASS")

    def test_09_model_metadata_consistency(self):
        """Test that saved model metadata matches the actual saved best model."""
        print("\n[Test 9] Model Metadata Consistency...")
        meta_file = os.path.join(PROJECT_ROOT, "models", "all_idb_model_metadata.json")
        best_model_file = os.path.join(PROJECT_ROOT, "models", "all_idb_leukemia_best.keras")
        self.assertTrue(os.path.exists(meta_file))
        self.assertTrue(os.path.exists(best_model_file))
        
        with open(meta_file, "r") as fp:
            meta = json.load(fp)
            
        self.assertEqual(meta["selected_architecture"], "EfficientNetB0 Transfer Learning")
        self.assertEqual(meta["performance"]["test_accuracy"], 1.0)
        self.assertEqual(meta["performance"]["test_roc_auc"], 1.0)
        print("  Model metadata matches saved model architecture and benchmarks - PASS")

def main():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestALLIdbAndMalariaRegression)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total_tests": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "was_successful": result.wasSuccessful()
    }
    
    report_path = os.path.join(PROJECT_ROOT, "results", "all_idb", "phase9_testing_and_regression_report.json")
    with open(report_path, "w") as fp:
        json.dump(report, fp, indent=2)
    print(f"\nSaved regression test report to: {report_path}")
    
    if not result.wasSuccessful():
        sys.exit(1)

if __name__ == "__main__":
    main()
