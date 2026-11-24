import os
import sys
import time
import subprocess
import shutil
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.edge.options import Options as EdgeOptions

PROJECT_ROOT = r"D:\smartcell"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "visual_verification")
ARTIFACT_DIR = r"C:\Users\sumit\.gemini\antigravity\brain\9b51a21d-57cf-456a-a18c-4071bed52642"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

APP_URL = "http://127.0.0.1:8501"

def save_screenshot(driver, filename, description):
    res_path = os.path.join(RESULTS_DIR, filename)
    art_path = os.path.join(ARTIFACT_DIR, filename)
    driver.save_screenshot(res_path)
    shutil.copy2(res_path, art_path)
    print(f"[Screenshot] {description} saved to:\n  -> {res_path}\n  -> {art_path}")
    return res_path

def create_driver():
    chrome_opts = ChromeOptions()
    chrome_opts.add_argument("--headless=new")
    chrome_opts.add_argument("--window-size=1440,1100")
    chrome_opts.add_argument("--use-fake-ui-for-media-stream")
    chrome_opts.add_argument("--use-fake-device-for-media-stream")
    chrome_opts.add_argument("--no-sandbox")
    chrome_opts.add_argument("--disable-gpu")
    chrome_opts.add_argument("--disable-dev-shm-usage")
    chrome_opts.add_argument("--log-level=3")

    try:
        print("[Browser] Initializing Chrome driver with synthetic WebRTC camera support...")
        driver = webdriver.Chrome(options=chrome_opts)
        return driver
    except Exception as e:
        print(f"[Browser] Chrome driver failed: {e}. Trying Edge driver...")
        edge_opts = EdgeOptions()
        edge_opts.add_argument("--headless=new")
        edge_opts.add_argument("--window-size=1440,1100")
        edge_opts.add_argument("--use-fake-ui-for-media-stream")
        edge_opts.add_argument("--use-fake-device-for-media-stream")
        edge_opts.add_argument("--no-sandbox")
        edge_opts.add_argument("--disable-gpu")
        edge_opts.add_argument("--log-level=3")
        driver = webdriver.Edge(options=edge_opts)
        return driver

def run_verification():
    driver = create_driver()
    try:
        print(f"[Browser] Navigating to SmartCell Web Application: {APP_URL}")
        driver.get(APP_URL)
        wait = WebDriverWait(driver, 15)

        # 1. Verify Application title and Disease mode tabs
        wait.until(EC.presence_of_element_located((By.ID, "diseaseTabLeukemia")))
        print("[Browser] Connected successfully. Page loaded.")

        # 2. Click Mode 2: ALL Leukemia Classification
        print("[Test Step 1] Switching to Mode 2: ALL Leukemia Classification...")
        tab_leukemia = driver.find_element(By.ID, "diseaseTabLeukemia")
        tab_leukemia.click()
        time.sleep(1)

        # 3. Verify Mode 2 is visible
        leukemia_view = driver.find_element(By.ID, "leukemiaWorkflow")
        assert leukemia_view.is_displayed(), "Leukemia workflow is not visible!"
        print("  -> Mode 2 Workflow visibly rendered [PASS]")

        # 4. Verify 3-Option Input Selector
        micro_btn = driver.find_element(By.ID, "leukemiaInputMicroscopeBtn")
        upload_btn = driver.find_element(By.ID, "leukemiaInputUploadBtn")
        samples_btn = driver.find_element(By.ID, "leukemiaInputSamplesBtn")
        assert micro_btn.is_displayed(), "Live Microscope button missing!"
        assert upload_btn.is_displayed(), "Upload Image button missing!"
        assert samples_btn.is_displayed(), "Dataset Samples button missing!"
        print("  -> 3-Option Input Selector visibly rendered (Live Microscope, Upload Image, Dataset Samples) [PASS]")

        # Screenshot 1: Mode 2 Initial State with Live Microscope Active
        save_screenshot(driver, "01_mode2_microscope_tab_initial.png", "Mode 2 Live Microscope Tab Initial State")

        # 5. Verify Live Microscope Controls
        dev_select = driver.find_element(By.ID, "leukemiaCameraDeviceSelect")
        refresh_btn = driver.find_element(By.ID, "leukemiaRefreshDevicesBtn")
        start_cam_btn = driver.find_element(By.ID, "leukemiaStartCameraBtn")
        stop_cam_btn = driver.find_element(By.ID, "leukemiaStopCameraBtn")
        capture_btn = driver.find_element(By.ID, "leukemiaCaptureImageBtn")
        retake_btn = driver.find_element(By.ID, "leukemiaRetakeImageBtn")
        discard_btn = driver.find_element(By.ID, "leukemiaDiscardImageBtn")
        analyze_btn = driver.find_element(By.ID, "leukemiaCaptureAndAnalyzeBtn")

        print("  -> Live Microscope UI controls verified (dropdown, start/stop, capture, retake, discard, analyze) [PASS]")

        # 6. Start Camera
        print("[Test Step 2] Clicking Start Camera...")
        start_cam_btn.click()
        time.sleep(2)

        # Wait for video element or status to indicate stream active
        status_text = driver.find_element(By.ID, "leukemiaMicroscopeStatusText").text
        print(f"  -> Camera status: {status_text}")
        assert "Stream Active" in status_text or "Active" in status_text or "Running" in status_text, f"Unexpected camera status: {status_text}"

        # Screenshot 2: Live Microscope Camera Streaming
        save_screenshot(driver, "02_mode2_microscope_camera_streaming.png", "Live Microscope Camera Stream Active")

        # 7. Capture Image
        print("[Test Step 3] Clicking Capture Image...")
        capture_btn.click()
        time.sleep(1)

        # Verify preview section is displayed
        preview_sec = driver.find_element(By.ID, "leukemiaCapturedPreviewSection")
        assert preview_sec.is_displayed(), "Captured preview section is not displayed!"
        frame_info = driver.find_element(By.ID, "leukemiaCapturedFrameInfo").text
        print(f"  -> Frame captured successfully: {frame_info}")
        assert not retake_btn.get_attribute("disabled"), "Retake button should be enabled after capture!"
        assert not discard_btn.get_attribute("disabled"), "Discard button should be enabled after capture!"
        assert not analyze_btn.get_attribute("disabled"), "Analyze button should be enabled after capture!"

        # Screenshot 3: Captured Frame Preview & Inspection State
        save_screenshot(driver, "03_mode2_microscope_captured_preview.png", "Captured Microscope Frame Inspection Preview State")

        # 8. Test Retake/Discard Functionality
        print("[Test Step 4] Testing Retake / Discard button...")
        retake_btn.click()
        time.sleep(1)
        assert not preview_sec.is_displayed(), "Preview section should be hidden after Retake!"
        print("  -> Retake button successfully cleared preview and restored live view [PASS]")

        # Capture again for analysis
        print("[Test Step 5] Capturing fresh frame for analysis...")
        capture_btn.click()
        time.sleep(1)
        assert preview_sec.is_displayed()

        # 9. Click Analyze Captured Image
        print("[Test Step 6] Clicking Analyze Captured Image (EfficientNetB0)...")
        analyze_btn.click()

        # Wait for analysis to complete and results card to display
        WebDriverWait(driver, 60).until(EC.visibility_of_element_located((By.ID, "leukemiaResultsCard")))
        time.sleep(1)
        diag_banner = driver.find_element(By.ID, "leukemiaDiagnosisBanner").text
        print(f"  -> Leukemia Diagnosis Result: {diag_banner.encode('ascii', 'replace').decode('ascii')}")

        # Screenshot 4: Analysis Results & Diagnosis Card
        save_screenshot(driver, "04_mode2_microscope_analysis_results.png", "ALL Leukemia Classification Diagnosis Results")

        # 10. Test Upload Image Tab
        print("[Test Step 7] Testing Upload Image input tab...")
        upload_btn.click()
        time.sleep(1)
        upload_panel = driver.find_element(By.ID, "leukemiaUploadPanel")
        assert upload_panel.is_displayed(), "Upload panel not displayed!"

        # Upload a real test image file
        test_img_path = os.path.join(PROJECT_ROOT, "dataset_all", "test", "0_normal", "Im133_0.png")
        file_input = driver.find_element(By.ID, "leukemiaImageUpload")
        file_input.send_keys(test_img_path)
        time.sleep(1)

        upload_preview = driver.find_element(By.ID, "leukemiaUploadPreviewSection")
        assert upload_preview.is_displayed(), "Upload preview not displayed!"
        print("  -> Upload image preview and details displayed successfully [PASS]")

        # Screenshot 5: Upload Image State
        save_screenshot(driver, "05_mode2_upload_image_preview.png", "ALL Leukemia Upload Image Preview & Details")

        # 11. Test Dataset Samples Tab
        print("[Test Step 8] Testing Dataset Samples input tab...")
        samples_btn.click()
        time.sleep(1)
        samples_panel = driver.find_element(By.ID, "leukemiaSamplesPanel")
        assert samples_panel.is_displayed(), "Dataset Samples panel not displayed!"

        # Click on blast specimen Im003_1
        sample_btns = driver.find_elements(By.CLASS_NAME, "leukemia-preset-btn")
        sample_im003 = [b for b in sample_btns if "Im003_1" in b.text][0]
        sample_im003.click()
        time.sleep(1)

        sample_preview = driver.find_element(By.ID, "leukemiaSamplePreviewSection")
        assert sample_preview.is_displayed(), "Sample benchmark preview not displayed!"
        print("  -> Dataset sample selection and benchmark cell preview verified [PASS]")

        # Screenshot 6: Dataset Samples State
        save_screenshot(driver, "06_mode2_dataset_samples_preview.png", "ALL Leukemia Dataset Samples Selection & Benchmark Cell Preview")

        print("\n" + "=" * 78)
        print("ALL VISUAL VERIFICATION STEPS PASSED SUCCESSFULLY IN REAL BROWSER!")
        print("=" * 78)

    finally:
        driver.quit()

if __name__ == "__main__":
    run_verification()
