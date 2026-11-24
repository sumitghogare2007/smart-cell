import os

def patch():
    app_path = r"D:\smartcell\scripts\smartcell_demo_app.py"
    with open(app_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Add active-leukemia button CSS
    css_needle = ".mode-btn.active {"
    css_add = """  .mode-btn.active-leukemia {
    background: linear-gradient(135deg, #b91c1c, #dc2626) !important;
    color: #fff !important;
    box-shadow: 0 2px 8px rgba(220, 38, 38, 0.4) !important;
  }
"""
    if ".mode-btn.active-leukemia" not in content:
        content = content.replace(css_needle, css_add + "  " + css_needle)
        print("1. Added CSS for active-leukemia mode button")

    # 2. Add discardBtn to getCameraElements
    get_cam_needle = "      retakeBtn: document.getElementById('leukemiaRetakeImageBtn'),"
    get_cam_add = """      retakeBtn: document.getElementById('leukemiaRetakeImageBtn'),
      discardBtn: document.getElementById('leukemiaDiscardImageBtn'),"""
    if "discardBtn: document.getElementById('leukemiaDiscardImageBtn')" not in content:
        content = content.replace(get_cam_needle, get_cam_add)
        print("2. Added discardBtn to getCameraElements leukemia")

    # 3. Add discard button in HTML
    html_needle = """        <!-- Capture / Retake Controls -->
        <div class="btn-row" style="margin-top:10px;">
          <button type="button" class="btn-secondary" id="leukemiaCaptureImageBtn" onclick="captureLeukemiaFrame()" disabled style="background:#2563eb; color:#fff; border-color:#3b82f6;">
            📸 Capture Image
          </button>
          <button type="button" class="btn-secondary" id="leukemiaRetakeImageBtn" onclick="retakeLeukemiaFrame()" disabled>
            🔄 Retake / Discard
          </button>
        </div>"""

    html_replacement = """        <!-- Capture / Retake / Discard Controls -->
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
        </div>"""

    if html_needle in content:
        content = content.replace(html_needle, html_replacement)
        print("3. Updated HTML capture/retake/discard button row in leukemiaLiveMicroscopePanel")

    # 4. Update Analyze Captured Button text to 'Analyze Captured Image'
    analyze_needle = "Analyze Captured Cell (EfficientNetB0)"
    if analyze_needle in content:
        content = content.replace(analyze_needle, "Analyze Captured Image (EfficientNetB0)")
        print("4. Updated analyze button label to Analyze Captured Image")

    # 5. Enable/disable discardBtn in captureMicroscopeFrame and retakeMicroscopeFrame
    cap_code_needle = "  if (els.retakeBtn) els.retakeBtn.disabled = false;"
    if "if (els.discardBtn) els.discardBtn.disabled = false;" not in content and cap_code_needle in content:
        content = content.replace(cap_code_needle, cap_code_needle + "\n  if (els.discardBtn) els.discardBtn.disabled = false;")
        print("5. Updated captureMicroscopeFrame to enable discardBtn")

    ret_code_needle = "  if (els.retakeBtn) els.retakeBtn.disabled = true;"
    if "if (els.discardBtn) els.discardBtn.disabled = true;" not in content and ret_code_needle in content:
        content = content.replace(ret_code_needle, ret_code_needle + "\n  if (els.discardBtn) els.discardBtn.disabled = true;")
        print("6. Updated retakeMicroscopeFrame to disable discardBtn")

    # 6. Add discardLeukemiaFrame function
    alias_needle = """function retakeLeukemiaFrame() {
  return retakeMicroscopeFrame('leukemia');
}"""
    alias_add = """function retakeLeukemiaFrame() {
  return retakeMicroscopeFrame('leukemia');
}

function discardLeukemiaFrame() {
  return retakeMicroscopeFrame('leukemia');
}"""
    if "function discardLeukemiaFrame" not in content:
        content = content.replace(alias_needle, alias_add)
        print("7. Added discardLeukemiaFrame alias")

    # 7. Update onLeukemiaFileChange for image details preview
    upload_fn_needle = """function onLeukemiaFileChange() {
  const fileInput = document.getElementById('leukemiaImageUpload');
  if (fileInput && fileInput.files.length > 0) {
    const file = fileInput.files[0];
    selectedLeukemiaPreset = '';
    document.querySelectorAll('.leukemia-preset-btn').forEach(b => b.classList.remove('selected'));
    document.getElementById('leukemiaSelectedName').textContent = 'Uploaded file: ' + file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)';

    const prevSection = document.getElementById('leukemiaUploadPreviewSection');
    const prevImg = document.getElementById('leukemiaUploadPreviewImg');
    if (prevSection && prevImg) {
      prevImg.src = URL.createObjectURL(file);
      prevSection.style.display = 'block';
    }
  }
}"""

    upload_fn_replacement = """function onLeukemiaFileChange() {
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
          metaBox.innerHTML = `<strong>File:</strong> ${file.name} &bull; <strong>Size:</strong> ${(file.size / 1024).toFixed(1)} KB &bull; <strong>Resolution:</strong> ${testImg.naturalWidth} × ${testImg.naturalHeight} px &bull; <strong>Format:</strong> ${file.type || file.name.split('.').pop().toUpperCase()}`;
        }
      };
      testImg.src = objUrl;
      prevSection.style.display = 'block';
    }
  }
}"""
    if upload_fn_needle in content:
        content = content.replace(upload_fn_needle, upload_fn_replacement)
        print("8. Updated onLeukemiaFileChange to show complete image details")

    # 8. Update selectLeukemiaPreset for benchmark preview
    preset_fn_needle = """function selectLeukemiaPreset(btn, filename) {
  selectedLeukemiaPreset = filename;
  document.querySelectorAll('.leukemia-preset-btn').forEach(b => b.classList.remove('selected'));
  btn.classList.add('selected');
  const fileInput = document.getElementById('leukemiaImageUpload');
  if (fileInput) fileInput.value = '';
  const hintEl = document.getElementById('leukemiaPresetSelectedHint');
  if (hintEl) hintEl.textContent = 'Selected: ' + filename;
  const prevSection = document.getElementById('leukemiaUploadPreviewSection');
  if (prevSection) prevSection.style.display = 'none';
}"""

    preset_fn_replacement = """function selectLeukemiaPreset(btn, filename) {
  selectedLeukemiaPreset = filename;
  document.querySelectorAll('.leukemia-preset-btn').forEach(b => b.classList.remove('selected'));
  btn.classList.add('selected');
  const fileInput = document.getElementById('leukemiaImageUpload');
  if (fileInput) fileInput.value = '';
  const isBlast = filename.includes('_1');
  const label = isBlast ? 'Lymphoblast (ALL Leukemia)' : 'Normal Lymphocyte (Healthy)';
  const hintEl = document.getElementById('leukemiaPresetSelectedHint');
  if (hintEl) hintEl.textContent = 'Selected: ' + filename + ' [' + label + ']';

  const prevSection = document.getElementById('leukemiaSamplePreviewSection');
  const prevImg = document.getElementById('leukemiaSamplePreviewImg');
  const badgeEl = document.getElementById('leukemiaSampleDetailsBadge');
  if (prevSection && prevImg) {
    prevImg.src = `/api/preset_image?filename=${encodeURIComponent(filename)}&disease=leukemia`;
    if (badgeEl) badgeEl.textContent = isBlast ? 'Blast Candidate (ALL)' : 'Normal Reference';
    prevSection.style.display = 'block';
  }
}"""
    if preset_fn_needle in content:
        content = content.replace(preset_fn_needle, preset_fn_replacement)
        print("9. Updated selectLeukemiaPreset to show benchmark preview")

    # 9. Update switchLeukemiaInputMode to apply active-leukemia class
    switch_fn_needle = """  if (btnMicro) btnMicro.classList.toggle('active', mode === 'microscope');
  if (btnUpload) btnUpload.classList.toggle('active', mode === 'upload');
  if (btnSamples) btnSamples.classList.toggle('active', mode === 'samples');"""

    switch_fn_replacement = """  if (btnMicro) {
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
  }"""
    if switch_fn_needle in content:
        content = content.replace(switch_fn_needle, switch_fn_replacement)
        print("10. Updated switchLeukemiaInputMode with active-leukemia styling")

    # 10. Add sample preview HTML and upload meta box
    upload_panel_old = """        <div id="leukemiaUploadPreviewSection" style="display:none; margin-top:10px;">
          <span style="font-size:0.8rem; font-weight:700; color:#93c5fd;">Upload Preview</span>
          <div class="video-stage" style="margin-top:4px; max-height:220px;">
            <img id="leukemiaUploadPreviewImg" alt="Uploaded Cell Preview" style="max-height:200px; width:auto; margin:0 auto; display:block;">
          </div>
        </div>"""

    upload_panel_new = """        <div id="leukemiaUploadPreviewSection" style="display:none; margin-top:10px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <span style="font-size:0.8rem; font-weight:700; color:#93c5fd;">Upload Preview &amp; Details</span>
            <span id="leukemiaUploadDetailsBadge" style="font-size:0.74rem; color:var(--muted);"></span>
          </div>
          <div class="video-stage" style="margin-top:2px; max-height:240px;">
            <img id="leukemiaUploadPreviewImg" alt="Uploaded Cell Preview" style="max-height:220px; width:auto; margin:0 auto; display:block;">
          </div>
          <div id="leukemiaUploadMetaBox" style="font-size:0.78rem; color:#cbd5e1; margin-top:6px; background:var(--panel-alt); padding:8px 12px; border-radius:6px; border:1px solid var(--border);"></div>
        </div>"""

    if upload_panel_old in content:
        content = content.replace(upload_panel_old, upload_panel_new)
        print("11. Updated leukemiaUploadPreviewSection in HTML")

    sample_panel_old = """        <div class="hint" id="leukemiaPresetSelectedHint" style="color:#86efac; font-weight:600; margin-bottom:10px;">Selected: None</div>

        <button type="button" class="btn btn-primary" id="btnRunLeukemiaSample\""""

    sample_panel_new = """        <div class="hint" id="leukemiaPresetSelectedHint" style="color:#86efac; font-weight:600; margin-bottom:10px;">Selected: None</div>

        <div id="leukemiaSamplePreviewSection" style="display:none; margin-top:8px; margin-bottom:10px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <span style="font-size:0.8rem; font-weight:700; color:#bae6fd;">Selected Benchmark Cell</span>
            <span id="leukemiaSampleDetailsBadge" style="font-size:0.74rem; color:var(--muted);"></span>
          </div>
          <div class="video-stage" style="margin-top:2px; max-height:220px;">
            <img id="leukemiaSamplePreviewImg" alt="Selected Benchmark Cell Preview" style="max-height:200px; width:auto; margin:0 auto; display:block;">
          </div>
        </div>

        <button type="button" class="btn btn-primary" id="btnRunLeukemiaSample\""""

    if sample_panel_old in content:
        content = content.replace(sample_panel_old, sample_panel_new)
        print("12. Added leukemiaSamplePreviewSection in HTML")

    # 11. Add /api/preset_image GET route in do_GET
    get_route_needle = """        if path == "/api/status":"""
    get_route_add = """        if path == "/api/preset_image":
            preset_fn = qs.get("filename", [""])[0]
            cand = resolve_preset_image_path(preset_fn)
            if not cand or not os.path.isfile(cand):
                self._send_json(404, {"error": "Preset image not found."})
                return
            ctype, _ = mimetypes.guess_type(cand)
            if not ctype:
                ctype = "image/png"
            with open(cand, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            self.wfile.write(data)
            return

        if path == "/api/status":"""

    if 'if path == "/api/preset_image":' not in content:
        content = content.replace(get_route_needle, get_route_add)
        print("13. Added /api/preset_image endpoint in do_GET")

    with open(app_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS: smartcell_demo_app.py patched completely.")

if __name__ == "__main__":
    patch()
