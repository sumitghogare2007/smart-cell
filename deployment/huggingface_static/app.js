/**
 * SmartCell — Client-Side Application Controller
 * Handles UI interactions, WebRTC digital microscope streaming,
 * optical quality validation, and external AI backend routing.
 */

// ============================================================================
// BACKEND CONNECTION MANAGER & SETTINGS MODAL
// ============================================================================
const SmartCellBackendManager = {
  status: 'unconfigured', // 'unconfigured' | 'offline' | 'connected' | 'processing' | 'failed' | 'timeout' | 'invalid'
  statusText: 'Backend: Not Configured',
  lastLatencyMs: null,

  init() {
    this.bindEvents();
    this.updateStatusDisplay();
    this.checkHealth();
  },

  /**
   * Validates and normalizes candidate backend URL.
   * Accepts only http:// and https:// schemes, rejects unsupported protocols & malformed strings.
   * Strips redundant trailing slashes.
   */
  validateUrl(rawInput) {
    if (rawInput === undefined || rawInput === null) {
      return { isValid: true, isEmpty: true, normalizedUrl: '', error: null };
    }
    const trimmed = String(rawInput).trim();
    if (trimmed === '') {
      return { isValid: true, isEmpty: true, normalizedUrl: '', error: null };
    }

    let parsed;
    try {
      parsed = new URL(trimmed);
    } catch (err) {
      return {
        isValid: false,
        isEmpty: false,
        normalizedUrl: null,
        error: 'Malformed URL: Please enter a valid address with protocol (e.g. http://127.0.0.1:8501 or https://my-backend.hf.space).'
      };
    }

    const protocol = parsed.protocol.toLowerCase();
    if (protocol !== 'http:' && protocol !== 'https:') {
      return {
        isValid: false,
        isEmpty: false,
        normalizedUrl: null,
        error: `Unsupported protocol "${parsed.protocol}": Only http:// and https:// URLs are permitted.`
      };
    }

    if (!parsed.hostname || parsed.hostname.trim() === '') {
      return {
        isValid: false,
        isEmpty: false,
        normalizedUrl: null,
        error: 'Invalid URL: Hostname cannot be empty.'
      };
    }

    let normalizedPath = parsed.pathname;
    if (normalizedPath === '/') {
      normalizedPath = '';
    } else {
      normalizedPath = normalizedPath.replace(/\/+$/, '');
    }

    const normalizedUrl = `${parsed.origin}${normalizedPath}`;
    return { isValid: true, isEmpty: false, normalizedUrl, error: null };
  },

  getApiBaseUrl() {
    if (window.SMARTCELL_CONFIG && typeof window.SMARTCELL_CONFIG.getActiveApiUrl === 'function') {
      return window.SMARTCELL_CONFIG.getActiveApiUrl();
    }
    return '';
  },

  /**
   * Public setter for active API URL with validation.
   */
  setActiveApiUrl(url) {
    const val = this.validateUrl(url);
    if (!val.isValid) {
      return { success: false, error: val.error, url: null };
    }
    const cleanUrl = val.normalizedUrl;
    if (window.SMARTCELL_CONFIG && typeof window.SMARTCELL_CONFIG.setActiveApiUrl === 'function') {
      window.SMARTCELL_CONFIG.setActiveApiUrl(cleanUrl);
    }
    const input = document.getElementById('backendUrlInput');
    if (input) input.value = cleanUrl;
    this.checkHealth();
    return { success: true, error: null, url: cleanUrl };
  },

  /**
   * Safely render status message into DOM without using unsafe innerHTML.
   * Completely eliminates XSS / HTML injection risks.
   */
  setSafeStatusMessage(container, alertClass, strongPrefix, messageText) {
    if (!container) return;
    container.className = `modal-test-status show ${alertClass}`;
    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }
    if (strongPrefix) {
      const strong = document.createElement('strong');
      strong.textContent = strongPrefix;
      container.appendChild(strong);
      container.appendChild(document.createTextNode(' '));
    }
    if (messageText) {
      container.appendChild(document.createTextNode(messageText));
    }
  },

  bindEvents() {
    const pill = document.getElementById('backendStatusPill');
    const openBtn = document.getElementById('openConfigModalBtn');
    const closeBtn = document.getElementById('closeConfigModalBtn');
    const modal = document.getElementById('backendConfigModal');
    const testBtn = document.getElementById('testBackendBtn');
    const saveBtn = document.getElementById('saveBackendBtn');

    if (pill) pill.addEventListener('click', () => this.openModal());
    if (openBtn) openBtn.addEventListener('click', () => this.openModal());
    if (closeBtn) closeBtn.addEventListener('click', () => this.closeModal());
    if (modal) {
      modal.addEventListener('click', (e) => {
        if (e.target === modal) this.closeModal();
      });
    }

    if (testBtn) testBtn.addEventListener('click', () => this.testConnectionFromModal());
    if (saveBtn) saveBtn.addEventListener('click', () => this.saveFromModal());
  },

  openModal() {
    const modal = document.getElementById('backendConfigModal');
    const input = document.getElementById('backendUrlInput');
    const statusBox = document.getElementById('modalTestStatus');
    if (input) input.value = this.getApiBaseUrl();
    if (statusBox) {
      statusBox.className = 'modal-test-status';
      while (statusBox.firstChild) {
        statusBox.removeChild(statusBox.firstChild);
      }
    }
    if (modal) modal.classList.add('active');
  },

  closeModal() {
    const modal = document.getElementById('backendConfigModal');
    if (modal) modal.classList.remove('active');
  },

  async testConnectionFromModal() {
    const input = document.getElementById('backendUrlInput');
    const statusBox = document.getElementById('modalTestStatus');
    const testBtn = document.getElementById('testBackendBtn');
    const rawVal = input ? input.value : '';

    const validation = this.validateUrl(rawVal);
    if (!validation.isValid) {
      this.setSafeStatusMessage(statusBox, 'alert-error', 'Invalid URL:', validation.error);
      return;
    }

    if (validation.isEmpty) {
      this.setSafeStatusMessage(statusBox, 'alert-warn', 'No URL Entered:', 'Please enter a valid HTTP/HTTPS address, or leave empty to operate in standalone/offline preview mode.');
      return;
    }

    const candidateUrl = validation.normalizedUrl;
    if (testBtn) {
      testBtn.disabled = true;
      testBtn.textContent = 'Testing...';
    }

    let timeoutId = null;
    const start = performance.now();
    try {
      const controller = new AbortController();
      timeoutId = setTimeout(() => controller.abort(), 6000);

      const resp = await fetch(`${candidateUrl}/api/health`, {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
        signal: controller.signal
      });

      const latency = Math.round(performance.now() - start);

      if (resp.ok) {
        let data = {};
        try { data = await resp.json(); } catch (err) { data = {}; }
        const serviceName = String(data.service || 'SmartCell Backend');

        // Accurate health-check wording: indicates reachability, not guaranteed model readiness
        this.setSafeStatusMessage(
          statusBox,
          'alert-success',
          `Reachable (${latency} ms):`,
          `Backend server "${serviceName}" is online and reachable. Note: AI model readiness and GPU status are verified by the server during screening.`
        );
      } else {
        const statusText = resp.statusText || 'Unexpected error status';
        this.setSafeStatusMessage(
          statusBox,
          'alert-error',
          `Server Error (HTTP ${resp.status}):`,
          `Endpoint responded with status ${resp.status} (${statusText}). Verify the backend service is running correctly.`
        );
      }
    } catch (err) {
      const isTimeout = (err && (err.name === 'AbortError' || err.message === 'The operation was aborted.'));
      if (isTimeout) {
        this.setSafeStatusMessage(
          statusBox,
          'alert-error',
          'Connection Timed Out:',
          `Request to "${candidateUrl}" timed out after 6 seconds. Check if the server is running and accessible over your network.`
        );
      } else {
        const errorMsg = err && err.message ? err.message : 'Network unreachable';
        this.setSafeStatusMessage(
          statusBox,
          'alert-error',
          'Connection Failed:',
          `Could not reach "${candidateUrl}" (${errorMsg}). If using a local server, ensure it is running and permits Cross-Origin Resource Sharing (CORS).`
        );
      }
    } finally {
      if (timeoutId) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }
      if (testBtn) {
        testBtn.disabled = false;
        testBtn.textContent = 'Test Connection';
      }
    }
  },

  async saveFromModal() {
    const input = document.getElementById('backendUrlInput');
    const statusBox = document.getElementById('modalTestStatus');
    const rawVal = input ? input.value : '';

    const validation = this.validateUrl(rawVal);
    if (!validation.isValid) {
      this.setSafeStatusMessage(statusBox, 'alert-error', 'Cannot Save Invalid URL:', validation.error);
      return;
    }

    const cleanUrl = validation.normalizedUrl;
    if (window.SMARTCELL_CONFIG && typeof window.SMARTCELL_CONFIG.setActiveApiUrl === 'function') {
      window.SMARTCELL_CONFIG.setActiveApiUrl(cleanUrl);
    }
    this.closeModal();
    await this.checkHealth();
  },

  async checkHealth() {
    const rawUrl = this.getApiBaseUrl();
    const validation = this.validateUrl(rawUrl);

    if (!validation.isValid) {
      this.status = 'invalid';
      this.statusText = 'Backend: Invalid URL';
      this.updateStatusDisplay();
      return;
    }

    if (validation.isEmpty) {
      this.status = 'unconfigured';
      this.statusText = 'Backend: Not Configured';
      this.updateStatusDisplay();
      return;
    }

    const apiUrl = validation.normalizedUrl;
    let timeoutId = null;
    const start = performance.now();

    try {
      const controller = new AbortController();
      timeoutId = setTimeout(() => controller.abort(), 5000);

      const resp = await fetch(`${apiUrl}/api/health`, {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
        signal: controller.signal
      });

      this.lastLatencyMs = Math.round(performance.now() - start);

      if (resp.ok) {
        this.status = 'connected';
        this.statusText = `Backend: Online (${this.lastLatencyMs}ms)`;
        this.updateStatusDisplay();
        await this.queryEnvironmentStatus();
      } else {
        this.status = 'failed';
        this.statusText = `Backend: Error (${resp.status})`;
        this.updateStatusDisplay();
      }
    } catch (e) {
      const isTimeout = (e && (e.name === 'AbortError' || e.message === 'The operation was aborted.'));
      if (isTimeout) {
        this.status = 'timeout';
        this.statusText = 'Backend: Timed Out';
      } else {
        this.status = 'offline';
        this.statusText = 'Backend: Offline';
      }
      this.updateStatusDisplay();
    } finally {
      if (timeoutId) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }
    }
  },

  updateStatusDisplay() {
    const dot = document.getElementById('statusIndicatorDot');
    const text = document.getElementById('statusIndicatorText');
    const pill = document.getElementById('backendStatusPill');

    if (dot) {
      let dotClass = 'dot-unconfigured';
      if (this.status === 'connected') dotClass = 'dot-connected';
      else if (this.status === 'offline' || this.status === 'timeout') dotClass = 'dot-offline';
      else if (this.status === 'failed' || this.status === 'invalid') dotClass = 'dot-failed';
      else if (this.status === 'processing') dotClass = 'dot-processing';

      dot.className = `status-indicator-dot ${dotClass}`;
    }
    if (text) {
      text.textContent = this.statusText;
    }
    if (pill) {
      pill.title = `${this.statusText} (Click to configure backend URL)`;
    }
  },

  async queryEnvironmentStatus() {
    const apiUrl = this.getApiBaseUrl();
    if (!apiUrl || this.status !== 'connected') return;

    try {
      const resp = await fetch(`${apiUrl}/api/status`);
      if (resp.ok) {
        const data = await resp.json();
        ENV_STATUS = data;

        const g = data && data.gpu_environment;
        const badges = document.getElementById('hwBadges');
        if (badges && g) {
          while (badges.firstChild) {
            badges.removeChild(badges.firstChild);
          }
          const gpuSpan = document.createElement('span');
          gpuSpan.className = `badge ${g.cuda_available ? 'gpu-ok' : 'cpu-ok'}`;
          gpuSpan.textContent = g.cuda_available ? `Stage 1 GPU: ${g.gpu_name}` : `Stage 1: ${g.gpu_name} (CPU)`;

          const cpuSpan = document.createElement('span');
          cpuSpan.className = 'badge cpu-ok';
          cpuSpan.textContent = 'Stage 2: Classifier Engine Active';

          badges.appendChild(gpuSpan);
          badges.appendChild(cpuSpan);
        }

        if (typeof populatePresetDropdown === 'function') {
          populatePresetDropdown();
        }
      }
    } catch (e) {
      // Non-critical background status poll
    }
  }
};

window.addEventListener('DOMContentLoaded', () => {
  SmartCellBackendManager.init();
});


// ============================================================================
// SMARTCELL CLIENT UI, WEBRTC MICROSCOPE & INFERENCE PIPELINE
// ============================================================================
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

    const apiUrl = SmartCellBackendManager.getApiBaseUrl();
    if (!apiUrl || SmartCellBackendManager.status === 'unconfigured' || SmartCellBackendManager.status === 'offline') {
      const errBox = (ACTIVE_DISEASE === 'malaria') ? els.errBox : document.getElementById('leukemiaErrorAlert');
      const msg = `<div class="alert-error" style="padding:16px; border-radius:var(--radius); margin-top:14px; text-align:left;">
        <strong>AI backend is not connected. Live analysis is currently unavailable.</strong><br>
        <span style="font-size:0.85rem; color:var(--text); display:inline-block; margin-top:6px; line-height:1.5;">
          SmartCell deep learning models (YOLOv8n and CNNs) require an active Python backend.
          Please configure an active backend API URL in <a href="javascript:void(0)" onclick="SmartCellBackendManager.openModal()" style="color:var(--accent); text-decoration:underline; font-weight:600;">Settings</a> or launch the local backend with <code>launch_smartcell_demo.bat</code>.
        </span>
      </div>`;
      if (errBox) errBox.innerHTML = msg;
      return;
    }
    const resp = await fetch(`${apiUrl}/api/analyze`, { method: 'POST', body: fd });
    const data = await resp.json();

    if (!data.success) {
      if (errBox) {
        errBox.innerHTML = `<strong>${data.error_type || 'Analysis Error'}:</strong> ${data.error_message || 'Inference failed.'}`;
        errBox.style.display = 'block';
      }
      setMicroscopeStatus('err', 'Analysis Error — See Details Below', 'Error', 'leukemia');
      return;
    }

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

    const apiUrl = SmartCellBackendManager.getApiBaseUrl();
    if (!apiUrl || SmartCellBackendManager.status === 'unconfigured' || SmartCellBackendManager.status === 'offline') {
      const errBox = (ACTIVE_DISEASE === 'malaria') ? els.errBox : document.getElementById('leukemiaErrorAlert');
      const msg = `<div class="alert-error" style="padding:16px; border-radius:var(--radius); margin-top:14px; text-align:left;">
        <strong>AI backend is not connected. Live analysis is currently unavailable.</strong><br>
        <span style="font-size:0.85rem; color:var(--text); display:inline-block; margin-top:6px; line-height:1.5;">
          SmartCell deep learning models (YOLOv8n and CNNs) require an active Python backend.
          Please configure an active backend API URL in <a href="javascript:void(0)" onclick="SmartCellBackendManager.openModal()" style="color:var(--accent); text-decoration:underline; font-weight:600;">Settings</a> or launch the local backend with <code>launch_smartcell_demo.bat</code>.
        </span>
      </div>`;
      if (errBox) errBox.innerHTML = msg;
      return;
    }
    const resp = await fetch(`${apiUrl}/api/analyze`, { method: 'POST', body: fd });
    const data = await resp.json();

    if (!data.success) {
      setMicroscopeStatus('err', 'Analysis Failed', 'Error');
      errBox.innerHTML = `<div class="alert-error"><strong>${data.error_type || 'Analysis Error'}:</strong> ${data.error_message || 'Unknown error occurred.'}</div>`;
      return;
    }

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
    const apiUrl = SmartCellBackendManager.getApiBaseUrl(); if (!apiUrl) return; const resp = await fetch(`${apiUrl}/api/status`);
    ENV_STATUS = await resp.json();
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

    const apiUrl = SmartCellBackendManager.getApiBaseUrl();
    if (!apiUrl || SmartCellBackendManager.status === 'unconfigured' || SmartCellBackendManager.status === 'offline') {
      const errBox = (ACTIVE_DISEASE === 'malaria') ? els.errBox : document.getElementById('leukemiaErrorAlert');
      const msg = `<div class="alert-error" style="padding:16px; border-radius:var(--radius); margin-top:14px; text-align:left;">
        <strong>AI backend is not connected. Live analysis is currently unavailable.</strong><br>
        <span style="font-size:0.85rem; color:var(--text); display:inline-block; margin-top:6px; line-height:1.5;">
          SmartCell deep learning models (YOLOv8n and CNNs) require an active Python backend.
          Please configure an active backend API URL in <a href="javascript:void(0)" onclick="SmartCellBackendManager.openModal()" style="color:var(--accent); text-decoration:underline; font-weight:600;">Settings</a> or launch the local backend with <code>launch_smartcell_demo.bat</code>.
        </span>
      </div>`;
      if (errBox) errBox.innerHTML = msg;
      return;
    }
    const resp = await fetch(`${apiUrl}/api/analyze`, { method: 'POST', body: fd });
    const data = await resp.json();

    if (!data.success) {
      errBox.innerHTML = `<div class="alert-error"><strong>${data.error_type || 'Analysis Error'}:</strong> ${data.error_message || 'Unknown error occurred.'}</div>`;
      return;
    }

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
    : `${SmartCellBackendManager.getApiBaseUrl()}/api/file?run_id=${encodeURIComponent(data.run_id)}&file=${encodeURIComponent(pdfFilename)}&download=1`;
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
    const cropUrl = `${SmartCellBackendManager.getApiBaseUrl()}/api/file?run_id=${CURRENT_RESULT.run_id}&file=${encodeURIComponent(r.crop_rel_path.replace(/\\/g, '/'))}`;
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
    const apiUrl = SmartCellBackendManager.getApiBaseUrl();
    if (!apiUrl || SmartCellBackendManager.status === 'unconfigured' || SmartCellBackendManager.status === 'offline') {
      const errBox = (ACTIVE_DISEASE === 'malaria') ? els.errBox : document.getElementById('leukemiaErrorAlert');
      const msg = `<div class="alert-error" style="padding:16px; border-radius:var(--radius); margin-top:14px; text-align:left;">
        <strong>AI backend is not connected. Live analysis is currently unavailable.</strong><br>
        <span style="font-size:0.85rem; color:var(--text); display:inline-block; margin-top:6px; line-height:1.5;">
          SmartCell deep learning models (YOLOv8n and CNNs) require an active Python backend.
          Please configure an active backend API URL in <a href="javascript:void(0)" onclick="SmartCellBackendManager.openModal()" style="color:var(--accent); text-decoration:underline; font-weight:600;">Settings</a> or launch the local backend with <code>launch_smartcell_demo.bat</code>.
        </span>
      </div>`;
      if (errBox) errBox.innerHTML = msg;
      return;
    }
    const resp = await fetch(`${apiUrl}/api/analyze`, { method: 'POST', body: fd });
    const data = await resp.json();

    if (!data.success) {
      if (errBox) {
        errBox.innerHTML = `<strong>${data.error_type || 'Analysis Error'}:</strong> ${data.error_message || 'Inference failed.'}`;
        errBox.style.display = 'block';
      }
      return;
    }

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