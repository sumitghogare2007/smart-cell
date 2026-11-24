/**
 * SmartCell — Hugging Face Static Deployment Configuration
 * 
 * In a static hosting environment (such as Hugging Face Static Spaces),
 * this configuration specifies the connection parameters for the external
 * SmartCell AI inference backend.
 * 
 * If API_BASE_URL is empty (""), the frontend runs in standalone/preview mode.
 * Users can also configure the backend URL dynamically in the web UI.
 */

window.SMARTCELL_CONFIG = {
  // Default Backend API Base URL (leave empty "" for unconfigured / standalone mode)
  // Example: "https://your-smartcell-backend.hf.space" or "http://127.0.0.1:8501"
  API_BASE_URL: "",

  // Standard SmartCell REST API Endpoints
  ENDPOINTS: {
    health: "/api/health",
    status: "/api/status",
    analyze: "/api/analyze",
    file: "/api/file"
  },

  // Connection & Inference Request Timeouts
  HEALTH_CHECK_TIMEOUT_MS: 5000,
  ANALYSIS_TIMEOUT_MS: 120000, // 2 minutes for deep-learning two-stage whole-smear inference

  // File Upload Guardrails
  MAX_UPLOAD_SIZE_BYTES: 35 * 1024 * 1024, // 35 MB
  ALLOWED_IMAGE_EXTENSIONS: [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"],

  // Optical Quality Thresholds (Pre-Inference Microscope Guardrails)
  QUALITY_GATES: {
    MIN_LUMINANCE: 15,    // Below this indicates a black frame / closed lens cap
    MAX_LUMINANCE: 245,   // Above this indicates an oversaturated blank field
    MIN_LAPLACIAN_VAR: 85 // Below this warns of severe optical defocus / blur
  },

  // Helper: Retrieve active API base URL (checks localStorage override first, then window.SMARTCELL_CONFIG)
  getActiveApiUrl: function() {
    try {
      const stored = localStorage.getItem("smartcell_api_base_url");
      if (stored !== null && stored.trim() !== "") {
        return stored.trim().replace(/\/+$/, "");
      }
    } catch (e) {
      // localStorage may be disabled in restricted iframe sandboxes
    }
    return (this.API_BASE_URL || "").trim().replace(/\/+$/, "");
  },

  // Helper: Save active API base URL to localStorage
  setActiveApiUrl: function(url) {
    if (url === undefined || url === null) {
      try { localStorage.removeItem("smartcell_api_base_url"); } catch (e) {}
      return "";
    }
    const trimmed = String(url).trim();
    if (!trimmed) {
      try { localStorage.removeItem("smartcell_api_base_url"); } catch (e) {}
      return "";
    }
    let cleanUrl = "";
    try {
      const parsed = new URL(trimmed);
      if (parsed.protocol === "http:" || parsed.protocol === "https:") {
        const normPath = (parsed.pathname === "/" ? "" : parsed.pathname.replace(/\/+$/, ""));
        cleanUrl = `${parsed.origin}${normPath}`;
      } else {
        console.warn("Rejected unsupported protocol in setActiveApiUrl:", parsed.protocol);
        return "";
      }
    } catch (err) {
      console.warn("Rejected malformed URL in setActiveApiUrl:", trimmed);
      return "";
    }

    try {
      if (cleanUrl) {
        localStorage.setItem("smartcell_api_base_url", cleanUrl);
      } else {
        localStorage.removeItem("smartcell_api_base_url");
      }
    } catch (e) {
      console.warn("Could not save to localStorage:", e);
    }
    return cleanUrl;
  }
};
