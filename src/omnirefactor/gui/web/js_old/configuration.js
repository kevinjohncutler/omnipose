window.addEventListener('error', (evt) => {
  const payload = {
    message: evt && evt.message ? evt.message : String(evt),
    filename: evt && evt.filename ? evt.filename : undefined,
    lineno: evt && evt.lineno ? evt.lineno : undefined,
    colno: evt && evt.colno ? evt.colno : undefined,
    stack: evt && evt.error && evt.error.stack ? String(evt.error.stack) : undefined,
  };
  try {
    console.error('[viewer] JS Error', payload);
  } catch (_) { /* ignore */ }
  try {
    fetch('/api/log', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type: 'JS_ERROR', payload })
    }).catch(() => {});
  } catch (err) {
    /* ignore */
  }
});

// Extracted from app.js lines 1-2112

const CONFIG = window.__OMNI_CONFIG__ || {};
const imgWidth = CONFIG.width || 0;
const imgHeight = CONFIG.height || 0;
const imageDataUrl = CONFIG.imageDataUrl || '';
const colorTable = CONFIG.colorTable || [];
const initialBrushRadius = CONFIG.brushRadius ?? 1; // default brush radius
const sessionId = CONFIG.sessionId || null;
const currentImagePath = CONFIG.imagePath || null;
const currentImageName = CONFIG.imageName || null;
const directoryEntries = Array.isArray(CONFIG.directoryEntries) ? CONFIG.directoryEntries : [];
const directoryIndex = typeof CONFIG.directoryIndex === 'number' ? CONFIG.directoryIndex : null;
const directoryPath = CONFIG.directoryPath || null;
const savedViewerState = CONFIG.savedViewerState || null;
const hasPrevImage = Boolean(CONFIG.hasPrev);
const hasNextImage = Boolean(CONFIG.hasNext);
let defaultPalette = [];
let nColorPalette = [];

// Centralized input policy so stylus, touch, and mouse behavior can be tuned per platform.
const POINTER_OPTIONS = {
  stylus: {
    allowSimultaneousTouchGestures: false,
    barrelButtonPan: true,
  },
  touch: {
    enablePan: true,
    enablePinchZoom: true,
    enableRotate: true,
    rotationDeadzoneDegrees: 0.1,
  },
  mouse: {
    primaryDraw: true,
    secondaryPan: true,
  },
};

const RAD_TO_DEG = 180 / Math.PI;
const USE_WEBGL_PIPELINE = CONFIG.useWebglPipeline ?? false;

const MAIN_WEBGL_CONTEXT_ATTRIBUTES = {
  alpha: true,
  antialias: true,
  premultipliedAlpha: true,
  preserveDrawingBuffer: false,
  depth: false,
  stencil: false,
  desynchronized: true,
};

function createPointerState(options = {}) {
  const merged = {
    stylus: {
      allowSimultaneousTouchGestures: false,
      barrelButtonPan: true,
      ...(options.stylus || {}),
    },
    touch: {
      enablePan: true,
      enablePinchZoom: true,
      enableRotate: true,
      rotationDeadzoneDegrees: 0.1,
      ...(options.touch || {}),
    },
    mouse: {
      primaryDraw: true,
      secondaryPan: true,
      ...(options.mouse || {}),
    },
  };
  let activePenId = null;
  let penButtons = 0;
  let penBarrelPan = false;

  function isStylusPointer(evt) {
    if (!evt) {
      return false;
    }
    if (evt.pointerType === 'pen') {
      return true;
    }
    if (evt.pointerType === 'touch' && typeof evt.touchType === 'string') {
      return evt.touchType.toLowerCase() === 'stylus';
    }
    return false;
  }

  function registerPenButtons(evt) {
    penButtons = typeof evt.buttons === 'number' ? evt.buttons : 0;
    penBarrelPan = merged.stylus.barrelButtonPan && (penButtons & ~1) !== 0;
  }

  return {
    options: merged,
    isStylusPointer,
    registerPointerDown(evt) {
      if (isStylusPointer(evt)) {
        activePenId = evt.pointerId;
        registerPenButtons(evt);
      }
    },
    registerPointerMove(evt) {
      if (isStylusPointer(evt) && evt.pointerId === activePenId) {
        registerPenButtons(evt);
      }
    },
    registerPointerUp(evt) {
      if (isStylusPointer(evt) && evt.pointerId === activePenId) {
        registerPenButtons(evt);
        activePenId = null;
        penBarrelPan = false;
        penButtons = 0;
      }
    },
    isActivePen(pointerId) {
      return activePenId !== null && pointerId === activePenId;
    },
    hasActivePen() {
      return activePenId !== null;
    },
    isBarrelPanActive() {
      return penBarrelPan;
    },
    shouldIgnoreTouch(evt) {
      if (!evt || evt.pointerType !== 'touch') {
        return false;
      }
      if (!merged.stylus.allowSimultaneousTouchGestures && activePenId !== null) {
        return true;
      }
      return false;
    },
    resetPen() {
      activePenId = null;
      penButtons = 0;
      penBarrelPan = false;
    },
  };
}

const supportsGestureEvents = typeof window !== 'undefined'
  && (typeof window.GestureEvent === 'function' || 'ongesturestart' in window);
const pointerOptionsOverride = CONFIG.pointerOptions || {};
const pointerState = createPointerState({
  stylus: { ...POINTER_OPTIONS.stylus, ...(pointerOptionsOverride.stylus || {}) },
  touch: { ...POINTER_OPTIONS.touch, ...(pointerOptionsOverride.touch || {}) },
  mouse: { ...POINTER_OPTIONS.mouse, ...(pointerOptionsOverride.mouse || {}) },
});
const debugTouchOverlay = CONFIG.debugTouchOverlay ?? false;

const canvas = document.getElementById('canvas');
let gl = null;
let ctx = null;
const webglPipelineRequested = USE_WEBGL_PIPELINE && typeof WebGL2RenderingContext !== 'undefined';
if (canvas && webglPipelineRequested) {
  try {
    gl = canvas.getContext('webgl2', MAIN_WEBGL_CONTEXT_ATTRIBUTES);
  } catch (err) {
    console.warn('WebGL pipeline context creation failed:', err);
    gl = null;
  }
  if (!gl) {
    console.warn('WebGL pipeline requested but unavailable, falling back to 2D canvas.');
  } else {
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  }
}
if (canvas && !gl) {
  ctx = canvas.getContext('2d');
}
const viewer = document.getElementById('viewer');
const dropOverlay = document.getElementById('dropOverlay');
if (viewer) {
  viewer.setAttribute('tabindex', '0');
  const focusViewer = () => {
    try {
      viewer.focus({ preventScroll: true });
    } catch (_) {
      viewer.focus();
    }
  };
  viewer.addEventListener('mousedown', focusViewer);
  viewer.addEventListener('touchstart', focusViewer, { passive: true });
  viewer.addEventListener('pointerdown', focusViewer);
  focusViewer();
}
if (canvas) {
  canvas.setAttribute('tabindex', '0');
}
const dpr = window.devicePixelRatio || 1;
const rootStyle = window.getComputedStyle(document.documentElement);
const sidebarWidthRaw = rootStyle.getPropertyValue('--sidebar-width');
const sidebarWidthValue = Number.parseFloat(sidebarWidthRaw || '');
const sidebarWidthDefault = Number.isFinite(sidebarWidthValue) ? Math.max(0, sidebarWidthValue) : 260;
const BRUSH_CROSSHAIR_ENABLED = CONFIG.showBrushCrosshair ?? true;
function getSidebarWidthPx() {
  const el = document.getElementById('sidebar');
  if (!el) return sidebarWidthDefault;
  const r = el.getBoundingClientRect();
  const w = Math.max(0, Math.round(r.width));
  return Number.isFinite(w) && w > 0 ? w : sidebarWidthDefault;
}

const leftPanelWidthRaw = rootStyle.getPropertyValue('--left-panel-width');
const leftPanelWidthValue = Number.parseFloat(leftPanelWidthRaw || '');
const leftPanelWidthDefault = Number.isFinite(leftPanelWidthValue) ? Math.max(0, leftPanelWidthValue) : 260;
function getLeftPanelWidthPx() {
  const el = document.getElementById('leftPanel');
  if (!el) return leftPanelWidthDefault;
  const r = el.getBoundingClientRect();
  const w = Math.max(0, Math.round(r.width));
  return Number.isFinite(w) && w > 0 ? w : leftPanelWidthDefault;
}

function getViewportSize() {
  // Prefer the viewer element's true bounds so we center/scale relative to the actual canvas slot.
  const viewerRect = viewer ? viewer.getBoundingClientRect() : null;
  if (viewerRect && viewerRect.width > 0 && viewerRect.height > 0) {
    const width = Math.max(1, Math.round(viewerRect.width));
    const height = Math.max(1, Math.round(viewerRect.height));
    const leftInset = Math.max(0, Math.round(viewerRect.left));
    return {
      width,
      height,
      visibleWidth: width,
      leftInset,
      rightInset: Math.max(0, Math.round(window.innerWidth - viewerRect.right)),
    };
  }
  const baseWidth = Math.max(1, Math.floor(window.innerWidth || document.documentElement.clientWidth || 1));
  const h = Math.max(1, Math.floor(window.innerHeight || document.documentElement.clientHeight || 1));
  const leftInset = getLeftPanelWidthPx();
  const rightInset = getSidebarWidthPx();
  const visibleWidth = Math.max(1, baseWidth - leftInset - rightInset);
  return {
    width: baseWidth,
    height: h,
    visibleWidth,
    leftInset,
    rightInset,
  };
}
const accentColor = (rootStyle.getPropertyValue('--accent-color') || '#d8a200').trim();
const histogramWindowColor = (rootStyle.getPropertyValue('--histogram-window-color') || 'rgba(140, 140, 140, 0.35)').trim();
const panelTextColor = (rootStyle.getPropertyValue('--panel-text-color') || '#f4f4f4').trim();
// Custom cursor assets and override control
let cursorOverride = null;
let cursorOverrideTimer = null;
function svgCursorUrl(svg) {
  return 'url("data:image/svg+xml;utf8,' + encodeURIComponent(svg) + '") 16 16, auto';
}
function buildDotCursorCss(size = 16, rgba = [128, 128, 128, 0.5]) {
  const [r, g, b, a] = rgba;
  const radius = Math.max(2, Math.floor(size / 4));
  const cx = Math.floor(size / 2);
  const cy = Math.floor(size / 2);
  const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
  <circle cx="${cx}" cy="${cy}" r="${radius}" fill="rgb(${r},${g},${b})" fill-opacity="${a}" />
</svg>`;
  // hotspot at center
  return 'url("data:image/svg+xml;utf8,' + encodeURIComponent(svg) + '") ' + cx + ' ' + cy + ', auto';
}
// 3x larger dot cursor (48px box, ~12px radius)
const dotCursorCss = buildDotCursorCss(48, [128, 128, 128, 0.5]);
function setCursorTemporary(style, durationMs = 400) {
  cursorOverride = style;
  updateCursor();
  if (cursorOverrideTimer) {
    clearTimeout(cursorOverrideTimer);
  }
  if (style) {
    cursorOverrideTimer = setTimeout(() => {
      if (cursorOverride === style) {
        cursorOverride = null;
        updateCursor();
      }
    }, Math.max(50, durationMs | 0));
  }
}
function setCursorHold(style) {
  if (cursorOverrideTimer) {
    clearTimeout(cursorOverrideTimer);
    cursorOverrideTimer = null;
  }
  cursorOverride = style;
  updateCursor();
}
function clearCursorOverride() {
  if (cursorOverrideTimer) {
    clearTimeout(cursorOverrideTimer);
    cursorOverrideTimer = null;
  }
  cursorOverride = null;
  updateCursor();
}

const offscreen = document.createElement('canvas');
offscreen.width = imgWidth;
offscreen.height = imgHeight;
const offCtx = offscreen.getContext('2d');

const maskCanvas = document.createElement('canvas');
maskCanvas.width = imgWidth;
maskCanvas.height = imgHeight;
const maskCtx = maskCanvas.getContext('2d');
const maskData = maskCtx.createImageData(imgWidth, imgHeight);
const maskValues = new Uint32Array(imgWidth * imgHeight);
let maskHasNonZero = false;
let floodVisited = null;
let floodVisitStamp = 1;
let floodStack = null;
let floodOutput = null;
// Outline state bitmap (1 = boundary), used to modulate per-pixel alpha in mask rendering
const outlineState = new Uint8Array(maskValues.length);
const brushSizeSlider = document.getElementById('brushSizeSlider');
const brushSizeInput = document.getElementById('brushSizeInput');
const brushKernelModeSelect = document.getElementById('brushKernelMode');
const gammaSlider = document.getElementById('gamma');
const gammaValue = document.getElementById('gammaValue');
const maskLabel = document.getElementById('maskLabel');
const labelValueInput = document.getElementById('labelValueInput');
const labelStepDown = document.getElementById('labelStepDown');
const labelStepUp = document.getElementById('labelStepUp');
const undoButton = document.getElementById('undoButton');
const redoButton = document.getElementById('redoButton');
const resetViewButton = document.getElementById('resetViewButton');
const maskVisibility = document.getElementById('maskVisibility');
const toolInfo = document.getElementById('toolInfo');
const segmentButton = document.getElementById('segmentButton');
const segmentStatus = document.getElementById('segmentStatus');
const clearMasksButton = document.getElementById('clearMasksButton');
const clearCacheButton = document.getElementById('clearCacheButton');
const colorMode = document.getElementById('colorMode');
const gammaInput = document.getElementById('gammaInput');
const histogramCanvas = document.getElementById('histogram');
const histRangeLabel = document.getElementById('histRange');
const hoverInfo = document.getElementById('hoverInfo');
const fpsDisplay = document.getElementById('fpsDisplay');
const maskOpacitySlider = document.getElementById('maskOpacity');
const maskOpacityInput = document.getElementById('maskOpacityInput');
const maskThresholdSlider = document.getElementById('maskThresholdSlider');
const maskThresholdInput = document.getElementById('maskThresholdInput');
const flowThresholdSlider = document.getElementById('flowThresholdSlider');
const flowThresholdInput = document.getElementById('flowThresholdInput');
const clusterToggle = document.getElementById('clusterToggle');
const affinityToggle = document.getElementById('affinityToggle');
const affinityGraphToggle = document.getElementById('affinityGraphToggle');
const flowOverlayToggle = document.getElementById('flowOverlayToggle');
const distanceOverlayToggle = document.getElementById('distanceOverlayToggle');
const imageVisibilityToggle = document.getElementById('imageVisibilityToggle');
const maskVisibilityToggle = document.getElementById('maskVisibilityToggle');
const toolStopButtons = Array.from(document.querySelectorAll('.tool-stop'));
const TOOL_MODE_ORDER = ['draw', 'erase', 'fill', 'picker'];
const PREVIEW_TOOL_TYPES = new Set(['brush', 'erase']);
const CROSSHAIR_TOOL_TYPES = new Set(['brush', 'erase', 'fill', 'picker']);

const HISTORY_LIMIT = 200;
const undoStack = [];
const redoStack = [];
const viewState = { scale: 1.0, offsetX: 0.0, offsetY: 0.0, rotation: 0.0 };
let maskVisible = true;
let imageVisible = true;
let currentLabel = 1;
let originalImageData = null;
let isPanning = false;
let isPainting = false;
let lastPoint = { x: 0, y: 0 };
let lastPaintPoint = null;
let strokeChanges = null;
let tool = 'brush';
let spacePan = false;
let paintStrokeQueue = [];

const FPS_SAMPLE_LIMIT = 30;
let fpsSamples = [];
let lastFrameTimestamp = (typeof performance !== 'undefined' ? performance.now() : Date.now());
let lastFpsUpdate = lastFrameTimestamp;

let hoverPoint = null;
let eraseActive = false;
let erasePreviousLabel = null;
let isSegmenting = false;
let histogramData = null;
let windowLow = 0;
let windowHigh = 255;
let currentGamma = 1.0;
let histDragTarget = null;
let histDragOffset = 0;
let cursorInsideCanvas = false;
let cursorInsideImage = false;
let maskOpacity = CONFIG.maskOpacity ?? 0.8;
const MASK_THRESHOLD_MIN = -5;
const MASK_THRESHOLD_MAX = 5;
const FLOW_THRESHOLD_MIN = 0;
const FLOW_THRESHOLD_MAX = 5;
let maskThreshold = clamp(
  typeof CONFIG.maskThreshold === 'number' ? CONFIG.maskThreshold : -2,
  MASK_THRESHOLD_MIN,
  MASK_THRESHOLD_MAX,
);
let flowThreshold = clamp(
  typeof CONFIG.flowThreshold === 'number' ? CONFIG.flowThreshold : 0,
  FLOW_THRESHOLD_MIN,
  FLOW_THRESHOLD_MAX,
);
let clusterEnabled = typeof CONFIG.cluster === 'boolean' ? CONFIG.cluster : true;
let affinitySegEnabled = typeof CONFIG.affinitySeg === 'boolean' ? CONFIG.affinitySeg : true;
let userAdjustedScale = false;
const touchPointers = new Map();
let pinchState = null;
let panPointerId = null;
let activePointerId = null;
let gestureState = null;
let wheelRotationBuffer = 0;
const MIN_GAMMA = 0.1;
const MAX_GAMMA = 6.0;
const DEFAULT_GAMMA = 1.0;
const HIST_HANDLE_THRESHOLD = 8;
let autoFitPending = true;

let nColorActive = false;
let nColorValues = null;
const rawColorMap = new Map();
const nColorColorMap = new Map();
const nColorAssignments = new Map();
const nColorColorToLabel = new Map();
let nColorMaxColorId = 0;
let lastLabelBeforeNColor = null;

function sinebowColor(t) {
  const angle = (t % 1) * 2 * Math.PI;
  const r = Math.sin(angle) * 0.5 + 0.5;
  const g = Math.sin(angle + (2 * Math.PI) / 3) * 0.5 + 0.5;
  const b = Math.sin(angle + (4 * Math.PI) / 3) * 0.5 + 0.5;
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255), 200];
}

function hashColorForLabel(label) {
  const golden = 0.61803398875;
  const t = ((label * golden) % 1 + 1) % 1;
  const base = sinebowColor(t);
  return [base[0], base[1], base[2]];
}

function clearColorCaches() {
  rawColorMap.clear();
  nColorColorMap.clear();
}



let outlinesVisible = true; // future UI toggle can control this
let stateSaveTimer = null;
let stateDirty = false;
let viewStateDirty = false;
let isRestoringState = false;
const imageInfo = document.getElementById('imageInfo');
let webglPipeline = null;
let webglPipelineReady = false;
const maskDirtyRegions = [];
const outlineDirtyRegions = [];
let maskTextureFullDirty = false;
let outlineTextureFullDirty = false;
let maskValueClampWarned = false;
let flowOverlayImage = null;
let flowOverlaySource = null;
let distanceOverlayImage = null;
let distanceOverlaySource = null;
let showFlowOverlay = false;
let showDistanceOverlay = false;
const DEFAULT_AFFINITY_STEPS = [
  [-1, -1],
  [-1, 0],
  [-1, 1],
  [0, -1],
  [0, 1],
  [1, -1],
  [1, 0],
  [1, 1],
];
// Overlay color: use pure white for all affinity segments
const AFFINITY_LINE_ALPHA = 1.0;
const AFFINITY_OVERLAY_COLOR = 'rgba(255, 255, 255, 1)';
// Enable WebGL overlay for affinity graph rendering
const USE_WEBGL_OVERLAY = true;
// WebGL overlay quality controls (configurable via __OMNI_CONFIG__)
const OVERLAY_MSAA_ENABLED = CONFIG.webglOverlayMsaa ?? false;
const OVERLAY_MSAA_SAMPLES = Math.max(1, Math.min(CONFIG.webglOverlayMsaaSamples ?? 2, 8));
const OVERLAY_FXAA_ENABLED = CONFIG.webglOverlayFxaa ?? false;
const OVERLAY_TAA_ENABLED = CONFIG.webglOverlayTaa ?? false;
const OVERLAY_TAA_BLEND = Math.max(0, Math.min(typeof CONFIG.webglOverlayTaaBlend === 'number' ? CONFIG.webglOverlayTaaBlend : 0.18, 1));
const FXAA_SUBPIX = CONFIG.webglOverlayFxaaSubpixel ?? 0.75;
const FXAA_EDGE_THRESHOLD = CONFIG.webglOverlayFxaaEdgeThreshold ?? 0.125;
const FXAA_EDGE_THRESHOLD_MIN = CONFIG.webglOverlayFxaaEdgeThresholdMin ?? 1 / 12;
const OVERLAY_PIXEL_FADE_CUTOFF = Math.max(0.0001, typeof CONFIG.webglOverlayPixelFadeCutoff === 'number' ? CONFIG.webglOverlayPixelFadeCutoff : 10);
const OVERLAY_GENERATE_MIPS = CONFIG.webglOverlayGenerateMips ?? false;
const OVERLAY_CONTEXT_ATTRIBUTES = {
  antialias: true,
  premultipliedAlpha: true,
  alpha: true,
  desynchronized: true,
  preserveDrawingBuffer: false,
};
const FLOOD_REBUILD_THRESHOLD_RAW = CONFIG.webglFloodRebuildThreshold ?? 0.35;
const FLOOD_REBUILD_THRESHOLD = Math.min(1, Math.max(0, FLOOD_REBUILD_THRESHOLD_RAW));
// Live WebGL overlay updates during painting (toggle visibility of edges)
const LIVE_AFFINITY_OVERLAY_UPDATES = true;
// Batch GPU uploads to once per frame
const BATCH_LIVE_OVERLAY_UPDATES = true;
// If live updates are enabled, do not defer
const DEFER_AFFINITY_OVERLAY_DURING_PAINT = !LIVE_AFFINITY_OVERLAY_UPDATES;

// Use coalesced pointer events for denser brush sampling on fast motion
const USE_COALESCED_EVENTS = true;
// Throttle mask redraw + draw() to once per animation frame during paint
const THROTTLE_DRAW_DURING_PAINT = true;
let paintingFrameScheduled = false;
let needsMaskRedraw = false;

function flushPendingAffinityUpdates() {
  if (!pendingAffinityIndexSet || pendingAffinityIndexSet.size === 0) {
    return false;
  }
  const merged = Array.from(pendingAffinityIndexSet);
  pendingAffinityIndexSet.clear();
  pendingAffinityIndexSet = null;
  updateAffinityGraphForIndices(merged);
  return true;
}

function requestPaintFrame() {
  if (!THROTTLE_DRAW_DURING_PAINT) {
    flushPendingAffinityUpdates();
    if (needsMaskRedraw) {
      if (isWebglPipelineActive()) {
        flushMaskTextureUpdates();
      } else {
        redrawMaskCanvas();
      }
      needsMaskRedraw = false;
    }
    draw();
    return;
  }
  if (paintingFrameScheduled) {
    return;
  }
  paintingFrameScheduled = true;
  requestAnimationFrame(() => {
    paintingFrameScheduled = false;
    flushPendingAffinityUpdates();
    if (needsMaskRedraw) {
      if (isWebglPipelineActive()) {
        flushMaskTextureUpdates();
      } else {
        redrawMaskCanvas();
      }
      needsMaskRedraw = false;
    }
    draw();
  });
}

function isWebglPipelineActive() {
  return Boolean(webglPipelineReady && webglPipeline && webglPipeline.gl);
}

function applyMaskRedrawImmediate() {
  if (!needsMaskRedraw) {
    return;
  }
  if (isWebglPipelineActive()) {
    flushMaskTextureUpdates();
  } else {
    redrawMaskCanvas();
  }
  needsMaskRedraw = false;
}

function scheduleStateSave(delay = 600) {
  if (!sessionId || isRestoringState) {
    return;
  }
  stateDirty = true;
  if (stateSaveTimer) {
    clearTimeout(stateSaveTimer);
  }
  stateSaveTimer = setTimeout(() => {
    stateSaveTimer = null;
    saveViewerState().catch((err) => {
      console.warn('saveViewerState failed', err);
    });
  }, Math.max(150, delay | 0));
}

async function saveViewerState({ immediate = false } = {}) {
  if (!sessionId) {
    return false;
  }
  if (!stateDirty && !immediate) {
    return true;
  }
  const viewerState = collectViewerState();
  const payload = {
    sessionId,
    imagePath: currentImagePath,
    viewerState,
  };
  const body = JSON.stringify(payload);
  try {
    if (immediate && typeof navigator !== 'undefined' && navigator.sendBeacon) {
      const blob = new Blob([body], { type: 'application/json' });
      navigator.sendBeacon('/api/save_state', blob);
      stateDirty = false;
      return true;
    }
    const response = await fetch('/api/save_state', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
      keepalive: immediate,
    });
    if (!response.ok) {
      throw new Error('HTTP ' + response.status);
    }
    stateDirty = false;
    return true;
  } catch (err) {
    if (!immediate) {
      stateDirty = true;
    }
    throw err;
  }
}

async function requestImageChange({ path = null, direction = null } = {}) {
  if (!sessionId) {
    console.warn('Session not initialized; cannot change image');
    return;
  }
  const payload = { sessionId };
  if (typeof path === 'string' && path) {
    payload.path = path;
  }
  if (typeof direction === 'string' && direction) {
    payload.direction = direction;
  }
  await saveViewerState({ immediate: true }).catch(() => {});
  try {
    const response = await fetch('/api/open_image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const message = await response.text().catch(() => 'unknown');
      console.warn('open_image failed', response.status, message);
      return;
    }
    const result = await response.json().catch(() => ({}));
    if (result && result.ok) {
      window.location.reload();
    } else if (result && result.error) {
      console.warn('open_image error', result.error);
    }
  } catch (err) {
    console.warn('open_image request failed', err);
  }
}

async function openImageByPath(path) {
  if (!path) {
    console.warn('No path provided for openImageByPath');
    return;
  }
  await requestImageChange({ path });
}

async function navigateDirectory(delta) {
  if (delta === 0) {
    return;
  }
  const direction = delta > 0 ? 'next' : 'prev';
  await requestImageChange({ direction });
}

function setupDragAndDrop() {
  if (!viewer || !dropOverlay || !sessionId) {
    return;
  }
  let dragDepth = 0;

  const showOverlay = () => {
    if (!dropOverlay) return;
    dropOverlay.classList.add('drop-overlay--visible');
  };

  const hideOverlay = () => {
    dragDepth = 0;
    if (!dropOverlay) return;
    dropOverlay.classList.remove('drop-overlay--visible');
  };

  viewer.addEventListener('dragenter', (evt) => {
    evt.preventDefault();
    dragDepth += 1;
    showOverlay();
  });

  viewer.addEventListener('dragover', (evt) => {
    evt.preventDefault();
    evt.dataTransfer.dropEffect = 'copy';
  });

  viewer.addEventListener('dragleave', (evt) => {
    evt.preventDefault();
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) {
      hideOverlay();
    }
  });

  viewer.addEventListener('drop', async (evt) => {
    evt.preventDefault();
    hideOverlay();
    const files = evt.dataTransfer && evt.dataTransfer.files;
    if (!files || files.length === 0) {
      return;
    }
    const file = files[0];
    if (file && typeof file.path === 'string') {
      await openImageByPath(file.path);
    } else if (file && file.webkitRelativePath) {
      await openImageByPath(file.webkitRelativePath);
    } else {
      console.warn('Dropped file has no accessible path; drag-and-drop requires file path support.');
    }
  });

  window.addEventListener('dragover', (evt) => {
    evt.preventDefault();
  });
  window.addEventListener('drop', (evt) => {
    evt.preventDefault();
    if (!viewer.contains(evt.target)) {
      hideOverlay();
    }
  });
}


function rectFromIndices(indices) {
  if (!indices) {
    return null;
  }
  const total = Array.isArray(indices) || indices instanceof Uint32Array || indices instanceof Uint16Array
    || indices instanceof Uint8Array ? indices.length : 0;
  if (total === 0) {
    return null;
  }
  let minX = imgWidth;
  let maxX = -1;
  let minY = imgHeight;
  let maxY = -1;
  const width = imgWidth | 0;
  for (let i = 0; i < total; i += 1) {
    const idx = indices[i] | 0;
    if (idx < 0 || idx >= maskValues.length) {
      continue;
    }
    const y = (idx / width) | 0;
    const x = idx - y * width;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  if (maxX < minX || maxY < minY) {
    return null;
  }
  return {
    x: minX,
    y: minY,
    width: (maxX - minX + 1),
    height: (maxY - minY + 1),
  };
}

function appendDirtyRect(list, rect) {
  if (!rect || !list) {
    return;
  }
  list.push(rect);
}

function markMaskTextureFullDirty() {
  if (!isWebglPipelineActive()) {
    return;
  }
  maskTextureFullDirty = true;
  maskDirtyRegions.length = 0;
  needsMaskRedraw = true;
}

function markOutlineTextureFullDirty() {
  if (!isWebglPipelineActive()) {
    return;
  }
  outlineTextureFullDirty = true;
  outlineDirtyRegions.length = 0;
  needsMaskRedraw = true;
}

function markMaskIndicesDirty(indices) {
  if (!isWebglPipelineActive()) {
    redrawMaskCanvas();
    return;
  }
  if (!indices || maskTextureFullDirty) {
    markMaskTextureFullDirty();
    return;
  }
  const rect = rectFromIndices(indices);
  if (!rect) {
    return;
  }
  const area = rect.width * rect.height;
  const totalArea = Math.max(1, imgWidth * imgHeight);
  if (area >= totalArea * 0.8) {
    markMaskTextureFullDirty();
    return;
  }
  appendDirtyRect(maskDirtyRegions, rect);
  needsMaskRedraw = true;
}

function markOutlineIndicesDirty(indices) {
  if (!isWebglPipelineActive()) {
    redrawMaskCanvas();
    return;
  }
  if (!indices || outlineTextureFullDirty) {
    markOutlineTextureFullDirty();
    return;
  }
  const rect = rectFromIndices(indices);
  if (!rect) {
    return;
  }
  const area = rect.width * rect.height;
  const totalArea = Math.max(1, imgWidth * imgHeight);
  if (area >= totalArea * 0.8) {
    markOutlineTextureFullDirty();
    return;
  }
  appendDirtyRect(outlineDirtyRegions, rect);
  needsMaskRedraw = true;
}

function getDisplayValueForIndex(index) {
  if (index < 0 || index >= maskValues.length) {
    return 0;
  }
  let value = maskValues[index] | 0;
  if (value < 0) {
    value = 0;
  }
  if (value > 0xffff) {
    if (!maskValueClampWarned) {
      console.warn('Mask value exceeds 16-bit range; clamping for GPU upload.');
      maskValueClampWarned = true;
    }
    value = 0xffff;
  }
  return value;
}

function ensureScratchBuffer(pipeline, kind, size) {
  if (!pipeline) {
    return null;
  }
  const channels = kind === 'mask' ? 2 : 1;
  const bytesNeeded = size * channels;
  const key = kind === 'mask' ? 'maskScratch' : 'outlineScratch';
  let buffer = pipeline[key];
  if (!buffer || buffer.length < bytesNeeded) {
    buffer = new Uint8Array(bytesNeeded);
    pipeline[key] = buffer;
  }
  return buffer;
}

function uploadMaskRegion(rect) {
  if (!isWebglPipelineActive() || !rect || !webglPipeline || !webglPipeline.maskTexture) {
    return;
  }
  const { gl } = webglPipeline;
  const area = rect.width * rect.height;
  if (area <= 0) {
    return;
  }
  const buffer = ensureScratchBuffer(webglPipeline, 'mask', area);
  let offset = 0;
  for (let row = 0; row < rect.height; row += 1) {
    const baseIndex = (rect.y + row) * imgWidth + rect.x;
    for (let col = 0; col < rect.width; col += 1) {
      const idx = baseIndex + col;
      const value = getDisplayValueForIndex(idx);
      buffer[offset] = value & 0xff;
      buffer[offset + 1] = (value >> 8) & 0xff;
      offset += 2;
    }
  }
  gl.bindTexture(gl.TEXTURE_2D, webglPipeline.maskTexture);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, rect.x, rect.y, rect.width, rect.height, gl.RG, gl.UNSIGNED_BYTE, buffer.subarray(0, area * 2));
  gl.bindTexture(gl.TEXTURE_2D, null);
}

function uploadOutlineRegion(rect) {
  if (!isWebglPipelineActive() || !rect || !webglPipeline || !webglPipeline.outlineTexture) {
    return;
  }
  const { gl } = webglPipeline;
  const area = rect.width * rect.height;
  if (area <= 0) {
    return;
  }
  const buffer = ensureScratchBuffer(webglPipeline, 'outline', area);
  let offset = 0;
  for (let row = 0; row < rect.height; row += 1) {
    const baseIndex = (rect.y + row) * imgWidth + rect.x;
    for (let col = 0; col < rect.width; col += 1) {
      const idx = baseIndex + col;
      buffer[offset] = outlineState[idx] ? 255 : 0;
      offset += 1;
    }
  }
  gl.bindTexture(gl.TEXTURE_2D, webglPipeline.outlineTexture);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, rect.x, rect.y, rect.width, rect.height, gl.RED, gl.UNSIGNED_BYTE, buffer.subarray(0, area));
  gl.bindTexture(gl.TEXTURE_2D, null);
}

function uploadFullMaskTexture() {
  if (!isWebglPipelineActive() || !webglPipeline) {
    return;
  }
  uploadMaskRegion({
    x: 0,
    y: 0,
    width: imgWidth,
    height: imgHeight,
  });
  maskTextureFullDirty = false;
}

function uploadFullOutlineTexture() {
  if (!isWebglPipelineActive() || !webglPipeline) {
    return;
  }
  uploadOutlineRegion({
    x: 0,
    y: 0,
    width: imgWidth,
    height: imgHeight,
  });
  outlineTextureFullDirty = false;
}

function flushMaskTextureUpdates() {
  if (!isWebglPipelineActive()) {
    return;
  }
  if (!webglPipeline || !webglPipeline.gl) {
    return;
  }
  if (maskTextureFullDirty) {
    uploadFullMaskTexture();
    maskDirtyRegions.length = 0;
  } else {
    for (let i = 0; i < maskDirtyRegions.length; i += 1) {
      uploadMaskRegion(maskDirtyRegions[i]);
    }
    maskDirtyRegions.length = 0;
  }
  if (outlineTextureFullDirty) {
    uploadFullOutlineTexture();
    outlineDirtyRegions.length = 0;
  } else {
    for (let i = 0; i < outlineDirtyRegions.length; i += 1) {
      uploadOutlineRegion(outlineDirtyRegions[i]);
    }
    outlineDirtyRegions.length = 0;
  }
}

const PIPELINE_VERTEX_SHADER = `#version 300 es
layout (location = 0) in vec2 a_position;
layout (location = 1) in vec2 a_texCoord;

uniform mat3 u_matrix;
out vec2 v_texCoord;

void main() {
  vec3 pos = u_matrix * vec3(a_position, 1.0);
  gl_Position = vec4(pos.xy, 0.0, 1.0);
  v_texCoord = a_texCoord;
}
`;

const PIPELINE_FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec2 v_texCoord;
out vec4 outColor;

uniform sampler2D u_baseSampler;
uniform sampler2D u_maskSampler;
uniform sampler2D u_outlineSampler;
uniform sampler2D u_flowSampler;
uniform sampler2D u_distanceSampler;

uniform float u_maskOpacity;
uniform float u_maskVisible;
uniform float u_outlinesVisible;
uniform float u_imageVisible;
uniform float u_flowVisible;
uniform float u_flowOpacity;
uniform float u_distanceVisible;
uniform float u_distanceOpacity;

vec3 sinebow(float t) {
  float angle = 6.28318530718 * fract(t);
  float r = sin(angle) * 0.5 + 0.5;
  float g = sin(angle + 2.09439510239) * 0.5 + 0.5;
  float b = sin(angle + 4.18879020479) * 0.5 + 0.5;
  return vec3(r, g, b);
}

vec3 hashColor(float label) {
  float golden = 0.61803398875;
  float t = fract(label * golden);
  return sinebow(t);
}

void main() {
  vec2 baseCoord = vec2(v_texCoord.x, 1.0 - v_texCoord.y);
  vec4 baseColor = vec4(0.0, 0.0, 0.0, 1.0);
  if (u_imageVisible > 0.5) {
    baseColor = texture(u_baseSampler, baseCoord);
  }
  vec3 color = baseColor.rgb;
  if (u_maskVisible > 0.5 && u_maskOpacity > 0.0) {
    vec2 packed = texture(u_maskSampler, v_texCoord).rg;
    float low = floor(packed.r * 255.0 + 0.5);
    float high = floor(packed.g * 255.0 + 0.5);
    float label = low + high * 256.0;
    if (label > 0.5) {
      float alpha = clamp(u_maskOpacity, 0.0, 1.0);
      if (u_outlinesVisible > 0.5) {
        float outlineSample = texture(u_outlineSampler, v_texCoord).r;
        float outline = outlineSample > 0.5 ? 1.0 : 0.0;
        alpha = mix(alpha * 0.5, alpha, outline);
      }
      vec3 maskColor = hashColor(label);
      color = mix(color, maskColor, alpha);
    }
  }
  if (u_flowVisible > 0.5) {
    vec4 overlay = texture(u_flowSampler, v_texCoord);
    float overlayAlpha = clamp(u_flowOpacity, 0.0, 1.0) * overlay.a;
    color = mix(color, overlay.rgb, overlayAlpha);
  }
  if (u_distanceVisible > 0.5) {
    vec4 overlay = texture(u_distanceSampler, v_texCoord);
    float overlayAlpha = clamp(u_distanceOpacity, 0.0, 1.0) * overlay.a;
    color = mix(color, overlay.rgb, overlayAlpha);
  }
  outColor = vec4(color, 1.0);
}
`;

const AFFINITY_LINE_VERTEX_SHADER = `#version 300 es
layout (location = 0) in vec2 a_position;

uniform mat3 u_matrix;

void main() {
  vec3 pos = u_matrix * vec3(a_position, 1.0);
  gl_Position = vec4(pos.xy, 0.0, 1.0);
}
`;

const AFFINITY_LINE_FRAGMENT_SHADER = `#version 300 es
precision mediump float;

uniform vec4 u_color;
uniform float u_alpha;
out vec4 outColor;

void main() {
  outColor = vec4(u_color.rgb, u_color.a * u_alpha);
}
`;

function initializeWebglPipelineResources(imageSource) {
  if (!gl || webglPipelineReady || !webglPipelineRequested) {
    return;
  }
  if (!imgWidth || !imgHeight) {
    return;
  }
  const program = createWebglProgram(gl, PIPELINE_VERTEX_SHADER, PIPELINE_FRAGMENT_SHADER);
  if (!program) {
    console.warn('Failed to create WebGL pipeline program; falling back to 2D rendering.');
    return;
  }
  const vao = gl.createVertexArray();
  const positionBuffer = gl.createBuffer();
  const texCoordBuffer = gl.createBuffer();
  const positions = new Float32Array([
    0, 0,
    imgWidth, 0,
    0, imgHeight,
    imgWidth, imgHeight,
  ]);
const texCoords = new Float32Array([
  0, 0,
  1, 0,
  0, 1,
  1, 1,
]);
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);

  const baseTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, baseTexture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  if (imageSource) {
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, imageSource);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  } else if (offscreen) {
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, offscreen);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  } else {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, imgWidth, imgHeight, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  }
  gl.bindTexture(gl.TEXTURE_2D, null);

  const maskTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, maskTexture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG8, imgWidth, imgHeight, 0, gl.RG, gl.UNSIGNED_BYTE, null);
  gl.bindTexture(gl.TEXTURE_2D, null);

  const outlineTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, outlineTexture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, imgWidth, imgHeight, 0, gl.RED, gl.UNSIGNED_BYTE, null);
  gl.bindTexture(gl.TEXTURE_2D, null);

  const emptyTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, emptyTexture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 0]));
  gl.bindTexture(gl.TEXTURE_2D, null);

  const uniforms = {
    matrix: gl.getUniformLocation(program, 'u_matrix'),
    baseSampler: gl.getUniformLocation(program, 'u_baseSampler'),
    maskSampler: gl.getUniformLocation(program, 'u_maskSampler'),
    outlineSampler: gl.getUniformLocation(program, 'u_outlineSampler'),
    maskOpacity: gl.getUniformLocation(program, 'u_maskOpacity'),
    maskVisible: gl.getUniformLocation(program, 'u_maskVisible'),
    imageVisible: gl.getUniformLocation(program, 'u_imageVisible'),
    outlinesVisible: gl.getUniformLocation(program, 'u_outlinesVisible'),
    flowSampler: gl.getUniformLocation(program, 'u_flowSampler'),
    flowVisible: gl.getUniformLocation(program, 'u_flowVisible'),
    flowOpacity: gl.getUniformLocation(program, 'u_flowOpacity'),
    distanceSampler: gl.getUniformLocation(program, 'u_distanceSampler'),
    distanceVisible: gl.getUniformLocation(program, 'u_distanceVisible'),
    distanceOpacity: gl.getUniformLocation(program, 'u_distanceOpacity'),
  };
  gl.useProgram(program);
  gl.uniform1i(uniforms.baseSampler, 0);
  gl.uniform1i(uniforms.maskSampler, 1);
  gl.uniform1i(uniforms.outlineSampler, 2);
  gl.uniform1i(uniforms.flowSampler, 3);
  gl.uniform1i(uniforms.distanceSampler, 4);
  gl.useProgram(null);

  const affinityProgram = createWebglProgram(gl, AFFINITY_LINE_VERTEX_SHADER, AFFINITY_LINE_FRAGMENT_SHADER);
  let affinityUniforms = null;
  let affinityVAO = null;
  let affinityBuffer = null;
  if (affinityProgram) {
    affinityUniforms = {
      matrix: gl.getUniformLocation(affinityProgram, 'u_matrix'),
      color: gl.getUniformLocation(affinityProgram, 'u_color'),
      alpha: gl.getUniformLocation(affinityProgram, 'u_alpha'),
    };
    affinityVAO = gl.createVertexArray();
    affinityBuffer = gl.createBuffer();
    if (!affinityVAO || !affinityBuffer) {
      if (affinityVAO) gl.deleteVertexArray(affinityVAO);
      if (affinityBuffer) gl.deleteBuffer(affinityBuffer);
      gl.deleteProgram(affinityProgram);
      console.warn('Failed to initialize affinity line resources; disabling GPU affinity rendering.');
    } else {
      gl.bindVertexArray(affinityVAO);
      gl.bindBuffer(gl.ARRAY_BUFFER, affinityBuffer);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      gl.bindVertexArray(null);
      gl.bindBuffer(gl.ARRAY_BUFFER, null);
    }
  }

  webglPipeline = {
    gl,
    program,
    vao,
    positionBuffer,
    texCoordBuffer,
    baseTexture,
    maskTexture,
    outlineTexture,
    flowTexture: null,
    distanceTexture: null,
    emptyTexture,
    uniforms,
    matrixCache: new Float32Array(9),
    maskScratch: null,
    outlineScratch: null,
    affinityProgram: affinityProgram && affinityVAO && affinityBuffer ? affinityProgram : null,
    affinityUniforms: affinityProgram && affinityVAO && affinityBuffer ? affinityUniforms : null,
    affinity: affinityProgram && affinityVAO && affinityBuffer ? {
      vao: affinityVAO,
      buffer: affinityBuffer,
      vertexCount: 0,
    } : null,
  };
  if (!webglPipeline.affinityProgram) {
    if (affinityProgram && (!affinityVAO || !affinityBuffer)) {
      // Program already deleted above; nothing to do.
    } else if (affinityProgram && webglPipeline.affinityProgram === null) {
      gl.deleteProgram(affinityProgram);
    }
  }
  webglPipelineReady = true;
  markMaskTextureFullDirty();
  markOutlineTextureFullDirty();
  ensureWebglOverlayReady();
  if (flowOverlayImage && flowOverlayImage.complete) {
    updateOverlayTexture('flow', flowOverlayImage);
  }
  if (distanceOverlayImage && distanceOverlayImage.complete) {
    updateOverlayTexture('distance', distanceOverlayImage);
  }
  affinityGeometryDirty = true;
}

function uploadBaseTextureFromCanvas() {
  if (!isWebglPipelineActive() || !webglPipeline || !webglPipeline.baseTexture) {
    return;
  }
  const { gl: pipelineGl, baseTexture } = webglPipeline;
  pipelineGl.bindTexture(pipelineGl.TEXTURE_2D, baseTexture);
  pipelineGl.pixelStorei(pipelineGl.UNPACK_FLIP_Y_WEBGL, true);
  pipelineGl.texSubImage2D(pipelineGl.TEXTURE_2D, 0, 0, 0, pipelineGl.RGBA, pipelineGl.UNSIGNED_BYTE, offscreen);
  pipelineGl.pixelStorei(pipelineGl.UNPACK_FLIP_Y_WEBGL, false);
  pipelineGl.bindTexture(pipelineGl.TEXTURE_2D, null);
}

function computeAffinityAlpha() {
  if (!showAffinityGraph || !affinityGraphInfo || !affinityGraphInfo.values) {
    return 0;
  }
  const scale = Math.max(0.0001, Number(viewState && viewState.scale ? viewState.scale : 1.0));
  const minStep = minAffinityStepLength > 0 ? minAffinityStepLength : 1.0;
  const minEdgePx = Math.max(0, scale * minStep);
  const dprSafe = Number.isFinite(dpr) && dpr > 0 ? dpr : 1;
  const cutoff = OVERLAY_PIXEL_FADE_CUTOFF * dprSafe;
  const t = cutoff <= 0 ? 1 : Math.max(0, Math.min(1, minEdgePx / cutoff));
  return Math.max(0, Math.min(1, t * t * (3 - 2 * t)));
}

function rebuildAffinityGeometry(gl) {
  if (!webglPipeline || !webglPipeline.affinity || !webglPipeline.affinity.buffer) {
    return;
  }
  if (!affinityGeometryDirty) {
    return;
  }
  affinityGeometryDirty = false;
  const affinity = webglPipeline.affinity;
  if (!showAffinityGraph || !affinityGraphInfo || !affinityGraphInfo.values) {
    gl.bindBuffer(gl.ARRAY_BUFFER, affinity.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(0), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    affinity.vertexCount = 0;
    return;
  }
  if (!affinityGraphInfo.segments) {
    buildAffinityGraphSegments();
  }
  const segments = affinityGraphInfo.segments;
  if (!segments) {
    gl.bindBuffer(gl.ARRAY_BUFFER, affinity.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(0), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    affinity.vertexCount = 0;
    return;
  }
  let totalEdges = 0;
  for (let i = 0; i < segments.length; i += 1) {
    const seg = segments[i];
    if (seg && seg.map && seg.map.size) {
      totalEdges += seg.map.size;
    }
  }
  if (totalEdges === 0) {
    gl.bindBuffer(gl.ARRAY_BUFFER, affinity.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(0), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    affinity.vertexCount = 0;
    return;
  }
  const data = new Float32Array(totalEdges * 4);
  let offset = 0;
  for (let i = 0; i < segments.length; i += 1) {
    const seg = segments[i];
    if (!seg || !seg.map) continue;
    seg.map.forEach((coords) => {
      data[offset + 0] = coords[0];
      data[offset + 1] = coords[1];
      data[offset + 2] = coords[2];
      data[offset + 3] = coords[3];
      offset += 4;
    });
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, affinity.buffer);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  affinity.vertexCount = totalEdges * 2;
}

function drawAffinityLines(matrix) {
  if (!webglPipeline || !webglPipeline.affinityProgram || !webglPipeline.affinityUniforms || !webglPipeline.affinity) {
    return;
  }
  if (!showAffinityGraph || !affinityGraphInfo || !affinityGraphInfo.values) {
    return;
  }
  const alpha = computeAffinityAlpha();
  if (alpha <= 0) {
    return;
  }
  const { gl, affinityProgram, affinityUniforms, affinity } = webglPipeline;
  rebuildAffinityGeometry(gl);
  if (!affinity.vertexCount || affinity.vertexCount <= 0) {
    return;
  }
  gl.useProgram(affinityProgram);
  gl.uniformMatrix3fv(affinityUniforms.matrix, false, matrix);
  const rgba = parseCssColorToRgba(AFFINITY_OVERLAY_COLOR, AFFINITY_LINE_ALPHA);
  gl.uniform4f(affinityUniforms.color, rgba[0], rgba[1], rgba[2], rgba[3]);
  gl.uniform1f(affinityUniforms.alpha, alpha);
  gl.bindVertexArray(affinity.vao);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.drawArrays(gl.LINES, 0, affinity.vertexCount);
  gl.disable(gl.BLEND);
  gl.bindVertexArray(null);
  gl.useProgram(null);
}

function updateOverlayTexture(kind, image) {
  if (!isWebglPipelineActive() || !webglPipeline) {
    return;
  }
  const key = kind === 'flow' ? 'flowTexture' : 'distanceTexture';
  const texUnit = kind === 'flow' ? 3 : 4;
  const { gl: pipelineGl } = webglPipeline;
  if (!image || !image.width || !image.height) {
    if (webglPipeline[key]) {
      pipelineGl.deleteTexture(webglPipeline[key]);
      webglPipeline[key] = null;
    }
    return;
  }
  let texture = webglPipeline[key];
  if (!texture) {
    texture = pipelineGl.createTexture();
    webglPipeline[key] = texture;
  }
  pipelineGl.activeTexture(pipelineGl.TEXTURE0 + texUnit);
  pipelineGl.bindTexture(pipelineGl.TEXTURE_2D, texture);
  pipelineGl.texParameteri(pipelineGl.TEXTURE_2D, pipelineGl.TEXTURE_MIN_FILTER, pipelineGl.NEAREST);
  pipelineGl.texParameteri(pipelineGl.TEXTURE_2D, pipelineGl.TEXTURE_MAG_FILTER, pipelineGl.NEAREST);
  pipelineGl.texParameteri(pipelineGl.TEXTURE_2D, pipelineGl.TEXTURE_WRAP_S, pipelineGl.CLAMP_TO_EDGE);
  pipelineGl.texParameteri(pipelineGl.TEXTURE_2D, pipelineGl.TEXTURE_WRAP_T, pipelineGl.CLAMP_TO_EDGE);
  pipelineGl.texImage2D(pipelineGl.TEXTURE_2D, 0, pipelineGl.RGBA8, image.width, image.height, 0, pipelineGl.RGBA, pipelineGl.UNSIGNED_BYTE, null);
  pipelineGl.texSubImage2D(pipelineGl.TEXTURE_2D, 0, 0, 0, pipelineGl.RGBA, pipelineGl.UNSIGNED_BYTE, image);
  pipelineGl.bindTexture(pipelineGl.TEXTURE_2D, null);
  pipelineGl.activeTexture(pipelineGl.TEXTURE0);
}

function bindOverlayTextureOrEmpty(texture, unit) {
  const { gl: pipelineGl, emptyTexture } = webglPipeline;
  pipelineGl.activeTexture(pipelineGl.TEXTURE0 + unit);
  pipelineGl.bindTexture(pipelineGl.TEXTURE_2D, texture || emptyTexture);
}

function drawWebglFrame() {
  if (!isWebglPipelineActive() || !webglPipeline) {
    return;
  }
  const {
    gl: pipelineGl,
    program,
    vao,
    uniforms,
    matrixCache,
    baseTexture,
    maskTexture,
    outlineTexture,
    flowTexture,
    distanceTexture,
  } = webglPipeline;
  pipelineGl.bindFramebuffer(pipelineGl.FRAMEBUFFER, null);
  pipelineGl.viewport(0, 0, canvas.width, canvas.height);
  pipelineGl.clearColor(0, 0, 0, 0);
  pipelineGl.clear(pipelineGl.COLOR_BUFFER_BIT);
  pipelineGl.useProgram(program);
  pipelineGl.bindVertexArray(vao);
  const matrix = computeWebglMatrix(matrixCache, canvas.width, canvas.height);
  pipelineGl.uniformMatrix3fv(uniforms.matrix, false, matrix);
  pipelineGl.uniform1f(uniforms.maskOpacity, Math.max(0, Math.min(1, maskOpacity)));
  pipelineGl.uniform1f(uniforms.maskVisible, maskVisible ? 1 : 0);
  pipelineGl.uniform1f(uniforms.imageVisible, imageVisible ? 1 : 0);
  pipelineGl.uniform1f(uniforms.outlinesVisible, outlinesVisible ? 1 : 0);
  pipelineGl.uniform1f(uniforms.flowVisible, showFlowOverlay && flowOverlayImage && flowOverlayImage.complete ? 1 : 0);
  pipelineGl.uniform1f(uniforms.flowOpacity, 0.7);
  pipelineGl.uniform1f(uniforms.distanceVisible, showDistanceOverlay && distanceOverlayImage && distanceOverlayImage.complete ? 1 : 0);
  pipelineGl.uniform1f(uniforms.distanceOpacity, 0.6);

  pipelineGl.activeTexture(pipelineGl.TEXTURE0);
  pipelineGl.bindTexture(pipelineGl.TEXTURE_2D, baseTexture || webglPipeline.emptyTexture);
  pipelineGl.activeTexture(pipelineGl.TEXTURE1);
  pipelineGl.bindTexture(pipelineGl.TEXTURE_2D, maskTexture || webglPipeline.emptyTexture);
  pipelineGl.activeTexture(pipelineGl.TEXTURE2);
  pipelineGl.bindTexture(pipelineGl.TEXTURE_2D, outlineTexture || webglPipeline.emptyTexture);
  bindOverlayTextureOrEmpty(flowTexture, 3);
  bindOverlayTextureOrEmpty(distanceTexture, 4);

  pipelineGl.drawArrays(pipelineGl.TRIANGLE_STRIP, 0, 4);

  pipelineGl.bindVertexArray(null);
  pipelineGl.useProgram(null);
  drawAffinityLines(matrix);
}

function drawAffinityGraphShared(matrix) {
  if (!webglOverlay || !webglOverlay.enabled) {
    return;
  }
  const glCanvas = getOverlayCanvasElement();
  if (!showAffinityGraph || !affinityGraphInfo || !affinityGraphInfo.values) {
    clearWebglOverlaySurface();
    return;
  }
  const { gl: overlayGl, program, attribs, uniforms } = webglOverlay;
  if (!overlayGl || !program || !uniforms || !attribs) {
    clearWebglOverlaySurface();
    return;
  }
  let mustBuild = Boolean(webglOverlay.needsGeometryRebuild)
    || !webglOverlay.positionsArray
    || !Number.isFinite(webglOverlay.vertexCount) || webglOverlay.vertexCount === 0
    || webglOverlay.width !== (affinityGraphInfo.width | 0)
    || webglOverlay.height !== (affinityGraphInfo.height | 0);
  if (mustBuild) {
    const shouldDefer = DEFER_AFFINITY_OVERLAY_DURING_PAINT && isPainting;
    if (!shouldDefer) {
      ensureWebglGeometry(affinityGraphInfo.width, affinityGraphInfo.height);
    }
  }
  if (!webglOverlay.positionsArray || !webglOverlay.positionsArray.length) {
    clearWebglOverlaySurface();
    return;
  }
  const targetWidth = glCanvas ? glCanvas.width : canvas.width;
  const targetHeight = glCanvas ? glCanvas.height : canvas.height;
  overlayGl.viewport(0, 0, targetWidth, targetHeight);
  if (!webglOverlay.shared) {
    overlayGl.bindFramebuffer(overlayGl.FRAMEBUFFER, null);
    overlayGl.clearColor(0, 0, 0, 0);
    overlayGl.clear(overlayGl.COLOR_BUFFER_BIT);
  }
  overlayGl.disable(overlayGl.DEPTH_TEST);
  overlayGl.lineWidth(1);
  overlayGl.useProgram(program);
  overlayGl.uniformMatrix3fv(uniforms.matrix, false, matrix);
  const s = Math.max(0.0001, Number(viewState && viewState.scale ? viewState.scale : 1.0));
  const minStep = minAffinityStepLength > 0 ? minAffinityStepLength : 1.0;
  const minEdgePx = Math.max(0, s * minStep);
  const dprSafe = Number.isFinite(dpr) && dpr > 0 ? dpr : 1;
  const cutoff = OVERLAY_PIXEL_FADE_CUTOFF * dprSafe;
  const t = cutoff <= 0 ? 1 : Math.max(0, Math.min(1, minEdgePx / cutoff));
  const alphaScale = t * t * (3 - 2 * t);
  const clampedAlpha = Math.max(0, Math.min(1, alphaScale));
  webglOverlay.displayAlpha = clampedAlpha;
  if (BATCH_LIVE_OVERLAY_UPDATES) {
    if (webglOverlay.dirtyPosSlots && webglOverlay.dirtyPosSlots.size) {
      overlayGl.bindBuffer(overlayGl.ARRAY_BUFFER, webglOverlay.positionBuffer);
      for (const slot of webglOverlay.dirtyPosSlots) {
        const basePos = slot * 4;
        overlayGl.bufferSubData(overlayGl.ARRAY_BUFFER, basePos * 4, webglOverlay.positionsArray.subarray(basePos, basePos + 4));
      }
      webglOverlay.dirtyPosSlots.clear();
    }
    if (webglOverlay.dirtyColSlots && webglOverlay.dirtyColSlots.size) {
      overlayGl.bindBuffer(overlayGl.ARRAY_BUFFER, webglOverlay.colorBuffer);
      for (const slot of webglOverlay.dirtyColSlots) {
        const baseCol = slot * 8;
        overlayGl.bufferSubData(overlayGl.ARRAY_BUFFER, baseCol * 4, webglOverlay.colorsArray.subarray(baseCol, baseCol + 8));
      }
      webglOverlay.dirtyColSlots.clear();
    }
  }
  overlayGl.bindBuffer(overlayGl.ARRAY_BUFFER, webglOverlay.positionBuffer);
  overlayGl.enableVertexAttribArray(attribs.position);
  overlayGl.vertexAttribPointer(attribs.position, 2, overlayGl.FLOAT, false, 0, 0);
  overlayGl.bindBuffer(overlayGl.ARRAY_BUFFER, webglOverlay.colorBuffer);
  overlayGl.enableVertexAttribArray(attribs.color);
  overlayGl.vertexAttribPointer(attribs.color, 4, overlayGl.FLOAT, false, 0, 0);
  overlayGl.enable(overlayGl.BLEND);
  overlayGl.blendFunc(overlayGl.SRC_ALPHA, overlayGl.ONE_MINUS_SRC_ALPHA);
  const edgesToDraw = Math.max(webglOverlay.edgeCount | 0, (webglOverlay.maxUsedSlotIndex | 0) + 1);
  const verticesToDraw = Math.max(0, edgesToDraw) * 2;
  let hasContent = false;
  if (verticesToDraw > 0) {
    overlayGl.drawArrays(overlayGl.LINES, 0, verticesToDraw);
    hasContent = true;
  }
  overlayGl.disableVertexAttribArray(attribs.position);
  overlayGl.disableVertexAttribArray(attribs.color);
  overlayGl.disable(overlayGl.BLEND);
  overlayGl.bindBuffer(overlayGl.ARRAY_BUFFER, null);
  overlayGl.useProgram(null);
  let finalAlpha = hasContent ? Math.max(0, Math.min(1, webglOverlay.displayAlpha || 0)) : 0;
  if (finalAlpha <= 0.001) {
    finalAlpha = 0;
  }
  if (webglOverlay.shared) {
    webglOverlay.displayAlpha = finalAlpha;
    return;
  }
  if (finalAlpha > 0) {
    setOverlayCanvasVisibility(true, finalAlpha);
  } else {
    clearWebglOverlaySurface();
  }
}
let affinityOppositeSteps = [];
let webglOverlay = null;

function getOverlayCanvasElement() {
  if (!webglOverlay || !webglOverlay.enabled) {
    return null;
  }
  return webglOverlay.canvas || null;
}

let affinityGeometryDirty = true;

function markAffinityGeometryDirty() {
  affinityGeometryDirty = true;
}

function setOverlayCanvasVisibility(visible, alpha = 1) {
  const glCanvas = getOverlayCanvasElement();
  if (!glCanvas || !glCanvas.style) {
    return;
  }
  if (visible) {
    if (glCanvas.style.display === 'none') {
      glCanvas.style.display = 'block';
    }
    glCanvas.style.opacity = String(Math.max(0, Math.min(1, alpha)));
  } else {
    if (glCanvas.style.display !== 'none') {
      glCanvas.style.display = 'none';
    }
    glCanvas.style.opacity = '0';
  }
}

function clearWebglOverlaySurface() {
  if (!webglOverlay || !webglOverlay.enabled) {
    return;
  }
  webglOverlay.displayAlpha = 0;
  if (!webglOverlay.shared && webglOverlay.gl) {
    const { gl } = webglOverlay;
    const target = webglOverlay.canvas;
    if (target) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      const w = target.width || gl.drawingBufferWidth || 0;
      const h = target.height || gl.drawingBufferHeight || 0;
      if (w > 0 && h > 0) {
        gl.viewport(0, 0, w, h);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
      }
    }
  }
  if (webglOverlay.canvas) {
    setOverlayCanvasVisibility(false, 0);
  }
}
function refreshOppositeStepMapping() {
  let minLen = Infinity;
  affinityOppositeSteps = affinitySteps.map((step) => {
    const dy = Number(step && step.length > 0 ? step[0] : 0) || 0;
    const dx = Number(step && step.length > 1 ? step[1] : 0) || 0;
    const length = Math.hypot(dy, dx);
    if (length > 0 && Number.isFinite(length) && length < minLen) {
      minLen = length;
    }
    const targetY = -dy;
    const targetX = -dx;
    for (let i = 0; i < affinitySteps.length; i += 1) {
      const candidate = affinitySteps[i];
      if ((candidate[0] | 0) === (targetY | 0) && (candidate[1] | 0) === (targetX | 0)) {
        return i;
      }
    }
    return -1;
  });
  if (!(minLen > 0 && Number.isFinite(minLen))) {
    minLen = 1.0;
  }
  minAffinityStepLength = minLen;
}


let minAffinityStepLength = 1.0;
const previewCanvas = document.getElementById('brushPreview');
const previewCtx = previewCanvas.getContext('2d');
previewCanvas.width = canvas.width;
previewCanvas.height = canvas.height;
  // Ensure brush preview is above WebGL overlay for visibility
  try { previewCanvas.style.zIndex = '2'; } catch (_) { /* ignore */ }

function updateFps(now) {
  const delta = now - lastFrameTimestamp;
  lastFrameTimestamp = now;
  if (delta <= 0 || delta > 2000) {
    return;
  }
  const fps = 1000 / delta;
  fpsSamples.push(fps);
  if (fpsSamples.length > FPS_SAMPLE_LIMIT) {
    fpsSamples.shift();
  }
  if (!fpsDisplay) {
    return;
  }
  if (now - lastFpsUpdate < 250) {
    return;
  }
  lastFpsUpdate = now;
  const average = fpsSamples.reduce((sum, value) => sum + value, 0) / fpsSamples.length;
  const text = `FPS: ${average.toFixed(1)}`;
  fpsDisplay.textContent = text;
  if (average >= 50) {
    fpsDisplay.style.color = '#8ff7c1';
  } else if (average >= 30) {
    fpsDisplay.style.color = '#f7d774';
  } else {
    fpsDisplay.style.color = '#ff8787';
  }
}

const loadingOverlay = null;
const loadingOverlayMessage = null;
let overlayDismissed = true;

function setLoadingOverlay() {}

function hideLoadingOverlay() {}

const pendingLogs = [];
let pywebviewReady = false;
let loggedPixelSample = false;
let drawLogCount = 0;
let logFlushTimer = null;
const SEGMENTATION_UPDATE_DELAY = 180;
let segmentationUpdateTimer = null;
let pendingMaskRebuild = false;
let canRebuildMask = false;
// Queue for segmentation payloads that arrive while painting; applied after stroke completes
let pendingSegmentationPayload = null;
let showAffinityGraph = true;
let affinityGraphInfo = null;
let affinityGraphNeedsLocalRebuild = false;
let affinityGraphSource = 'none';
let affinitySteps = DEFAULT_AFFINITY_STEPS.map((step) => step.slice());
refreshOppositeStepMapping();
const startTime = (typeof performance !== 'undefined' && performance.now)
  ? performance.now()
  : Date.now();
let shuttingDown = false;
const pywebviewPoll = setInterval(() => {
  if (pywebviewReady) {
    clearInterval(pywebviewPoll);
    return;
  }
  const api = window.pywebview ? window.pywebview.api : null;
  if (api && api.log) {
    pywebviewReady = true;
    clearInterval(pywebviewPoll);
    log('pywebview api detected via poll');
    flushLogs();
  }
}, 50);

window.addEventListener('error', (evt) => {
  try {
    const message = evt && evt.message ? evt.message : String(evt);
    log('uncaught error: ' + message);
  } catch (_) {
    /* ignore */
  }
});

function pushToQueue(msg) {
  pendingLogs.push(msg);
  if (pendingLogs.length > 200) {
    pendingLogs.shift();
  }
  scheduleLogFlush();
}

function flushLogs() {
  if (logFlushTimer !== null) {
    clearTimeout(logFlushTimer);
    logFlushTimer = null;
  }
  if (!pendingLogs.length) {
    return;
  }
  if (pywebviewReady) {
    const api = window.pywebview && window.pywebview.api && window.pywebview.api.log
      ? window.pywebview.api
      : null;
    if (!api || !api.log) {
      return;
    }
    while (pendingLogs.length) {
      api.log(pendingLogs.shift());
    }
    return;
  }
  if (typeof fetch !== 'function') {
    return;
  }
  const payload = { messages: pendingLogs.splice(0, pendingLogs.length) };
  const body = JSON.stringify(payload);
  try {
    if (navigator && typeof navigator.sendBeacon === 'function') {
      const ok = navigator.sendBeacon('/api/log', new Blob([body], { type: 'application/json' }));
      if (!ok) {
        pendingLogs.unshift(...payload.messages);
        scheduleLogFlush();
      }
    } else {
      fetch('/api/log', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
        keepalive: true,
      }).catch(() => {
        pendingLogs.unshift(...payload.messages);
        scheduleLogFlush();
      });
    }
  } catch (_) {
    pendingLogs.unshift(...payload.messages);
    scheduleLogFlush();
  }
}

function scheduleLogFlush(delay = 200) {
  if (logFlushTimer !== null) {
    return;
  }
  logFlushTimer = setTimeout(() => {
    logFlushTimer = null;
    flushLogs();
  }, delay);
}

function formatMessage(message) {
  const now = (typeof performance !== 'undefined' && performance.now)
    ? (performance.now() - startTime)
    : (Date.now() - startTime);
  const timestamp = now.toFixed(1).padStart(7, ' ');
  return '[' + timestamp + ' ms] ' + message;
}

function log(message) {
  const formatted = formatMessage(String(message));
  try {
    console.log('[pywebview]', formatted);
  } catch (_) {
    /* console unavailable */
  }
  const api = window.pywebview ? window.pywebview.api : null;
  if (api && api.log) {
    api.log(formatted);
  } else {
    pushToQueue(formatted);
  }
}

function clearHoverPreview() {
  hoverPoint = null;
  hoverScreenPoint = null;
  drawBrushPreview(null);
  updateHoverInfo(null);
}

const sliderRegistry = new Map();
const dropdownRegistry = new Map();
let dropdownOpenId = null;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function valueToPercent(input) {
  const min = Number(input.min || 0);
  const max = Number(input.max || 1);
  const span = max - min;
  if (!Number.isFinite(span) || span === 0) {
    return 0;
  }
  const value = Number(input.value || min);
  return clamp((value - min) / span, 0, 1);
}

function percentToValue(percent, input) {
  const min = Number(input.min || 0);
  const max = Number(input.max || 1);
  const span = max - min;
  const raw = min + clamp(percent, 0, 1) * span;
  const step = Number(input.step || '1');
  if (!Number.isFinite(step) || step <= 0) {
    return clamp(raw, min, max);
  }
  const snapped = Math.round((raw - min) / step) * step + min;
  const precision = (step.toString().split('.')[1] || '').length;
  const factor = 10 ** precision;
  return clamp(Math.round(snapped * factor) / factor, min, max);
}

function pointerPercent(evt, container) {
  const rect = container.getBoundingClientRect();
  if (rect.width <= 0) {
    return 0;
  }
  const ratio = (evt.clientX - rect.left) / rect.width;
  return clamp(ratio, 0, 1);
}

function registerSlider(root) {
  const id = root.dataset.sliderId || root.dataset.slider || root.id;
  if (!id) {
    return;
  }
  const type = (root.dataset.sliderType || 'single').toLowerCase();
  const inputs = Array.from(root.querySelectorAll('input[type="range"]'));
  if (!inputs.length) {
    return;
  }
  if (type === 'dual' && inputs.length < 2) {
    console.warn(`slider ${id} configured as dual but only one range input found`);
    return;
  }

  root.innerHTML = '';
  const track = document.createElement('div');
  track.className = 'slider-track';
  const fill = document.createElement('div');
  fill.className = 'slider-fill';
  root.appendChild(track);
  root.appendChild(fill);

  inputs.forEach((input) => {
    input.classList.add('slider-input');
    input.setAttribute('aria-hidden', 'true');
    root.appendChild(input);
  });

  const thumbs = inputs.map((_, idx) => {
    const thumb = document.createElement('div');
    thumb.className = 'slider-thumb';
    thumb.dataset.index = String(idx);
    root.appendChild(thumb);
    return thumb;
  });

  const entry = {
    id,
    type: type === 'dual' ? 'dual' : 'single',
    root,
    inputs,
    track,
    fill,
    thumbs,
    activePointer: null,
    activeThumb: null,
  };

  const apply = () => {
    if (entry.type === 'dual') {
      const minInput = entry.inputs[0];
      const maxInput = entry.inputs[1];
      let minValue = Number(minInput.value);
      let maxValue = Number(maxInput.value);
      if (minValue > maxValue) {
        const temp = minValue;
        minValue = maxValue;
        maxValue = temp;
        minInput.value = String(minValue);
        maxInput.value = String(maxValue);
      }
      const minPercent = valueToPercent(minInput);
      const maxPercent = valueToPercent(maxInput);
      const left = (minPercent * 100).toFixed(3) + '%';
      const rightPercent = (maxPercent * 100).toFixed(3) + '%';
      const width = Math.max(0, (maxPercent - minPercent) * 100).toFixed(3) + '%';
      entry.fill.style.left = left;
      entry.fill.style.width = width;
      entry.thumbs[0].style.left = left;
      entry.thumbs[1].style.left = rightPercent;
    } else {
      const input = entry.inputs[0];
      const percent = valueToPercent(input);
      const position = (percent * 100).toFixed(3) + '%';
      entry.fill.style.left = '0%';
      entry.fill.style.width = position;
      entry.thumbs[0].style.left = position;
    }
  };

  const setValueFromPercent = (index, percent) => {
    const input = entry.inputs[index];
    if (!input) {
      return;
    }
    let value = percentToValue(percent, input);
    if (entry.type === 'dual') {
      if (index === 0) {
        const other = Number(entry.inputs[1].value);
        if (value > other) {
          value = other;
        }
      } else {
        const other = Number(entry.inputs[0].value);
        if (value < other) {
          value = other;
        }
      }
    }
    input.value = String(value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
    apply();
  };

  const pickThumb = (percent) => {
    if (entry.type !== 'dual') {
      return 0;
    }
    const distances = entry.inputs.map((input) => Math.abs(percent - valueToPercent(input)));
    let bestIndex = 0;
    let bestDistance = distances[0];
    for (let i = 1; i < distances.length; i += 1) {
      if (distances[i] < bestDistance) {
        bestDistance = distances[i];
        bestIndex = i;
      }
    }
    return bestIndex;
  };

  const onPointerDown = (evt) => {
    evt.preventDefault();
    const percent = pointerPercent(evt, entry.root);
    const thumbIndex = entry.type === 'dual' ? pickThumb(percent) : 0;
    entry.activePointer = evt.pointerId;
    entry.activeThumb = thumbIndex;
    entry.root.setPointerCapture(entry.activePointer);
    entry.root.dataset.active = 'true';
    const targetInput = entry.inputs[thumbIndex];
    if (targetInput) {
      targetInput.focus();
    }
    setValueFromPercent(thumbIndex, percent);
  };

  const onPointerMove = (evt) => {
    if (entry.activePointer === null || evt.pointerId !== entry.activePointer) {
      return;
    }
    const percent = pointerPercent(evt, entry.root);
    setValueFromPercent(entry.activeThumb ?? 0, percent);
  };

  const onPointerRelease = (evt) => {
    if (entry.activePointer === null || evt.pointerId !== entry.activePointer) {
      return;
    }
    try {
      entry.root.releasePointerCapture(entry.activePointer);
    } catch (_) {
      /* ignore */
    }
    const percent = pointerPercent(evt, entry.root);
    setValueFromPercent(entry.activeThumb ?? 0, percent);
    entry.activePointer = null;
    entry.activeThumb = null;
    entry.root.dataset.active = 'false';
  };

  entry.root.addEventListener('pointerdown', onPointerDown);
  entry.root.addEventListener('pointermove', onPointerMove);
  entry.root.addEventListener('pointerup', onPointerRelease);
  entry.root.addEventListener('pointercancel', onPointerRelease);

  inputs.forEach((input) => {
    input.addEventListener('focus', () => {
      entry.root.dataset.focused = 'true';
    });
    input.addEventListener('blur', () => {
      entry.root.dataset.focused = 'false';
    });
    input.addEventListener('input', apply);
    input.addEventListener('change', apply);
  });

  entry.apply = apply;
  apply();
  sliderRegistry.set(id, entry);
}

function refreshSlider(id) {
  const entry = sliderRegistry.get(id);
  if (entry && typeof entry.apply === 'function') {
    entry.apply();
  }
}

function attachNumberInputStepper(input, onAdjust) {
  if (!input || typeof onAdjust !== 'function') {
    return;
  }
  input.addEventListener('keydown', (evt) => {
    if (evt.key !== 'ArrowUp' && evt.key !== 'ArrowDown') {
      return;
    }
    evt.preventDefault();
    const base = Number(input.step || '1');
    const step = Number.isFinite(base) && base > 0 ? base : 1;
    const factor = evt.shiftKey ? 5 : 1;
    const direction = evt.key === 'ArrowUp' ? 1 : -1;
    onAdjust(step * factor * direction);
  });
}

function closeDropdown(entry) {
  if (!entry) {
    return;
  }
  entry.root.dataset.open = 'false';
  entry.button.setAttribute('aria-expanded', 'false');
  if (entry.menu) {
    entry.menu.setAttribute('aria-hidden', 'true');
    entry.menu.scrollTop = 0;
  }
  dropdownOpenId = null;
}

function openDropdown(entry) {
  if (!entry) {
    return;
  }
  if (dropdownOpenId && dropdownOpenId !== entry.id) {
    closeDropdown(dropdownRegistry.get(dropdownOpenId));
  }
  entry.root.dataset.open = 'true';
  positionDropdown(entry);
  entry.button.setAttribute('aria-expanded', 'true');
  if (entry.menu) {
    entry.menu.setAttribute('aria-hidden', 'false');
  }
  dropdownOpenId = entry.id;
}

function toggleDropdown(entry) {
  if (!entry) {
    return;
  }
  const isOpen = entry.root.dataset.open === 'true';
  if (isOpen) {
    closeDropdown(entry);
  } else {
    openDropdown(entry);
  }
}

function positionDropdown(entry) {
  if (!entry || !entry.menu) {
    return;
  }
  entry.menu.style.minWidth = '100%';
}

function registerDropdown(root) {
  const select = root.querySelector('select');
  if (!select) {
    return;
  }
  const id = root.dataset.dropdownId || select.id || `dropdown-${dropdownRegistry.size}`;
  root.dataset.dropdownId = id;
  root.dataset.open = root.dataset.open || 'false';

  const originalOptions = Array.from(select.options).map((opt) => ({
    value: opt.value,
    label: opt.textContent || opt.value,
    disabled: opt.disabled,
  }));

  select.classList.add('dropdown-input');
  root.innerHTML = '';
  root.appendChild(select);

  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'dropdown-toggle';
  button.setAttribute('aria-haspopup', 'listbox');
  button.setAttribute('aria-expanded', 'false');
  const menu = document.createElement('div');
  menu.className = 'dropdown-menu';
  menu.setAttribute('role', 'listbox');
  menu.setAttribute('aria-hidden', 'true');
  menu.id = `${id}-menu`;
  button.setAttribute('aria-controls', menu.id);
  root.appendChild(button);
  const menuWrapper = document.createElement('div');
  menuWrapper.className = 'dropdown-menu-wrap';
  menuWrapper.appendChild(menu);
  root.appendChild(menuWrapper);

  const entry = {
    id,
    root,
    select,
    button,
    menu,
    menuWrapper,
    options: originalOptions,
  };

  const applySelection = () => {
    const selectedOption = select.options[select.selectedIndex];
    button.textContent = selectedOption ? selectedOption.textContent : 'Select';
    menu.querySelectorAll('.dropdown-option').forEach((child) => {
      child.dataset.selected = child.dataset.value === select.value ? 'true' : 'false';
    });
  };

  const buildMenu = () => {
    menu.innerHTML = '';
    entry.options.forEach((opt) => {
      const item = document.createElement('div');
      item.className = 'dropdown-option';
      item.dataset.value = opt.value;
      item.textContent = opt.label;
      item.setAttribute('role', 'option');
      if (opt.disabled) {
        item.setAttribute('aria-disabled', 'true');
        item.style.opacity = '0.45';
        item.style.pointerEvents = 'none';
      }
      item.addEventListener('pointerdown', (evt) => {
        evt.preventDefault();
        if (opt.disabled) {
          return;
        }
        select.value = opt.value;
        select.dispatchEvent(new Event('change', { bubbles: true }));
        applySelection();
        closeDropdown(entry);
      });
      menu.appendChild(item);
    });
    applySelection();
  };

  button.addEventListener('click', () => {
    toggleDropdown(entry);
  });

  select.addEventListener('change', () => {
    applySelection();
  });

  buildMenu();
  entry.applySelection = applySelection;
  entry.buildMenu = buildMenu;
  positionDropdown(entry);
  dropdownRegistry.set(id, entry);
}

document.addEventListener('pointerdown', (evt) => {
  if (!dropdownOpenId) {
    return;
  }
  const entry = dropdownRegistry.get(dropdownOpenId);
  if (!entry) {
    dropdownOpenId = null;
    return;
  }
  if (!entry.root.contains(evt.target)) {
    closeDropdown(entry);
  }
});

function refreshDropdown(id) {
  const entry = dropdownRegistry.get(id);
  if (entry && typeof entry.applySelection === 'function') {
    entry.applySelection();
  }
}

window.addEventListener('blur', () => {
  touchPointers.clear();
  pinchState = null;
  panPointerId = null;
  pointerState.resetPen();
  gestureState = null;
  wheelRotationBuffer = 0;
  clearHoverPreview();
  if (dropdownOpenId) {
    closeDropdown(dropdownRegistry.get(dropdownOpenId));
  }
});

document.addEventListener('mouseleave', (evt) => {
  if (!evt.relatedTarget && !evt.toElement) {
    clearHoverPreview();
  }
});
