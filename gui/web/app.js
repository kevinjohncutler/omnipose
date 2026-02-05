const CONFIG = window.__OMNI_CONFIG__ || {};
const imgWidth = CONFIG.width || 0;
const imgHeight = CONFIG.height || 0;
const imageDataUrl = CONFIG.imageDataUrl || '';
const colorTable = CONFIG.colorTable || [];
const hasNColor = Boolean(
  CONFIG.hasNColor
  ?? CONFIG.nColor
  ?? CONFIG.ncolor
  ?? CONFIG.enableNColor
  ?? CONFIG.nColorMask
  ?? CONFIG.ncolorMask
  ?? false,
);
const initialBrushRadius = CONFIG.brushRadius ?? 6;
const sessionId = CONFIG.sessionId || null;
const currentImagePath = CONFIG.imagePath || null;
const currentImageName = CONFIG.imageName || null;
const localStateKey = `OMNI_VIEWER_STATE:${CONFIG.imagePath || CONFIG.imageName || 'default'}`;

function loadLocalViewerState() {
  if (typeof window === 'undefined' || !window.localStorage) {
    return null;
  }
  try {
    const raw = window.localStorage.getItem(localStateKey);
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object') {
      return null;
    }
    delete parsed.imageVisible;
    delete parsed.maskVisible;
    return parsed;
  } catch (_) {
    return null;
  }
}
const directoryEntries = Array.isArray(CONFIG.directoryEntries) ? CONFIG.directoryEntries : [];
const directoryIndex = typeof CONFIG.directoryIndex === 'number' ? CONFIG.directoryIndex : null;
const directoryPath = CONFIG.directoryPath || null;
const savedViewerState = CONFIG.savedViewerState || loadLocalViewerState();
let hasPendingSavedNiter = false;
let pendingSavedNiter = null;
const hasPrevImage = Boolean(CONFIG.hasPrev);
const hasNextImage = Boolean(CONFIG.hasNext);
let nColorPalette = [];
let nColorPaletteColors = [];
let labelColormap = 'sinebow'; // Default to sinebow (cyclic with hue offset support)
let nColorColormap = 'sinebow'; // Colormap for N-color mode (legacy, now uses labelColormap)
let nColorHueOffset = 0; // Hue offset for N-color palette
let labelShuffle = true;
let labelShuffleSeed = 0;
// Image colormap (for grayscale images)
let imageColormap = 'gray'; // Default grayscale
const IMAGE_COLORMAPS = [
  { value: 'gray', label: 'grayscale' },
  { value: 'gray-clip', label: 'grayclip' },
  { value: 'magma', label: 'magma' },
  { value: 'viridis', label: 'viridis' },
  { value: 'inferno', label: 'inferno' },
  { value: 'plasma', label: 'plasma' },
  { value: 'hot', label: 'hot' },
  { value: 'turbo', label: 'turbo' },
];
let imageCmapLutTexture = null;
let imageCmapLutDirty = true;
// Track current max label for dynamic palette sizing
let currentMaxLabel = 128;
// Permutation cache for shuffle - ensures bijective mapping (no color collisions)
// Maps [1..N] to [1..N] using golden ratio for optimal visual separation
let shufflePermutation = null;
let shufflePermutationSize = 0;
let shufflePermutationSeed = 0;
const PALETTE_TEXTURE_SIZE = 1024;
let paletteTextureDirty = true;
const DEFAULT_NCOLOR_COUNT = 4;
if (!Array.isArray(colorTable) || colorTable.length === 0) {
  labelColormap = 'sinebow';
}
const DEBUG_FILL_PERF = Boolean(
  CONFIG.debugFillPerf
  ?? CONFIG.debugFillPerformance
  ?? false,
);
const ENABLE_MASK_PIPELINE_V2 = Boolean(
  CONFIG.enableMaskPipelineV2
  ?? CONFIG.maskPipelineV2
  ?? false,
);
const DEBUG_STATE_SAVE = Boolean(
  CONFIG.debugStateSave
  ?? CONFIG.debugViewerState
  ?? false,
);
if (typeof window !== 'undefined') {
  window.__OMNI_FILL_DEBUG__ = DEBUG_FILL_PERF;
  const hasConfigForceGrid = Object.prototype.hasOwnProperty.call(CONFIG, 'debugForceGridMask');
  const hasWindowForceGrid = Object.prototype.hasOwnProperty.call(window, '__OMNI_FORCE_GRID_MASK__');
  const forceGridMask = hasConfigForceGrid
    ? Boolean(CONFIG.debugForceGridMask)
    : (hasWindowForceGrid ? Boolean(window.__OMNI_FORCE_GRID_MASK__) : false);
  window.__OMNI_FORCE_GRID_MASK__ = forceGridMask;
  if (ENABLE_MASK_PIPELINE_V2) {
    window.__OMNI_MASK_PIPELINE_V2__ = true;
  }
  if (DEBUG_STATE_SAVE) {
    window.__OMNI_SAVE_DEBUG__ = true;
  }
}

const OmniPointer = window.OmniPointer || {};
const POINTER_OPTIONS = OmniPointer.POINTER_OPTIONS;
const createPointerState = typeof OmniPointer.createPointerState === 'function'
  ? OmniPointer.createPointerState
  : null;
if (!POINTER_OPTIONS || !createPointerState) {
  throw new Error('OmniPointer helpers missing; ensure /static/js/pointer-state.js loads before app.js');
}

const RAD_TO_DEG = 180 / Math.PI;
// Force the WebGL pipeline on for every build; the legacy canvas renderer is
// intentionally disabled so we exercise a single code path everywhere.
const USE_WEBGL_PIPELINE = true;
const MAIN_WEBGL_CONTEXT_ATTRIBUTES = {
  alpha: true,
  antialias: true,
  premultipliedAlpha: true,
  preserveDrawingBuffer: false,
  depth: false,
  stencil: false,
  desynchronized: true,
};

const supportsGestureEvents = typeof window !== 'undefined'
  && (typeof window.GestureEvent === 'function' || 'ongesturestart' in window);
const isIOSDevice = typeof navigator !== 'undefined' && (
  /iPad|iPhone|iPod/.test(navigator.userAgent)
  || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1)
);
const isSafariWebKit = typeof navigator !== 'undefined'
  && /AppleWebKit/.test(navigator.userAgent)
  && !/Chrome|CriOS|Edg|Firefox|FxiOS/.test(navigator.userAgent);
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
if (!webglPipelineRequested) {
  throw new Error('WebGL2 support is required: the legacy canvas renderer has been retired');
}
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
const viewer = document.getElementById('viewer');
const ENABLE_WEBGL_LOGGING = Boolean(window.__OMNI_WEBGL_LOGGING__);
function perfNow() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}
function logWebgl(kind, data) {
  if (!ENABLE_WEBGL_LOGGING) {
    return;
  }
  try {
    if (typeof window !== 'undefined' && typeof window.__omniLogPush === 'function') {
      window.__omniLogPush(kind, data);
    } else if (console && typeof console.debug === 'function') {
      console.debug('[webgl]', kind, data);
    }
  } catch (_) {
    /* ignore */
  }
}
const AFFINITY_DRAW_LOG_INTERVAL = 30;
let affinityDrawLogCounter = 0;
let webglUnavailableNotified = false;
const dropOverlay = document.getElementById('dropOverlay');
let debugFillOverlay = null;
let debugFillOverlayHideTimer = null;
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
  debugFillOverlay = document.createElement('div');
  Object.assign(debugFillOverlay.style, {
    position: 'absolute',
    border: '2px solid rgba(0, 196, 255, 0.85)',
    background: 'rgba(0, 196, 255, 0.2)',
    pointerEvents: 'none',
    zIndex: 60,
    display: 'none',
    opacity: '0',
    transition: 'opacity 0.2s ease',
  });
  viewer.appendChild(debugFillOverlay);
}
if (canvas) {
  canvas.setAttribute('tabindex', '0');
}
const dpr = window.devicePixelRatio || 1;
const rootStyle = window.getComputedStyle(document.documentElement);
const rootStyleWrite = document.documentElement && document.documentElement.style ? document.documentElement.style : null;
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
const rightPanelWidthRaw = rootStyle.getPropertyValue('--sidebar-width');
const rightPanelWidthValue = Number.parseFloat(rightPanelWidthRaw || '');
const rightPanelWidthDefault = Number.isFinite(rightPanelWidthValue) ? Math.max(0, rightPanelWidthValue) : 260;
function getLeftPanelWidthPx() {
  const el = document.getElementById('leftPanel');
  if (!el) return leftPanelWidthDefault;
  const r = el.getBoundingClientRect();
  const w = Math.max(0, Math.round(r.width));
  return Number.isFinite(w) && w > 0 ? w : leftPanelWidthDefault;
}
function getRightPanelWidthPx() {
  const el = document.getElementById('sidebar');
  if (!el) return rightPanelWidthDefault;
  const r = el.getBoundingClientRect();
  const w = Math.max(0, Math.round(r.width));
  return Number.isFinite(w) && w > 0 ? w : rightPanelWidthDefault;
}

function getViewportSize() {
  // Use the true viewport size to avoid layout effects from scrolling sidebar
  const baseWidth = Math.max(1, Math.floor(window.innerWidth || document.documentElement.clientWidth || 1));
  const h = Math.max(1, Math.floor(window.innerHeight || document.documentElement.clientHeight || 1));
  const leftInset = getLeftPanelWidthPx();
  const rightInset = getRightPanelWidthPx();
  const visibleWidth = Math.max(1, baseWidth - leftInset - rightInset);
  return {
    width: baseWidth,
    height: h,
    visibleWidth,
    leftInset,
    rightInset,
  };
}

function rgbToCss(rgb) {
  return 'rgb(' + (rgb[0] | 0) + ', ' + (rgb[1] | 0) + ', ' + (rgb[2] | 0) + ')';
}

function parseCssColor(value) {
  if (!value || typeof value !== 'string') {
    return null;
  }
  const trimmed = value.trim();
  if (trimmed.startsWith('#')) {
    const hex = trimmed.slice(1);
    if (hex.length === 3) {
      return [
        parseInt(hex[0] + hex[0], 16),
        parseInt(hex[1] + hex[1], 16),
        parseInt(hex[2] + hex[2], 16),
      ];
    }
    if (hex.length === 6) {
      return [
        parseInt(hex.slice(0, 2), 16),
        parseInt(hex.slice(2, 4), 16),
        parseInt(hex.slice(4, 6), 16),
      ];
    }
  }
  const match = trimmed.match(/rgba?\(([^)]+)\)/i);
  if (match) {
    const parts = match[1].split(',').map((part) => Number(part.trim()));
    if (parts.length >= 3 && parts.every((val) => Number.isFinite(val))) {
      return parts.slice(0, 3).map((val) => Math.max(0, Math.min(255, val)));
    }
  }
  return null;
}

function readableTextColor(rgb) {
  const [r, g, b] = rgb;
  const luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
  return luminance > 0.62 ? 'rgba(20, 20, 20, 0.86)' : '#f6f6f6';
}

function lightenRgb(rgb, amount = 0.25) {
  const clampChannel = (value) => Math.max(0, Math.min(255, value));
  const mix = (a, b) => Math.round(a + (b - a) * amount);
  return [
    clampChannel(mix(rgb[0], 255)),
    clampChannel(mix(rgb[1], 255)),
    clampChannel(mix(rgb[2], 255)),
  ];
}

function updateAccentColorsFromRgb(rgb) {
  if (!rootStyleWrite || !Array.isArray(rgb) || rgb.length < 3) {
    return;
  }
  const base = rgbToCss(rgb);
  const hover = rgbToCss(lightenRgb(rgb));
  const ink = readableTextColor(rgb);
  rootStyleWrite.setProperty('--accent-color', base);
  rootStyleWrite.setProperty('--accent-hover', hover);
  rootStyleWrite.setProperty('--accent-ink', ink);
  accentColor = base;
  if (typeof renderHistogram === 'function') {
    renderHistogram();
  }
}

function resetAccentColors() {
  if (!rootStyleWrite) {
    return;
  }
  rootStyleWrite.setProperty('--accent-color', accentColorDefault);
  rootStyleWrite.setProperty('--accent-hover', accentHoverDefault);
  const parsed = parseCssColor(accentColorDefault);
  if (parsed) {
    rootStyleWrite.setProperty('--accent-ink', readableTextColor(parsed));
  }
  accentColor = accentColorDefault;
  if (typeof renderHistogram === 'function') {
    renderHistogram();
  }
}
const accentColorDefault = (rootStyle.getPropertyValue('--accent-color') || '#d8a200').trim();
const accentHoverDefault = (rootStyle.getPropertyValue('--accent-hover') || '#f3c54a').trim();
let accentColor = accentColorDefault;
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
let maskValues = new Uint32Array(imgWidth * imgHeight);
let maskHasNonZero = false;
// Outline state bitmap (1 = boundary), used to modulate per-pixel alpha in mask rendering
const outlineState = new Uint8Array(maskValues.length);
const MASK_DISPLAY_MODES = {
  OUTLINED: 'outlined',
  SOLID: 'solid',
  OUTLINE: 'outline',
  HIDDEN: 'hidden',
};
let maskDisplayMode = MASK_DISPLAY_MODES.OUTLINED;
let outlinesVisible = true; // derives from maskDisplayMode

function normalizeMaskDisplayMode(value) {
  if (value === MASK_DISPLAY_MODES.SOLID || value === MASK_DISPLAY_MODES.OUTLINE || value === MASK_DISPLAY_MODES.HIDDEN) {
    return value;
  }
  return MASK_DISPLAY_MODES.OUTLINED;
}
let stateSaveTimer = null;
let stateDirty = false;
let stateDirtySeq = 0;
let lastSavedSeq = 0;
let viewStateDirty = false;
let isRestoringState = false;
const imageInfo = document.getElementById('imageInfo');
let webglPipeline = null;
let webglPipelineReady = false;
let pendingMaskTextureFull = false;
let pendingOutlineTextureFull = false;
const maskDirtyRegions = [];
const outlineDirtyRegions = [];
let maskTextureFullDirty = false;
let outlineTextureFullDirty = false;
let maskValueClampWarned = false;
let flowOverlayImage = null;
let flowOverlaySource = null;
let distanceOverlayImage = null;
let distanceOverlaySource = null;
let pointsOverlayInfo = null;
let pointsOverlayData = null;
let selectedPointCoords = [];
let selectedPointIndices = new Set();
const POINT_SELECT_COLOR = [255, 64, 64, 255];
const POINT_DEFAULT_COLOR = [255, 255, 255, 255];
const POINT_PICK_RADIUS = 6;
let vectorOverlayInfo = null;
let vectorOverlayData = null;
let showFlowOverlay = false;
let showDistanceOverlay = false;
let showPointsOverlay = false;
let showVectorOverlay = false;
let vectorOverlayPreferred = false;
function stateDebugEnabled() {
  if (DEBUG_STATE_SAVE) {
    return true;
  }
  if (typeof window !== 'undefined' && window.__OMNI_SAVE_DEBUG__) {
    return Boolean(window.__OMNI_SAVE_DEBUG__);
  }
  return false;
}

function formatStateDebugPart(part) {
  if (part === null || part === undefined) {
    return String(part);
  }
  if (typeof part === 'object') {
    try {
      return JSON.stringify(part);
    } catch (_) {
      return Object.prototype.toString.call(part);
    }
  }
  return String(part);
}

function stateDebugLog(...parts) {
  if (!stateDebugEnabled()) {
    return;
  }
  const message = parts.map((part) => formatStateDebugPart(part)).join(' ');
  try {
    log('[state-save] ' + message);
  } catch (err) {
    if (typeof console !== 'undefined' && typeof console.debug === 'function') {
      console.debug('[state-save]', message);
    }
  }
}
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
const POINTS_OVERLAY_ALPHA = 0.35;
const VECTOR_OVERLAY_ALPHA = 0.35;
const VECTOR_ARROW_LENGTH = 4;
const VECTOR_ARROW_WIDTH = 2;
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
const DEBUG_COUNTERS = {
  draw: 0,
  requestPaintFrame: 0,
  flushMaskTextureUpdates: 0,
  applyMaskRedrawImmediate: 0,
  affinityUpdateLastMs: 0,
};
const DEBUG_DIRTY_TRACKER = {
  maskCalls: 0,
  maskLastCount: 0,
  maskLastRect: null,
  maskFullDirtyTransitions: 0,
  maskPendingRegions: 0,
  maskUploads: 0,
  maskLastUploadRect: null,
  outlineCalls: 0,
  outlineLastCount: 0,
  outlineLastRect: null,
  outlineFullDirtyTransitions: 0,
  outlinePendingRegions: 0,
  outlineUploads: 0,
  outlineLastUploadRect: null,
  flushCalls: 0,
  lastFlushUsedFullMask: false,
  lastFlushMaskRegions: 0,
  lastFlushUsedFullOutline: false,
  lastFlushOutlineRegions: 0,
};

function requestPaintFrame() {
  DEBUG_COUNTERS.requestPaintFrame += 1;
  if (!THROTTLE_DRAW_DURING_PAINT) {
    if (typeof paintingApi.flushPendingAffinityUpdates === 'function') {
      paintingApi.flushPendingAffinityUpdates();
    }
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
    if (typeof paintingApi.flushPendingAffinityUpdates === 'function') {
      paintingApi.flushPendingAffinityUpdates();
    }
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
  DEBUG_COUNTERS.applyMaskRedrawImmediate += 1;
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

function nextStateDirtySeq() {
  stateDirtySeq += 1;
  return stateDirtySeq;
}

function scheduleStateSave(delay = 600) {
  if (isRestoringState) {
    stateDebugLog('skip schedule (restoring)', {});
    return;
  }
  stateDirty = true;
  const scheduledSeq = nextStateDirtySeq();
  const delayMs = Math.max(150, delay | 0);
  stateDebugLog('schedule', { seq: scheduledSeq, delay: delayMs, immediate: false });
  if (stateSaveTimer) {
    clearTimeout(stateSaveTimer);
  }
  stateSaveTimer = setTimeout(() => {
    stateSaveTimer = null;
    stateDebugLog('timer fire', { seq: scheduledSeq });
    saveViewerState({ seq: scheduledSeq }).catch((err) => {
      console.warn('saveViewerState failed', err);
      stateDebugLog('save error (scheduled)', { seq: scheduledSeq, message: err && err.message ? err.message : String(err) });
    });
  }, delayMs);
}

async function saveViewerState({ immediate = false, seq = null } = {}) {
  if (!stateDirty && !immediate) {
    stateDebugLog('skip save (clean state)', { seq: stateDirtySeq, immediate });
    return true;
  }
  const requestSeq = typeof seq === 'number' ? seq : stateDirtySeq;
  stateDebugLog('save start', {
    seq: requestSeq,
    immediate,
    dirtySeq: stateDirtySeq,
    lastSavedSeq,
  });
  const viewerState = collectViewerState();
  // Always save to localStorage (works without sessionId)
  if (typeof window !== 'undefined' && window.localStorage) {
    try {
      window.localStorage.setItem(localStateKey, JSON.stringify(viewerState));
    } catch (_) {
      /* ignore */
    }
  }
  // Server-side save requires sessionId
  if (!sessionId) {
    stateDebugLog('skip server save (no session)', { immediate, seq });
    stateDirty = false;
    return true;
  }
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
      lastSavedSeq = Math.max(lastSavedSeq, requestSeq);
      if (requestSeq === stateDirtySeq) {
        stateDirty = false;
        stateDebugLog('state clean (beacon)', { seq: requestSeq });
      } else {
        stateDebugLog('stale beacon save', { seq: requestSeq, latest: stateDirtySeq });
      }
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
    lastSavedSeq = Math.max(lastSavedSeq, requestSeq);
    if (requestSeq === stateDirtySeq) {
      stateDirty = false;
      stateDebugLog('state clean (fetch)', { seq: requestSeq, status: response.status });
    } else {
      stateDebugLog('stale fetch save', { seq: requestSeq, latest: stateDirtySeq, status: response.status });
    }
    return true;
  } catch (err) {
    if (!immediate) {
      stateDirty = true;
      stateDebugLog('save failed, state remains dirty', {
        seq: requestSeq,
        message: err && err.message ? err.message : String(err),
      });
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
  await saveViewerState({ immediate: true, seq: stateDirtySeq }).catch(() => {});
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



async function selectImageFolder() {
  await saveViewerState({ immediate: true, seq: stateDirtySeq }).catch(() => {});
  try {
    const response = await fetch('/api/select_image_folder', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sessionId }),
    });
    const result = await response.json().catch(async () => {
      const message = await response.text().catch(() => 'unknown');
      return { error: message || 'unknown' };
    });
    if (!response.ok) {
      console.warn('select_image_folder failed', response.status, result);
      return;
    }
    if (result && result.ok) {
      window.location.reload();
    } else if (result && result.error) {
      console.warn('select_image_folder error', result.error);
    }
  } catch (err) {
    console.warn('select_image_folder request failed', err);
  }
}

async function selectImageFile() {
  await saveViewerState({ immediate: true, seq: stateDirtySeq }).catch(() => {});
  try {
    const response = await fetch('/api/select_image_file', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sessionId }),
    });
    const result = await response.json().catch(async () => {
      const message = await response.text().catch(() => 'unknown');
      return { error: message || 'unknown' };
    });
    if (!response.ok) {
      console.warn('select_image_file failed', response.status, result);
      return;
    }
    if (result && result.ok) {
      window.location.reload();
    } else if (result && result.error) {
      console.warn('select_image_file error', result.error);
    }
  } catch (err) {
    console.warn('select_image_file request failed', err);
  }
}

async function openImageFolder(path) {
  if (!path) {
    console.warn('No path provided for openImageFolder');
    return;
  }
  await saveViewerState({ immediate: true, seq: stateDirtySeq }).catch(() => {});
  try {
    const response = await fetch('/api/open_image_folder', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sessionId, path }),
    });
    if (!response.ok) {
      const message = await response.text().catch(() => 'unknown');
      console.warn('open_image_folder failed', response.status, message);
      return;
    }
    const result = await response.json().catch(() => ({}));
    if (result && result.ok) {
      window.location.reload();
    } else if (result && result.error) {
      console.warn('open_image_folder error', result.error);
    }
  } catch (err) {
    console.warn('open_image_folder request failed', err);
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
  DEBUG_DIRTY_TRACKER.maskFullDirtyTransitions += 1;
  if (!isWebglPipelineActive()) {
    pendingMaskTextureFull = true;
    needsMaskRedraw = true;
    return;
  }
  pendingMaskTextureFull = false;
  maskTextureFullDirty = true;
  maskDirtyRegions.length = 0;
  DEBUG_DIRTY_TRACKER.maskPendingRegions = 0;
  needsMaskRedraw = true;
}

function markOutlineTextureFullDirty() {
  DEBUG_DIRTY_TRACKER.outlineFullDirtyTransitions += 1;
  if (!isWebglPipelineActive()) {
    pendingOutlineTextureFull = true;
    needsMaskRedraw = true;
    return;
  }
  pendingOutlineTextureFull = false;
  outlineTextureFullDirty = true;
  outlineDirtyRegions.length = 0;
  DEBUG_DIRTY_TRACKER.outlinePendingRegions = 0;
  needsMaskRedraw = true;
}

function applyPendingFullTextureUpdates() {
  if (!isWebglPipelineActive()) {
    return;
  }
  if (pendingMaskTextureFull) {
    pendingMaskTextureFull = false;
    maskTextureFullDirty = true;
    maskDirtyRegions.length = 0;
    needsMaskRedraw = true;
  }
  if (pendingOutlineTextureFull) {
    pendingOutlineTextureFull = false;
    outlineTextureFullDirty = true;
    outlineDirtyRegions.length = 0;
    needsMaskRedraw = true;
  }
}

function markMaskIndicesDirty(indices) {
  DEBUG_DIRTY_TRACKER.maskCalls += 1;
  DEBUG_DIRTY_TRACKER.maskLastCount = indices && typeof indices.length === 'number' ? (indices.length | 0) : 0;
  DEBUG_DIRTY_TRACKER.maskLastRect = null;
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
  DEBUG_DIRTY_TRACKER.maskLastRect = Object.assign({}, rect);
  const area = rect.width * rect.height;
  const totalArea = Math.max(1, imgWidth * imgHeight);
  if (area >= totalArea * 0.8) {
    markMaskTextureFullDirty();
    return;
  }
  appendDirtyRect(maskDirtyRegions, rect);
  DEBUG_DIRTY_TRACKER.maskPendingRegions = maskDirtyRegions.length;
  needsMaskRedraw = true;
}

function markOutlineIndicesDirty(indices) {
  DEBUG_DIRTY_TRACKER.outlineCalls += 1;
  DEBUG_DIRTY_TRACKER.outlineLastCount = indices && typeof indices.length === 'number' ? (indices.length | 0) : 0;
  DEBUG_DIRTY_TRACKER.outlineLastRect = null;
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
  DEBUG_DIRTY_TRACKER.outlineLastRect = Object.assign({}, rect);
  const area = rect.width * rect.height;
  const totalArea = Math.max(1, imgWidth * imgHeight);
  if (area >= totalArea * 0.8) {
    markOutlineTextureFullDirty();
    return;
  }
  appendDirtyRect(outlineDirtyRegions, rect);
  DEBUG_DIRTY_TRACKER.outlinePendingRegions = outlineDirtyRegions.length;
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
  DEBUG_DIRTY_TRACKER.maskUploads += 1;
  DEBUG_DIRTY_TRACKER.maskLastUploadRect = Object.assign({}, rect);
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
  DEBUG_DIRTY_TRACKER.outlineUploads += 1;
  DEBUG_DIRTY_TRACKER.outlineLastUploadRect = Object.assign({}, rect);
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
  DEBUG_DIRTY_TRACKER.maskUploads += 1;
  DEBUG_DIRTY_TRACKER.maskLastUploadRect = { x: 0, y: 0, width: imgWidth, height: imgHeight };
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
  DEBUG_DIRTY_TRACKER.outlineUploads += 1;
  DEBUG_DIRTY_TRACKER.outlineLastUploadRect = { x: 0, y: 0, width: imgWidth, height: imgHeight };
}

function flushMaskTextureUpdates() {
  DEBUG_COUNTERS.flushMaskTextureUpdates += 1;
  DEBUG_DIRTY_TRACKER.flushCalls += 1;
  applyPendingFullTextureUpdates();
  if (!isWebglPipelineActive()) {
    return;
  }
  if (!webglPipeline || !webglPipeline.gl) {
    return;
  }
  const usedFullMask = maskTextureFullDirty;
  const usedFullOutline = outlineTextureFullDirty;
  const maskRegionsBefore = maskDirtyRegions.length;
  const outlineRegionsBefore = outlineDirtyRegions.length;
  if (maskTextureFullDirty) {
    uploadFullMaskTexture();
    maskDirtyRegions.length = 0;
    DEBUG_DIRTY_TRACKER.maskPendingRegions = 0;
  } else {
    for (let i = 0; i < maskDirtyRegions.length; i += 1) {
      uploadMaskRegion(maskDirtyRegions[i]);
    }
    maskDirtyRegions.length = 0;
    DEBUG_DIRTY_TRACKER.maskPendingRegions = 0;
  }
  if (outlineTextureFullDirty) {
    uploadFullOutlineTexture();
    outlineDirtyRegions.length = 0;
    DEBUG_DIRTY_TRACKER.outlinePendingRegions = 0;
  } else {
    for (let i = 0; i < outlineDirtyRegions.length; i += 1) {
      uploadOutlineRegion(outlineDirtyRegions[i]);
    }
    outlineDirtyRegions.length = 0;
    DEBUG_DIRTY_TRACKER.outlinePendingRegions = 0;
  }
  DEBUG_DIRTY_TRACKER.lastFlushUsedFullMask = usedFullMask;
  DEBUG_DIRTY_TRACKER.lastFlushMaskRegions = maskRegionsBefore;
  DEBUG_DIRTY_TRACKER.lastFlushUsedFullOutline = usedFullOutline;
  DEBUG_DIRTY_TRACKER.lastFlushOutlineRegions = outlineRegionsBefore;
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
uniform sampler2D u_pointsSampler;
uniform sampler2D u_paletteSampler;
uniform sampler2D u_imageCmapSampler;

uniform float u_maskOpacity;
uniform float u_maskVisible;
uniform float u_outlinesVisible;
uniform float u_maskStyle;
uniform float u_imageVisible;
uniform float u_flowVisible;
uniform float u_flowOpacity;
uniform float u_distanceVisible;
uniform float u_distanceOpacity;
uniform float u_pointsVisible;
uniform float u_pointsOpacity;
uniform float u_colorOffset;
uniform float u_paletteSize;
uniform float u_usePalette;
uniform float u_imageCmapType;

vec3 sinebow(float t) {
  float angle = 6.28318530718 * fract(t);
  float r = sin(angle) * 0.5 + 0.5;
  float g = sin(angle + 2.09439510239) * 0.5 + 0.5;
  float b = sin(angle + 4.18879020479) * 0.5 + 0.5;
  return vec3(r, g, b);
}

vec3 hashColor(float label) {
  float golden = 0.61803398875;
  float t = fract(label * golden + u_colorOffset);
  return sinebow(t);
}

vec3 paletteColor(float label) {
  float size = max(u_paletteSize, 1.0);
  // palette[0] = background, palette[1] = color for label 1, etc.
  // So use label directly as index (no -1 offset)
  float idx = mod(label, size);
  float u = (idx + 0.5) / size;
  return texture(u_paletteSampler, vec2(u, 0.5)).rgb;
}

// Apply image colormap to grayscale value
vec3 applyImageCmap(float intensity) {
  // 0 = grayscale (passthrough)
  if (u_imageCmapType < 0.5) {
    return vec3(intensity);
  }
  // 1 = grayscale with red clipping indicator
  if (u_imageCmapType < 1.5) {
    if (intensity > 0.999) {
      return vec3(1.0, 0.0, 0.0); // Red for clipped pixels
    }
    return vec3(intensity);
  }
  // 2+ = use LUT texture
  return texture(u_imageCmapSampler, vec2(intensity, 0.5)).rgb;
}

void main() {
  vec2 baseCoord = vec2(v_texCoord.x, 1.0 - v_texCoord.y);
  vec4 baseColor = vec4(0.0, 0.0, 0.0, 1.0);
  if (u_imageVisible > 0.5) {
    vec4 rawColor = texture(u_baseSampler, baseCoord);
    // Convert to grayscale intensity (use luminance for color images, or just R for grayscale)
    float intensity = dot(rawColor.rgb, vec3(0.299, 0.587, 0.114));
    // Apply colormap
    baseColor = vec4(applyImageCmap(intensity), rawColor.a);
  }
  vec3 color = baseColor.rgb;
  if (u_maskVisible > 0.5 && u_maskOpacity > 0.0) {
    vec2 packed = texture(u_maskSampler, v_texCoord).rg;
    float low = floor(packed.r * 255.0 + 0.5);
    float high = floor(packed.g * 255.0 + 0.5);
    float label = low + high * 256.0;
    if (label > 0.5) {
      float alpha = clamp(u_maskOpacity, 0.0, 1.0);
      float outline = 0.0;
      if (u_outlinesVisible > 0.5) {
        float outlineSample = texture(u_outlineSampler, v_texCoord).r;
        outline = outlineSample > 0.5 ? 1.0 : 0.0;
      }
      if (u_maskStyle > 1.5) {
        alpha = alpha * outline;
      } else if (u_maskStyle < 0.5 && u_outlinesVisible > 0.5) {
        alpha = mix(alpha * 0.5, alpha, outline);
      }
      vec3 maskColor = (u_usePalette > 0.5) ? paletteColor(label) : hashColor(label);
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
  if (u_pointsVisible > 0.5) {
    vec4 overlay = texture(u_pointsSampler, v_texCoord);
    float overlayAlpha = clamp(u_pointsOpacity, 0.0, 1.0) * overlay.a;
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

  const paletteTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, paletteTexture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const paletteInitData = buildPaletteTextureData();
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, PALETTE_TEXTURE_SIZE, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, paletteInitData);
  gl.bindTexture(gl.TEXTURE_2D, null);
  paletteTextureDirty = false;

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
    maskStyle: gl.getUniformLocation(program, 'u_maskStyle'),
    flowSampler: gl.getUniformLocation(program, 'u_flowSampler'),
    flowVisible: gl.getUniformLocation(program, 'u_flowVisible'),
    flowOpacity: gl.getUniformLocation(program, 'u_flowOpacity'),
    distanceSampler: gl.getUniformLocation(program, 'u_distanceSampler'),
    distanceVisible: gl.getUniformLocation(program, 'u_distanceVisible'),
    distanceOpacity: gl.getUniformLocation(program, 'u_distanceOpacity'),
    pointsSampler: gl.getUniformLocation(program, 'u_pointsSampler'),
    paletteSampler: gl.getUniformLocation(program, 'u_paletteSampler'),
    pointsVisible: gl.getUniformLocation(program, 'u_pointsVisible'),
    pointsOpacity: gl.getUniformLocation(program, 'u_pointsOpacity'),
    colorOffset: gl.getUniformLocation(program, 'u_colorOffset'),
    paletteSize: gl.getUniformLocation(program, 'u_paletteSize'),
    usePalette: gl.getUniformLocation(program, 'u_usePalette'),
    imageCmapSampler: gl.getUniformLocation(program, 'u_imageCmapSampler'),
    imageCmapType: gl.getUniformLocation(program, 'u_imageCmapType'),
  };
  gl.useProgram(program);
  gl.uniform1i(uniforms.baseSampler, 0);
  gl.uniform1i(uniforms.maskSampler, 1);
  gl.uniform1i(uniforms.outlineSampler, 2);
  gl.uniform1i(uniforms.flowSampler, 3);
  gl.uniform1i(uniforms.distanceSampler, 4);
  gl.uniform1i(uniforms.pointsSampler, 5);
  gl.uniform1i(uniforms.paletteSampler, 6);
  gl.uniform1i(uniforms.imageCmapSampler, 7);
  gl.uniform1f(uniforms.paletteSize, PALETTE_TEXTURE_SIZE);
  gl.uniform1f(uniforms.imageCmapType, 0.0); // Default to grayscale
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
    paletteTexture,
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
  updatePaletteTextureIfNeeded();
  applyPendingFullTextureUpdates();
  markMaskTextureFullDirty();
  markOutlineTextureFullDirty();
  applyMaskRedrawImmediate();
  draw();
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
  const startTime = perfNow();
  if (!webglPipeline || !webglPipeline.affinity || !webglPipeline.affinity.buffer) {
    logWebgl('affinity-geometry', { reason: 'missing-buffer' });
    return;
  }
  if (!affinityGeometryDirty) {
    return;
  }
  if (USE_WEBGL_OVERLAY && webglOverlay && webglOverlay.enabled) {
    affinityGeometryDirty = false;
    return;
  }
  const affinity = webglPipeline.affinity;
  if (!affinityGraphInfo) {
    gl.bindBuffer(gl.ARRAY_BUFFER, affinity.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(0), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    affinity.vertexCount = 0;
    affinityGeometryDirty = false;
    logWebgl('affinity-geometry', { reason: 'hidden', duration: perfNow() - startTime });
    return;
  }
  if (!affinityGraphInfo.vertices || !affinityGraphInfo.vertices.length) {
    buildAffinityGraphSegments();
  }
  const info = affinityGraphInfo;
  if (!showAffinityGraph || !info || !info.vertices || !info.vertices.length) {
    gl.bindBuffer(gl.ARRAY_BUFFER, affinity.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(0), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    affinity.vertexCount = 0;
    affinityGeometryDirty = false;
    logWebgl('affinity-geometry', { reason: 'hidden', duration: perfNow() - startTime });
    return;
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, affinity.buffer);
  gl.bufferData(gl.ARRAY_BUFFER, info.vertices, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  affinity.vertexCount = info.vertices.length / 2;
  affinityGeometryDirty = false;
  logWebgl('affinity-geometry', {
    edges: info.vertices.length / 4,
    vertices: affinity.vertexCount,
    byteLength: info.vertices.byteLength,
    duration: perfNow() - startTime,
  });
}

function drawAffinityLines(matrix) {
  const startTime = perfNow();
  if (!webglPipeline || !webglPipeline.affinityProgram || !webglPipeline.affinityUniforms || !webglPipeline.affinity) {
    return;
  }
  if (!showAffinityGraph || !affinityGraphInfo || !affinityGraphInfo.values) {
    return;
  }
  const alpha = computeAffinityAlpha();
  if (alpha <= 0) {
    logWebgl('affinity-draw', { reason: 'alpha-zero', alpha });
    return;
  }
  const { gl, affinityProgram, affinityUniforms, affinity } = webglPipeline;
  rebuildAffinityGeometry(gl);
  if (!affinity.vertexCount || affinity.vertexCount <= 0) {
    logWebgl('affinity-draw', { reason: 'no-vertices' });
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
  if ((affinityDrawLogCounter++ % AFFINITY_DRAW_LOG_INTERVAL) === 0) {
    logWebgl('affinity-draw', {
      vertexCount: affinity.vertexCount,
      alpha,
      scale: viewState.scale,
      duration: perfNow() - startTime,
    });
  }
}

function updateOverlayTexture(kind, image) {
  if (!isWebglPipelineActive() || !webglPipeline) {
    return;
  }
  if (kind !== 'flow' && kind !== 'distance') {
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
    paletteTexture,
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
  const maskStyleValue = maskDisplayMode === MASK_DISPLAY_MODES.SOLID
    ? 1
    : (maskDisplayMode === MASK_DISPLAY_MODES.OUTLINE ? 2 : 0);
  pipelineGl.uniform1f(uniforms.maskStyle, maskStyleValue);
  pipelineGl.uniform1f(uniforms.colorOffset, nColorActive ? 0.0 : 0.0);
  pipelineGl.uniform1f(uniforms.flowVisible, showFlowOverlay && flowOverlayImage && flowOverlayImage.complete ? 1 : 0);
  pipelineGl.uniform1f(uniforms.flowOpacity, 0.7);
  pipelineGl.uniform1f(uniforms.distanceVisible, showDistanceOverlay && distanceOverlayImage && distanceOverlayImage.complete ? 1 : 0);
  pipelineGl.uniform1f(uniforms.distanceOpacity, 0.6);
  pipelineGl.uniform1f(uniforms.pointsVisible, 0);
  pipelineGl.uniform1f(uniforms.pointsOpacity, 0);
  pipelineGl.uniform1f(uniforms.usePalette, 1);
  pipelineGl.uniform1f(uniforms.imageCmapType, getImageCmapTypeValue());

  pipelineGl.activeTexture(pipelineGl.TEXTURE0);
  pipelineGl.bindTexture(pipelineGl.TEXTURE_2D, baseTexture || webglPipeline.emptyTexture);
  pipelineGl.activeTexture(pipelineGl.TEXTURE1);
  pipelineGl.bindTexture(pipelineGl.TEXTURE_2D, maskTexture || webglPipeline.emptyTexture);
  pipelineGl.activeTexture(pipelineGl.TEXTURE2);
  pipelineGl.bindTexture(pipelineGl.TEXTURE_2D, outlineTexture || webglPipeline.emptyTexture);
  pipelineGl.activeTexture(pipelineGl.TEXTURE6);
  pipelineGl.bindTexture(pipelineGl.TEXTURE_2D, webglPipeline.paletteTexture || webglPipeline.emptyTexture);
  pipelineGl.activeTexture(pipelineGl.TEXTURE7);
  pipelineGl.bindTexture(pipelineGl.TEXTURE_2D, webglPipeline.imageCmapTexture || webglPipeline.emptyTexture);
  bindOverlayTextureOrEmpty(flowTexture, 3);
  bindOverlayTextureOrEmpty(distanceTexture, 4);

  pipelineGl.drawArrays(pipelineGl.TRIANGLE_STRIP, 0, 4);

  pipelineGl.bindVertexArray(null);
  pipelineGl.useProgram(null);
  const overlayHandled = drawAffinityGraphWebgl();
  if (!overlayHandled) {
    drawAffinityLines(matrix);
  }
  if (vectorOverlay && vectorOverlay.enabled) {
    const matrix = computeWebglMatrix((webglOverlay && webglOverlay.matrixCache) || new Float32Array(9), canvas.width, canvas.height);
    drawVectorOverlay(matrix);
  }
  if (pointsOverlay && pointsOverlay.enabled) {
    const matrix = computeWebglMatrix((webglOverlay && webglOverlay.matrixCache) || new Float32Array(9), canvas.width, canvas.height);
    drawPointsOverlay(matrix);
  }
}


function shouldApplyOverlayZoomFade() {
  return !affinitySegEnabled;
}


function updateVectorOverlayVisibility() {
  if (!vectorOverlay || !vectorOverlay.enabled) {
    return;
  }
  const show = affinitySegEnabled && showVectorOverlay && vectorOverlay.vertexCount > 0;
  if (vectorOverlay.canvas) {
    vectorOverlay.canvas.style.display = show ? 'block' : 'none';
    vectorOverlay.canvas.style.opacity = show ? '1' : '0';
  }
}

function updateOverlayVisibility() {
  if (!webglOverlay || !webglOverlay.enabled) {
    return;
  }
  const hasLines = showAffinityGraph && webglOverlay.edgeCount > 0;
  const visible = hasLines;
  const alpha = hasLines
    ? (Number.isFinite(webglOverlay.displayAlpha) ? webglOverlay.displayAlpha : 1)
    : 0;
  setOverlayCanvasVisibility(visible, alpha);
  updateVectorOverlayVisibility();
}

function drawAffinityGraphShared(matrix) {
  if (!webglOverlay || !webglOverlay.enabled) {
    return;
  }
  const glCanvas = getOverlayCanvasElement();
  const showLines = showAffinityGraph && affinityGraphInfo && affinityGraphInfo.values;
  const showPoints = false;
  if (!showLines && !showPoints) {
    clearWebglOverlaySurface();
    return;
  }
  const {
    gl: overlayGl,
    lineProgram,
    lineAttribs,
    lineUniforms,
    pointProgram,
    pointAttribs,
    pointUniforms,
  } = webglOverlay;
  if (!overlayGl || !lineProgram || !lineUniforms || !lineAttribs) {
    clearWebglOverlaySurface();
    return;
  }
  let mustBuild = showLines && Boolean(webglOverlay.needsGeometryRebuild)
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
  if (showLines && (!webglOverlay.positionsArray || !webglOverlay.positionsArray.length)) {
    if (!showPoints) {
      clearWebglOverlaySurface();
      return;
    }
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
  overlayGl.useProgram(lineProgram);
  overlayGl.uniformMatrix3fv(lineUniforms.matrix, false, matrix);
  const s = Math.max(0.0001, Number(viewState && viewState.scale ? viewState.scale : 1.0));
  const minStep = minAffinityStepLength > 0 ? minAffinityStepLength : 1.0;
  const minEdgePx = Math.max(0, s * minStep);
  const dprSafe = Number.isFinite(dpr) && dpr > 0 ? dpr : 1;
  const cutoff = OVERLAY_PIXEL_FADE_CUTOFF * dprSafe;
  const t = cutoff <= 0 ? 1 : Math.max(0, Math.min(1, minEdgePx / cutoff));
  const alphaScale = t * t * (3 - 2 * t);
  const clampedAlpha = Math.max(0, Math.min(1, alphaScale));
  const scaledAlpha = clampedAlpha;
  webglOverlay.lineAlpha = clampedAlpha;
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
        overlayGl.bufferSubData(overlayGl.ARRAY_BUFFER, baseCol, webglOverlay.colorsArray.subarray(baseCol, baseCol + 8));
      }
      webglOverlay.dirtyColSlots.clear();
    }
  }
  overlayGl.bindBuffer(overlayGl.ARRAY_BUFFER, webglOverlay.positionBuffer);
  overlayGl.enableVertexAttribArray(lineAttribs.position);
  overlayGl.vertexAttribPointer(lineAttribs.position, 2, overlayGl.FLOAT, false, 0, 0);
  overlayGl.bindBuffer(overlayGl.ARRAY_BUFFER, webglOverlay.colorBuffer);
  overlayGl.enableVertexAttribArray(lineAttribs.color);
  overlayGl.vertexAttribPointer(lineAttribs.color, 4, overlayGl.UNSIGNED_BYTE, true, 0, 0);
  overlayGl.enable(overlayGl.BLEND);
  overlayGl.blendFunc(overlayGl.SRC_ALPHA, overlayGl.ONE_MINUS_SRC_ALPHA);
  const edgesToDraw = Math.max(webglOverlay.edgeCount | 0, (webglOverlay.maxUsedSlotIndex | 0) + 1);
  const verticesToDraw = Math.max(0, edgesToDraw) * 2;
  let hasContent = false;
  if (showLines && verticesToDraw > 0) {
    overlayGl.drawArrays(overlayGl.LINES, 0, verticesToDraw);
    hasContent = true;
  }
  overlayGl.disableVertexAttribArray(lineAttribs.position);
  overlayGl.disableVertexAttribArray(lineAttribs.color);
  overlayGl.disable(overlayGl.BLEND);
  overlayGl.bindBuffer(overlayGl.ARRAY_BUFFER, null);
  overlayGl.useProgram(null);
  let finalAlpha = hasContent ? Math.max(0, Math.min(1, webglOverlay.displayAlpha || 0)) : 0;
  if (finalAlpha <= 0.001) {
    finalAlpha = 0;
  }
  if (webglOverlay.shared) {
    webglOverlay.displayAlpha = finalAlpha;
    updateOverlayVisibility();
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
let pointsOverlay = null;
let vectorOverlay = null;

function getOverlayCanvasElement() {
  if (!webglOverlay || !webglOverlay.enabled) {
    return null;
  }
  return webglOverlay.canvas || null;
}

let affinityGeometryDirty = true;

function markAffinityGeometryDirty() {
  affinityGeometryDirty = true;
  if (affinityGraphInfo) {
    affinityGraphInfo.path = null;
    affinityGraphInfo.vertices = null;
  }
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
let minAffinityStepLength = 1.0;
const previewCanvas = document.getElementById('brushPreview');
const previewCtx = previewCanvas.getContext('2d');
previewCanvas.width = canvas.width;
previewCanvas.height = canvas.height;
  // Ensure brush preview is above WebGL overlay for visibility
  try { previewCanvas.style.zIndex = '3'; } catch (_) { /* ignore */ }

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
  const average = fpsSamples.reduce((sum, value) => sum + value, 0) / fpsSamples.length;
  const interval = 1000 / Math.max(average, 1);
  const smoothing = 0.3;
  const blendedInterval = (frameIntervalEstimateMs * (1 - smoothing)) + (interval * smoothing);
  frameIntervalEstimateMs = Math.max(MIN_FRAME_INTERVAL_MS, blendedInterval);
  if (!fpsDisplay) {
    return;
  }
  if (now - lastFpsUpdate < 250) {
    return;
  }
  lastFpsUpdate = now;
  let text = `FPS: ${average.toFixed(1)}`;
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

const OmniLog = window.OmniLog || {};
const log = typeof OmniLog.log === 'function' ? OmniLog.log : () => {};
const flushLogs = typeof OmniLog.flushLogs === 'function' ? OmniLog.flushLogs : () => {};
const scheduleLogFlush = typeof OmniLog.scheduleLogFlush === 'function' ? OmniLog.scheduleLogFlush : () => {};
const setPywebviewReady = typeof OmniLog.setPywebviewReady === 'function' ? OmniLog.setPywebviewReady : () => {};
const clearLogQueue = typeof OmniLog.clearQueue === 'function' ? OmniLog.clearQueue : () => {};
const OmniHistory = window.OmniHistory || {};
let loggedPixelSample = false;
let drawLogCount = 0;
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
let savedAffinityGraphPayload = null;
let affinitySteps = DEFAULT_AFFINITY_STEPS.map((step) => step.slice());
refreshOppositeStepMapping();
let shuttingDown = false;

function clearHoverPreview() {
  if (interactionsApi && typeof interactionsApi.clearHoverPreview === 'function') {
    interactionsApi.clearHoverPreview();
    return;
  }
  drawBrushPreview(null);
  updateHoverInfo(null);
}

function clamp(value, min, max) {
  if (!Number.isFinite(value)) {
    return min;
  }
  return Math.min(max, Math.max(min, value));
}

const sliderRegistry = new Map();
const dropdownRegistry = new Map();
let dropdownOpenId = null;

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


const FILENAME_TRUNCATE = 10;

function truncateFilename(name, keep = FILENAME_TRUNCATE) {
  if (!name || typeof name !== 'string') {
    return '';
  }
  const lastDot = name.lastIndexOf('.');
  const hasExt = lastDot > 0 && lastDot < name.length - 1;
  const base = hasExt ? name.slice(0, lastDot) : name;
  const ext = hasExt ? name.slice(lastDot) : '';
  if (base.length <= keep * 2) {
    return base + ext;
  }
  return `${base.slice(0, keep)}...${base.slice(-keep)}${ext}`;
}

function pointerPercent(evt, container) {
  const rect = container.getBoundingClientRect();
  if (rect.width <= 0) {
    return 0;
  }
  const ratio = (evt.clientX - rect.left) / rect.width;
  return clamp(ratio, 0, 1);
}

function updateNativeRangeFill(input) {
  if (!input || !(isIOSDevice && isSafariWebKit)) {
    return;
  }
  const percent = valueToPercent(input);
  const root = input.closest('.slider');
  if (!root) {
    return;
  }
  const trackRadius = parseFloat(getComputedStyle(root).getPropertyValue('--slider-track-radius'))
    || Math.round(root.clientHeight / 2);
  const usable = Math.max(0, root.clientWidth - trackRadius * 2);
  const fillPx = Math.round(usable * percent);
  root.style.setProperty('--slider-fill-px', `${fillPx}px`);
  const knob = root.querySelector('.slider-native-knob');
  if (knob) {
    knob.style.left = `${trackRadius + fillPx}px`;
  }
}

function registerSlider(root) {
  const id = root.dataset.sliderId || root.dataset.slider || root.id;
  if (!id) {
    return;
  }
  const type = (root.dataset.sliderType || 'single').toLowerCase();
  const inputs = Array.from(root.querySelectorAll('input[type=\"range\"]'));
  if (!inputs.length) {
    return;
  }
  if (type === 'dual' && inputs.length < 2) {
    console.warn(`slider ${id} configured as dual but only one range input found`);
    return;
  }
  if (isIOSDevice && isSafariWebKit) {
    root.classList.add('slider-native');
    if (!root.querySelector('.slider-native-track')) {
      const track = document.createElement('div');
      track.className = 'slider-native-track';
      const fill = document.createElement('div');
      fill.className = 'slider-native-fill';
      track.appendChild(fill);
      const knob = document.createElement('div');
      knob.className = 'slider-native-knob';
      root.appendChild(track);
      root.appendChild(knob);
    }
    inputs.forEach((input) => {
      updateNativeRangeFill(input);
      input.addEventListener('input', () => updateNativeRangeFill(input));
      input.addEventListener('change', () => updateNativeRangeFill(input));
    });
    return;
  }

  root.innerHTML = '';
  const track = document.createElement('div');
  track.className = 'slider-track';
  root.appendChild(track);
  const thumbs = inputs.map(() => {
    const thumb = document.createElement('div');
    thumb.className = 'slider-thumb';
    root.appendChild(thumb);
    return thumb;
  });

  const entry = {
    id,
    type: type === 'dual' ? 'dual' : 'single',
    root,
    inputs,
    track,
    thumbs,
    activePointer: null,
    activeThumb: null,
  };

  if (!entry.root.hasAttribute('tabindex')) {
    entry.root.tabIndex = 0;
  }

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
      entry.track.style.setProperty('--slider-fill-start', left);
      entry.track.style.setProperty('--slider-fill-end', rightPercent);
      entry.thumbs[0].style.left = left;
      entry.thumbs[1].style.left = rightPercent;
    } else {
      const input = entry.inputs[0];
      const percent = valueToPercent(input);
      const trackStyle = getComputedStyle(entry.track);
      const trackRadius = parseFloat(trackStyle.getPropertyValue('--slider-track-radius'))
        || Math.round(entry.track.clientHeight / 2);
      const usable = Math.max(0, entry.track.clientWidth - trackRadius * 2);
      const fillPx = Math.round(usable * percent);
      entry.track.style.setProperty('--slider-fill-px', `${fillPx}px`);
      entry.track.style.setProperty('--slider-track-radius', `${trackRadius}px`);
      entry.thumbs[0].style.left = `${trackRadius + fillPx}px`;
    }
  };

  const stepInputValue = (input, direction) => {
    if (!input) {
      return;
    }
    const min = Number(input.min || 0);
    const max = Number(input.max || 1);
    let step = Number(input.step || '1');
    if (!Number.isFinite(step) || step <= 0) {
      step = 1;
    }
    const precision = (step.toString().split('.')[1] || '').length;
    const factor = 10 ** precision;
    const current = Number(input.value || min);
    let next = current + direction * step;
    next = clamp(next, min, max);
    next = Math.round(next * factor) / factor;
    input.value = String(next);
    input.dispatchEvent(new Event('input', { bubbles: true }));
    apply();
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
    if (entry.root.focus) {
      entry.root.focus({ preventScroll: true });
    }
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

  entry.root.addEventListener('keydown', (evt) => {
    if (evt.key === 'ArrowLeft' || evt.key === 'ArrowDown') {
      evt.preventDefault();
      const index = entry.type === 'dual' ? (entry.activeThumb ?? 0) : 0;
      stepInputValue(entry.inputs[index], -1);
      return;
    }
    if (evt.key === 'ArrowRight' || evt.key === 'ArrowUp') {
      evt.preventDefault();
      const index = entry.type === 'dual' ? (entry.activeThumb ?? 0) : 0;
      stepInputValue(entry.inputs[index], 1);
      return;
    }
    if (evt.key === 'Home') {
      evt.preventDefault();
      const index = entry.type === 'dual' ? (entry.activeThumb ?? 0) : 0;
      const input = entry.inputs[index];
      if (input) {
        input.value = String(input.min || 0);
        input.dispatchEvent(new Event('input', { bubbles: true }));
        apply();
      }
      return;
    }
    if (evt.key === 'End') {
      evt.preventDefault();
      const index = entry.type === 'dual' ? (entry.activeThumb ?? 0) : 0;
      const input = entry.inputs[index];
      if (input) {
        input.value = String(input.max || 1);
        input.dispatchEvent(new Event('input', { bubbles: true }));
        apply();
      }
    }
  });

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
  if (entry.menuWrapper) {
    entry.menuWrapper.focus({ preventScroll: true });
  }
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
  // Create label span for gradient text support
  const labelSpan = document.createElement('span');
  labelSpan.className = 'dropdown-label';
  button.appendChild(labelSpan);
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
  menuWrapper.tabIndex = -1;
  root.appendChild(menuWrapper);

  const entry = {
    id,
    root,
    select,
    button,
    menu,
    menuWrapper,
    options: originalOptions,
    loop: root.dataset.loop === 'true' ? { size: 5, mode: 'loop' } : null,
    countLabel: root.dataset.countLabel || 'items',
    confirm: root.dataset.apply === 'confirm',
    tooltipDisabled: root.dataset.tooltipDisabled === 'true',
  };

    const applySelection = () => {
    const selectedOption = select.options[select.selectedIndex];
    const displayLabel = selectedOption ? selectedOption.textContent : 'Select';
    labelSpan.textContent = displayLabel;
    if (selectedOption) {
      const fullLabel = selectedOption.dataset.fullPath || selectedOption.dataset.fullLabel || selectedOption.title || selectedOption.textContent;
      if (fullLabel) {
        if (entry.tooltipDisabled) {
          button.removeAttribute('title');
          button.removeAttribute('data-tooltip');
        } else if (entry.id === 'imageNavigator') {
          button.dataset.tooltip = fullLabel;
          button.removeAttribute('title');
        } else {
          button.removeAttribute('title');
          button.removeAttribute('data-tooltip');
        }
      }
    }
    menu.querySelectorAll('.dropdown-option').forEach((child) => {
      const isSelected = child.dataset.value === select.value;
      child.dataset.selected = isSelected ? 'true' : 'false';
      const color = isSelected ? 'var(--accent-ink, #161616)' : 'var(--panel-text-color)';
      child.style.setProperty('color', color, 'important');
    });
  };

  const buildOption = (opt) => {
    const item = document.createElement('div');
    item.className = 'dropdown-option';
    item.dataset.value = opt.value;
    item.textContent = opt.label;
    item.setAttribute('role', 'option');
    if (opt.title && !entry.tooltipDisabled && entry.id === 'imageNavigator') {
      item.dataset.tooltip = opt.title;
    }
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
    return item;
  };

    const buildMenu = () => {
    menu.innerHTML = '';
    const opts = entry.options || [];
    const loopEnabled = Boolean(entry.loop && entry.loop.mode === 'loop');
    if (loopEnabled && opts.length) {
      const loopOptions = opts.filter((opt) => !['__add__','__open_folder__'].includes(opt.value));
      const addOption = opts.find((opt) => opt.value === '__add__');
      const openFolderOption = opts.find((opt) => opt.value === '__open_folder__');
      const size = entry.loop.size || 5;
      const half = Math.floor(size / 2);
      const total = loopOptions.length;
      const currentIndex = Math.max(0, loopOptions.findIndex((opt) => opt.value === select.value));
      for (let i = -half; i <= half; i += 1) {
        const idx = total > 0 ? (currentIndex + i + total) % total : 0;
        const opt = loopOptions[idx];
        if (!opt) continue;
        menu.appendChild(buildOption(opt));
      }
      if (addOption) {
        const addRow = buildOption(addOption);
        addRow.classList.add('dropdown-add');
        menu.appendChild(addRow);
      }
      if (openFolderOption) {
        const openRow = buildOption(openFolderOption);
        openRow.classList.add('dropdown-add');
        menu.appendChild(openRow);
      }
      const footer = document.createElement('div');
      footer.className = 'dropdown-footer';
      const count = document.createElement('span');
      count.className = 'dropdown-count';
      count.textContent = total ? `${total} ${entry.countLabel}` : `0 ${entry.countLabel}`;
      const toggleBtn = document.createElement('button');
      toggleBtn.type = 'button';
      toggleBtn.className = 'dropdown-expand';
      toggleBtn.textContent = 'Show all';
      toggleBtn.addEventListener('click', (evt) => {
        evt.preventDefault();
        entry.loop.mode = 'full';
        buildMenu();
      });
      footer.appendChild(count);
      footer.appendChild(toggleBtn);
      menu.appendChild(footer);
    } else {
      opts.forEach((opt) => {
        menu.appendChild(buildOption(opt));
      });
      if (entry.loop) {
        const footer = document.createElement('div');
        footer.className = 'dropdown-footer';
        const count = document.createElement('span');
        count.className = 'dropdown-count';
        count.textContent = opts.filter((opt) => !['__add__','__open_folder__'].includes(opt.value)).length + ` ${entry.countLabel}`;
        const toggleBtn = document.createElement('button');
        toggleBtn.type = 'button';
        toggleBtn.className = 'dropdown-expand';
        toggleBtn.textContent = 'Show less';
        toggleBtn.addEventListener('click', (evt) => {
          evt.preventDefault();
          entry.loop.mode = 'loop';
          buildMenu();
        });
        footer.appendChild(count);
        footer.appendChild(toggleBtn);
        menu.appendChild(footer);
      }
    }
    applySelection();
  };

  const shiftSelection = (delta) => {
    if (!entry.loop || entry.loop.mode !== 'loop') return;
    const loopOptions = entry.options.filter((opt) => !['__add__','__open_folder__'].includes(opt.value));
    if (!loopOptions.length) return;
    const currentIndex = Math.max(0, loopOptions.findIndex((opt) => opt.value === select.value));
    const nextIndex = (currentIndex + delta + loopOptions.length) % loopOptions.length;
    const nextValue = loopOptions[nextIndex].value;
    select.value = nextValue;
    if (!entry.confirm) {
      select.dispatchEvent(new Event('change', { bubbles: true }));
    }
    buildMenu();
  };

  button.addEventListener('click', () => {
    toggleDropdown(entry);
  });

  select.addEventListener('change', () => {
    if (entry.loop) {
      buildMenu();
    } else {
      applySelection();
    }
  });

  if (entry.loop) {
    let wheelVelocity = 0;
    let wheelAccumulator = 0;
    let wheelAnimating = false;
    let wheelLastTime = 0;
    let wheelRaf = 0;

    const getStepPx = () => {
      const styles = getComputedStyle(menuWrapper);
      const raw = styles.getPropertyValue('--slider-track-height')
        || getComputedStyle(document.documentElement).getPropertyValue('--slider-track-height');
      const parsed = parseFloat(raw);
      return Number.isFinite(parsed) && parsed > 0 ? parsed : 32;
    };

    const animateWheel = (ts) => {
      if (!entry.loop || entry.loop.mode !== 'loop') {
        wheelAnimating = false;
        wheelVelocity = 0;
        wheelAccumulator = 0;
        return;
      }
      if (!wheelLastTime) {
        wheelLastTime = ts;
      }
      const dt = Math.min(48, ts - wheelLastTime);
      wheelLastTime = ts;
      const friction = Math.pow(0.9, dt / 16);
      wheelVelocity *= friction;
      wheelAccumulator += wheelVelocity * (dt / 16);

      const stepPx = getStepPx();
      while (Math.abs(wheelAccumulator) >= stepPx) {
        const direction = wheelAccumulator > 0 ? 1 : -1;
        shiftSelection(direction);
        wheelAccumulator -= direction * stepPx;
      }

      if (Math.abs(wheelVelocity) < 0.05) {
        wheelAnimating = false;
        wheelVelocity = 0;
        wheelAccumulator = 0;
        wheelLastTime = 0;
        return;
      }
      wheelRaf = requestAnimationFrame(animateWheel);
    };

    menuWrapper.addEventListener('wheel', (evt) => {
      if (!entry.loop || entry.loop.mode !== 'loop') {
        return;
      }
      evt.preventDefault();
      wheelVelocity += evt.deltaY * 0.6;
      if (!wheelAnimating) {
        wheelAnimating = true;
        wheelLastTime = 0;
        wheelRaf = requestAnimationFrame(animateWheel);
      }
    }, { passive: false });
  }
  menuWrapper.addEventListener('keydown', (evt) => {
    if (evt.key === 'ArrowDown' || evt.key === 'ArrowUp') {
      evt.preventDefault();
      const delta = evt.key === 'ArrowDown' ? 1 : -1;
      if (entry.loop && entry.loop.mode === 'loop') {
        shiftSelection(delta);
        return;
      }
      const opts = entry.options.filter((opt) => !opt.disabled);
      if (!opts.length) return;
      const currentIndex = Math.max(0, opts.findIndex((opt) => opt.value === select.value));
      const nextIndex = Math.min(opts.length - 1, Math.max(0, currentIndex + delta));
      const nextValue = opts[nextIndex].value;
      select.value = nextValue;
      if (!entry.confirm) {
        select.dispatchEvent(new Event('change', { bubbles: true }));
      }
      applySelection();
      return;
    }
    if (evt.key === 'Enter') {
      evt.preventDefault();
      if (entry.confirm) {
        select.dispatchEvent(new Event('change', { bubbles: true }));
      }
      closeDropdown(entry);
      return;
    }
  });

  buildMenu();
  entry.applySelection = applySelection;
  entry.buildMenu = buildMenu;
  positionDropdown(entry);
  dropdownRegistry.set(id, entry);
}


function refreshDropdown(id) {
  const entry = dropdownRegistry.get(id);
  if (!entry) return;
  if (entry.loop && typeof entry.buildMenu === 'function') {
    entry.buildMenu();
    return;
  }
  if (typeof entry.applySelection === 'function') {
    entry.applySelection();
  }
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

function handleWindowBlur() {
  if (touchPointers && typeof touchPointers.clear === 'function') {
    touchPointers.clear();
  }
  pinchState = null;
  panPointerId = null;
  pointerState.resetPen();
  gestureState = null;
  wheelRotationBuffer = 0;
  clearHoverPreview();
  if (dropdownOpenId) {
    const entry = dropdownRegistry.get(dropdownOpenId);
    if (entry) {
      closeDropdown(entry);
    } else {
      dropdownOpenId = null;
    }
  }
}

document.addEventListener('mouseleave', (evt) => {
  if (!evt.relatedTarget && !evt.toElement) {
    clearHoverPreview();
  }
});

function currentTouchPoints() {
  return Array.from(touchPointers.entries()).map(([id, data]) => ({ id, x: data.x, y: data.y }));
}

function beginPinchGesture() {
  if (touchPointers.size < 2) {
    return;
  }
  const rect = canvas.getBoundingClientRect();
  const points = currentTouchPoints().slice(0, 2);
  const dx = points[0].x - points[1].x;
  const dy = points[0].y - points[1].y;
  const distance = Math.hypot(dx, dy) || 1;
  const angle = Math.atan2(dy, dx);
  const midpoint = {
    x: (points[0].x + points[1].x) / 2,
    y: (points[0].y + points[1].y) / 2,
  };
  const canvasMid = {
    x: midpoint.x - rect.left,
    y: midpoint.y - rect.top,
  };
  const imageMid = screenToImage(canvasMid);
  pinchState = {
    pointers: [points[0].id, points[1].id],
    startDistance: distance,
    startScale: viewState.scale,
    startAngle: angle,
    startRotation: normalizeAngle(viewState.rotation),
    imageMid,
    lastLoggedRotation: 0,
  };
  log('pinch gesture start distance=' + distance.toFixed(2) + ' angle=' + (angle * RAD_TO_DEG).toFixed(2) + ' deg');
  wheelRotationBuffer = 0;
  isPanning = false;
  isPainting = false;
  setHoverState(imageMid, canvasMid);
  updateHoverInfo(imageMid);
  renderHoverPreview();
}

function handleTouchPointerDown(evt, pointer) {
  if (pointerState.shouldIgnoreTouch(evt)) {
    evt.preventDefault();
    return;
  }
  evt.preventDefault();
  touchPointers.set(evt.pointerId, { x: evt.clientX, y: evt.clientY });
  try {
    canvas.setPointerCapture(evt.pointerId);
  } catch (_) {
    /* ignore */
  }
  if (pinchState) {
    // already in pinch gesture, ignore additional touches
    return;
  }
  if (touchPointers.size >= 2) {
    if (!pointerState.options.touch.enablePinchZoom && !pointerState.options.touch.enableRotate) {
      return;
    }
    if (panPointerId !== null) {
      try {
        canvas.releasePointerCapture(panPointerId);
      } catch (_) {
        /* ignore */
      }
      panPointerId = null;
    }
    beginPinchGesture();
    return;
  }
  // single finger: treat as pan gesture
  if (!pointerState.options.touch.enablePan) {
    return;
  }
  isPainting = false;
  isPanning = true;
  panPointerId = evt.pointerId;
  try {
    canvas.setPointerCapture(evt.pointerId);
  } catch (_) {
    /* ignore */
  }
  const worldPoint = screenToImage(pointer);
  setHoverState(worldPoint, { x: pointer.x, y: pointer.y });
  updateHoverInfo(worldPoint);
  renderHoverPreview();
  lastPoint = pointer;
  updateCursor();
}

function handleGestureStart(evt) {
  if (!pointerState.options.touch.enableRotate && !pointerState.options.touch.enablePinchZoom) {
    return;
  }
  if (!canvas) {
    return;
  }
  evt.preventDefault();
  const origin = resolveGestureOrigin(evt);
  const imagePoint = screenToImage(origin);
  gestureState = {
    startScale: viewState.scale,
    startRotation: normalizeAngle(viewState.rotation),
    origin,
    imagePoint,
  };
  if (interactionsApi && typeof interactionsApi.resetGestureScheduling === 'function') {
    interactionsApi.resetGestureScheduling();
  }
  wheelRotationBuffer = 0;
  setHoverState(imagePoint, origin);
  updateHoverInfo(imagePoint);
  renderHoverPreview();
  isPanning = false;
  spacePan = false;
  // Set a helpful cursor at gesture start
  setCursorHold(dotCursorCss);
  // Show simple dot cursor during gesture
  setCursorHold(dotCursorCss);
}

function handleGestureChange(evt) {
  if (!gestureState) {
    return;
  }
  evt.preventDefault();
  if (interactionsApi && typeof interactionsApi.setPendingGestureUpdate === 'function') {
    interactionsApi.setPendingGestureUpdate({
      origin: resolveGestureOrigin(evt),
      scale: Number.isFinite(evt.scale) ? evt.scale : 1,
      rotation: Number.isFinite(evt.rotation) ? evt.rotation : 0,
    });
  }
  scheduleGestureUpdate();
}

function handleGestureEnd(evt) {
  if (!gestureState) {
    return;
  }
  if (evt && typeof evt.preventDefault === 'function') {
    evt.preventDefault();
  }
  applyPendingGestureUpdate();
  gestureState = null;
  if (interactionsApi && typeof interactionsApi.resetGestureScheduling === 'function') {
    interactionsApi.resetGestureScheduling();
  }
  clearCursorOverride();
  updateHoverInfo(getHoverPoint() || null);
  renderHoverPreview();
}

const MIN_BRUSH_DIAMETER = 1;
const MAX_BRUSH_DIAMETER = 25;

const BRUSH_KERNEL_MODES = {
  SMOOTH: 'smooth',
  SNAPPED: 'snapped',
};
let brushKernelMode = BRUSH_KERNEL_MODES.SMOOTH;

function clampBrushDiameter(value) {
  const numeric = Number(value);
  if (Number.isNaN(numeric)) {
    return MIN_BRUSH_DIAMETER;
  }
  return Math.max(MIN_BRUSH_DIAMETER, Math.min(MAX_BRUSH_DIAMETER, Math.round(numeric)));
}

const defaultDiameter = clampBrushDiameter(initialBrushRadius * 2 + 1);
let brushDiameter = defaultDiameter;

const brushSizeSlider = document.getElementById('brushSizeSlider');
const brushSizeInput = document.getElementById('brushSizeInput');
const brushKernelModeSelect = document.getElementById('brushKernelMode');
if (brushSizeSlider) {
  brushSizeSlider.min = String(MIN_BRUSH_DIAMETER);
  brushSizeSlider.max = String(MAX_BRUSH_DIAMETER);
  brushSizeSlider.step = '1';
}
if (brushSizeInput) {
  brushSizeInput.min = String(MIN_BRUSH_DIAMETER);
  brushSizeInput.max = String(MAX_BRUSH_DIAMETER);
  brushSizeInput.step = '1';
}
if (brushKernelModeSelect) {
  brushKernelModeSelect.value = brushKernelMode;
}

window.addEventListener('resize', () => {
  if (dropdownOpenId) {
    const entry = dropdownRegistry.get(dropdownOpenId);
    if (entry) {
      positionDropdown(entry);
    } else {
      dropdownOpenId = null;
    }
  }
});

const gammaSlider = document.getElementById('gamma');
const gammaValue = document.getElementById('gammaValue');
const maskLabel = document.getElementById('maskLabel');
const labelValueInput = document.getElementById('labelValueInput');
const labelStepDown = document.getElementById('labelStepDown');
const labelStepUp = document.getElementById('labelStepUp');
const undoButton = document.getElementById('undoButton');
const redoButton = document.getElementById('redoButton');
const resetViewButton = document.getElementById('resetViewButton');
const saveStateButton = document.getElementById('saveStateButton');
const rotateLeftButton = document.getElementById('rotateLeftButton');
const rotateRightButton = document.getElementById('rotateRightButton');
const prevImageButton = document.getElementById('prevImageButton');
const nextImageButton = document.getElementById('nextImageButton');
const imageNavigator = document.getElementById('imageNavigator');
const leftPanelEl = document.getElementById('leftPanel');
const sidebarEl = document.getElementById('sidebar');
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
const segmentationModelSelect = document.getElementById('segmentationModel');
const segmentationModelFile = document.getElementById('segmentationModelFile');
const niterSlider = document.getElementById('niterSlider');
const niterInput = document.getElementById('niterInput');
if (niterSlider) {
  niterSlider.min = '-1';
}
if (niterInput) {
  niterInput.min = '-1';
}

const hoverInfo = document.getElementById('hoverInfo');
const fpsDisplay = document.getElementById('fpsDisplay');
const hoverValueDisplay = document.getElementById('hoverValueDisplay');
const hoverCoordDisplay = document.getElementById('hoverCoordDisplay');
const histogramValueMarker = document.getElementById('histogramValueMarker');
const maskOpacitySlider = document.getElementById('maskOpacity');
const maskOpacityInput = document.getElementById('maskOpacityInput');
const maskThresholdSlider = document.getElementById('maskThresholdSlider');
const maskThresholdInput = document.getElementById('maskThresholdInput');
const flowThresholdSlider = document.getElementById('flowThresholdSlider');
const flowThresholdInput = document.getElementById('flowThresholdInput');
const clusterToggle = document.getElementById('clusterToggle');
const affinityToggle = document.getElementById('affinityToggle');
const segModeToggle = document.getElementById('segModeToggle');
const segModeButtons = segModeToggle
  ? Array.from(segModeToggle.querySelectorAll('.kernel-option'))
  : [];
const affinityGraphToggle = document.getElementById('affinityGraphToggle');
const flowOverlayToggle = document.getElementById('flowOverlayToggle');
const distanceOverlayToggle = document.getElementById('distanceOverlayToggle');
const pointsOverlayToggle = document.getElementById('pointsOverlayToggle');
const vectorOverlayToggle = document.getElementById('vectorOverlayToggle');
const vectorOverlayRow = document.getElementById('vectorOverlayRow');
const imageVisibilityToggle = document.getElementById('imageVisibilityToggle');
const maskVisibilityToggle = document.getElementById('maskVisibilityToggle');
const intensityPanel = document.getElementById('intensityPanel');
const labelStylePanel = document.getElementById('labelStylePanel');
const autoNColorToggle = document.getElementById('autoNColorToggle');
const cmapPanel = document.getElementById('cmapPanel');
const cmapSelect = document.getElementById('cmapSelect');
const cmapHueOffsetSlider = document.getElementById('cmapHueOffset');
const cmapHueOffsetWrapper = document.querySelector('[data-slider-id="cmapHueOffset"]');
const cmapPreviewPill = document.getElementById('cmapPreviewPill');
const imageCmapPanel = document.getElementById('imageCmapPanel');
const imageCmapSelect = document.getElementById('imageCmapSelect');
const imageCmapPreviewPill = document.getElementById('imageCmapPreviewPill');
const ncolorSubsection = document.getElementById('ncolorSubsection');
const ncolorSwatches = document.getElementById('ncolorSwatches');
const ncolorAddColor = document.getElementById('ncolorAddColor');
const ncolorRemoveColor = document.getElementById('ncolorRemoveColor');
// Legacy references for compatibility
const ncolorPanel = cmapPanel;
const ncolorHueOffsetSlider = cmapHueOffsetSlider;
const ncolorColormapPreview = cmapPreviewPill;
const instanceColormapPreview = cmapPreviewPill;
const labelColormapSelect = cmapSelect;
const labelShuffleToggle = document.getElementById('labelShuffleToggle');
const labelShuffleSeedInput = document.getElementById('labelShuffleSeed');
const brushKernelToggle = document.getElementById('brushKernelToggle');
const systemRamEl = document.getElementById('systemRam');
const systemCpuEl = document.getElementById('systemCpu');
const systemGpuEl = document.getElementById('systemGpu');
const useGpuToggle = document.getElementById('useGpuToggle');
const maskStyleToggle = document.getElementById('maskStyleToggle');
const maskStyleButtons = maskStyleToggle
  ? Array.from(maskStyleToggle.querySelectorAll('.kernel-option'))
  : [];
const toolStopButtons = Array.from(document.querySelectorAll('.tool-stop'));
const toolOptionsBlocks = Array.from(document.querySelectorAll('.tool-options'));
const TOOL_MODE_ORDER = ['draw', 'erase', 'fill', 'picker'];
const PREVIEW_TOOL_TYPES = new Set(['brush', 'erase']);
const CROSSHAIR_TOOL_TYPES = new Set(['brush', 'erase', 'fill', 'picker']);

if (savedViewerState) {
  if (brushSizeSlider && typeof savedViewerState.brushDiameter === 'number') {
    brushSizeSlider.value = String(savedViewerState.brushDiameter);
    updateNativeRangeFill(brushSizeSlider);
  }
  if (maskOpacitySlider && typeof savedViewerState.maskOpacity === 'number') {
    maskOpacitySlider.value = Number(savedViewerState.maskOpacity).toFixed(2);
    updateNativeRangeFill(maskOpacitySlider);
  }
  if (maskThresholdSlider && typeof savedViewerState.maskThreshold === 'number') {
    maskThresholdSlider.value = Number(savedViewerState.maskThreshold).toFixed(1);
    updateNativeRangeFill(maskThresholdSlider);
  }
  if (flowThresholdSlider && typeof savedViewerState.flowThreshold === 'number') {
    flowThresholdSlider.value = Number(savedViewerState.flowThreshold).toFixed(1);
    updateNativeRangeFill(flowThresholdSlider);
  }
  if (typeof savedViewerState.maskDisplayMode === 'string') {
    maskDisplayMode = normalizeMaskDisplayMode(savedViewerState.maskDisplayMode);
    outlinesVisible = maskDisplayMode === MASK_DISPLAY_MODES.OUTLINED || maskDisplayMode === MASK_DISPLAY_MODES.OUTLINE;
  }
  if (gammaSlider && typeof savedViewerState.gamma === 'number') {
    const sliderValue = Math.round(savedViewerState.gamma * 100);
    gammaSlider.value = String(Math.min(600, Math.max(10, sliderValue)));
    updateNativeRangeFill(gammaSlider);
  }
  if ('niter' in savedViewerState) {
    hasPendingSavedNiter = true;
    pendingSavedNiter = savedViewerState.niter;
  }
}

document.querySelectorAll('[data-slider-id]').forEach((root) => {
  registerSlider(root);
});
document.querySelectorAll('[data-dropdown-id]').forEach((root) => {
  registerDropdown(root);
});
updateBrushControls();
updateKernelToggle();

if (labelValueInput) {
  labelValueInput.addEventListener('change', (evt) => {
    applyCurrentLabel(evt.target.value);
  });
  labelValueInput.addEventListener('keydown', (evt) => {
    if (evt.key === 'Enter') {
      applyCurrentLabel(evt.target.value);
    }
  });
}
if (labelStepDown) {
  labelStepDown.addEventListener('click', () => {
    applyCurrentLabel(currentLabel - 1);
  });
}
if (labelStepUp) {
  labelStepUp.addEventListener('click', () => {
    applyCurrentLabel(currentLabel + 1);
  });
}
if (undoButton) {
  undoButton.addEventListener('click', () => {
    undo();
  });
}
if (redoButton) {
  redoButton.addEventListener('click', () => {
    redo();
  });
}
if (resetViewButton) {
  resetViewButton.addEventListener('click', () => {
    resetView();
  });
}
if (saveStateButton) {
  saveStateButton.addEventListener('click', () => {
    saveViewerState({ immediate: true, seq: stateDirtySeq }).catch(() => {});
  });
}
if (rotateLeftButton) {
  rotateLeftButton.addEventListener('click', () => {
    rotateView(-Math.PI / 2);
  });
}
if (rotateRightButton) {
  rotateRightButton.addEventListener('click', () => {
    rotateView(Math.PI / 2);
  });
}
setupImageNavigator();


function setupImageNavigator() {
  if (!imageNavigator) {
    return;
  }
  if (!Array.isArray(directoryEntries) || directoryEntries.length === 0) {
    imageNavigator.innerHTML = '';
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = currentImageName || 'Select';
    imageNavigator.appendChild(opt);
    const openFileOption = document.createElement('option');
    openFileOption.value = '__open_file__';
    openFileOption.textContent = 'Open image file...';
    openFileOption.title = 'Open image file';
    imageNavigator.appendChild(openFileOption);
    const openFolderOption = document.createElement('option');
    openFolderOption.value = '__open_folder__';
    openFolderOption.textContent = 'Open image folder...';
    openFolderOption.title = 'Open image folder';
    imageNavigator.appendChild(openFolderOption);
    return;
  }
  imageNavigator.innerHTML = '';
  directoryEntries.forEach((entry) => {
    const opt = document.createElement('option');
    opt.value = entry.path || entry.name;
    opt.textContent = truncateFilename(entry.name);
    opt.dataset.fullLabel = entry.name;
    opt.dataset.fullPath = entry.path || entry.name;
    opt.title = entry.path || entry.name;
    if (entry.isCurrent) {
      opt.selected = true;
    }
    imageNavigator.appendChild(opt);
  });
  const openFileOption = document.createElement('option');
  openFileOption.value = '__open_file__';
  openFileOption.textContent = 'Open image file...';
  openFileOption.title = 'Open image file';
  imageNavigator.appendChild(openFileOption);
  const openFolderOption = document.createElement('option');
  openFolderOption.value = '__open_folder__';
  openFolderOption.textContent = 'Open image folder...';
  openFolderOption.title = 'Open image folder';
  imageNavigator.appendChild(openFolderOption);
  const dropdownEntry = dropdownRegistry.get('imageNavigator');
  if (dropdownEntry) {
    dropdownEntry.options = Array.from(imageNavigator.options).map((opt) => ({
      value: opt.value,
      label: opt.textContent || opt.value,
      disabled: opt.disabled,
      title: opt.dataset.fullPath || opt.dataset.fullLabel || opt.title || opt.textContent || opt.value,
    }));
    if (typeof dropdownEntry.buildMenu === 'function') {
      dropdownEntry.buildMenu();
    }
  }
  imageNavigator.addEventListener('change', async () => {
    const nextPath = imageNavigator.value;
    if (!nextPath || nextPath === currentImagePath) {
      return;
    }
    if (nextPath === '__open_file__') {
      await selectImageFile();
      imageNavigator.value = currentImagePath || '';
      refreshDropdown('imageNavigator');
      return;
    }
    if (nextPath === '__open_folder__') {
      await selectImageFolder();
      imageNavigator.value = currentImagePath || '';
      refreshDropdown('imageNavigator');
      return;
    }
    openImageByPath(nextPath).catch((err) => {
      console.warn('openImageByPath failed', err);
    });
  });
}

suppressDoubleTapZoom(leftPanelEl);
suppressDoubleTapZoom(sidebarEl);

attachNumberInputStepper(brushSizeInput, (delta) => {
  setBrushDiameter(brushDiameter + delta, true);
});

attachNumberInputStepper(gammaInput, (delta) => {
  setGamma(currentGamma + delta);
});

const HISTORY_LIMIT = 200;
if (typeof OmniHistory.init === 'function') {
  try {
    OmniHistory.init({ limit: HISTORY_LIMIT });
  } catch (err) {
    console.warn('OmniHistory init failed', err);
  }
}
const undoStack = typeof OmniHistory.getUndoStack === 'function'
  ? OmniHistory.getUndoStack()
  : [];
const redoStack = typeof OmniHistory.getRedoStack === 'function'
  ? OmniHistory.getRedoStack()
  : [];
const viewState = { scale: 1.0, offsetX: 0.0, offsetY: 0.0, rotation: 0.0 };
let maskVisible = true;
let imageVisible = true;
let currentLabel = 1;
if (savedViewerState && typeof savedViewerState.currentLabel === 'number' && savedViewerState.currentLabel > 0) {
  currentLabel = savedViewerState.currentLabel;
}
let originalImageData = null;
let isPanning = false;
let isPainting = false;
let lastPoint = { x: 0, y: 0 };
let tool = 'brush';
let spacePan = false;

const FPS_SAMPLE_LIMIT = 30;
let fpsSamples = [];
let lastFrameTimestamp = (typeof performance !== 'undefined' ? performance.now() : Date.now());
let lastFpsUpdate = lastFrameTimestamp;
const DEFAULT_FRAME_INTERVAL_MS = 1000 / 60;
const MIN_FRAME_INTERVAL_MS = 1000 / 240;
let frameIntervalEstimateMs = DEFAULT_FRAME_INTERVAL_MS;
let drawRequestHandle = 0;
let drawRequestPending = false;
let lastDrawCompletedAt = lastFrameTimestamp;

let eraseActive = false;
let erasePreviousLabel = null;
let isSegmenting = false;
let histogramData = null;
let windowLow = 0;
let windowHigh = 255;
let currentGamma = 1.0;
const NITER_MIN = -1;
const NITER_MAX = 400;
let niter = typeof CONFIG.niter === 'number' ? clamp(CONFIG.niter, 0, NITER_MAX) : 0;
let niterAuto = typeof CONFIG.niter !== 'number';
if (hasPendingSavedNiter) {
  setNiter(pendingSavedNiter, { silent: true });
  hasPendingSavedNiter = false;
  pendingSavedNiter = null;
}

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
const DEFAULT_SEGMENTATION_MODEL = 'bact_phase_affinity';
// Colormaps with hasOffset=true are cyclic and support hue rotation
// Colormaps with hasOffset=false are linear gradients (no offset control)
const LABEL_COLORMAPS = [
  { value: 'sinebow', label: 'sinebow', hasOffset: true },
  { value: 'viridis', label: 'viridis', hasOffset: false },
  { value: 'magma', label: 'magma', hasOffset: false },
  { value: 'plasma', label: 'plasma', hasOffset: false },
  { value: 'inferno', label: 'inferno', hasOffset: false },
  { value: 'cividis', label: 'cividis', hasOffset: false },
  { value: 'turbo', label: 'turbo', hasOffset: false },
  { value: 'gist_ncar', label: 'gist ncar', hasOffset: false },
  { value: 'vivid', label: 'vivid', hasOffset: true },
  { value: 'pastel', label: 'pastel', hasOffset: true },
  { value: 'gray', label: 'grayscale', hasOffset: false },
];

const SEGMENTATION_MODELS = [
  'bact_phase_omni',
  'bact_fluor_omni',
  'worm_omni',
  'worm_bact_omni',
  'worm_high_res_omni',
  'cyto2_omni',
  'bact_phase_cp',
  'bact_fluor_cp',
  'plant_cp',
  'worm_cp',
  'plant_omni',
  'bact_phase_affinity',
];
let segmentationModel = typeof CONFIG.model === 'string'
  ? CONFIG.model
  : DEFAULT_SEGMENTATION_MODEL;
let customSegmentationModelPath = null;
let flowThreshold = clamp(
  typeof CONFIG.flowThreshold === 'number' ? CONFIG.flowThreshold : 0,
  FLOW_THRESHOLD_MIN,
  FLOW_THRESHOLD_MAX,
);
let clusterEnabled = typeof CONFIG.cluster === 'boolean' ? CONFIG.cluster : true;
let affinitySegEnabled = typeof CONFIG.affinitySeg === 'boolean' ? CONFIG.affinitySeg : true;
let segMode = clusterEnabled ? (affinitySegEnabled ? 'none' : 'cluster') : (affinitySegEnabled ? 'affinity' : 'none');
let userAdjustedScale = false;
let viewStateRestored = false;
const touchPointers = new Map();
window.addEventListener('blur', handleWindowBlur);
const brushTapHistory = { pen: 0, mouse: 0 };
const brushTapLastPos = { pen: null, mouse: null };
let brushTapUndoBaseline = null;
let pendingBrushDoubleTap = null;
const BRUSH_DOUBLE_TAP_MAX_INTERVAL_MOUSE_MS = 220;
const BRUSH_DOUBLE_TAP_MAX_INTERVAL_STYLUS_MS = 420;
const BRUSH_DOUBLE_TAP_MAX_POINTER_DISTANCE = 24;
window.OmniBrush = window.OmniBrush || {};
const brushApi = window.OmniBrush;
window.OmniPainting = window.OmniPainting || {};
const paintingApi = window.OmniPainting;
if (typeof brushApi.init === 'function') {
  try {
    brushApi.init({
      getBrushDiameter: () => brushDiameter,
      getBrushKernelMode: () => brushKernelMode,
      modes: BRUSH_KERNEL_MODES,
      getImageDimensions: () => ({ width: imgWidth, height: imgHeight }),
      previewCanvas,
      previewCtx,
      canvas,
      viewer,
      applyViewTransform,
      getViewState: () => viewState,
      getDpr: () => dpr,
      getTool: () => tool,
      previewToolTypes: PREVIEW_TOOL_TYPES,
      crosshairToolTypes: CROSSHAIR_TOOL_TYPES,
      isCrosshairEnabled: () => BRUSH_CROSSHAIR_ENABLED,
      isDebugTouchOverlay: () => debugTouchOverlay,
      getTouchPointers: () => touchPointers,
    });
  } catch (err) {
    console.warn('OmniBrush init failed', err);
  }
}
const paintingInitOptions = {
  maskValues,
  outlineState,
  viewState,
  getImageDimensions: () => ({ width: imgWidth, height: imgHeight }),
  getCurrentLabel: () => currentLabel,
  setCurrentLabel: (value) => {
    currentLabel = value;
    if (value > currentMaxLabel) {
      currentMaxLabel = value;
      shufflePermutation = null;
      paletteTextureDirty = true;
      clearColorCaches();
    }
    updateMaskLabel();
  },
  hasNColor,
  isNColorActive: () => nColorActive,
  getMaskHasNonZero: () => maskHasNonZero,
  setMaskHasNonZero: (value) => {
    maskHasNonZero = Boolean(value);
  },
  markMaskIndicesDirty,
  markMaskTextureFullDirty,
  markOutlineTextureFullDirty,
  markOutlineIndicesDirty,
  updateAffinityGraphForIndices,
  rebuildLocalAffinityGraph,
  markAffinityGeometryDirty,
  getAffinityGraphInfo: () => affinityGraphInfo,
  getAffinitySteps: () => affinitySteps,
  hasLiveAffinityOverlay: () => Boolean(webglOverlay && webglOverlay.enabled && LIVE_AFFINITY_OVERLAY_UPDATES),
  isWebglPipelineActive,
  clearColorCaches,
  requestPaintFrame: () => requestPaintFrame(),
  scheduleStateSave: () => scheduleStateSave(),
  pushHistory,
  log,
  draw,
  scheduleDraw: (options) => scheduleDraw(options),
  redrawMaskCanvas,
  markNeedsMaskRedraw: () => {
    needsMaskRedraw = true;
  },
  applySegmentationMask,
  getPendingSegmentationPayload: () => pendingSegmentationPayload,
  setPendingSegmentationPayload: (value) => {
    pendingSegmentationPayload = value;
  },
  getPendingMaskRebuild: () => pendingMaskRebuild,
  setPendingMaskRebuild: (value) => {
    pendingMaskRebuild = Boolean(value);
  },
  getSegmentationTimer: () => segmentationUpdateTimer,
  setSegmentationTimer: (value) => {
    segmentationUpdateTimer = value;
  },
  canRebuildMask: () => canRebuildMask,
  triggerMaskRebuild,
  applyMaskRedrawImmediate,
  collectBrushIndices: (target, x, y) => {
    if (typeof brushApi.collectBrushIndices === 'function') {
      brushApi.collectBrushIndices(target, x, y);
    }
  },
  enqueueAffinityIndexBatch: (buffer, length) => enqueueAffinityIndexBatch(buffer, length),
  onMaskBufferReplaced: (next) => {
    maskValues = next;
    if (!next || !next.length) {
      return;
    }
    if (affinityGraphSource === 'remote') {
      return;
    }
    if (!maskHasNonZero) {
      return;
    }
    const lastBuild = typeof window !== 'undefined' && window.__OMNI_DEBUG__
      ? window.__OMNI_DEBUG__.lastLocalAffinityBuild
      : null;
    const localGraphEmpty = Boolean(lastBuild && lastBuild.hasValues && lastBuild.nonZero === false);
    const missingGraph = !affinityGraphInfo || !affinityGraphInfo.values || !affinityGraphInfo.stepCount;
    if (!localGraphEmpty && !missingGraph) {
      return;
    }
    affinityGraphNeedsLocalRebuild = true;
    if (showAffinityGraph || affinitySegEnabled || outlinesVisible) {
      try {
        rebuildLocalAffinityGraph();
      } catch (err) {
        log('affinity rebuild after mask buffer replace failed', err);
      }
    }
  },
  debugFillPerformance: () => DEBUG_FILL_PERF,
  enableMaskPipelineV2: ENABLE_MASK_PIPELINE_V2,
};
let paintingInitApplied = false;
let debugGridBootstrapped = false;

function logDebugGridStatus(reason) {
  if (!window.__OMNI_DEBUG__ || !window.__OMNI_DEBUG__.gridLogs) {
    return;
  }
  if (!paintingApi || typeof paintingApi.__debugGetState !== 'function') {
    return;
  }
  try {
    const state = paintingApi.__debugGetState();
    const ctx = state && state.ctx;
    if (!ctx || !ctx.maskValues) {
      console.log('[debug-grid]', reason, 'no ctx/mask');
      return;
    }
    const mask = ctx.maskValues;
    let nonZero = 0;
    for (let i = 0; i < mask.length; i += 1) {
      if ((mask[i] | 0) > 0) {
        nonZero += 1;
      }
    }
    console.log('[debug-grid]', reason, {
      total: mask.length,
      nonZero,
      trackerReady: Boolean(state.componentTracker && state.componentTracker.components),
      affinityInfo: typeof window.__OMNI_DEBUG__ === 'object' && window.__OMNI_DEBUG__
        && typeof window.__OMNI_DEBUG__.getAffinityInfo === 'function'
        ? window.__OMNI_DEBUG__.getAffinityInfo()
        : null,
    });
  } catch (err) {
    console.log('[debug-grid]', reason, 'error', err);
  }
}

function ensureDebugGridBootstrap() {
  if (!window.__OMNI_FORCE_GRID_MASK__ || debugGridBootstrapped) {
    return;
  }
  if (!paintingApi || typeof paintingApi.__debugGetState !== 'function') {
    return;
  }
  const paintingState = paintingApi.__debugGetState();
  if (!paintingState || !paintingState.ctx) {
    return;
  }
  const ctx = paintingState.ctx;
  const execute = () => {
    if (paintingState.componentTracker && typeof paintingState.componentTracker.rebuild === 'function') {
      try {
        paintingState.componentTracker.rebuild(ctx);
      } catch (err) {
        log('debug grid tracker bootstrap failed', err);
      }
    }
    try {
      rebuildLocalAffinityGraph();
    } catch (err) {
      log('debug grid affinity bootstrap failed', err);
    }
    if (typeof markAffinityGeometryDirty === 'function') {
      markAffinityGeometryDirty();
    }
    markMaskTextureFullDirty();
    markOutlineTextureFullDirty();
    needsMaskRedraw = true;
    if (typeof ctx.scheduleStateSave === 'function') {
      try {
        ctx.scheduleStateSave();
      } catch (err) {
        log('debug grid bootstrap scheduleStateSave failed', err);
      }
    }
    try {
      requestPaintFrame();
    } catch (err) {
      if (err && /nColorActive/.test(String(err))) {
        if (typeof setTimeout === 'function') {
          setTimeout(() => {
            try { requestPaintFrame(); } catch (_) { /* ignore */ }
          }, 0);
        }
      } else {
        throw err;
      }
    }
    if (window.__OMNI_DEBUG__) {
      window.__OMNI_DEBUG__.gridBootstrapCompleted = true;
    }
  };
  debugGridBootstrapped = true;
  if (typeof setTimeout === 'function') {
    setTimeout(execute, 0);
    setTimeout(() => logDebugGridStatus('post-bootstrap'), 1000);
  } else if (typeof requestAnimationFrame === 'function') {
    requestAnimationFrame(() => {
      execute();
      logDebugGridStatus('post-bootstrap-ra');
    });
  } else {
    execute();
    logDebugGridStatus('post-bootstrap-sync');
  }
}
function applyPaintingInit() {
  if (paintingInitApplied) {
    return true;
  }
  if (typeof paintingApi.init !== 'function') {
    return false;
  }
  try {
    paintingApi.init(paintingInitOptions);
    paintingInitApplied = true;
    if (typeof setTimeout === 'function') {
      setTimeout(() => ensureDebugGridBootstrap(), 0);
    } else {
      ensureDebugGridBootstrap();
    }
  } catch (err) {
    console.warn('OmniPainting init failed', err);
  }
  return paintingInitApplied;
}
if (!applyPaintingInit()) {
  Object.defineProperty(paintingApi, 'init', {
    configurable: true,
    enumerable: true,
    get: () => undefined,
    set(fn) {
      Object.defineProperty(paintingApi, 'init', {
        configurable: true,
        enumerable: true,
        writable: true,
        value: fn,
      });
      applyPaintingInit();
    },
  });
}

if (typeof window !== 'undefined') {
  window.__OMNI_DEBUG__ = window.__OMNI_DEBUG__ || {};
  window.__OMNI_DEBUG__.walkGridFill = async function walkGridFill(options = {}) {
    const painting = window.OmniPainting;
    if (!painting || typeof painting.__debugGetState !== 'function') {
      throw new Error('OmniPainting debug state unavailable');
    }
    const state = painting.__debugGetState();
    const ctx = state.ctx;
    if (!ctx || !ctx.maskValues || typeof ctx.getImageDimensions !== 'function') {
      throw new Error('Painting context missing required helpers');
    }
    const dims = ctx.getImageDimensions();
    const width = dims.width | 0;
    const height = dims.height | 0;
    if (options.forceGrid && typeof painting.__debugApplyGridIfNeeded === 'function') {
      painting.__debugApplyGridIfNeeded(true);
      await new Promise((resolve) => setTimeout(resolve, options.settleDelay || 50));
    }
    const mask = ctx.maskValues;
    const resolvedTargetLabel = Number.isFinite(options.targetLabel)
      ? (options.targetLabel | 0)
      : ((window.__OMNI_DEBUG__ && typeof window.__OMNI_DEBUG__.gridBackgroundLabel === 'number')
        ? window.__OMNI_DEBUG__.gridBackgroundLabel | 0
        : 0);
    const targetLabel = resolvedTargetLabel;
    const paintLabel = Number.isFinite(options.paintLabel)
      ? (options.paintLabel | 0)
      : 1;
    if (targetLabel === paintLabel) {
      throw new Error('walkGridFill paintLabel matches targetLabel; choose a different paintLabel.');
    }
    const visited = new Set();
    const components = [];
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const idx = y * width + x;
        if ((mask[idx] | 0) !== targetLabel || visited.has(idx)) {
          continue;
        }
        const comp = [];
        const queue = [idx];
        visited.add(idx);
        while (queue.length) {
          const current = queue.pop();
          if ((mask[current] | 0) !== targetLabel) {
            continue;
          }
          comp.push(current);
          const cx = current % width;
          const cy = (current / width) | 0;
          const neighbors = [
            cy > 0 ? current - width : -1,
            cy < height - 1 ? current + width : -1,
            cx > 0 ? current - 1 : -1,
            cx < width - 1 ? current + 1 : -1,
          ];
          for (const n of neighbors) {
            if (n >= 0 && !visited.has(n)) {
              visited.add(n);
              queue.push(n);
            }
          }
        }
        if (comp.length) {
          components.push(comp);
        }
      }
    }
    const fastMode = Boolean(options.fast || options.instant || options.noDelay);
    const delay = fastMode
      ? 0
      : (Number.isFinite(options.delay) ? Math.max(0, options.delay) : 35);
    const useAnimationFrameDelay = !fastMode && delay <= 0 && typeof requestAnimationFrame === 'function';
    const documentHidden = typeof document !== 'undefined'
      && typeof document.visibilityState === 'string'
      && document.visibilityState === 'hidden';
    const waitForFrame = () => new Promise((resolve) => {
      if (typeof requestAnimationFrame === 'function') {
        let settled = false;
        const finish = () => {
          if (settled) {
            return;
          }
          settled = true;
          resolve();
        };
        requestAnimationFrame(() => finish());
        // Fallback in case RAF never fires (hidden tabs / throttled timers)
        setTimeout(finish, 32);
        return;
      }
      setTimeout(resolve, 0);
    });
    const waitForDelay = (ms) => new Promise((resolve) => setTimeout(resolve, Math.max(0, ms | 0)));
    const results = [];
    const wantsVisualFastUpdates = fastMode
      && !(options.visual === false || options.visualize === false);
    for (let tile = 0; tile < components.length; tile += 1) {
      const comp = components[tile];
      const sampleIdx = comp[Math.floor(comp.length / 2)] | 0;
      const sx = sampleIdx % width;
      const sy = (sampleIdx / width) | 0;
      const maskBefore = ctx.maskValues[sampleIdx] | 0;
      ctx.setCurrentLabel(paintLabel);
      painting.floodFill({ x: sx, y: sy });
      if (fastMode) {
        if (wantsVisualFastUpdates && !documentHidden) {
          await waitForFrame();
        }
      } else if (useAnimationFrameDelay) {
        await waitForFrame();
      } else if (delay > 0) {
        await waitForDelay(delay);
      } else {
        await waitForFrame();
      }
      let failures = 0;
      for (let i = 0; i < comp.length; i += 1) {
        const idx = comp[i] | 0;
        if ((ctx.maskValues[idx] | 0) !== paintLabel) {
          failures += 1;
        }
      }
      const lastResult = painting.__debugGetState().lastFillResult;
      const entry = {
        tile,
        componentSize: comp.length,
        sample: { x: sx, y: sy, idx: sampleIdx },
        failures,
        lastFill: lastResult ? Object.assign({}, lastResult) : null,
        finalizeCount: state && typeof state.finalizeCallCount === 'number'
          ? state.finalizeCallCount
          : null,
        maskBefore,
      };
      console.log('[debug-walk-grid]', entry);
      results.push(entry);
      if (failures > 0) {
        break;
      }
    }
    return results;
  };
}
window.OmniInteractions = window.OmniInteractions || {};
const interactionsApi = window.OmniInteractions;
if (!interactionsApi || typeof interactionsApi.init !== 'function') {
  console.warn('OmniInteractions helpers missing; ensure /static/js/interactions.js loads before app.js');
}
let pinchState = null;
let panPointerId = null;
let activePointerId = null;
let gestureState = null;
let wheelRotationBuffer = 0;

// Interaction dot overlay removed; dot is now the cursor itself

const MIN_GAMMA = 0.1;
const MAX_GAMMA = 6.0;
const DEFAULT_GAMMA = 1.0;
const HIST_HANDLE_THRESHOLD = 8;
let autoFitPending = true;

function resizePreviewCanvas() {
  if (typeof brushApi.resizePreviewCanvas === 'function') {
    brushApi.resizePreviewCanvas();
  }
}

function drawBrushPreview(point, options = {}) {
  if (typeof brushApi.drawBrushPreview === 'function') {
    brushApi.drawBrushPreview(point, options);
  }
}

function getBrushKernelCenter(x, y) {
  if (typeof brushApi.getBrushKernelCenter === 'function') {
    return brushApi.getBrushKernelCenter(x, y);
  }
  return { x, y };
}

function enumerateBrushPixels(centerX, centerY) {
  if (typeof brushApi.enumerateBrushPixels === 'function') {
    return brushApi.enumerateBrushPixels(centerX, centerY);
  }
  return [];
}

function collectBrushIndices(target, centerX, centerY) {
  if (typeof brushApi.collectBrushIndices === 'function') {
    brushApi.collectBrushIndices(target, centerX, centerY);
  }
}

function scheduleGestureUpdate() {
  if (interactionsApi && typeof interactionsApi.scheduleGestureUpdate === 'function') {
    interactionsApi.scheduleGestureUpdate();
  }
}

function applyPendingGestureUpdate() {
  if (interactionsApi && typeof interactionsApi.applyPendingGestureUpdate === 'function') {
    interactionsApi.applyPendingGestureUpdate();
  }
}

function renderHoverPreview() {
  if (interactionsApi && typeof interactionsApi.renderHoverPreview === 'function') {
    interactionsApi.renderHoverPreview();
    return;
  }
  const point = getHoverPoint();
  if (!point) {
    drawBrushPreview(null);
    return;
  }
  if (PREVIEW_TOOL_TYPES.has(tool)) {
    drawBrushPreview(point);
    return;
  }
  if (CROSSHAIR_TOOL_TYPES.has(tool) && cursorInsideImage) {
    drawBrushPreview(point, { crosshairOnly: true });
    return;
  }
  drawBrushPreview(null);
}

function getHoverPoint() {
  if (interactionsApi && typeof interactionsApi.getHoverPoint === 'function') {
    return interactionsApi.getHoverPoint();
  }
  return null;
}

function getHoverScreenPoint() {
  if (interactionsApi && typeof interactionsApi.getHoverScreenPoint === 'function') {
    return interactionsApi.getHoverScreenPoint();
  }
  return null;
}

function setHoverState(worldPoint, screenPoint = null) {
  if (interactionsApi && typeof interactionsApi.setHoverState === 'function') {
    interactionsApi.setHoverState(worldPoint, screenPoint);
  }
}

function clearHoverState() {
  if (interactionsApi && typeof interactionsApi.clearHoverState === 'function') {
    interactionsApi.clearHoverState();
  }
}

function queuePanPointer(point) {
  if (interactionsApi && typeof interactionsApi.queuePanPointer === 'function') {
    interactionsApi.queuePanPointer(point);
  }
}

function queueHoverUpdate(point, hasPreview = false) {
  if (interactionsApi && typeof interactionsApi.queueHoverUpdate === 'function') {
    interactionsApi.queueHoverUpdate(point, { hasPreview });
  }
}

function clearInteractionPending() {
  if (interactionsApi && typeof interactionsApi.clearPending === 'function') {
    interactionsApi.clearPending();
  }
}

if (interactionsApi && typeof interactionsApi.init === 'function') {
  interactionsApi.init({
    pointerState,
    getTool: () => tool,
    getPreviewToolTypes: () => PREVIEW_TOOL_TYPES,
    getCrosshairToolTypes: () => CROSSHAIR_TOOL_TYPES,
    isCursorInsideImage: () => cursorInsideImage,
    drawBrushPreview: (point, options) => drawBrushPreview(point, options),
    updateHoverInfo: (point) => updateHoverInfo(point),
    draw: () => draw(),
    scheduleDraw: (options) => scheduleDraw(options),
    screenToImage: (point) => screenToImage(point),
    setCursorHold: (style) => setCursorHold(style),
    cursorStyles: { dot: dotCursorCss },
    normalizeAngle: (value) => normalizeAngle(value),
    setOffsetForImagePoint: (imagePoint, origin) => setOffsetForImagePoint(imagePoint, origin),
    getViewState: () => viewState,
    setViewStateScale: (value) => {
      viewState.scale = value;
    },
    setViewStateRotation: (value) => {
      viewState.rotation = value;
    },
    applyPanDelta: (dx, dy) => {
      viewState.offsetX += dx;
      viewState.offsetY += dy;
    },
    markViewStateDirty: () => {
      viewStateDirty = true;
    },
    setUserAdjustedScale: (value) => {
      userAdjustedScale = Boolean(value);
    },
    setAutoFitPending: (value) => {
      autoFitPending = Boolean(value);
    },
    processPaintQueue: () => (typeof paintingApi.processPaintQueue === 'function'
      ? paintingApi.processPaintQueue()
      : null),
    isPainting: () => isPainting,
    isPanning: () => isPanning,
    getLastPointer: () => ({ x: lastPoint.x, y: lastPoint.y }),
    setLastPointer: (value) => {
      if (value && Number.isFinite(value.x) && Number.isFinite(value.y)) {
        lastPoint = { x: value.x, y: value.y };
      }
    },
    getGestureState: () => gestureState,
    setGestureState: (value) => {
      gestureState = value;
    },
    requestAnimationFrame: (callback) => requestAnimationFrame(callback),
  });
}

function schedulePointerFrame() {
  if (interactionsApi && typeof interactionsApi.schedulePointerFrame === 'function') {
    interactionsApi.schedulePointerFrame();
  }
}

function queuePaintPoint(world) {
  if (typeof paintingApi.queuePaintPoint === 'function') {
    paintingApi.queuePaintPoint(world);
  }
}

function updateCursor() {
  if (cursorOverride) {
    canvas.style.cursor = cursorOverride;
    return;
  }
  if (!cursorInsideCanvas) {
    canvas.style.cursor = 'default';
    return;
  }
  // Prefer dot cursor during any interaction
  if (isPanning || spacePan || gestureState) {
    canvas.style.cursor = dotCursorCss;
  } else if (CROSSHAIR_TOOL_TYPES.has(tool) && cursorInsideImage) {
    canvas.style.cursor = 'none';
  } else {
    canvas.style.cursor = 'default';
  }
}

function updateMaskLabel() {
  if (maskLabel) {
    const prefix = nColorActive ? 'Mask Group' : 'Mask Label';
    maskLabel.textContent = prefix + ': ' + currentLabel;
  }
  updateLabelControls();
}

function updateHistoryButtons() {
  if (undoButton) {
    const count = (typeof OmniHistory.getUndoCount === 'function')
      ? OmniHistory.getUndoCount()
      : 0;
    undoButton.disabled = count === 0;
  }
  if (redoButton) {
    const count = (typeof OmniHistory.getRedoCount === 'function')
      ? OmniHistory.getRedoCount()
      : 0;
    redoButton.disabled = count === 0;
  }
}

function updateLabelControls() {
  if (!labelValueInput) {
    return;
  }
  labelValueInput.value = String(currentLabel);
  if (currentLabel > 0) {
    let rgb = null;
    if (typeof nColorActive !== 'undefined' && typeof rawColorMap !== 'undefined') {
      rgb = nColorActive
        ? getNColorLabelColor(currentLabel)
        : getRawLabelColor(currentLabel);
    } else {
      rgb = hashColorForLabel(currentLabel, 0.0);
    }
    if (Array.isArray(rgb) && rgb.length >= 3) {
      const [r, g, b] = rgb;
      const color = 'rgb(' + (r | 0) + ', ' + (g | 0) + ', ' + (b | 0) + ')';
      const textColor = readableTextColor([r, g, b]);
      labelValueInput.style.setProperty('background', color);
      labelValueInput.style.setProperty('background-clip', 'padding-box');
      labelValueInput.style.setProperty('border-color', 'var(--slider-track-border)');
      labelValueInput.style.setProperty('color', textColor);
      labelValueInput.style.setProperty('caret-color', textColor);
      updateAccentColorsFromRgb(rgb);
      return;
    }
  }
  labelValueInput.style.removeProperty('background');
  labelValueInput.style.removeProperty('background-clip');
  labelValueInput.style.removeProperty('border-color');
  labelValueInput.style.removeProperty('color');
  labelValueInput.style.removeProperty('caret-color');
  resetAccentColors();
}

function updateMaskVisibilityLabel() {
  if (!maskVisibility) {
    return;
  }
  maskVisibility.textContent = 'Mask Layer: ' + (maskVisible ? 'On' : 'Off') + " (toggle with 'M')";
}

function updateMaskOpacityLabel() {
}

function updateToolInfo() {
  if (!toolInfo) {
    return;
  }
  const mode = getActiveToolMode();
  let description = 'Brush (B)';
  if (mode === 'erase') {
    description = 'Erase (E)';
  } else if (mode === 'fill') {
    description = 'Fill (G)';
  } else if (mode === 'picker') {
    description = 'Eyedropper (I)';
  }
  toolInfo.textContent = 'Tool: ' + description;
}

function applyCurrentLabel(nextLabel, { scheduleSave = true } = {}) {
  const parsed = Number(nextLabel);
  if (!Number.isFinite(parsed)) {
    updateLabelControls();
    return;
  }
  const normalized = Math.max(0, Math.floor(parsed));
  if (normalized === currentLabel) {
    updateLabelControls();
    return;
  }
  if (eraseActive && normalized !== 0) {
    stopEraseOverride();
  }
  currentLabel = normalized;
  // Update max label tracking if selecting a higher label
  if (normalized > currentMaxLabel) {
    currentMaxLabel = normalized;
    shufflePermutation = null;
    paletteTextureDirty = true;
    clearColorCaches();
  }
  updateMaskLabel();
  if (scheduleSave) {
    scheduleStateSave();
  }
}

function updateImageInfo() {
  if (!imageInfo) {
    return;
  }
  if (currentImageName) {
    if (directoryPath) {
      imageInfo.textContent = currentImageName + ' — ' + directoryPath;
    } else {
      imageInfo.textContent = currentImageName;
    }
  } else {
    imageInfo.textContent = 'Sample Image';
  }
}

function startEraseOverride() {
  if (eraseActive) {
    return;
  }
  eraseActive = true;
  erasePreviousLabel = currentLabel;
  setTool('brush');
  currentLabel = 0;
  updateMaskLabel();
  updateToolButtons();
}

function stopEraseOverride() {
  if (!eraseActive) {
    return;
  }
  if (erasePreviousLabel !== null) {
    currentLabel = erasePreviousLabel;
  }
  eraseActive = false;
  erasePreviousLabel = null;
  updateMaskLabel();
  updateToolButtons();
}

function setTool(nextTool) {
  if (tool === nextTool) {
    updateToolButtons();
    return;
  }
  tool = nextTool;
  updateToolInfo();
  updateCursor();
  updateToolButtons();
}

function getActiveToolMode() {
  if (eraseActive) {
    return 'erase';
  }
  if (tool === 'fill') {
    return 'fill';
  }
  if (tool === 'picker') {
    return 'picker';
  }
  return 'draw';
}


function updateToolOptions() {
  if (!toolOptionsBlocks.length) {
    return;
  }
  const mode = getActiveToolMode();
  toolOptionsBlocks.forEach((block) => {
    const tools = (block.getAttribute('data-tool') || '')
      .split(',')
      .map((value) => value.trim())
      .filter(Boolean);
    const active = tools.includes(mode);
    block.style.display = active ? 'flex' : 'none';
  });
}

function updateToolButtons() {
  if (!toolStopButtons || !toolStopButtons.length) {
    return;
  }
  const mode = getActiveToolMode();
  toolStopButtons.forEach((btn) => {
    const btnMode = btn.getAttribute('data-mode');
    const isActive = btnMode === mode;
    if (isActive) {
      btn.setAttribute('data-active', 'true');
      btn.setAttribute('aria-pressed', 'true');
    } else {
      btn.removeAttribute('data-active');
      btn.setAttribute('aria-pressed', 'false');
    }
  });
  updateToolOptions();
}

function selectToolMode(mode) {
  switch (mode) {
    case 'erase':
      startEraseOverride();
      break;
    case 'fill':
      if (eraseActive) stopEraseOverride();
      setTool('fill');
      break;
    case 'picker':
      if (eraseActive) stopEraseOverride();
      setTool('picker');
      break;
    case 'draw':
    default:
      if (eraseActive) stopEraseOverride();
      setTool('brush');
      break;
  }
  updateToolButtons();
}

function updateBrushControls() {
  if (brushSizeSlider) {
    brushSizeSlider.value = String(brushDiameter);
    updateNativeRangeFill(brushSizeSlider);
    refreshSlider('brushSizeSlider');
  }
  if (brushSizeInput) {
    brushSizeInput.value = String(brushDiameter);
  }
}

function setBrushDiameter(nextDiameter, fromUser = false) {
  const clamped = clampBrushDiameter(nextDiameter);
  if (brushDiameter === clamped) {
    if (fromUser) {
      setTool('brush');
    }
    return;
  }
  brushDiameter = clamped;
  updateBrushControls();
  log('brush size set to ' + brushDiameter);
  if (fromUser) {
    selectToolMode('draw');
    scheduleStateSave();
  }
  refreshSlider('brushSizeSlider');
  drawBrushPreview(getHoverPoint());
}

function syncMaskOpacityControls() {
  if (maskOpacitySlider) {
    maskOpacitySlider.value = maskOpacity.toFixed(2);
    updateNativeRangeFill(maskOpacitySlider);
    refreshSlider('maskOpacity');
  }
  if (maskOpacityInput && document.activeElement !== maskOpacityInput) {
    maskOpacityInput.value = maskOpacity.toFixed(2);
  }
  updateMaskOpacityLabel();
}

function setMaskOpacity(value, { silent = false } = {}) {
  const numeric = typeof value === 'number' ? value : parseFloat(value);
  if (Number.isNaN(numeric)) {
    syncMaskOpacityControls();
    return;
  }
  const clamped = clamp(numeric, 0, 1);
  if (maskOpacity === clamped && !silent) {
    syncMaskOpacityControls();
    return;
  }
  maskOpacity = clamped;
  syncMaskOpacityControls();
  if (!silent) {
    if (!isWebglPipelineActive()) {
      redrawMaskCanvas();
    }
    scheduleDraw();
    scheduleStateSave();
  }
}

function updateMaskThresholdLabel() {
}

function syncMaskThresholdControls() {
  if (maskThresholdSlider) {
    maskThresholdSlider.value = maskThreshold.toFixed(1);
    updateNativeRangeFill(maskThresholdSlider);
    refreshSlider('maskThreshold');
  }
  if (maskThresholdInput && document.activeElement !== maskThresholdInput) {
    maskThresholdInput.value = maskThreshold.toFixed(1);
  }
  updateMaskThresholdLabel();
}

function setMaskThreshold(value, { silent = false } = {}) {
  const numeric = typeof value === 'number' ? value : parseFloat(value);
  if (Number.isNaN(numeric)) {
    syncMaskThresholdControls();
    return;
  }
  const clamped = clamp(numeric, MASK_THRESHOLD_MIN, MASK_THRESHOLD_MAX);
  if (maskThreshold === clamped && !silent) {
    syncMaskThresholdControls();
    return;
  }
  maskThreshold = clamped;
  syncMaskThresholdControls();
  if (!silent) {
    scheduleMaskRebuild({ interactive: true });
    scheduleStateSave();
  }
}

function updateFlowThresholdLabel() {
}


function syncNiterControls() {
  const isAuto = Boolean(niterAuto);
  if (niterSlider) {
    if (niterSlider.min !== String(NITER_MIN)) {
      niterSlider.min = '-1';
    }
    niterSlider.value = String(isAuto ? -1 : (niter | 0));
    updateNativeRangeFill(niterSlider);
    refreshSlider('niter');
  }
  if (niterInput) {
    if (niterInput.min !== String(NITER_MIN)) {
      niterInput.min = '-1';
    }
    if (isAuto) {
      niterInput.value = 'auto';
    } else {
      niterInput.value = String(niter | 0);
    }
  }
}

function setNiter(value, { silent = false } = {}) {
  if (value === null || value === undefined) {
    niterAuto = true;
    syncNiterControls();
    if (!silent) {
      scheduleMaskRebuild({ interactive: true });
      scheduleStateSave();
    }
    return;
  }
  const raw = typeof value === 'string' ? value.trim().toLowerCase() : value;
  if (raw === 'auto' || raw === 'none') {
    niterAuto = true;
    syncNiterControls();
    if (!silent) {
      scheduleMaskRebuild({ interactive: true });
      scheduleStateSave();
    }
    return;
  }
  const numeric = typeof value === 'number' ? value : parseInt(value, 10);
  if (Number.isNaN(numeric)) {
    syncNiterControls();
    return;
  }
  if (numeric <= NITER_MIN) {
    niterAuto = true;
  } else {
    niterAuto = false;
    niter = clamp(numeric, 0, NITER_MAX);
  }
  syncNiterControls();
  if (!silent) {
    scheduleMaskRebuild({ interactive: true });
    scheduleStateSave();
  }
}

function syncFlowThresholdControls() {
  if (flowThresholdSlider) {
    flowThresholdSlider.value = flowThreshold.toFixed(1);
    updateNativeRangeFill(flowThresholdSlider);
    refreshSlider('flowThreshold');
  }
  if (flowThresholdInput && document.activeElement !== flowThresholdInput) {
    flowThresholdInput.value = flowThreshold <= 0 ? 'off' : flowThreshold.toFixed(1);
  }
  updateFlowThresholdLabel();
}

function setFlowThreshold(value, { silent = false } = {}) {
  const numeric = typeof value === 'number' ? value : parseFloat(value);
  if (Number.isNaN(numeric)) {
    syncFlowThresholdControls();
    return;
  }
  const clamped = clamp(numeric, FLOW_THRESHOLD_MIN, FLOW_THRESHOLD_MAX);
  if (flowThreshold === clamped && !silent) {
    syncFlowThresholdControls();
    return;
  }
  flowThreshold = clamped;
  syncFlowThresholdControls();
  if (!silent) {
    scheduleMaskRebuild({ interactive: true });
    scheduleStateSave();
  }
}

function setClusterEnabled(value, { silent = false } = {}) {
  const next = Boolean(value);
  if (clusterEnabled === next && !silent) {
    return;
  }
  clusterEnabled = next;
  if (clusterToggle) {
    clusterToggle.checked = clusterEnabled;
  }
  if (!silent) {
    scheduleMaskRebuild();
    scheduleStateSave();
  }
}

function setAffinitySegEnabled(value, { silent = false } = {}) {
  const next = Boolean(value);
  if (affinitySegEnabled === next && !silent) {
    return;
  }
  affinitySegEnabled = next;
  if (affinityToggle) {
    affinityToggle.checked = affinitySegEnabled;
  }
  if (vectorOverlayToggle) {
    vectorOverlayToggle.disabled = !affinitySegEnabled || !vectorOverlayData;
    vectorOverlayToggle.checked = showVectorOverlay;
  }
  if (vectorOverlayRow) {
    vectorOverlayRow.style.display = affinitySegEnabled ? '' : 'none';
  }
  updateVectorOverlayVisibility();
  if (!silent) {
    scheduleMaskRebuild();
    scheduleStateSave();
  }
}


function setPanelCollapsed(panel, collapsed) {
  if (!panel) {
    return;
  }
  if (collapsed) {
    panel.classList.add('panel-collapsed');
  } else {
    panel.classList.remove('panel-collapsed');
  }
}

function setImageVisible(value, { silent = false } = {}) {
  const next = Boolean(value);
  if (imageVisible === next) {
    if (!silent) {
      draw();
    }
    return;
  }
  imageVisible = next;
  if (imageVisibilityToggle) {
    imageVisibilityToggle.checked = imageVisible;
  }
  setPanelCollapsed(intensityPanel, !imageVisible);
  if (!silent) {
    draw();
    scheduleStateSave();
  }
}

function pushHistory(indices, before, after) {
  if (typeof OmniHistory.push === 'function') {
    OmniHistory.push(indices, before, after);
  }
  updateHistoryButtons();
}

function applyHistoryEntry(entry, useAfter) {
  const values = useAfter ? entry.after : entry.before;
  const idxs = entry.indices;
  let sawPositive = maskHasNonZero;
  if (!idxs) {
    sawPositive = false;
    for (let i = 0; i < maskValues.length; i += 1) {
      const next = values[i] | 0;
      maskValues[i] = next;
      if (!sawPositive && next > 0) {
        sawPositive = true;
      }
    }
    maskHasNonZero = sawPositive;
    if (paintingApi && typeof paintingApi.rebuildComponents === 'function') {
      paintingApi.rebuildComponents();
    }
    return;
  }
  for (let i = 0; i < idxs.length; i += 1) {
    const idx = idxs[i];
    const next = values[i];
    maskValues[idx] = next;
    if (!sawPositive && (next | 0) > 0) {
      sawPositive = true;
    }
  }
  if (sawPositive) {
    maskHasNonZero = true;
  } else if (!useAfter && idxs.length === maskValues.length) {
    let sequential = true;
    for (let i = 0; i < idxs.length; i += 1) {
      if ((idxs[i] | 0) !== i) {
        sequential = false;
        break;
      }
    }
    if (sequential) {
      maskHasNonZero = false;
    }
  }
  if (paintingApi && typeof paintingApi.rebuildComponents === 'function') {
    paintingApi.rebuildComponents();
  }
}

function undo() {
  const entry = (typeof OmniHistory.undo === 'function') ? OmniHistory.undo() : null;
  if (!entry) {
    return;
  }
  applyHistoryEntry(entry, false);
  // For large operations (>35% of image), do a full affinity rebuild like fill does
  // Otherwise, use incremental update
  const totalPixels = maskValues ? maskValues.length : 0;
  const indicesCount = entry.indices ? entry.indices.length : totalPixels;
  const isLargeOperation = totalPixels > 0 && indicesCount >= totalPixels * 0.35;
  try {
    if (isLargeOperation || !entry.indices) {
      rebuildLocalAffinityGraph();
    } else {
      updateAffinityGraphForIndices(entry.indices);
    }
  } catch (_) {}
  clearColorCaches();
  if (isWebglPipelineActive()) {
    markMaskIndicesDirty(entry.indices);
    markOutlineIndicesDirty(entry.indices);
  } else {
    redrawMaskCanvas();
  }
  draw();
  scheduleStateSave();
  updateHistoryButtons();
}

function redo() {
  const entry = (typeof OmniHistory.redo === 'function') ? OmniHistory.redo() : null;
  if (!entry) {
    return;
  }
  applyHistoryEntry(entry, true);
  // For large operations (>35% of image), do a full affinity rebuild like fill does
  // Otherwise, use incremental update
  const totalPixels = maskValues ? maskValues.length : 0;
  const indicesCount = entry.indices ? entry.indices.length : totalPixels;
  const isLargeOperation = totalPixels > 0 && indicesCount >= totalPixels * 0.35;
  try {
    if (isLargeOperation || !entry.indices) {
      rebuildLocalAffinityGraph();
    } else {
      updateAffinityGraphForIndices(entry.indices);
    }
  } catch (_) {}
  clearColorCaches();
  if (isWebglPipelineActive()) {
    markMaskIndicesDirty(entry.indices);
    markOutlineIndicesDirty(entry.indices);
  } else {
    redrawMaskCanvas();
  }
  draw();
  scheduleStateSave();
  updateHistoryButtons();
}


function hasAnyMaskPixels() {
  if (maskHasNonZero) {
    return true;
  }
  for (let i = 0; i < maskValues.length; i += 1) {
    if ((maskValues[i] | 0) !== 0) {
      return true;
    }
  }
  return false;
}

const CLEAR_MASK_CONFIRM_MESSAGE = 'Clear all masks? This removes every mask label. You can undo (Cmd/Ctrl+Z) to restore.';

function performClearMasks({ recordHistory = true } = {}) {
  if (!maskValues || maskValues.length === 0) {
    return false;
  }
  stopInteraction(null);
  if (typeof paintingApi.cancelStroke === 'function') {
    paintingApi.cancelStroke();
  }
  const hadPixels = hasAnyMaskPixels();
  const shouldRecord = Boolean(recordHistory && hadPixels);
  if (shouldRecord) {
    const before = maskValues.slice();
    const after = new Uint32Array(before.length);
    pushHistory(null, before, after);
  }
  maskValues.fill(0);
  if (paintingApi && typeof paintingApi.rebuildComponents === 'function') {
    paintingApi.rebuildComponents();
  }
  maskHasNonZero = false;
  outlineState.fill(0);
  nColorValues = null;
  nColorInstanceMask = null;  // Clear instance mask to prevent ghost affinity graph
  nColorLabelToGroup.clear();
  clearColorCaches();
  clearAffinityGraphData();
  applyPointsPayload(null);
  if (window.__OMNI_FORCE_GRID_MASK__ && window.OmniPainting
    && typeof window.OmniPainting.__debugApplyGridIfNeeded === 'function') {
    try {
      window.OmniPainting.__debugApplyGridIfNeeded(true);
    } catch (err) {
      console.warn('debug grid reapply after clear masks failed', err);
    }
  }
  if (webglOverlay && webglOverlay.enabled) {
    webglOverlay.needsGeometryRebuild = true;
  }
  markAffinityGeometryDirty();
  if (isWebglPipelineActive()) {
    markMaskTextureFullDirty();
    markOutlineTextureFullDirty();
  } else {
    redrawMaskCanvas();
  }
  draw();
  drawBrushPreview(getHoverPoint());
  updateMaskLabel();
  updateColorModeLabel();
  log('Masks cleared');
  scheduleStateSave();
  updatePointsOverlayBuffers();
  return true;
}

async function promptClearMasks({ skipConfirm = false } = {}) {
  if (!skipConfirm && !hasAnyMaskPixels() && !nColorActive) {
    return false;
  }
  if (!skipConfirm) {
    const confirmed = await showConfirmDialog(CLEAR_MASK_CONFIRM_MESSAGE, { confirmText: 'Clear', cancelText: 'Cancel' });
    if (!confirmed) {
      return false;
    }
  }
  return performClearMasks({ recordHistory: true });
}


function updateKernelToggle() {
  if (!brushKernelToggle) {
    return;
  }
  brushKernelToggle.checked = brushKernelMode === BRUSH_KERNEL_MODES.SNAPPED;
}

function updateSegModeControls() {
  if (!segModeButtons || !segModeButtons.length) {
    return;
  }
  segModeButtons.forEach((btn) => {
    const mode = btn.getAttribute('data-seg-mode');
    const isActive = mode === segMode;
    if (isActive) {
      btn.setAttribute('data-active', 'true');
      btn.setAttribute('aria-pressed', 'true');
    } else {
      btn.removeAttribute('data-active');
      btn.setAttribute('aria-pressed', 'false');
    }
  });
}

function setSegMode(nextMode, { silent = false } = {}) {
  const normalized = nextMode === 'cluster' || nextMode === 'none' ? nextMode : 'affinity';
  if (segMode === normalized) {
    updateSegModeControls();
    return;
  }
  segMode = normalized;
  if (segMode === 'cluster') {
    setClusterEnabled(true, { silent: true });
    setAffinitySegEnabled(false, { silent: true });
  } else if (segMode === 'affinity') {
    setClusterEnabled(false, { silent: true });
    setAffinitySegEnabled(true, { silent: true });
  } else {
    setClusterEnabled(false, { silent: true });
    setAffinitySegEnabled(false, { silent: true });
  }
  updateSegModeControls();
  if (segMode !== 'affinity') {
    affinityGraphSource = 'local';
    affinityGraphNeedsLocalRebuild = true;
    rebuildLocalAffinityGraph();
  }
  if (!silent) {
    scheduleMaskRebuild();
    scheduleStateSave();
  }
}

function updateMaskStyleControls() {
  if (!maskStyleButtons || !maskStyleButtons.length) {
    return;
  }
  maskStyleButtons.forEach((btn) => {
    const mode = btn.getAttribute('data-mask-style');
    const isActive = mode === maskDisplayMode;
    if (isActive) {
      btn.setAttribute('data-active', 'true');
      btn.setAttribute('aria-pressed', 'true');
    } else {
      btn.removeAttribute('data-active');
      btn.setAttribute('aria-pressed', 'false');
    }
  });
}

function setMaskDisplayMode(nextMode, { silent = false } = {}) {
  const normalized = normalizeMaskDisplayMode(nextMode);
  if (maskDisplayMode === normalized) {
    updateMaskStyleControls();
    return;
  }
  maskDisplayMode = normalized;
  if (maskDisplayMode === MASK_DISPLAY_MODES.HIDDEN) {
    maskVisible = false;
    outlinesVisible = false;
    setPanelCollapsed(labelStylePanel, true);
  } else {
    maskVisible = true;
    outlinesVisible = maskDisplayMode === MASK_DISPLAY_MODES.OUTLINED || maskDisplayMode === MASK_DISPLAY_MODES.OUTLINE;
    setPanelCollapsed(labelStylePanel, false);
  }
  if (outlinesVisible) {
    if (affinityGraphInfo && affinityGraphInfo.values) {
      rebuildOutlineFromAffinity();
    } else {
      scheduleAffinityRebuildIfStale('mask-style');
    }
  }
  updateMaskStyleControls();
  if (isWebglPipelineActive()) {
    markMaskTextureFullDirty();
    markOutlineTextureFullDirty();
  } else {
    redrawMaskCanvas();
  }
  draw();
  if (!silent) {
    scheduleStateSave();
  }
}

function setBrushKernelMode(nextMode) {
  const normalized = nextMode === BRUSH_KERNEL_MODES.SNAPPED
    ? BRUSH_KERNEL_MODES.SNAPPED
    : BRUSH_KERNEL_MODES.SMOOTH;
  if (brushKernelMode === normalized) {
    return;
  }
  brushKernelMode = normalized;
  if (brushKernelModeSelect) {
    brushKernelModeSelect.value = brushKernelMode;
    refreshDropdown('brushKernelMode');
  }
  updateKernelToggle();
  log('brush kernel mode set to ' + brushKernelMode);
  drawBrushPreview(getHoverPoint());
}

function floodFill(point) {
  if (typeof paintingApi.floodFill === 'function') {
    paintingApi.floodFill(point);
  }
}

function pickColor(point) {
  if (typeof paintingApi.pickColor === 'function') {
    paintingApi.pickColor(point);
  }
}

function labelAtPoint(point) {
  if (typeof paintingApi.labelAtPoint === 'function') {
    return paintingApi.labelAtPoint(point);
  }
  return 0;
}

function redrawMaskCanvas() {
  if (isWebglPipelineActive()) {
    markMaskTextureFullDirty();
    markOutlineTextureFullDirty();
    return;
  }
  const data = maskData.data;
  for (let i = 0; i < maskValues.length; i += 1) {
    const label = maskValues[i];
    const p = i * 4;
    if (label <= 0) {
      data[p] = 0;
      data[p + 1] = 0;
      data[p + 2] = 0;
      data[p + 3] = 0;
      continue;
    }
    const rgb = getDisplayColor(i);
    if (!rgb) {
      data[p] = 0;
      data[p + 1] = 0;
      data[p + 2] = 0;
      data[p + 3] = 0;
      continue;
    }
    const maxAlpha = Math.round(Math.max(0, Math.min(1, maskOpacity)) * 255);
    let alpha = maxAlpha;
    if (maskDisplayMode === MASK_DISPLAY_MODES.OUTLINE) {
      alpha = outlineState[i] ? maxAlpha : 0;
    } else if (maskDisplayMode === MASK_DISPLAY_MODES.OUTLINED) {
      alpha = outlineState[i] ? maxAlpha : Math.max(0, Math.min(255, (maxAlpha >> 1)));
    }
    data[p] = rgb[0];
    data[p + 1] = rgb[1];
    data[p + 2] = rgb[2];
    data[p + 3] = alpha;
  }
  maskCtx.putImageData(maskData, 0, 0);
}

function decodeBase64ToUint8(encoded) {
  if (!encoded) {
    return new Uint8Array(0);
  }
  const binary = atob(encoded);
  const len = binary.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

function base64FromUint32(array) {
  if (!array || array.length === 0) {
    return '';
  }
  const bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
  return base64FromUint8(bytes);
}

function uint32FromBase64(encoded, expectedLength = null) {
  const bytes = decodeBase64ToUint8(encoded);
  if (bytes.length % 4 !== 0) {
    throw new Error('uint32FromBase64 length mismatch');
  }
  const view = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  if (typeof expectedLength === 'number' && expectedLength > 0 && view.length !== expectedLength) {
    const copy = new Uint32Array(expectedLength);
    copy.set(view.subarray(0, Math.min(view.length, expectedLength)));
    return copy;
  }
  return view;
}

function base64FromUint8Array(array) {
  if (!array || array.length === 0) {
    return '';
  }
  return base64FromUint8(array instanceof Uint8Array ? array : new Uint8Array(array.buffer));
}

function encodeHistoryField(array) {
  if (!array || array.length === 0) {
    return '';
  }
  const view = array instanceof Uint32Array ? array : new Uint32Array(array);
  return base64FromUint32(view);
}

function decodeHistoryField(value, expectedLength) {
  if (value == null || value === '') {
    if (typeof expectedLength === 'number' && expectedLength > 0) {
      return new Uint32Array(expectedLength);
    }
    return new Uint32Array(0);
  }
  if (value instanceof Uint32Array) {
    if (typeof expectedLength === 'number' && expectedLength > 0 && value.length !== expectedLength) {
      const copy = new Uint32Array(expectedLength);
      copy.set(value.subarray(0, Math.min(value.length, expectedLength)));
      return copy;
    }
    return value;
  }
  if (Array.isArray(value)) {
    return new Uint32Array(value);
  }
  if (typeof value === 'string') {
    return uint32FromBase64(value, expectedLength);
  }
  return new Uint32Array(0);
}

function serializeHistoryState() {
  if (typeof OmniHistory.serialize === 'function') {
    try {
      return OmniHistory.serialize(encodeHistoryField);
    } catch (err) {
      console.warn('serializeHistoryState failed', err);
    }
  }
  return { undo: [], redo: [] };
}

function restoreHistoryState(serialized, expectedLength) {
  if (typeof OmniHistory.restore === 'function') {
    try {
      OmniHistory.restore(serialized || {}, decodeHistoryField, expectedLength);
      updateHistoryButtons();
      return;
    } catch (err) {
      console.warn('restoreHistoryState failed', err);
    }
  }
  if (typeof OmniHistory.clear === 'function') {
    OmniHistory.clear();
  }
  updateHistoryButtons();
}

function collectViewerState() {
  const historyState = serializeHistoryState();
  return {
    width: imgWidth,
    height: imgHeight,
    mask: base64FromUint32(maskValues),
    maskHasNonZero: Boolean(maskHasNonZero),
    outline: base64FromUint8Array(outlineState),
    undoStack: historyState.undo,
    redoStack: historyState.redo,
    currentLabel,
    maskOpacity,
    maskThreshold,
    flowThreshold,
    niter: niterAuto ? null : niter,
    affinityGraph: (savedAffinityGraphPayload && savedAffinityGraphPayload.encoded)
      ? savedAffinityGraphPayload
      : ((affinityGraphSource !== 'none' && affinityGraphInfo && affinityGraphInfo.values && affinityGraphInfo.values.length)
        ? {
          width: affinityGraphInfo.width,
          height: affinityGraphInfo.height,
          steps: affinitySteps.map((p) => [p[0] | 0, p[1] | 0]),
          encoded: base64FromUint8(affinityGraphInfo.values),
        }
        : null),
    segmentationModel,
    customSegmentationModelPath,
    viewState: {
      scale: viewState.scale,
      offsetX: viewState.offsetX,
      offsetY: viewState.offsetY,
      rotation: viewState.rotation,
    },
    tool,
    brushDiameter,
    maskVisible,
    maskDisplayMode,
    nColorActive,
    nColorValues: nColorActive && nColorValues ? base64FromUint32(nColorValues) : null,
    nColorInstanceMask: nColorInstanceMask ? base64FromUint32(nColorInstanceMask) : null,
    nColorPaletteColors: nColorPaletteColors && nColorPaletteColors.length ? nColorPaletteColors : null,
    nColorHueOffset,
    nColorColormap,
    labelColormap,
    imageColormap,
    labelShuffle,
    labelShuffleSeed,
    currentMaxLabel,
    clusterEnabled,
    affinitySegEnabled,
    showFlowOverlay,
    showDistanceOverlay,
    showPointsOverlay,
    pointsPayload: pointsOverlayData || null,
    showAffinityGraph: Boolean(showAffinityGraph),
    segMode,
    useGpu: useGpuToggle ? Boolean(useGpuToggle.checked) : undefined,
    timestamp: Date.now(),
  };
}

function restoreViewerState(saved) {
  if (!saved || typeof saved !== 'object') {
    return;
  }
  isRestoringState = true;
  try {
    const expectedLength = maskValues.length;
    if (saved.mask) {
      const restoredMask = uint32FromBase64(saved.mask, expectedLength);
      maskValues.set(restoredMask.subarray(0, expectedLength));
      maskHasNonZero = Boolean(saved.maskHasNonZero);
      // Update max label for dynamic palette sizing
      updateMaxLabelFromMask();
    } else {
      maskValues.fill(0);
      maskHasNonZero = false;
    }
    if (window.OmniPainting && typeof window.OmniPainting.__debugApplyGridIfNeeded === 'function') {
      try {
        window.OmniPainting.__debugApplyGridIfNeeded(false);
      } catch (err) {
        console.warn('debug grid reapply after restore failed', err);
      }
    }
    if (saved.outline) {
      const outlineBytes = decodeBase64ToUint8(saved.outline);
      if (outlineBytes.length === outlineState.length) {
        outlineState.set(outlineBytes);
      } else {
        outlineState.fill(0);
      }
    } else {
      outlineState.fill(0);
    }
    restoreHistoryState(
      { undo: saved.undoStack, redo: saved.redoStack },
      expectedLength,
    );
    if (typeof saved.currentLabel === 'number') {
      currentLabel = saved.currentLabel;
    }
    if (typeof saved.maskOpacity === 'number') {
      maskOpacity = saved.maskOpacity;
      syncMaskOpacityControls();
      updateMaskOpacityLabel();
    }
    if (typeof saved.maskThreshold === 'number') {
      maskThreshold = saved.maskThreshold;
      syncMaskThresholdControls();
    }
    if (typeof saved.flowThreshold === 'number') {
      flowThreshold = saved.flowThreshold;
      syncFlowThresholdControls();
    }
    if (saved.niter === null) {
      niterAuto = true;
      syncNiterControls();
    } else if (typeof saved.niter === 'number') {
      niterAuto = false;
      niter = clamp(saved.niter, 0, NITER_MAX);
      syncNiterControls();
    }
    if (typeof saved.segmentationModel === 'string') {
      segmentationModel = saved.segmentationModel;
    }
    if (typeof saved.customSegmentationModelPath === 'string') {
      customSegmentationModelPath = saved.customSegmentationModelPath;
    }
    if (saved.viewState) {
      const vs = saved.viewState;
      if (typeof vs.scale === 'number') viewState.scale = vs.scale;
      if (typeof vs.offsetX === 'number') viewState.offsetX = vs.offsetX;
      if (typeof vs.offsetY === 'number') viewState.offsetY = vs.offsetY;
      if (typeof vs.rotation === 'number') viewState.rotation = vs.rotation;
      // Prevent auto-fit and recenter from overriding restored view
      userAdjustedScale = true;
      autoFitPending = false;
      viewStateRestored = true;
    }
    if (typeof saved.brushDiameter === 'number') {
      setBrushDiameter(saved.brushDiameter, false);
    }
    eraseActive = false;
    erasePreviousLabel = null;
    setTool('brush');
    if (typeof saved.maskVisible === 'boolean') {
      maskVisible = saved.maskVisible;
      if (maskVisibilityToggle) {
        maskVisibilityToggle.checked = maskVisible;
      }
      updateMaskVisibilityLabel();
    }
    if (typeof saved.maskDisplayMode === 'string') {
      maskDisplayMode = normalizeMaskDisplayMode(saved.maskDisplayMode);
      outlinesVisible = maskDisplayMode === MASK_DISPLAY_MODES.OUTLINED || maskDisplayMode === MASK_DISPLAY_MODES.OUTLINE;
      if (maskDisplayMode === MASK_DISPLAY_MODES.HIDDEN) {
        maskVisible = false;
        setPanelCollapsed(labelStylePanel, true);
      } else {
        maskVisible = true;
        setPanelCollapsed(labelStylePanel, false);
      }
    }
    if (typeof saved.clusterEnabled === 'boolean') {
      setClusterEnabled(saved.clusterEnabled, { silent: true });
    }
    if (typeof saved.affinitySegEnabled === 'boolean') {
      setAffinitySegEnabled(saved.affinitySegEnabled, { silent: true });
    }
    if (typeof saved.showFlowOverlay === 'boolean') {
      showFlowOverlay = saved.showFlowOverlay;
      if (flowOverlayToggle) flowOverlayToggle.checked = showFlowOverlay;
    }
    if (typeof saved.showDistanceOverlay === 'boolean') {
      showDistanceOverlay = saved.showDistanceOverlay;
      if (distanceOverlayToggle) distanceOverlayToggle.checked = showDistanceOverlay;
    }
    if (typeof saved.showPointsOverlay === 'boolean') {
      showPointsOverlay = saved.showPointsOverlay;
      if (pointsOverlayToggle) pointsOverlayToggle.checked = showPointsOverlay;
    }
    if (saved.affinityGraph && saved.affinityGraph.encoded) {
      savedAffinityGraphPayload = saved.affinityGraph;
    }
    if (typeof saved.showVectorOverlay === 'boolean') {
      vectorOverlayPreferred = saved.showVectorOverlay;
      showVectorOverlay = affinitySegEnabled ? vectorOverlayPreferred : false;
      if (vectorOverlayToggle) vectorOverlayToggle.checked = vectorOverlayPreferred;
    }
    if (!affinitySegEnabled) {
      showVectorOverlay = false;
      if (vectorOverlayToggle) vectorOverlayToggle.checked = vectorOverlayPreferred;
    }
    if (saved.pointsPayload) {
      applyPointsPayload(saved.pointsPayload);
    }
    if (typeof saved.showAffinityGraph === 'boolean') {
      showAffinityGraph = saved.showAffinityGraph;
      if (affinityGraphToggle) {
        affinityGraphToggle.checked = showAffinityGraph;
      }
      if (!showAffinityGraph && !showPointsOverlay) {
        clearWebglOverlaySurface();
      } else {
        updateOverlayVisibility();
      }
    }
    if (saved.nColorActive) {
      nColorActive = true;
      nColorValues = null;
    } else if (typeof saved.nColorActive === 'boolean') {
      nColorActive = false;
      nColorValues = null;
    } else {
      nColorActive = true;
      nColorValues = null;
    }
    if (saved.nColorInstanceMask) {
      const restored = uint32FromBase64(saved.nColorInstanceMask, expectedLength);
      if (restored.length === expectedLength) {
        nColorInstanceMask = new Uint32Array(restored);
      }
    }
    if (Number.isFinite(saved.nColorHueOffset)) {
      nColorHueOffset = saved.nColorHueOffset;
    }
    if (typeof saved.nColorColormap === 'string') {
      nColorColormap = saved.nColorColormap;
    }
    if (Array.isArray(saved.nColorPaletteColors)) {
      setNColorPaletteColors(saved.nColorPaletteColors, { render: false, schedule: false });
    }
    if (typeof saved.labelColormap === 'string') {
      labelColormap = saved.labelColormap;
    }
    if (typeof saved.imageColormap === 'string') {
      const validCmaps = IMAGE_COLORMAPS.map(c => c.value);
      if (validCmaps.includes(saved.imageColormap)) {
        imageColormap = saved.imageColormap;
        imageCmapLutDirty = true;
      }
    }
    if (typeof saved.labelShuffle === 'boolean') {
      labelShuffle = saved.labelShuffle;
    }
    if (Number.isFinite(saved.labelShuffleSeed)) {
      labelShuffleSeed = saved.labelShuffleSeed | 0;
    }
    if (Number.isFinite(saved.currentMaxLabel) && saved.currentMaxLabel > 0) {
      currentMaxLabel = saved.currentMaxLabel | 0;
    }
    if (segmentationModelSelect) {
      if (segmentationModel && segmentationModel.startsWith('file:')) {
        const label = segmentationModel.replace(/^file:/, '') || 'Custom Model';
        let option = null;
        for (const opt of segmentationModelSelect.options) {
          if (opt.value === segmentationModel) {
            option = opt;
            break;
          }
        }
        if (!option) {
          option = document.createElement('option');
          option.value = segmentationModel;
          option.textContent = label;
          segmentationModelSelect.insertBefore(option, segmentationModelSelect.lastElementChild);
        }
      }
      segmentationModelSelect.value = segmentationModel || DEFAULT_SEGMENTATION_MODEL;
      refreshDropdown('segmentationModel');
    }
    if (!Number.isFinite(currentLabel) || currentLabel <= 0) {
      currentLabel = 1;
    }
    updateColorModeLabel();
    updateMaskLabel();
    updateMaskVisibilityLabel();
    if (typeof saved.segMode === "string") {
      segMode = saved.segMode;
    }
    setSegMode(segMode, { silent: true });
    maybeApplySavedAffinityGraph();
    updateMaskStyleControls();
    updateToolInfo();
    updateBrushControls();
    if (flowOverlayToggle) flowOverlayToggle.checked = showFlowOverlay;
    if (distanceOverlayToggle) distanceOverlayToggle.checked = showDistanceOverlay;
    if (pointsOverlayToggle) pointsOverlayToggle.checked = showPointsOverlay;
    if (autoNColorToggle) autoNColorToggle.checked = nColorActive;
    if (labelColormapSelect) {
      labelColormapSelect.value = labelColormap || 'sinebow';
      refreshDropdown('labelColormap');
    }
    if (imageCmapSelect) {
      imageCmapSelect.value = imageColormap || 'gray';
      refreshDropdown('imageCmapSelect');
    }
    updateImageCmapPanelUI();
    updateImageCmapTexture();
    if (labelShuffleToggle) labelShuffleToggle.checked = labelShuffle;
    if (labelShuffleSeedInput) labelShuffleSeedInput.value = String(labelShuffleSeed);
    if (ncolorHueOffsetSlider) ncolorHueOffsetSlider.value = Math.round(nColorHueOffset * 100);
    updateLabelShuffleControls();
    // Invalidate shuffle permutation cache since seed may have changed
    shufflePermutation = null;
    // Force palette rebuild to match restored shuffle settings
    paletteTextureDirty = true;
    clearColorCaches();
    renderNColorSwatches();
    updateNColorPanel();
    updateInstanceColormapPreview();
    if (isWebglPipelineActive()) {
      markMaskTextureFullDirty();
      markOutlineTextureFullDirty();
    } else {
      redrawMaskCanvas();
    }
    updateToolButtons();
    markAffinityGeometryDirty();
  } finally {
    isRestoringState = false;
  }
  // Apply saved affinity graph if not already applied (e.g., when not in affinity seg mode)
  if (savedAffinityGraphPayload && savedAffinityGraphPayload.encoded && affinityGraphSource === 'none') {
    applyAffinityGraphPayload(savedAffinityGraphPayload);
    affinityGraphSource = 'remote';
  }
  if (affinityGraphSource === 'remote' && affinityGraphInfo && affinityGraphInfo.values) {
    buildAffinityGraphSegments();
    rebuildOutlineFromAffinity();
  } else {
    rebuildLocalAffinityGraph();
  }
  logDebugGridStatus('after-restore');
  if (webglOverlay && webglOverlay.enabled) {
    webglOverlay.needsGeometryRebuild = true;
  }
  if (!showAffinityGraph) {
    clearWebglOverlaySurface();
  }
  needsMaskRedraw = true;
  applyMaskRedrawImmediate();
  draw();
  scheduleAffinityRebuildIfStale('restore');
  stateDirty = false;
  stateDirtySeq = 0;
  lastSavedSeq = 0;
  stateDebugLog('restore reset sequences');
  updateImageInfo();
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

function ensureWebglOverlayReady() {
  if (!USE_WEBGL_OVERLAY) {
    return false;
  }
  if (webglOverlay && webglOverlay.enabled) {
    return true;
  }
  if (webglOverlay && webglOverlay.failed) {
    return false;
  }
  return initializeWebglOverlay();
}

function initializeSharedOverlay(gl) {
  if (!gl) {
    webglOverlay = { enabled: false, failed: true };
    return false;
  }
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  const lineVertexSource = `#version 300 es
layout (location = 0) in vec2 a_position;
layout (location = 1) in vec4 a_color;

uniform mat3 u_matrix;
out vec4 v_color;

void main() {
  vec3 pos = u_matrix * vec3(a_position, 1.0);
  gl_Position = vec4(pos.xy, 0.0, 1.0);
  v_color = a_color;
}
`;
  const lineFragmentSource = `#version 300 es
precision mediump float;
in vec4 v_color;
out vec4 outColor;

void main() {
  outColor = v_color;
}
`;
  const pointVertexSource = `#version 300 es
layout (location = 0) in vec2 a_position;
layout (location = 1) in vec4 a_color;

uniform mat3 u_matrix;
out vec4 v_color;

void main() {
  vec3 pos = u_matrix * vec3(a_position, 1.0);
  gl_Position = vec4(pos.xy, 0.0, 1.0);
  gl_PointSize = 10.0;
  v_color = a_color;
}
`;
  const pointFragmentSource = `#version 300 es
precision mediump float;
in vec4 v_color;
uniform float u_alpha;
out vec4 outColor;

void main() {
  vec2 coord = gl_PointCoord - vec2(0.5);
  if (dot(coord, coord) > 0.25) {
    discard;
  }
  outColor = vec4(v_color.rgb, v_color.a * u_alpha);
}
`;
  const lineProgram = createWebglProgram(gl, lineVertexSource, lineFragmentSource);
  if (!lineProgram) {
    webglOverlay = { enabled: false, failed: true };
    return false;
  }
  const pointProgram = createWebglProgram(gl, pointVertexSource, pointFragmentSource);
  const positionBuffer = gl.createBuffer();
  const colorBuffer = gl.createBuffer();
  const pointsPositionBuffer = gl.createBuffer();
  const pointsColorBuffer = gl.createBuffer();
  gl.clearColor(0, 0, 0, 0);
  gl.lineWidth(1);
  const lineAttribs = {
    position: gl.getAttribLocation(lineProgram, 'a_position'),
    color: gl.getAttribLocation(lineProgram, 'a_color'),
  };
  const lineUniforms = {
    matrix: gl.getUniformLocation(lineProgram, 'u_matrix'),
  };
  const pointAttribs = pointProgram ? {
    position: gl.getAttribLocation(pointProgram, 'a_position'),
    color: gl.getAttribLocation(pointProgram, 'a_color'),
  } : null;
  const pointUniforms = pointProgram ? {
    matrix: gl.getUniformLocation(pointProgram, 'u_matrix'),
    alpha: gl.getUniformLocation(pointProgram, 'u_alpha'),
  } : null;
  webglOverlay = {
    enabled: true,
    failed: false,
    shared: true,
    canvas: null,
    gl,
    lineProgram,
    pointProgram,
    lineAttribs,
    lineUniforms,
    pointAttribs,
    pointUniforms,
    positionBuffer,
    colorBuffer,
    pointsPositionBuffer,
    pointsColorBuffer,
    positionsArray: null,
    colorsArray: null,
    pointsPositions: null,
    pointsColors: null,
    pointsCount: 0,
    edgeCount: 0,
    vertexCount: 0,
    width: 0,
    height: 0,
    displayAlpha: 1,
    targetWidth: 0,
    targetHeight: 0,
    targetsReady: true,
    matrixCache: new Float32Array(9),
    needsGeometryRebuild: true,
    capacityEdges: 0,
    nextSlot: 0,
    maxUsedSlotIndex: -1,
    freeSlots: [],
    dirtyPosSlots: new Set(),
    dirtyColSlots: new Set(),
    useMsaa: false,
    useFxaa: false,
    useTaa: false,
    resolve: { framebuffer: null, texture: null, levels: 1 },
    postProcess: { framebuffer: null, texture: null },
    taa: { framebuffer: null, texture: null, historyTexture: null, hasHistory: false },
    msaa: { framebuffer: null, renderbuffer: null, samples: 0 },
    quad: null,
    fxaaProgram: null,
    fxaaUniforms: null,
    taaProgram: null,
    taaUniforms: null,
  };
  return true;
}


function initializePointsOverlay() {
  if (pointsOverlay && pointsOverlay.enabled) {
    return true;
  }
  if (!viewer || typeof WebGL2RenderingContext === 'undefined') {
    pointsOverlay = { enabled: false, failed: true };
    return false;
  }
  const canvasEl = document.createElement('canvas');
  canvasEl.id = 'pointsWebgl';
  canvasEl.style.position = 'absolute';
  canvasEl.style.inset = '0';
  canvasEl.style.pointerEvents = 'none';
  canvasEl.style.zIndex = '2';
  canvasEl.style.display = 'none';
  canvasEl.style.opacity = '0';
  viewer.appendChild(canvasEl);
  const gl = canvasEl.getContext('webgl2', OVERLAY_CONTEXT_ATTRIBUTES);
  if (!gl) {
    viewer.removeChild(canvasEl);
    pointsOverlay = { enabled: false, failed: true };
    return false;
  }
  if (!(gl instanceof WebGL2RenderingContext)) {
    viewer.removeChild(canvasEl);
    pointsOverlay = { enabled: false, failed: true };
    return false;
  }
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  const pointVertexSource = `#version 300 es
layout (location = 0) in vec2 a_position;
layout (location = 1) in vec4 a_color;

uniform mat3 u_matrix;
out vec4 v_color;

void main() {
  vec3 pos = u_matrix * vec3(a_position, 1.0);
  gl_Position = vec4(pos.xy, 0.0, 1.0);
  gl_PointSize = 10.0;
  v_color = a_color;
}
`;
  const pointFragmentSource = `#version 300 es
precision mediump float;
in vec4 v_color;
uniform float u_alpha;
out vec4 outColor;

void main() {
  vec2 coord = gl_PointCoord - vec2(0.5);
  if (dot(coord, coord) > 0.25) {
    discard;
  }
  outColor = vec4(v_color.rgb, v_color.a * u_alpha);
}
`;
  const program = createWebglProgram(gl, pointVertexSource, pointFragmentSource);
  if (!program) {
    viewer.removeChild(canvasEl);
    pointsOverlay = { enabled: false, failed: true };
    return false;
  }
  const attribs = {
    position: gl.getAttribLocation(program, 'a_position'),
    color: gl.getAttribLocation(program, 'a_color'),
  };
  const uniforms = {
    matrix: gl.getUniformLocation(program, 'u_matrix'),
    alpha: gl.getUniformLocation(program, 'u_alpha'),
  };
  const positionBuffer = gl.createBuffer();
  const colorBuffer = gl.createBuffer();
  pointsOverlay = {
    enabled: true,
    failed: false,
    canvas: canvasEl,
    gl,
    program,
    attribs,
    uniforms,
    positionBuffer,
    colorBuffer,
    pointsCount: 0,
    width: 0,
    height: 0,
  };
  updatePointsOverlayBuffers();
  return true;
}



function initializeVectorOverlay() {
  if (vectorOverlay && vectorOverlay.enabled) {
    return true;
  }
  if (!viewer || typeof WebGL2RenderingContext === 'undefined') {
    vectorOverlay = { enabled: false, failed: true };
    return false;
  }
  const canvasEl = document.createElement('canvas');
  canvasEl.id = 'vectorsWebgl';
  canvasEl.style.position = 'absolute';
  canvasEl.style.inset = '0';
  canvasEl.style.pointerEvents = 'none';
  canvasEl.style.zIndex = '1.5';
  canvasEl.style.display = 'none';
  canvasEl.style.opacity = '0';
  viewer.appendChild(canvasEl);
  const gl = canvasEl.getContext('webgl2', OVERLAY_CONTEXT_ATTRIBUTES);
  if (!gl) {
    viewer.removeChild(canvasEl);
    vectorOverlay = { enabled: false, failed: true };
    return false;
  }
  if (!(gl instanceof WebGL2RenderingContext)) {
    viewer.removeChild(canvasEl);
    vectorOverlay = { enabled: false, failed: true };
    return false;
  }
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  const vertexSource = `#version 300 es
layout (location = 0) in vec2 a_position;
layout (location = 1) in vec4 a_color;

uniform mat3 u_matrix;
out vec4 v_color;

void main() {
  vec3 pos = u_matrix * vec3(a_position, 1.0);
  gl_Position = vec4(pos.xy, 0.0, 1.0);
  v_color = a_color;
}
`;
  const fragmentSource = `#version 300 es
precision mediump float;

in vec4 v_color;
uniform float u_alpha;
out vec4 outColor;

void main() {
  outColor = vec4(v_color.rgb, v_color.a * u_alpha);
}
`;
  const program = createWebglProgram(gl, vertexSource, fragmentSource);
  if (!program) {
    viewer.removeChild(canvasEl);
    vectorOverlay = { enabled: false, failed: true };
    return false;
  }
  const attribs = {
    position: gl.getAttribLocation(program, 'a_position'),
    color: gl.getAttribLocation(program, 'a_color'),
  };
  const uniforms = {
    matrix: gl.getUniformLocation(program, 'u_matrix'),
    alpha: gl.getUniformLocation(program, 'u_alpha'),
  };
  const positionBuffer = gl.createBuffer();
  const colorBuffer = gl.createBuffer();
  vectorOverlay = {
    enabled: true,
    failed: false,
    canvas: canvasEl,
    gl,
    program,
    attribs,
    uniforms,
    positionBuffer,
    colorBuffer,
    vertexCount: 0,
    width: 0,
    height: 0,
  };
  updateVectorOverlayBuffers();
  return true;
}

function resizeVectorOverlay() {
  if (!vectorOverlay || !vectorOverlay.enabled || !vectorOverlay.canvas) {
    return;
  }
  const v = getViewportSize();
  const renderWidth = Math.max(1, v.width);
  const canvasEl = vectorOverlay.canvas;
  canvasEl.width = Math.max(1, Math.round(renderWidth * dpr));
  canvasEl.height = Math.max(1, Math.round(v.height * dpr));
  canvasEl.style.width = renderWidth + 'px';
  canvasEl.style.height = v.height + 'px';
}

function drawVectorOverlay(matrix) {
  if (!vectorOverlay || !vectorOverlay.enabled) {
    return;
  }
  const showVectors = showVectorOverlay && affinitySegEnabled && vectorOverlay.vertexCount > 0;
  if (!showVectors) {
    if (vectorOverlay.canvas) {
      vectorOverlay.canvas.style.display = 'none';
      vectorOverlay.canvas.style.opacity = '0';
    }
    return;
  }
  const gl = vectorOverlay.gl;
  const canvasEl = vectorOverlay.canvas;
  if (!gl || !canvasEl) {
    return;
  }
  canvasEl.style.display = 'block';
  canvasEl.style.opacity = '1';
  gl.viewport(0, 0, canvasEl.width, canvasEl.height);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.useProgram(vectorOverlay.program);
  gl.uniformMatrix3fv(vectorOverlay.uniforms.matrix, false, matrix);
  if (vectorOverlay.uniforms.alpha) {
    gl.uniform1f(vectorOverlay.uniforms.alpha, VECTOR_OVERLAY_ALPHA);
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, vectorOverlay.positionBuffer);
  gl.enableVertexAttribArray(vectorOverlay.attribs.position);
  gl.vertexAttribPointer(vectorOverlay.attribs.position, 2, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, vectorOverlay.colorBuffer);
  gl.enableVertexAttribArray(vectorOverlay.attribs.color);
  gl.vertexAttribPointer(vectorOverlay.attribs.color, 4, gl.UNSIGNED_BYTE, true, 0, 0);
  gl.drawArrays(gl.LINES, 0, vectorOverlay.vertexCount);
  gl.disableVertexAttribArray(vectorOverlay.attribs.position);
  gl.disableVertexAttribArray(vectorOverlay.attribs.color);
}

function updateVectorOverlayBuffers() {
  if (!vectorOverlay || !vectorOverlay.enabled) {
    return;
  }
  const gl = vectorOverlay.gl;
  if (!gl || !vectorOverlay.positionBuffer || !vectorOverlay.colorBuffer) {
    return;
  }
  if (!vectorOverlay.positions || !vectorOverlay.colors || !vectorOverlay.vertexCount) {
    vectorOverlay.vertexCount = 0;
    gl.bindBuffer(gl.ARRAY_BUFFER, vectorOverlay.positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, vectorOverlay.colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(), gl.STATIC_DRAW);
    return;
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, vectorOverlay.positionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, vectorOverlay.positions, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, vectorOverlay.colorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, vectorOverlay.colors, gl.STATIC_DRAW);
}

function initializeWebglOverlay() {
  if (!viewer || typeof WebGL2RenderingContext === 'undefined') {
    webglOverlay = { enabled: false, failed: true };
    return false;
  }
  if (!viewer.style.position) {
    viewer.style.position = 'relative';
  }
  const canvasEl = document.createElement('canvas');
  canvasEl.id = 'affinityWebgl';
  canvasEl.style.position = 'absolute';
  canvasEl.style.inset = '0';
  canvasEl.style.pointerEvents = 'none';
  canvasEl.style.zIndex = '1';
  canvasEl.style.display = 'none';
  canvasEl.style.opacity = '0';
  viewer.appendChild(canvasEl);
  const gl = canvasEl.getContext('webgl2', OVERLAY_CONTEXT_ATTRIBUTES);
  if (!gl) {
    viewer.removeChild(canvasEl);
    webglOverlay = { enabled: false, failed: true };
    return false;
  }
  if (!(gl instanceof WebGL2RenderingContext)) {
    viewer.removeChild(canvasEl);
    webglOverlay = { enabled: false, failed: true };
    return false;
  }
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  const success = initializeSharedOverlay(gl);
  initializePointsOverlay();
  initializeVectorOverlay();
  if (!success) {
    viewer.removeChild(canvasEl);
    return false;
  }
  webglOverlay.canvas = canvasEl;
  webglOverlay.shared = false;
  webglOverlay.targetsReady = false;
  webglOverlay.useMsaa = OVERLAY_MSAA_ENABLED;
  webglOverlay.useFxaa = OVERLAY_FXAA_ENABLED;
  webglOverlay.useTaa = OVERLAY_TAA_ENABLED;
  webglOverlay.requestedMsaaSamples = OVERLAY_MSAA_SAMPLES;
  webglOverlay.generateMips = OVERLAY_GENERATE_MIPS;
  webglOverlay.resolve = { framebuffer: null, texture: null, levels: 1 };
  webglOverlay.postProcess = { framebuffer: null, texture: null };
  webglOverlay.taa = { framebuffer: null, texture: null, historyTexture: null, hasHistory: false };
  webglOverlay.msaa = { framebuffer: null, renderbuffer: null, samples: 0 };
  webglOverlay.fxaaProgram = null;
  webglOverlay.quad = null;
  webglOverlay.fxaaUniforms = null;
  webglOverlay.taaProgram = null;
  webglOverlay.taaUniforms = null;
  webglOverlay.displayAlpha = 1;
  webglOverlay.freeSlots = [];
  setupOverlayPostProcessing(gl, webglOverlay);
  if (!webglOverlay.enabled) {
    viewer.removeChild(canvasEl);
    webglOverlay = { enabled: false, failed: true };
    return false;
  }
  resizeWebglOverlay();
  if (affinityGraphInfo && affinityGraphInfo.vertices) {
    affinityGraphInfo.vertices = null;
    affinityGeometryDirty = true;
  }
  return true;
}

function setupOverlayPostProcessing(gl, overlay) {
  overlay.quad = createFullscreenQuad(gl);
  if (!overlay.quad) {
    overlay.enabled = false;
    overlay.failed = true;
    return;
  }
  overlay.presentProgram = createPresentProgram(gl);
  if (!overlay.presentProgram) {
    overlay.enabled = false;
    overlay.failed = true;
    return;
  }
  if (overlay.useFxaa) {
    overlay.fxaaProgram = createFxaaProgram(gl);
    if (!overlay.fxaaProgram) {
      overlay.useFxaa = false;
    }
  }
  if (overlay.useTaa) {
    overlay.taaProgram = createTaaProgram(gl);
    if (!overlay.taaProgram) {
      overlay.useTaa = false;
    }
  }
}

function createFullscreenQuad(gl) {
  const vao = gl.createVertexArray();
  const buffer = gl.createBuffer();
  if (!vao || !buffer) {
    if (vao) gl.deleteVertexArray(vao);
    if (buffer) gl.deleteBuffer(buffer);
    return null;
  }
  const vertices = new Float32Array([
    -1, -1,
     1, -1,
    -1,  1,
     1,  1,
  ]);
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  return {
    vao,
    buffer,
    vertexCount: 4,
  };
}

function createOverlayColorTarget(gl, width, height, { generateMips = false } = {}) {
  const texture = gl.createTexture();
  const framebuffer = gl.createFramebuffer();
  if (!texture || !framebuffer) {
    if (texture) gl.deleteTexture(texture);
    if (framebuffer) gl.deleteFramebuffer(framebuffer);
    return null;
  }
  const levels = generateMips ? Math.floor(Math.log2(Math.max(width, height))) + 1 : 1;
  gl.bindTexture(gl.TEXTURE_2D, texture);
  if (levels > 1 && gl.texStorage2D) {
    gl.texStorage2D(gl.TEXTURE_2D, levels, gl.RGBA8, width, height);
  } else {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  }
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  if (generateMips) {
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  } else {
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  }
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.bindTexture(gl.TEXTURE_2D, null);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    gl.deleteTexture(texture);
    gl.deleteFramebuffer(framebuffer);
    return null;
  }
  return {
    texture,
    framebuffer,
    levels,
  };
}

function createOverlayRenderbuffer(gl, width, height, samples) {
  const framebuffer = gl.createFramebuffer();
  const renderbuffer = gl.createRenderbuffer();
  if (!framebuffer || !renderbuffer) {
    if (framebuffer) gl.deleteFramebuffer(framebuffer);
    if (renderbuffer) gl.deleteRenderbuffer(renderbuffer);
    return null;
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.bindRenderbuffer(gl.RENDERBUFFER, renderbuffer);
  gl.renderbufferStorageMultisample(gl.RENDERBUFFER, samples, gl.RGBA8, width, height);
  gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, renderbuffer);
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.bindRenderbuffer(gl.RENDERBUFFER, null);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    gl.deleteFramebuffer(framebuffer);
    gl.deleteRenderbuffer(renderbuffer);
    return null;
  }
  return {
    framebuffer,
    renderbuffer,
    samples,
  };
}

function deleteOverlayTarget(gl, target) {
  if (!target) return;
  if (target.texture) {
    gl.deleteTexture(target.texture);
    target.texture = null;
  }
  if (target.framebuffer) {
    gl.deleteFramebuffer(target.framebuffer);
    target.framebuffer = null;
  }
}

function deleteOverlayRenderbuffer(gl, msaa) {
  if (!msaa) return;
  if (msaa.renderbuffer) {
    gl.deleteRenderbuffer(msaa.renderbuffer);
    msaa.renderbuffer = null;
  }
  if (msaa.framebuffer) {
    gl.deleteFramebuffer(msaa.framebuffer);
    msaa.framebuffer = null;
  }
  msaa.samples = 0;
}

function ensureOverlayTargets(width, height) {
  if (!webglOverlay || !webglOverlay.enabled) {
    return;
  }
  const w = Math.max(1, width | 0);
  const h = Math.max(1, height | 0);
  if (webglOverlay.targetWidth === w && webglOverlay.targetHeight === h && webglOverlay.targetsReady) {
    return;
  }
  if (webglOverlay.shared) {
    webglOverlay.targetWidth = w;
    webglOverlay.targetHeight = h;
    webglOverlay.targetsReady = true;
    return;
  }
  const { gl } = webglOverlay;
  releaseOverlayTargets();
  // Resolve target is the baseline texture everyone samples from.
  const resolveTarget = createOverlayColorTarget(gl, w, h, { generateMips: webglOverlay.generateMips });
  if (!resolveTarget) {
    webglOverlay.enabled = false;
    webglOverlay.failed = true;
    return;
  }
  webglOverlay.resolve.framebuffer = resolveTarget.framebuffer;
  webglOverlay.resolve.texture = resolveTarget.texture;
  webglOverlay.resolve.levels = resolveTarget.levels;
  // Optional post-process target (FXAA / intermediate).
  if (webglOverlay.useFxaa || webglOverlay.useTaa) {
    const ppTarget = createOverlayColorTarget(gl, w, h, { generateMips: false });
    if (!ppTarget) {
      webglOverlay.useFxaa = false;
    } else {
      webglOverlay.postProcess.framebuffer = ppTarget.framebuffer;
      webglOverlay.postProcess.texture = ppTarget.texture;
    }
  }
  // TAA output + history textures.
  if (webglOverlay.useTaa) {
    const taaTarget = createOverlayColorTarget(gl, w, h, { generateMips: false });
    if (!taaTarget) {
      webglOverlay.useTaa = false;
      webglOverlay.taa.hasHistory = false;
    } else {
      webglOverlay.taa.framebuffer = taaTarget.framebuffer;
      webglOverlay.taa.texture = taaTarget.texture;
      const historyTexture = gl.createTexture();
      if (!historyTexture) {
        webglOverlay.useTaa = false;
        webglOverlay.taa.hasHistory = false;
      } else {
        gl.bindTexture(gl.TEXTURE_2D, historyTexture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.bindTexture(gl.TEXTURE_2D, null);
        if (webglOverlay.taa.historyTexture) {
          gl.deleteTexture(webglOverlay.taa.historyTexture);
        }
        webglOverlay.taa.historyTexture = historyTexture;
        webglOverlay.taa.hasHistory = false;
      }
    }
  }
  // MSAA target (optional)
  if (webglOverlay.useMsaa) {
    const maxSamples = gl.getParameter(gl.MAX_SAMPLES) | 0;
    const samples = Math.max(1, Math.min(webglOverlay.requestedMsaaSamples | 0, maxSamples));
    if (samples > 1) {
      const msaaTarget = createOverlayRenderbuffer(gl, w, h, samples);
      if (msaaTarget) {
        webglOverlay.msaa.framebuffer = msaaTarget.framebuffer;
        webglOverlay.msaa.renderbuffer = msaaTarget.renderbuffer;
        webglOverlay.msaa.samples = samples;
      } else {
        webglOverlay.msaa.framebuffer = null;
        webglOverlay.msaa.renderbuffer = null;
        webglOverlay.msaa.samples = 0;
      }
    } else {
      webglOverlay.msaa.framebuffer = null;
      webglOverlay.msaa.renderbuffer = null;
      webglOverlay.msaa.samples = 0;
    }
  }
  webglOverlay.targetWidth = w;
  webglOverlay.targetHeight = h;
  webglOverlay.targetsReady = true;
}

function releaseOverlayTargets() {
  if (!webglOverlay || !webglOverlay.gl) {
    return;
  }
  if (webglOverlay.shared) {
    webglOverlay.targetsReady = true;
    return;
  }
  const { gl } = webglOverlay;
  deleteOverlayTarget(gl, webglOverlay.resolve);
  deleteOverlayTarget(gl, webglOverlay.postProcess);
  if (webglOverlay.taa && webglOverlay.taa.texture) {
    gl.deleteTexture(webglOverlay.taa.texture);
    gl.deleteFramebuffer(webglOverlay.taa.framebuffer);
    webglOverlay.taa.texture = null;
    webglOverlay.taa.framebuffer = null;
  }
  if (webglOverlay.taa && webglOverlay.taa.historyTexture) {
    gl.deleteTexture(webglOverlay.taa.historyTexture);
    webglOverlay.taa.historyTexture = null;
    webglOverlay.taa.hasHistory = false;
  }
  if (webglOverlay.msaa) {
    deleteOverlayRenderbuffer(gl, webglOverlay.msaa);
  }
  webglOverlay.targetsReady = false;
  webglOverlay.targetWidth = 0;
  webglOverlay.targetHeight = 0;
}

function createFxaaProgram(gl) {
  const vertexSource = `#version 300 es
layout (location = 0) in vec2 a_position;
out vec2 v_uv;

void main() {
  v_uv = a_position * 0.5 + 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;
  const fragmentSource = `#version 300 es
precision mediump float;
in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_input;
uniform vec2 u_texelSize;
uniform float u_subpixel;
uniform float u_edgeThreshold;
uniform float u_edgeThresholdMin;

const vec3 LUMA_WEIGHTS = vec3(0.299, 0.587, 0.114);

void main() {
  vec4 centerSample = texture(u_input, v_uv);
  float lumaM = dot(centerSample.rgb, LUMA_WEIGHTS);
  float lumaN = dot(texture(u_input, v_uv + vec2(0.0, -u_texelSize.y)).rgb, LUMA_WEIGHTS);
  float lumaS = dot(texture(u_input, v_uv + vec2(0.0, u_texelSize.y)).rgb, LUMA_WEIGHTS);
  float lumaE = dot(texture(u_input, v_uv + vec2(u_texelSize.x, 0.0)).rgb, LUMA_WEIGHTS);
  float lumaW = dot(texture(u_input, v_uv + vec2(-u_texelSize.x, 0.0)).rgb, LUMA_WEIGHTS);
  float lumaMin = min(lumaM, min(min(lumaN, lumaS), min(lumaE, lumaW)));
  float lumaMax = max(lumaM, max(max(lumaN, lumaS), max(lumaE, lumaW)));
  float lumaRange = lumaMax - lumaMin;
  if (lumaRange < max(u_edgeThresholdMin, lumaMax * u_edgeThreshold)) {
    outColor = centerSample;
    return;
  }
  vec3 rgbNW = texture(u_input, v_uv + vec2(-u_texelSize.x, -u_texelSize.y)).rgb;
  vec3 rgbNE = texture(u_input, v_uv + vec2(u_texelSize.x, -u_texelSize.y)).rgb;
  vec3 rgbSW = texture(u_input, v_uv + vec2(-u_texelSize.x, u_texelSize.y)).rgb;
  vec3 rgbSE = texture(u_input, v_uv + vec2(u_texelSize.x, u_texelSize.y)).rgb;
  float lumaNW = dot(rgbNW, LUMA_WEIGHTS);
  float lumaNE = dot(rgbNE, LUMA_WEIGHTS);
  float lumaSW = dot(rgbSW, LUMA_WEIGHTS);
  float lumaSE = dot(rgbSE, LUMA_WEIGHTS);
  vec2 dir;
  dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
  dir.y = ((lumaNW + lumaSW) - (lumaNE + lumaSE));
  float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * u_subpixel), u_edgeThresholdMin);
  float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
  dir = clamp(dir * rcpDirMin, vec2(-8.0), vec2(8.0));
  dir *= u_texelSize;
  vec3 rgbA = 0.5 * (
    texture(u_input, v_uv + dir * (1.0 / 3.0 - 0.5)).rgb +
    texture(u_input, v_uv + dir * (-1.0 / 3.0 - 0.5)).rgb
  );
  vec3 rgbB = rgbA * 0.5 + 0.25 * (
    texture(u_input, v_uv + dir * (-0.5)).rgb +
    texture(u_input, v_uv + dir * (0.5)).rgb
  );
  float lumaB = dot(rgbB, LUMA_WEIGHTS);
  if (lumaB < lumaMin || lumaB > lumaMax) {
    outColor = vec4(rgbA, centerSample.a);
  } else {
    outColor = vec4(rgbB, centerSample.a);
  }
}
`;
  const program = createWebglProgram(gl, vertexSource, fragmentSource);
  if (!program) {
    return null;
  }
  gl.useProgram(program);
  const uniforms = {
    inputTexture: gl.getUniformLocation(program, 'u_input'),
    texelSize: gl.getUniformLocation(program, 'u_texelSize'),
    subpixel: gl.getUniformLocation(program, 'u_subpixel'),
    edgeThreshold: gl.getUniformLocation(program, 'u_edgeThreshold'),
    edgeThresholdMin: gl.getUniformLocation(program, 'u_edgeThresholdMin'),
  };
  gl.uniform1i(uniforms.inputTexture, 0);
  return { program, uniforms };
}

function createPresentProgram(gl) {
  const vertexSource = `#version 300 es
layout (location = 0) in vec2 a_position;
out vec2 v_uv;

void main() {
  v_uv = a_position * 0.5 + 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;
  const fragmentSource = `#version 300 es
precision mediump float;
in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_texture;

void main() {
  outColor = texture(u_texture, v_uv);
}
`;
  const program = createWebglProgram(gl, vertexSource, fragmentSource);
  if (!program) {
    return null;
  }
  gl.useProgram(program);
  const uniforms = {
    texture: gl.getUniformLocation(program, 'u_texture'),
  };
  gl.uniform1i(uniforms.texture, 0);
  return { program, uniforms };
}

function createTaaProgram(gl) {
  const vertexSource = `#version 300 es
layout (location = 0) in vec2 a_position;
out vec2 v_uv;

void main() {
  v_uv = a_position * 0.5 + 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;
  const fragmentSource = `#version 300 es
precision mediump float;
in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_current;
uniform sampler2D u_history;
uniform float u_alpha;
uniform int u_hasHistory;

void main() {
  vec4 currentSample = texture(u_current, v_uv);
  vec4 historySample = texture(u_history, v_uv);
  if (u_hasHistory == 1) {
    outColor = mix(currentSample, historySample, u_alpha);
  } else {
    outColor = currentSample;
  }
}
`;
  const program = createWebglProgram(gl, vertexSource, fragmentSource);
  if (!program) {
    return null;
  }
  gl.useProgram(program);
  const uniforms = {
    current: gl.getUniformLocation(program, 'u_current'),
    history: gl.getUniformLocation(program, 'u_history'),
    alpha: gl.getUniformLocation(program, 'u_alpha'),
    hasHistory: gl.getUniformLocation(program, 'u_hasHistory'),
  };
  gl.uniform1i(uniforms.current, 0);
  gl.uniform1i(uniforms.history, 1);
  gl.uniform1f(uniforms.alpha, OVERLAY_TAA_BLEND);
  gl.uniform1i(uniforms.hasHistory, 0);
  return { program, uniforms };
}

function runOverlayFxaa(sourceTexture, width, height) {
  if (!webglOverlay || !webglOverlay.fxaaProgram || !webglOverlay.postProcess || !webglOverlay.postProcess.framebuffer) {
    return sourceTexture;
  }
  const { gl, fxaaProgram, postProcess, quad } = webglOverlay;
  gl.bindFramebuffer(gl.FRAMEBUFFER, postProcess.framebuffer);
  gl.viewport(0, 0, width, height);
  gl.disable(gl.BLEND);
  gl.useProgram(fxaaProgram.program);
  gl.uniform2f(fxaaProgram.uniforms.texelSize, 1 / width, 1 / height);
  gl.uniform1f(fxaaProgram.uniforms.subpixel, FXAA_SUBPIX);
  gl.uniform1f(fxaaProgram.uniforms.edgeThreshold, FXAA_EDGE_THRESHOLD);
  gl.uniform1f(fxaaProgram.uniforms.edgeThresholdMin, FXAA_EDGE_THRESHOLD_MIN);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
  if (quad && quad.vao) {
    gl.bindVertexArray(quad.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, quad.vertexCount);
    gl.bindVertexArray(null);
  }
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return postProcess.texture || sourceTexture;
}

function runOverlayTaa(sourceTexture, width, height, hasContent) {
  if (!webglOverlay || !webglOverlay.taaProgram || !webglOverlay.taa || !webglOverlay.taa.framebuffer) {
    return sourceTexture;
  }
  const { gl, taaProgram, taa, quad } = webglOverlay;
  gl.bindFramebuffer(gl.FRAMEBUFFER, taa.framebuffer);
  gl.viewport(0, 0, width, height);
  gl.disable(gl.BLEND);
  gl.useProgram(taaProgram.program);
  gl.uniform1f(taaProgram.uniforms.alpha, webglOverlay.taaBlend);
  gl.uniform1i(taaProgram.uniforms.hasHistory, taa.hasHistory && hasContent ? 1 : 0);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, taa.historyTexture);
  if (quad && quad.vao) {
    gl.bindVertexArray(quad.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, quad.vertexCount);
    gl.bindVertexArray(null);
  }
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, null);
  // Copy freshly rendered frame into history for the next pass
  if (taa.historyTexture) {
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, taa.framebuffer);
    gl.bindTexture(gl.TEXTURE_2D, taa.historyTexture);
    gl.copyTexSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 0, 0, width, height);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
    taa.hasHistory = hasContent;
  } else {
    taa.hasHistory = false;
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return taa.texture || sourceTexture;
}

function presentOverlayTexture(sourceTexture, width, height, hasContent) {
  if (!webglOverlay || !webglOverlay.presentProgram) {
    return;
  }
  const { gl, presentProgram, quad } = webglOverlay;
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, width, height);
  gl.disable(gl.BLEND);
  if (!hasContent || !sourceTexture) {
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    return;
  }
  gl.useProgram(presentProgram.program);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
  if (quad && quad.vao) {
    gl.bindVertexArray(quad.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, quad.vertexCount);
    gl.bindVertexArray(null);
  }
  gl.bindTexture(gl.TEXTURE_2D, null);
}

function finalizeOverlayFrame(hasContent) {
  if (!webglOverlay || !webglOverlay.enabled || !webglOverlay.targetsReady) {
    if (webglOverlay && webglOverlay.gl) {
      webglOverlay.gl.bindFramebuffer(webglOverlay.gl.FRAMEBUFFER, null);
    }
    if (webglOverlay && webglOverlay.canvas && webglOverlay.canvas.style) {
      webglOverlay.canvas.style.opacity = '0';
    }
    return;
  }
  if (webglOverlay.shared) {
    return;
  }
  const { gl, canvas: glCanvas, resolve, msaa, generateMips } = webglOverlay;
  const width = glCanvas.width || 1;
  const height = glCanvas.height || 1;
  if (msaa && msaa.framebuffer && msaa.samples > 1 && resolve && resolve.framebuffer) {
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, msaa.framebuffer);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, resolve.framebuffer);
    gl.blitFramebuffer(0, 0, width, height, 0, 0, width, height, gl.COLOR_BUFFER_BIT, gl.LINEAR);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }
  if (generateMips && resolve && resolve.texture) {
    gl.bindTexture(gl.TEXTURE_2D, resolve.texture);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }
  let currentTexture = resolve ? resolve.texture : null;
  if (webglOverlay.useFxaa && webglOverlay.fxaaProgram && currentTexture) {
    currentTexture = runOverlayFxaa(currentTexture, width, height);
  }
  if (webglOverlay.useTaa && webglOverlay.taaProgram && currentTexture) {
    currentTexture = runOverlayTaa(currentTexture, width, height, hasContent);
  } else if (webglOverlay.taa) {
    webglOverlay.taa.hasHistory = false;
  }
  presentOverlayTexture(currentTexture, width, height, hasContent);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  if (glCanvas && glCanvas.style) {
    let finalAlpha = hasContent ? (typeof webglOverlay.displayAlpha === 'number' ? webglOverlay.displayAlpha : 1) : 0;
    if (!Number.isFinite(finalAlpha)) {
      finalAlpha = hasContent ? 1 : 0;
    }
    finalAlpha = Math.max(0, Math.min(1, finalAlpha));
    glCanvas.style.opacity = String(finalAlpha);
    if (hasContent && finalAlpha > 0) {
      setOverlayCanvasVisibility(true, finalAlpha);
    } else {
      setOverlayCanvasVisibility(false, 0);
    }
  }
}

function overlayEnsureCapacity(requiredEdges) {
  if (!webglOverlay || !webglOverlay.enabled) return;
  const { gl } = webglOverlay;
  let cap = webglOverlay.capacityEdges | 0;
  // If buffers are missing, force allocation even if requiredEdges <= cap
  const missingBuffers = !webglOverlay.positionsArray || !webglOverlay.colorsArray;
  if (!missingBuffers && requiredEdges <= cap) {
    return;
  }
  let newCap = cap > 0 ? cap : 1;
  while (newCap < requiredEdges) newCap = newCap * 2;
  const newPositions = new Float32Array(newCap * 4);
  const newColors = new Uint8Array(newCap * 8);
  if (webglOverlay.positionsArray) {
    newPositions.set(webglOverlay.positionsArray);
  }
  if (webglOverlay.colorsArray) {
    newColors.set(webglOverlay.colorsArray);
  }
  webglOverlay.positionsArray = newPositions;
  webglOverlay.colorsArray = newColors;
  webglOverlay.capacityEdges = newCap;
  webglOverlay.vertexCount = newCap * 2;
  gl.bindBuffer(gl.ARRAY_BUFFER, webglOverlay.positionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, newPositions, gl.DYNAMIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, webglOverlay.colorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, newColors, gl.DYNAMIC_DRAW);
}

function overlayAllocSlot() {
  if (!webglOverlay || !webglOverlay.enabled) return -1;
  let slot = -1;
  if (webglOverlay.freeSlots && webglOverlay.freeSlots.length) {
    slot = webglOverlay.freeSlots.pop();
  } else {
    slot = webglOverlay.nextSlot | 0;
    overlayEnsureCapacity(slot + 1);
    webglOverlay.nextSlot = slot + 1;
  }
  if (slot > (webglOverlay.maxUsedSlotIndex | 0)) {
    webglOverlay.maxUsedSlotIndex = slot;
  }
  return slot;
}

function overlayWriteSlotPosition(slot, coords) {
  if (!webglOverlay || !webglOverlay.enabled || slot < 0) return;
  const basePos = slot * 4;
  webglOverlay.positionsArray[basePos] = coords[0];
  webglOverlay.positionsArray[basePos + 1] = coords[1];
  webglOverlay.positionsArray[basePos + 2] = coords[2];
  webglOverlay.positionsArray[basePos + 3] = coords[3];
  if (BATCH_LIVE_OVERLAY_UPDATES) {
    webglOverlay.dirtyPosSlots.add(slot);
  } else {
    const { gl } = webglOverlay;
    gl.bindBuffer(gl.ARRAY_BUFFER, webglOverlay.positionBuffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, basePos * 4, webglOverlay.positionsArray.subarray(basePos, basePos + 4));
  }
}

function overlaySetSlotVisibility(slot, rgba, visible) {
  if (!webglOverlay || !webglOverlay.enabled || slot < 0) return;
  const baseCol = slot * 8;
  const a = visible ? (rgba && rgba.length ? rgba[3] : AFFINITY_LINE_ALPHA) : 0.0;
  const r = rgba && rgba.length ? rgba[0] : 1.0;
  const g = rgba && rgba.length ? rgba[1] : 1.0;
  const b = rgba && rgba.length ? rgba[2] : 1.0;
  const rByte = Math.max(0, Math.min(255, Math.round(r * 255)));
  const gByte = Math.max(0, Math.min(255, Math.round(g * 255)));
  const bByte = Math.max(0, Math.min(255, Math.round(b * 255)));
  const aByte = Math.max(0, Math.min(255, Math.round(a * 255)));
  // Two vertices per edge
  webglOverlay.colorsArray[baseCol] = rByte;
  webglOverlay.colorsArray[baseCol + 1] = gByte;
  webglOverlay.colorsArray[baseCol + 2] = bByte;
  webglOverlay.colorsArray[baseCol + 3] = aByte;
  webglOverlay.colorsArray[baseCol + 4] = rByte;
  webglOverlay.colorsArray[baseCol + 5] = gByte;
  webglOverlay.colorsArray[baseCol + 6] = bByte;
  webglOverlay.colorsArray[baseCol + 7] = aByte;
  if (BATCH_LIVE_OVERLAY_UPDATES) {
    webglOverlay.dirtyColSlots.add(slot);
  } else {
    const { gl } = webglOverlay;
    gl.bindBuffer(gl.ARRAY_BUFFER, webglOverlay.colorBuffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, baseCol, webglOverlay.colorsArray.subarray(baseCol, baseCol + 8));
  }
}

function getSegmentRgba(segments, s) {
  const seg = segments[s];
  if (!seg) return [1, 1, 1, AFFINITY_LINE_ALPHA];
  if (!seg.rgba) {
    seg.rgba = parseCssColorToRgba(seg.color, AFFINITY_LINE_ALPHA);
  }
  return seg.rgba;
}

function createWebglProgram(gl, vertexSource, fragmentSource) {
  const vertexShader = compileWebglShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fragmentShader = compileWebglShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  if (!vertexShader || !fragmentShader) {
    return null;
  }
  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.warn('WebGL program link failed:', gl.getProgramInfoLog(program));
    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);
    gl.deleteProgram(program);
    return null;
  }
  gl.detachShader(program, vertexShader);
  gl.detachShader(program, fragmentShader);
  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);
  return program;
}

function compileWebglShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.warn('WebGL shader compile failed:', gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}


function resizePointsOverlay() {
  if (!pointsOverlay || !pointsOverlay.enabled || !pointsOverlay.canvas) {
    return;
  }
  const v = getViewportSize();
  const renderWidth = Math.max(1, v.width);
  pointsOverlay.canvas.width = Math.max(1, Math.round(renderWidth * dpr));
  pointsOverlay.canvas.height = Math.max(1, Math.round(v.height * dpr));
  pointsOverlay.canvas.style.width = renderWidth + 'px';
  pointsOverlay.canvas.style.height = v.height + 'px';
}

function resizeWebglOverlay() {
  if (!webglOverlay || !webglOverlay.enabled) {
    return;
  }
  if (webglOverlay.shared) {
    return;
  }
  const { canvas: glCanvas } = webglOverlay;
  glCanvas.width = canvas.width;
  glCanvas.height = canvas.height;
  glCanvas.style.width = canvas.style.width;
  glCanvas.style.height = canvas.style.height;
  glCanvas.style.transform = canvas.style.transform || '';
  ensureOverlayTargets(glCanvas.width, glCanvas.height);
}

// Overlay fills the viewer; no additional positioning function needed

// Minimal CSS color parser to rgba floats
function parseCssColorToRgba(color, fallbackAlpha = AFFINITY_LINE_ALPHA) {
  if (typeof color !== 'string') {
    return [1, 1, 1, fallbackAlpha];
  }
  const s = color.trim().toLowerCase();
  // rgba(r,g,b,a) or rgb(r,g,b)
  let m = s.match(/^rgba?\(([^)]+)\)$/i);
  if (m) {
    const parts = m[1].split(',').map((x) => x.trim());
    const r = Math.max(0, Math.min(255, parseFloat(parts[0])));
    const g = Math.max(0, Math.min(255, parseFloat(parts[1])));
    const b = Math.max(0, Math.min(255, parseFloat(parts[2])));
    const a = parts[3] !== undefined ? Math.max(0, Math.min(1, parseFloat(parts[3]))) : fallbackAlpha;
    return [r / 255, g / 255, b / 255, a];
  }
  // #rgb, #rrggbb
  if (s[0] === '#') {
    const hex = s.slice(1);
    if (hex.length === 3) {
      const r = parseInt(hex[0] + hex[0], 16);
      const g = parseInt(hex[1] + hex[1], 16);
      const b = parseInt(hex[2] + hex[2], 16);
      return [r / 255, g / 255, b / 255, fallbackAlpha];
    }
    if (hex.length === 6) {
      const r = parseInt(hex.slice(0, 2), 16);
      const g = parseInt(hex.slice(2, 4), 16);
      const b = parseInt(hex.slice(4, 6), 16);
      return [r / 255, g / 255, b / 255, fallbackAlpha];
    }
  }
  return [1, 1, 1, fallbackAlpha];
}

// Rebuild geometry to include only active edges from segments to avoid huge buffers
function ensureWebglGeometry(width, height) {
  if (!webglOverlay || !webglOverlay.enabled) {
    return;
  }
  const info = affinityGraphInfo;
  const segments = info && info.segments ? info.segments : null;
  if (!segments) {
    // Clear buffers
    const { gl, positionBuffer, colorBuffer } = webglOverlay;
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(), gl.DYNAMIC_DRAW);
    webglOverlay.positionsArray = null;
    webglOverlay.colorsArray = null;
    webglOverlay.edgeCount = 0;
    webglOverlay.vertexCount = 0;
    webglOverlay.width = width | 0;
    webglOverlay.height = height | 0;
    webglOverlay.needsGeometryRebuild = false;
    return;
  }
  // Build slot-based geometry and establish slot mapping per segment for live updates
  let totalEdges = 0;
  for (let s = 0; s < segments.length; s += 1) {
    const seg = segments[s];
    if (seg && seg.indices && seg.indices.size) {
      totalEdges += seg.indices.size;
    }
  }
  const { gl } = webglOverlay;
  overlayEnsureCapacity(totalEdges);
  webglOverlay.width = width | 0;
  webglOverlay.height = height | 0;
  webglOverlay.nextSlot = 0;
  webglOverlay.maxUsedSlotIndex = -1;
  webglOverlay.freeSlots.length = 0;
  // Reset segment slot maps
  for (let s = 0; s < segments.length; s += 1) {
    const seg = segments[s];
    if (seg) {
      seg.slots = new Map();
      seg.rgba = parseCssColorToRgba(seg.color, AFFINITY_LINE_ALPHA);
    }
  }
  let cursor = 0;
  for (let s = 0; s < segments.length; s += 1) {
    const seg = segments[s];
    const indices = seg && seg.indices;
    if (!seg || !indices || indices.size === 0) continue;
    const [dyRaw, dxRaw] = affinitySteps[s];
    const dx = dxRaw | 0;
    const dy = dyRaw | 0;
    const rgba = seg.rgba || parseCssColorToRgba(seg.color, AFFINITY_LINE_ALPHA);
    for (const index of indices) {
      const slot = cursor;
      const x = index % width;
      const y = (index / width) | 0;
      const cx1 = x + 0.5;
      const cy1 = y + 0.5;
      const cx2 = x + dx + 0.5;
      const cy2 = y + dy + 0.5;
      // write positions
      const basePos = slot * 4;
      webglOverlay.positionsArray[basePos] = cx1;
      webglOverlay.positionsArray[basePos + 1] = cy1;
      webglOverlay.positionsArray[basePos + 2] = cx2;
      webglOverlay.positionsArray[basePos + 3] = cy2;
      // write colors
      const r = Math.max(0, Math.min(255, Math.round(rgba[0] * 255)));
      const g = Math.max(0, Math.min(255, Math.round(rgba[1] * 255)));
      const b = Math.max(0, Math.min(255, Math.round(rgba[2] * 255)));
      const aByte = Math.max(0, Math.min(255, Math.round(rgba[3] * 255)));
      const baseCol = slot * 8;
      webglOverlay.colorsArray[baseCol] = r;
      webglOverlay.colorsArray[baseCol + 1] = g;
      webglOverlay.colorsArray[baseCol + 2] = b;
      webglOverlay.colorsArray[baseCol + 3] = aByte;
      webglOverlay.colorsArray[baseCol + 4] = r;
      webglOverlay.colorsArray[baseCol + 5] = g;
      webglOverlay.colorsArray[baseCol + 6] = b;
      webglOverlay.colorsArray[baseCol + 7] = aByte;
      seg.slots.set(index, slot);
      cursor += 1;
    }
  }
  // Upload full buffers
  gl.bindBuffer(gl.ARRAY_BUFFER, webglOverlay.positionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, webglOverlay.positionsArray, gl.DYNAMIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, webglOverlay.colorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, webglOverlay.colorsArray, gl.DYNAMIC_DRAW);
  webglOverlay.edgeCount = totalEdges;
  webglOverlay.vertexCount = webglOverlay.capacityEdges * 2; // capacity vertices; invisible lines have alpha 0
  webglOverlay.nextSlot = totalEdges;
  webglOverlay.maxUsedSlotIndex = totalEdges - 1;
  webglOverlay.needsGeometryRebuild = false;
}

function syncWebglColors() {
  // Geometry now includes only active edges, with fixed per-segment colors.
  // Color sync is implicit in ensureWebglGeometry/rebuild.
  if (!webglOverlay || !webglOverlay.enabled || !affinityGraphInfo) {
    return;
  }
  ensureWebglGeometry(affinityGraphInfo.width, affinityGraphInfo.height);
}

// No color delta uploads are needed; colors are baked into geometry buffers

function computeWebglMatrix(out, width, height) {
  // Compose clip transform with the exact view transform (scale, rotate, translate) in device pixels.
  // Image -> Device: M = [ a c e; b d f; 0 0 1 ] where
  // a = s*cos, b = s*sin, c = -s*sin, d = s*cos, e = offsetX*dpr, f = offsetY*dpr
  const cosR = Math.cos(viewState.rotation);
  const sinR = Math.sin(viewState.rotation);
  const s = viewState.scale * (dpr || 1);
  const a = s * cosR;
  const b = s * sinR;
  const c = -s * sinR;
  const d = s * cosR;
  const e = viewState.offsetX * (dpr || 1);
  const f = viewState.offsetY * (dpr || 1);
  // Device -> NDC clip transform C (row-major):
  // [ 2/W  0  -1 ]
  // [ 0  -2/H  1 ]
  // [ 0    0   1 ]
  // U = C * M. We output column-major for WebGL.
  const invW2 = 2 / (width || 1);
  const invH2 = 2 / (height || 1);
  const U00 = invW2 * a;
  const U01 = invW2 * c;
  const U02 = invW2 * e - 1;
  const U10 = -invH2 * b;
  const U11 = -invH2 * d;
  const U12 = -invH2 * f + 1;
  const U20 = 0;
  const U21 = 0;
  const U22 = 1;
  // Column-major order
  out[0] = U00; out[1] = U10; out[2] = U20;
  out[3] = U01; out[4] = U11; out[5] = U21;
  out[6] = U02; out[7] = U12; out[8] = U22;
  return out;
}


function drawPointsOverlay(matrix) {
  if (!pointsOverlay || !pointsOverlay.enabled) {
    return;
  }
  const showPoints = showPointsOverlay && pointsOverlay.pointsCount > 0;
  if (!showPoints) {
    if (pointsOverlay.canvas) {
      pointsOverlay.canvas.style.display = 'none';
      pointsOverlay.canvas.style.opacity = '0';
    }
    return;
  }
  const gl = pointsOverlay.gl;
  const canvasEl = pointsOverlay.canvas;
  if (!gl || !canvasEl) {
    return;
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, canvasEl.width, canvasEl.height);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.disable(gl.DEPTH_TEST);
  gl.useProgram(pointsOverlay.program);
  gl.uniformMatrix3fv(pointsOverlay.uniforms.matrix, false, matrix);
  if (pointsOverlay.uniforms.alpha) {
    gl.uniform1f(pointsOverlay.uniforms.alpha, POINTS_OVERLAY_ALPHA);
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, pointsOverlay.positionBuffer);
  gl.enableVertexAttribArray(pointsOverlay.attribs.position);
  gl.vertexAttribPointer(pointsOverlay.attribs.position, 2, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, pointsOverlay.colorBuffer);
  gl.enableVertexAttribArray(pointsOverlay.attribs.color);
  gl.vertexAttribPointer(pointsOverlay.attribs.color, 4, gl.UNSIGNED_BYTE, true, 0, 0);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
  gl.drawArrays(gl.POINTS, 0, pointsOverlay.pointsCount);
  gl.disableVertexAttribArray(pointsOverlay.attribs.position);
  gl.disableVertexAttribArray(pointsOverlay.attribs.color);
  gl.disable(gl.BLEND);
  if (canvasEl.style.display !== 'block') {
    canvasEl.style.display = 'block';
  }
  canvasEl.style.opacity = '1';
}

function drawAffinityGraphWebgl() {
  if (!ensureWebglOverlayReady() || !webglOverlay || !webglOverlay.enabled) {
    return false;
  }
  const showLines = showAffinityGraph && affinityGraphInfo && affinityGraphInfo.values;
  const showPoints = false;
  if (webglOverlay.shared) {
    const matrix = computeWebglMatrix(webglOverlay.matrixCache, canvas.width, canvas.height);
    drawAffinityGraphShared(matrix);
    return true;
  }
  const {
    gl, canvas: glCanvas, lineProgram, lineAttribs, lineUniforms, pointProgram, pointAttribs, pointUniforms, matrixCache, msaa, resolve,
  } = webglOverlay;
  ensureOverlayTargets(glCanvas.width, glCanvas.height);
  if (!webglOverlay.enabled || !webglOverlay.targetsReady) {
    finalizeOverlayFrame(false);
    return true;
  }
  const drawFramebuffer = (webglOverlay.useMsaa && msaa && msaa.framebuffer && msaa.samples > 1)
    ? msaa.framebuffer
    : (resolve && resolve.framebuffer ? resolve.framebuffer : null);
  gl.bindFramebuffer(gl.FRAMEBUFFER, drawFramebuffer);
  gl.viewport(0, 0, glCanvas.width, glCanvas.height);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  let hasContent = false;
  if (!showLines && !showPoints) {
    if (webglOverlay) {
      webglOverlay.displayAlpha = 0;
    }
    finalizeOverlayFrame(false);
    return true;
  }
  // Build geometry if marked dirty, or if buffers are empty/mismatched
  let mustBuild = showLines && (
    Boolean(webglOverlay.needsGeometryRebuild)
    || !webglOverlay.positionsArray
    || !Number.isFinite(webglOverlay.vertexCount) || webglOverlay.vertexCount == 0
    || webglOverlay.width != (affinityGraphInfo.width | 0)
    || webglOverlay.height != (affinityGraphInfo.height | 0)
  );
  if (mustBuild) {
    const shouldDefer = DEFER_AFFINITY_OVERLAY_DURING_PAINT && isPainting;
    if (!shouldDefer) {
      ensureWebglGeometry(affinityGraphInfo.width, affinityGraphInfo.height);
    }
  }
  if (showLines && (!webglOverlay.positionsArray || !webglOverlay.positionsArray.length)) {
    if (!showPoints) {
      webglOverlay.displayAlpha = 0;
      finalizeOverlayFrame(false);
      return true;
    }
  }
  gl.useProgram(lineProgram);
  const matrix = computeWebglMatrix(matrixCache, glCanvas.width, glCanvas.height);
  gl.uniformMatrix3fv(lineUniforms.matrix, false, matrix);
  // Compute a global alpha based on the projected pixel size of the shortest affinity edge
  const s = Math.max(0.0001, Number(viewState && viewState.scale ? viewState.scale : 1.0));
  const minStep = minAffinityStepLength > 0 ? minAffinityStepLength : 1.0;
  const minEdgePx = Math.max(0, s * minStep);
  const dprSafe = Number.isFinite(dpr) && dpr > 0 ? dpr : 1;
  const cutoff = OVERLAY_PIXEL_FADE_CUTOFF * dprSafe;
  const t = cutoff <= 0 ? 1 : Math.max(0, Math.min(1, minEdgePx / cutoff));
  const alphaScale = t * t * (3 - 2 * t);
  const clampedAlpha = Math.max(0, Math.min(1, alphaScale));
  const scaledAlpha = clampedAlpha;
  webglOverlay.lineAlpha = clampedAlpha;
  webglOverlay.displayAlpha = clampedAlpha;
  if (DEBUG_AFFINITY && typeof window !== 'undefined') {
    const prev = Number.isFinite(window.__debugAlpha) ? window.__debugAlpha : NaN;
    if (!Number.isFinite(prev) || Math.abs(prev - clampedAlpha) > 0.01) {
      console.log('[affinity] alphaScale', clampedAlpha.toFixed(4), 'zoom', s.toFixed(3), 'minEdgePx', minEdgePx.toFixed(3));
      window.__debugAlpha = clampedAlpha;
    }
  }
  // Flush any batched slot updates
  if (BATCH_LIVE_OVERLAY_UPDATES) {
    if (webglOverlay.dirtyPosSlots && webglOverlay.dirtyPosSlots.size) {
      gl.bindBuffer(gl.ARRAY_BUFFER, webglOverlay.positionBuffer);
      for (const slot of webglOverlay.dirtyPosSlots) {
        const basePos = slot * 4;
        gl.bufferSubData(gl.ARRAY_BUFFER, basePos * 4, webglOverlay.positionsArray.subarray(basePos, basePos + 4));
      }
      webglOverlay.dirtyPosSlots.clear();
    }
    if (webglOverlay.dirtyColSlots && webglOverlay.dirtyColSlots.size) {
      gl.bindBuffer(gl.ARRAY_BUFFER, webglOverlay.colorBuffer);
      for (const slot of webglOverlay.dirtyColSlots) {
        const baseCol = slot * 8;
        gl.bufferSubData(gl.ARRAY_BUFFER, baseCol, webglOverlay.colorsArray.subarray(baseCol, baseCol + 8));
      }
      webglOverlay.dirtyColSlots.clear();
    }
  }
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  const edgesToDraw = Math.max(webglOverlay.edgeCount | 0, (webglOverlay.maxUsedSlotIndex | 0) + 1);
  const verticesToDraw = Math.max(0, edgesToDraw) * 2;
  if (showLines && verticesToDraw > 0) {
    gl.bindBuffer(gl.ARRAY_BUFFER, webglOverlay.positionBuffer);
    gl.enableVertexAttribArray(lineAttribs.position);
    gl.vertexAttribPointer(lineAttribs.position, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, webglOverlay.colorBuffer);
    gl.enableVertexAttribArray(lineAttribs.color);
    gl.vertexAttribPointer(lineAttribs.color, 4, gl.UNSIGNED_BYTE, true, 0, 0);
    hasContent = true;
    gl.drawArrays(gl.LINES, 0, verticesToDraw);
  }
  gl.disableVertexAttribArray(lineAttribs.position);
  gl.disableVertexAttribArray(lineAttribs.color);
  gl.disable(gl.BLEND);
  finalizeOverlayFrame(hasContent);
  return true;
}


// Affinity graph rendering is handled exclusively by WebGL.

function clearAffinityGraphData() {
  resetAffinityUpdateQueue();
  affinityGraphInfo = null;
  affinityGraphNeedsLocalRebuild = true;
  affinityGraphSource = 'none';
  savedAffinityGraphPayload = null;
  affinitySteps = DEFAULT_AFFINITY_STEPS.map((step) => step.slice());
  refreshOppositeStepMapping();
  clearOutline();
  if (webglOverlay && webglOverlay.enabled) {
    const { gl, positionBuffer, colorBuffer, canvas: glCanvas } = webglOverlay;
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(), gl.DYNAMIC_DRAW);
    webglOverlay.positionsArray = null;
    webglOverlay.colorsArray = null;
    webglOverlay.edgeCount = 0;
    webglOverlay.vertexCount = 0;
    webglOverlay.capacityEdges = 0;
    webglOverlay.nextSlot = 0;
    webglOverlay.maxUsedSlotIndex = -1;
    if (webglOverlay.freeSlots) webglOverlay.freeSlots.length = 0;
    webglOverlay.needsGeometryRebuild = true;
  }
  clearWebglOverlaySurface();
  if (pointsOverlay && pointsOverlay.enabled) {
    const gl = pointsOverlay.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, pointsOverlay.positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, pointsOverlay.colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(), gl.STATIC_DRAW);
    pointsOverlay.pointsCount = 0;
  }
  markAffinityGeometryDirty();
}

let affinityRebuildScheduled = false;

function sampledHasNonZero(values) {
  if (!values || values.length === 0) {
    return false;
  }
  const step = Math.max(1, Math.floor(values.length / 4096));
  for (let i = 0; i < values.length; i += step) {
    if ((values[i] | 0) !== 0) {
      return true;
    }
  }
  return false;
}

function scheduleAffinityRebuildIfStale(reason) {
  if (affinityGraphSource === 'remote' && affinityGraphInfo && affinityGraphInfo.values) {
    return;
  }
  if (affinityRebuildScheduled) {
    return;
  }
  affinityRebuildScheduled = true;
  const schedule = typeof requestAnimationFrame === 'function'
    ? requestAnimationFrame
    : (cb) => setTimeout(cb, 0);
  schedule(() => {
    affinityRebuildScheduled = false;
    if (!maskValues || !maskValues.length) {
      return;
    }
    if (affinityGraphSource === 'remote') {
      return;
    }
    if (!showAffinityGraph && !affinitySegEnabled && !outlinesVisible) {
      return;
    }
    let maskNonZero = maskHasNonZero;
    if (!maskNonZero) {
      maskNonZero = sampledHasNonZero(maskValues);
    }
    if (!maskNonZero) {
      return;
    }
    const lastBuild = typeof window !== 'undefined' && window.__OMNI_DEBUG__
      ? window.__OMNI_DEBUG__.lastLocalAffinityBuild
      : null;
    const localGraphEmpty = Boolean(lastBuild && lastBuild.hasValues && lastBuild.nonZero === false);
    if (!localGraphEmpty && affinityGraphInfo && affinityGraphInfo.values) {
      if (sampledHasNonZero(affinityGraphInfo.values)) {
        return;
      }
    }
    affinityGraphNeedsLocalRebuild = true;
    try {
      rebuildLocalAffinityGraph();
    } catch (err) {
      log('affinity rebuild after stale graph (' + reason + ') failed', err);
    }
  });
}

function maybeApplySavedAffinityGraph() {
  if (!affinitySegEnabled || !savedAffinityGraphPayload || !savedAffinityGraphPayload.encoded) {
    return false;
  }
  applyAffinityGraphPayload(savedAffinityGraphPayload);
  affinityGraphSource = 'remote';
  affinityGraphNeedsLocalRebuild = false;
  return true;
}

function applyAffinityGraphPayload(payload) {
  resetAffinityUpdateQueue();
  // If the backend did not provide an affinity graph, fall back to local
  if (!payload || !payload.encoded || !payload.steps || !payload.steps.length) {
    if (affinityGraphInfo && affinityGraphInfo.values) {
      // Keep existing remote graph; just ensure overlays are rebuilt.
      if (showAffinityGraph) {
        buildAffinityGraphSegments();
      }
      rebuildOutlineFromAffinity();
      return;
    }
    clearAffinityGraphData();
    rebuildLocalAffinityGraph();
    if (webglOverlay && webglOverlay.enabled) {
      webglOverlay.needsGeometryRebuild = true;
    }
    if (showAffinityGraph && affinityGraphInfo && affinityGraphInfo.values) {
      buildAffinityGraphSegments();
    }
    if (affinityGraphInfo && affinityGraphInfo.values) {
      rebuildOutlineFromAffinity();
    }
    return;
  }
  const width = Number(payload.width) || imgWidth;
  const height = Number(payload.height) || imgHeight;
  const stepsInput = Array.isArray(payload.steps) ? payload.steps : [];
  if (!stepsInput.length) {
    clearAffinityGraphData();
    if (showAffinityGraph) {
      affinityGraphNeedsLocalRebuild = true;
    }
    return;
  }
  affinitySteps = stepsInput.map((pair) => {
    if (!Array.isArray(pair) || pair.length < 2) {
      return [0, 0];
    }
    return [pair[0] | 0, pair[1] | 0];
  });
  refreshOppositeStepMapping();
  if (ensureWebglOverlayReady()) {
    resizeWebglOverlay();
    if (webglOverlay) {
      webglOverlay.needsGeometryRebuild = true;
    }
  }
  const values = decodeBase64ToUint8(payload.encoded);
  const planeStride = width * height;
  const expectedLength = affinitySteps.length * planeStride;
  if (values.length !== expectedLength) {
    console.warn('affinityGraph payload length mismatch', values.length, expectedLength);
    clearAffinityGraphData();
    if (showAffinityGraph) {
      affinityGraphNeedsLocalRebuild = true;
    }
    return;
  }
  affinityGraphInfo = {
    width,
    height,
    values,
    stepCount: affinitySteps.length,
    segments: null,
  };
  if (webglOverlay && webglOverlay.enabled) {
    webglOverlay.needsGeometryRebuild = true;
  }
  buildAffinityGraphSegments();
  rebuildOutlineFromAffinity();
  affinityGraphNeedsLocalRebuild = false;
  affinityGraphSource = 'remote';
}

function rebuildLocalAffinityGraph() {
  resetAffinityUpdateQueue();
  refreshOppositeStepMapping();
  // Use maskValues directly - this is the current editing state
  const sourceMask = maskValues;
  if (!sourceMask || sourceMask.length === 0) {
    clearAffinityGraphData();
    affinityGraphSource = 'local';
    affinityGraphNeedsLocalRebuild = false;
    return;
  }
  const width = imgWidth;
  const height = imgHeight;
  const stepCount = affinitySteps.length;
  const planeStride = width * height;
  if (stepCount === 0) {
    affinityGraphInfo = null;
    affinityGraphNeedsLocalRebuild = false;
    affinityGraphSource = 'local';
    return;
  }
  const values = new Uint8Array(stepCount * planeStride);
  // Always compute affinity based on raw instance labels (mode-invariant)
  for (let index = 0; index < planeStride; index += 1) {
    const rawLabel = (sourceMask[index] | 0);
    if (rawLabel <= 0) {
      continue;
    }
    const x = index % width;
    const y = (index / width) | 0;
    for (let s = 0; s < stepCount; s += 1) {
      const [dyRaw, dxRaw] = affinitySteps[s];
      const dx = dxRaw | 0;
      const dy = dyRaw | 0;
      const nx = x + dx;
      const ny = y + dy;
      const planeOffset = s * planeStride + index;
      let value = 0;
      if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
        const neighborIndex = ny * width + nx;
        const neighborRaw = (sourceMask[neighborIndex] | 0);
        if (neighborRaw > 0 && neighborRaw === rawLabel) {
          value = 1;
        }
        const oppIdx = affinityOppositeSteps[s];
        if (value && oppIdx >= 0) {
          values[oppIdx * planeStride + neighborIndex] = 1;
        }
      }
      values[planeOffset] = value;
    }
  }
  affinityGraphInfo = {
    width,
    height,
    values,
    stepCount,
    segments: null,
  };
  buildAffinityGraphSegments();
  rebuildOutlineFromAffinity();
  if (typeof window !== 'undefined' && window.__OMNI_DEBUG__) {
    window.__OMNI_DEBUG__.lastLocalAffinityBuild = {
      stepCount: affinitySteps.length | 0,
      hasValues: Boolean(values && values.length),
      nonZero: values ? values.some((v) => v !== 0) : false,
    };
  }
  if (webglOverlay && webglOverlay.enabled) {
    webglOverlay.needsGeometryRebuild = true;
  }
  affinityGraphNeedsLocalRebuild = false;
  affinityGraphSource = 'local';
  markAffinityGeometryDirty();
}

function buildAffinityGraphSegments() {
  if (!affinityGraphInfo) {
    return;
  }
  const { width, height, values, stepCount } = affinityGraphInfo;
  if (!values || values.length === 0 || stepCount === 0) {
    affinityGraphInfo.segments = null;
    affinityGraphInfo.path = null;
    affinityGraphInfo.vertices = null;
    markAffinityGeometryDirty();
    return;
  }
  const planeStride = width * height;
  const segments = new Array(stepCount);
  let totalEdges = 0;
  let path = null;
  const canUsePath = typeof Path2D === 'function';
  if (canUsePath) {
    try {
      path = new Path2D();
    } catch (err) {
      path = null;
    }
  }
  for (let s = 0; s < stepCount; s += 1) {
    const [dyRaw, dxRaw] = affinitySteps[s];
    const dy = dyRaw | 0;
    const dx = dxRaw | 0;
    const planeOffset = s * planeStride;
    const indices = new Set();
    for (let idx = 0; idx < planeStride; idx += 1) {
      if (!values[planeOffset + idx]) {
        continue;
      }
      const x = idx % width;
      const y = (idx / width) | 0;
      const nx = x + dx;
      const ny = y + dy;
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
        continue;
      }
      indices.add(idx);
      totalEdges += 1;
      if (path) {
        path.moveTo(x + 0.5, y + 0.5);
        path.lineTo(nx + 0.5, ny + 0.5);
      }
    }
    segments[s] = {
      color: AFFINITY_OVERLAY_COLOR,
      indices,
    };
  }
  const overlayActiveForGeometry = Boolean(USE_WEBGL_OVERLAY && webglOverlay && webglOverlay.enabled);
  let vertices = null;
  if (totalEdges > 0 && !overlayActiveForGeometry) {
    vertices = new Float32Array(totalEdges * 4);
    let offset = 0;
    for (let i = 0; i < segments.length; i += 1) {
      const seg = segments[i];
      if (!seg || !seg.map) continue;
      seg.map.forEach((coords) => {
        vertices[offset + 0] = coords[0];
        vertices[offset + 1] = coords[1];
        vertices[offset + 2] = coords[2];
        vertices[offset + 3] = coords[3];
        offset += 4;
      });
    }
  }
  affinityGraphInfo.segments = segments;
  affinityGraphInfo.path = path || null;
  affinityGraphInfo.vertices = vertices;
  affinityGeometryDirty = true;
}

function clearOutline() {
  outlineState.fill(0);
  if (isWebglPipelineActive()) {
    markOutlineTextureFullDirty();
  }
}

function setOutlinePixel(index, isBoundary) {
  outlineState[index] = isBoundary ? 1 : 0;
}

function updateOutlineForIndex(index) {
  // Outline from affinity graph with off-screen continuation:
  // If the pixel's label continues off the image edge, treat that step as connected.
  if (!affinityGraphInfo || !affinityGraphInfo.values || !affinityGraphInfo.stepCount) {
    setOutlinePixel(index, false);
    return;
  }
  const { values, width, height, stepCount } = affinityGraphInfo;
  if (index < 0 || index >= (width * height)) {
    setOutlinePixel(index, false);
    return;
  }
  // Determine outline presence using raw label connectivity from affinity values.
  const label = (maskValues[index] | 0);
  if (label <= 0) {
    setOutlinePixel(index, false);
    return;
  }
  const x = index % width;
  const y = (index / width) | 0;
  const planeStride = width * height;
  let connected = 0;
  for (let s = 0; s < stepCount; s += 1) {
    const [dyRaw, dxRaw] = affinitySteps[s];
    const dx = dxRaw | 0;
    const dy = dyRaw | 0;
    const nx = x + dx;
    const ny = y + dy;
    if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
      // Off-image: assume continuation for foreground labels
      connected += 1;
      continue;
    }
    if (values[s * planeStride + index]) {
      connected += 1;
    }
  }
  const isBoundary = connected > 0 && connected < stepCount;
  setOutlinePixel(index, isBoundary);
}

function updateOutlineForIndices(indicesSet) {
  if (!indicesSet || !indicesSet.size) {
    return;
  }
  const width = imgWidth | 0;
  const height = imgHeight | 0;
  const planeStride = width * height;
  // Expand the update set by a 1-pixel halo to avoid leaving stale
  // pixels along stair-step edges where connectivity flips at neighbors.
  const expanded = new Set();
  const push = (i) => { if (i >= 0 && i < planeStride) expanded.add(i | 0); };
  indicesSet.forEach((idx) => {
    idx = idx | 0;
    if (idx < 0 || idx >= planeStride) return;
    const x = idx % width;
    const y = (idx / width) | 0;
    for (let dy = -1; dy <= 1; dy += 1) {
      for (let dx = -1; dx <= 1; dx += 1) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
        push(ny * width + nx);
      }
    }
  });
  let changed = false;
  expanded.forEach((idx) => {
    const before = outlineState[idx];
    updateOutlineForIndex(idx);
    if (outlineState[idx] !== before) changed = true;
  });
  if (changed) {
    if (isWebglPipelineActive()) {
      const rectIndices = Array.from(expanded);
      markOutlineIndicesDirty(rectIndices);
    } else {
      redrawMaskCanvas();
    }
  }
}

function rebuildOutlineFromAffinity() {
  clearOutline();
  if (!affinityGraphInfo || !affinityGraphInfo.values) {
    return;
  }
  for (let i = 0; i < maskValues.length; i += 1) {
    updateOutlineForIndex(i);
  }
  // After rebuilding outline bitmap, refresh mask rendering
  if (isWebglPipelineActive()) {
    markOutlineTextureFullDirty();
  } else {
    redrawMaskCanvas();
  }
}

function updateAffinityGraphForIndices(indices) {
  const needsAffinity = showAffinityGraph || affinitySegEnabled || outlinesVisible;
  if (!needsAffinity) {
    affinityGraphNeedsLocalRebuild = true;
    return;
  }
  if (!indices || !indices.length) {
    rebuildLocalAffinityGraph();
    return;
  }
  const updateStart = typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
  debugAffinity(`[affinity] incremental update start (indices=${indices.length})`);
  if (!affinityGraphInfo || !affinityGraphInfo.values || !affinityGraphInfo.stepCount) {
    // Initialize the local affinity graph from the current mask so we can incrementally update it
    rebuildLocalAffinityGraph();
    if (!affinityGraphInfo || !affinityGraphInfo.values || !affinityGraphInfo.stepCount) {
      return;
    }
  }
  const info = affinityGraphInfo;
  // Use maskValues directly - this is the current editing state
  const sourceMask = maskValues;
  if (!sourceMask || sourceMask.length === 0) {
    return;
  }
  if (showAffinityGraph && !info.segments) {
    buildAffinityGraphSegments();
    if (!info.segments) {
      return;
    }
  }
  const width = info.width;
  const height = info.height;
  const stepCount = info.stepCount;
  if (width <= 0 || height <= 0 || stepCount <= 0) {
    return;
  }
  ensureWebglOverlayReady();
  const planeStride = width * height;
  if (indices.length === planeStride) {
    let sequential = true;
    for (let i = 0; i < planeStride; i += 1) {
      if ((indices[i] | 0) !== i) {
        sequential = false;
        break;
      }
    }
    if (sequential) {
      rebuildLocalAffinityGraph();
      return;
    }
  }
  ensureAffinityTouchedStorage(planeStride);
  let stamp = affinityTouchedStamp++;
  if (stamp === 0xffffffff) {
    affinityTouchedMask.fill(0);
    stamp = 1;
    affinityTouchedStamp = 2;
  }
  const touchedMask = affinityTouchedMask;
  const touchedList = affinityTouchedList;
  let touchedCount = 0;
  const pushIndex = (idx) => {
    if (idx < 0 || idx >= planeStride) return;
    if (touchedMask[idx] === stamp) return;
    touchedMask[idx] = stamp;
    touchedList[touchedCount++] = idx;
  };
  for (let i = 0; i < indices.length; i += 1) {
    const baseIndex = Number(indices[i]) | 0;
    if (baseIndex < 0 || baseIndex >= planeStride) continue;
    pushIndex(baseIndex);
    const x = baseIndex % width;
    const y = (baseIndex / width) | 0;
    for (let s = 0; s < stepCount; s += 1) {
      const [dyRaw, dxRaw] = affinitySteps[s];
      const dx = dxRaw | 0;
      const dy = dyRaw | 0;
      const nx = x + dx;
      const ny = y + dy;
      if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
        pushIndex(ny * width + nx);
      }
      const px = x - dx;
      const py = y - dy;
      if (px >= 0 && px < width && py >= 0 && py < height) {
        pushIndex(py * width + px);
      }
    }
  }
  if (touchedCount === 0) {
    return;
  }
  const { values } = info;
  const segments = showAffinityGraph ? info.segments : null;
  let outlineChanged = false;
  for (let listIdx = 0; listIdx < touchedCount; listIdx += 1) {
    const index = touchedList[listIdx];
    const label = (sourceMask[index] | 0);
    const x = index % width;
    const y = (index / width) | 0;
    for (let s = 0; s < stepCount; s += 1) {
      const [dyRaw, dxRaw] = affinitySteps[s];
      const dx = dxRaw | 0;
      const dy = dyRaw | 0;
      const nx = x + dx;
      const ny = y + dy;
      const planeOffset = s * planeStride + index;
      let value = 0;
      let neighborIndex = -1;
      if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
        neighborIndex = ny * width + nx;
        const neighborRaw = (sourceMask[neighborIndex] | 0);
        if (label > 0 && neighborRaw > 0 && neighborRaw === label) {
          value = 1;
        }
      }
      values[planeOffset] = value;
      // Maintain symmetric edge in the opposite step direction
      if (neighborIndex >= 0) {
        const oppIdx = affinityOppositeSteps[s] | 0;
        if (oppIdx >= 0) {
          values[oppIdx * planeStride + neighborIndex] = value;
        }
      }
      const segment = segments ? segments[s] : null;
      if (segment) {
        if (!segment.rgba) segment.rgba = parseCssColorToRgba(segment.color, AFFINITY_LINE_ALPHA);
        if (!segment.indices) segment.indices = new Set();
        if (neighborIndex >= 0) {
          const cx1 = x + 0.5;
          const cy1 = y + 0.5;
          const cx2 = (x + (dx | 0)) + 0.5;
          const cy2 = (y + (dy | 0)) + 0.5;
          if (value) {
            segment.indices.add(index);
          } else {
            segment.indices.delete(index);
          }
          if (webglOverlay && webglOverlay.enabled && LIVE_AFFINITY_OVERLAY_UPDATES) {
            if (!segment.slots) segment.slots = new Map();
            let slot = segment.slots.get(index);
            if (value) {
              if (slot === undefined) {
                slot = overlayAllocSlot();
                segment.slots.set(index, slot);
              }
              overlayWriteSlotPosition(slot, [cx1, cy1, cx2, cy2]);
              overlaySetSlotVisibility(slot, segment.rgba, true);
            } else if (slot !== undefined) {
              overlaySetSlotVisibility(slot, segment.rgba, false);
            }
          }
        }
      }
      // Mirror segment for the opposite step direction
      const oppIdx = affinityOppositeSteps[s] | 0;
      if (neighborIndex >= 0 && oppIdx >= 0) {
        const oppSegment = segments ? segments[oppIdx] : null;
        if (oppSegment) {
          if (!oppSegment.rgba) oppSegment.rgba = parseCssColorToRgba(oppSegment.color, AFFINITY_LINE_ALPHA);
          if (!oppSegment.indices) oppSegment.indices = new Set();
          const cx1o = (x + (dx | 0)) + 0.5;
          const cy1o = (y + (dy | 0)) + 0.5;
          const cx2o = x + 0.5;
          const cy2o = y + 0.5;
          if (value) {
            oppSegment.indices.add(neighborIndex);
          } else {
            oppSegment.indices.delete(neighborIndex);
          }
          if (webglOverlay && webglOverlay.enabled && LIVE_AFFINITY_OVERLAY_UPDATES) {
            if (!oppSegment.slots) oppSegment.slots = new Map();
            let slot2 = oppSegment.slots.get(neighborIndex);
            if (value) {
              if (slot2 === undefined) {
                slot2 = overlayAllocSlot();
                oppSegment.slots.set(neighborIndex, slot2);
              }
              overlayWriteSlotPosition(slot2, [cx1o, cy1o, cx2o, cy2o]);
              overlaySetSlotVisibility(slot2, oppSegment.rgba, true);
            } else if (slot2 !== undefined) {
              overlaySetSlotVisibility(slot2, oppSegment.rgba, false);
            }
          }
        }
      }
    }
  }
  // Recompute outline state for the touched region to avoid stale pixels
  // along stair-step edges and ensure outline stays in sync with mask writes.
  if (touchedCount > 0) {
    const indicesSet = new Set();
    for (let i = 0; i < touchedCount; i += 1) {
      indicesSet.add(touchedList[i]);
    }
    updateOutlineForIndices(indicesSet);
  }
  affinityGraphNeedsLocalRebuild = false;
  // Mark as locally modified since we've updated the affinity data
  affinityGraphSource = 'local';
  for (let listIdx = 0; listIdx < touchedCount; listIdx += 1) {
    const idx = touchedList[listIdx];
    const before = outlineState[idx];
    updateOutlineForIndex(idx);
    if (outlineState[idx] !== before) {
      outlineChanged = true;
    }
  }
  if (outlineChanged) {
    const slice = touchedList.subarray(0, touchedCount);
    if (isWebglPipelineActive()) {
      markOutlineIndicesDirty(slice);
    } else {
      redrawMaskCanvas();
    }
  }
  // Force affinity graph persistence to use the latest edited graph (always clear saved payload on edit)
  savedAffinityGraphPayload = null;
  scheduleStateSave();
  if (showAffinityGraph && (!webglOverlay || !webglOverlay.enabled || !LIVE_AFFINITY_OVERLAY_UPDATES)) {
    markAffinityGeometryDirty();
  }
  const updateEnd = typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
  DEBUG_COUNTERS.affinityUpdateLastMs = updateEnd - updateStart;
  if (DEBUG_AFFINITY) {
    debugAffinity(`[affinity] incremental update complete (touched=${touchedCount}, outlineChanged=${outlineChanged}, ${Math.round(updateEnd - updateStart)}ms)`);
  }
}
function markAffinityGraphStale() {
  affinityGraphInfo = null;
  affinityGraphNeedsLocalRebuild = true;
  affinityGraphSource = 'local';
  markAffinityGeometryDirty();
}

function shouldLogDraw() {
  if (drawLogCount < 20) {
    drawLogCount += 1;
    return true;
  }
  drawLogCount += 1;
  return drawLogCount % 50 === 0;
}

function applyViewTransform(context, { includeDpr = false } = {}) {
  const cos = Math.cos(viewState.rotation);
  const sin = Math.sin(viewState.rotation);
  const scaleFactor = viewState.scale * (includeDpr ? dpr : 1);
  const tx = viewState.offsetX * (includeDpr ? dpr : 1);
  const ty = viewState.offsetY * (includeDpr ? dpr : 1);
  context.setTransform(
    scaleFactor * cos,
    scaleFactor * sin,
    -scaleFactor * sin,
    scaleFactor * cos,
    tx,
    ty,
  );
}

function setOffsetForImagePoint(imagePoint, canvasPoint) {
  const cos = Math.cos(viewState.rotation);
  const sin = Math.sin(viewState.rotation);
  const scaledX = imagePoint.x * viewState.scale;
  const scaledY = imagePoint.y * viewState.scale;
  const rotatedX = scaledX * cos - scaledY * sin;
  const rotatedY = scaledX * sin + scaledY * cos;
  viewState.offsetX = canvasPoint.x - rotatedX;
  viewState.offsetY = canvasPoint.y - rotatedY;
  viewStateDirty = true;
}

function normalizeAngleDelta(delta) {
  if (!Number.isFinite(delta)) {
    return 0;
  }
  return Math.atan2(Math.sin(delta), Math.cos(delta));
}

function normalizeAngle(angle) {
  if (!Number.isFinite(angle)) {
    return 0;
  }
  return Math.atan2(Math.sin(angle), Math.cos(angle));
}

function suppressDoubleTapZoom(element) {
  if (!element || typeof element.addEventListener !== 'function') {
    return;
  }
  let lastTouchTime = 0;
  let lastTouchX = 0;
  let lastTouchY = 0;
  const reset = () => {
    lastTouchTime = 0;
    lastTouchX = 0;
    lastTouchY = 0;
  };
  element.addEventListener('touchend', (evt) => {
    if (!evt || (evt.touches && evt.touches.length > 0)) {
      return;
    }
    const changed = evt.changedTouches && evt.changedTouches[0];
    if (!changed) {
      return;
    }
    if (evt.target && typeof evt.target.closest === 'function') {
      const interactive = evt.target.closest('input, textarea, select, [contenteditable="true"]');
      if (interactive) {
        reset();
        return;
      }
    }
    const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
    const dt = now - lastTouchTime;
    const dx = changed.clientX - lastTouchX;
    const dy = changed.clientY - lastTouchY;
    const distanceSq = (dx * dx) + (dy * dy);
    if (Number.isFinite(dt) && dt > 0 && dt < 350 && distanceSq < 400) {
      evt.preventDefault();
      evt.stopPropagation();
      reset();
      return;
    }
    lastTouchTime = now;
    lastTouchX = changed.clientX;
    lastTouchY = changed.clientY;
  }, { passive: false });
}

function rotateView(deltaRadians) {
  if (!Number.isFinite(deltaRadians) || deltaRadians === 0) {
    return;
  }
  if (!canvas) {
    return;
  }
  if (!Number.isFinite(imgWidth) || !Number.isFinite(imgHeight) || imgWidth <= 0 || imgHeight <= 0) {
    return;
  }
  const rect = canvas.getBoundingClientRect();
  if (!rect || rect.width <= 0 || rect.height <= 0) {
    return;
  }
  const imageCenter = {
    x: imgWidth / 2,
    y: imgHeight / 2,
  };
  const anchorScreen = imageToScreen(imageCenter);
  viewState.rotation = normalizeAngle(viewState.rotation + deltaRadians);
  log('rotate view delta=' + (deltaRadians * RAD_TO_DEG).toFixed(1) + ' deg, now ' + (viewState.rotation * RAD_TO_DEG).toFixed(1) + ' deg');
  setOffsetForImagePoint(imageCenter, anchorScreen);
  userAdjustedScale = true;
  autoFitPending = false;
  viewStateRestored = false;
  draw();
  drawBrushPreview(getHoverPoint());
  viewStateDirty = true;
  scheduleStateSave();
  // Briefly show dot cursor for keyboard rotation
  setCursorTemporary(dotCursorCss, 700);
  // Show a simple dot cursor briefly
  setCursorTemporary(dotCursorCss, 600);
}

function scheduleDraw(options = {}) {
  if (shuttingDown) {
    return;
  }
  const opts = options || {};
  const immediate = Boolean(opts.immediate);
  if (immediate) {
    if (drawRequestHandle && typeof cancelAnimationFrame === 'function') {
      cancelAnimationFrame(drawRequestHandle);
    }
    drawRequestHandle = 0;
    drawRequestPending = false;
    draw();
    return;
  }
  if (drawRequestPending) {
    return;
  }
  if (typeof requestAnimationFrame !== 'function') {
    draw();
    return;
  }
  drawRequestPending = true;
  drawRequestHandle = requestAnimationFrame(() => {
    drawRequestPending = false;
    drawRequestHandle = 0;
    draw();
  });
}

const AFFINITY_UPDATE_BATCH_SIZE = 24000;
const AFFINITY_UPDATE_MAX_CHUNKS_PER_TICK = 3;
const affinityUpdateQueue = [];
let affinityBatchScheduled = false;
let affinityBatchHandle = null;
let affinityBatchQueuedTotal = 0;
let affinityBatchProcessedTotal = 0;
let affinityBatchChunkCounter = 0;
const idleScheduler = typeof requestIdleCallback === 'function'
  ? {
    request: (cb) => requestIdleCallback(cb, { timeout: 32 }),
    cancel: (handle) => cancelIdleCallback(handle),
  }
  : {
    request: (cb) => setTimeout(() => cb({ didTimeout: true, timeRemaining: () => 0 }), 16),
    cancel: (handle) => clearTimeout(handle),
  };

function enqueueAffinityIndexBatch(source, length) {
  if (!source || typeof length !== 'number') {
    return;
  }
  const total = Math.max(0, Math.min(source.length, length | 0));
  if (total === 0) {
    return;
  }
  const view = total === source.length ? source : source.subarray(0, total);
  if (total <= AFFINITY_UPDATE_BATCH_SIZE) {
    affinityUpdateQueue.push(view.subarray(0, total));
  } else {
    for (let offset = 0; offset < total; offset += AFFINITY_UPDATE_BATCH_SIZE) {
      const end = Math.min(total, offset + AFFINITY_UPDATE_BATCH_SIZE);
      affinityUpdateQueue.push(view.subarray(offset, end));
    }
  }
  affinityBatchQueuedTotal += total;
  debugAffinity(`[affinity] queued ${total} indices (${affinityUpdateQueue.length} chunks pending, total ${affinityBatchQueuedTotal})`);
  if (!affinityBatchScheduled) {
    affinityBatchScheduled = true;
    scheduleAffinityBatch();
  }
}

function scheduleAffinityBatch() {
  debugAffinity('[affinity] scheduling batch processing');
  affinityBatchHandle = idleScheduler.request(processAffinityBatch);
}

function resetAffinityUpdateQueue() {
  affinityUpdateQueue.length = 0;
  if (affinityBatchScheduled && affinityBatchHandle !== null) {
    idleScheduler.cancel(affinityBatchHandle);
  }
  affinityBatchHandle = null;
  affinityBatchScheduled = false;
  affinityBatchQueuedTotal = 0;
  affinityBatchProcessedTotal = 0;
  affinityBatchChunkCounter = 0;
}

function processAffinityBatch(deadline) {
  let processed = 0;
  const allowMore = () => {
    if (!deadline || typeof deadline.timeRemaining !== 'function') {
      return processed < AFFINITY_UPDATE_MAX_CHUNKS_PER_TICK;
    }
    return deadline.timeRemaining() > 1 || deadline.didTimeout || processed < AFFINITY_UPDATE_MAX_CHUNKS_PER_TICK;
  };
  while (affinityUpdateQueue.length && allowMore()) {
    const chunk = affinityUpdateQueue.shift();
    try {
      updateAffinityGraphForIndices(chunk);
    } catch (err) {
      console.warn('incremental affinity update failed', err);
    }
    affinityBatchProcessedTotal += chunk.length;
    affinityBatchChunkCounter += 1;
    debugAffinity(`[affinity] processed chunk ${affinityBatchChunkCounter} (${chunk.length} indices, remaining chunks ${affinityUpdateQueue.length})`);
    processed += 1;
  }
  if (affinityUpdateQueue.length) {
    scheduleAffinityBatch();
  } else {
    affinityBatchScheduled = false;
    affinityBatchHandle = null;
    debugAffinity(`[affinity] batch complete (processed ${affinityBatchProcessedTotal}/${affinityBatchQueuedTotal} indices across ${affinityBatchChunkCounter} chunks)`);
    scheduleDraw();
    drawBrushPreview(getHoverPoint());
  }
}

let affinityTouchedMask = null;
let affinityTouchedStamp = 1;
let affinityTouchedList = new Uint32Array(0);

function ensureAffinityTouchedStorage(length) {
  if (!affinityTouchedMask || affinityTouchedMask.length !== length) {
    affinityTouchedMask = new Uint32Array(length);
    affinityTouchedStamp = 1;
  }
  if (!affinityTouchedList || affinityTouchedList.length < length) {
    affinityTouchedList = new Uint32Array(length);
  }
}

function draw() {
  DEBUG_COUNTERS.draw += 1;
  if (shuttingDown) {
    return;
  }
  const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
  updateFps(now);
  lastDrawCompletedAt = now;
  if (shouldLogDraw()) {
    log('draw start scale=' + viewState.scale.toFixed(3) + ' offset=' + viewState.offsetX.toFixed(1) + ',' + viewState.offsetY.toFixed(1));
  }
  if (isWebglPipelineActive()) {
    // CRITICAL: Must update palette texture before drawing if dirty (e.g., after segmentation or mode change)
    updatePaletteTextureIfNeeded();
    if (needsMaskRedraw) {
      flushMaskTextureUpdates();
      needsMaskRedraw = false;
    }
    drawWebglFrame();
    if (!isPanning) {
      drawBrushPreview(getHoverPoint());
    }
    if (!loggedPixelSample && canvas.width > 0 && canvas.height > 0) {
      loggedPixelSample = true;
      hideLoadingOverlay();
    }
    return;
  }
  if (webglPipelineRequested) {
    if (!webglPipelineReady) {
      return;
    }
  }
  if (!webglUnavailableNotified) {
    webglUnavailableNotified = true;
    console.error('WebGL2 pipeline unavailable; OmniPose viewer requires WebGL2.');
    setLoadingOverlay('WebGL2 rendering unavailable. Please enable WebGL2 support and reload.', true);
  }
}

function base64FromUint8(bytes) {
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    const sub = bytes.subarray(i, i + chunk);
    binary += String.fromCharCode.apply(null, sub);
  }
  return btoa(binary);
}

// Recompute N-color mapping from the CURRENT client mask for display grouping.
function recomputeNColorFromCurrentMask(forceActive = false) {
  return new Promise((resolve) => {
    try {
      if (!nColorActive && !forceActive) { resolve(false); return; }
      const buf = maskValues.buffer.slice(maskValues.byteOffset, maskValues.byteOffset + maskValues.byteLength);
      const bytes = new Uint8Array(buf);
      const b64 = base64FromUint8(bytes);
      const payload = { mask: b64, width: imgWidth, height: imgHeight, expand: true };
      const applyMapping = (obj) => {
        try {
          if (!obj || !obj.nColorMask) { resolve(false); return; }
          const bin = atob(obj.nColorMask);
          if (bin.length !== maskValues.length * 4) { resolve(false); return; }
          const buffer = new ArrayBuffer(bin.length);
          const arr = new Uint8Array(buffer);
          for (let i = 0; i < bin.length; i += 1) arr[i] = bin.charCodeAt(i);
          const groups = new Uint32Array(buffer);
          if (maskValues && maskValues.length === groups.length) {
            const sampleMap = {};
            const missingLabels = new Set();
            let missing = 0;
            const sampleLimit = 10;
            for (let i = 0; i < groups.length; i += 1) {
              const label = maskValues[i] | 0;
              if (label > 0 && label <= sampleLimit && sampleMap[label] === undefined) {
                sampleMap[label] = groups[i] | 0;
              }
              if (label > 0 && groups[i] === 0) {
                missing += 1;
                if (missingLabels.size < sampleLimit) {
                  missingLabels.add(label);
                }
              }
            }
            console.warn('[ncolor] sample label->group:', sampleMap);
            if (missing > 0) {
              console.warn('[ncolor] group mapping returned 0 for labeled pixels:', missing, 'labels:', Array.from(missingLabels));
            }
          }
          // Only save instance mask if we don't already have one with more unique labels
          // (prevents overwriting valid instance data with N-color groups)
          const currentMaskUniqueCount = (() => {
            const s = new Set();
            for (let i = 0; i < maskValues.length; i++) {
              if (maskValues[i] > 0) s.add(maskValues[i]);
            }
            return s.size;
          })();
          const existingInstanceUniqueCount = nColorInstanceMask ? (() => {
            const s = new Set();
            for (let i = 0; i < nColorInstanceMask.length; i++) {
              if (nColorInstanceMask[i] > 0) s.add(nColorInstanceMask[i]);
            }
            return s.size;
          })() : 0;
          if (!nColorInstanceMask || nColorInstanceMask.length !== maskValues.length) {
            nColorInstanceMask = new Uint32Array(maskValues);
          } else if (currentMaskUniqueCount > existingInstanceUniqueCount) {
            // Only overwrite if current mask has MORE labels (true instance data)
            nColorInstanceMask.set(maskValues);
          }
          // else: keep existing instance mask (it has more labels, so it's the real instance data)
          let maxGroup = 0;
          for (let i = 0; i < groups.length; i += 1) {
            const g = groups[i] | 0;
            if (g > maxGroup) maxGroup = g;
          }
          if (maxGroup > 0) {
            const updatedPalette = ensureNColorPaletteLength(maxGroup);
            if (updatedPalette.length !== nColorPaletteColors.length) {
              setNColorPaletteColors(updatedPalette, { render: true, schedule: false });
            }
          }
          // Overwrite maskValues with group IDs; no separate overlay state
          let hasNonZero = false;
          for (let i = 0; i < groups.length; i += 1) {
            maskValues[i] = groups[i];
            if (groups[i] > 0) {
              hasNonZero = true;
            }
          }
          maskHasNonZero = hasNonZero;
          nColorValues = null;
          nColorActive = true;
          if (savedAffinityGraphPayload && savedAffinityGraphPayload.encoded) {
            applyAffinityGraphPayload(savedAffinityGraphPayload);
          }
          clearColorCaches();
          paletteTextureDirty = true;
          if (isWebglPipelineActive()) {
            markMaskTextureFullDirty();
            markOutlineTextureFullDirty();
          } else {
            redrawMaskCanvas();
          }
          // Do not modify the affinity graph when toggling N-color
          draw();
          resolve(true);
        } catch (e) { resolve(false); }
      };
      if (window.pywebview && window.pywebview.api && typeof window.pywebview.api.ncolor_from_mask === 'function') {
        window.pywebview.api.ncolor_from_mask(payload).then(applyMapping).catch(() => resolve(false));
        return;
      }
      if (typeof fetch === 'function') {
        fetch('/api/ncolor_from_mask', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
          .then((res) => (res.ok ? res.json() : Promise.reject(new Error('HTTP ' + res.status))))
          .then(applyMapping)
          .catch(() => resolve(false));
        return;
      }
      resolve(false);
    } catch (_) {
      resolve(false);
    }
  });
}

async function relabelFromAffinity() {
  const hasPywebview = Boolean(window.pywebview && window.pywebview.api && (window.pywebview.api.relabel_from_affinity));
  const canHttp = typeof fetch === 'function';
  if (!hasPywebview && !canHttp) return false;
  log(`[ncolor] relabel_from_affinity start hasPywebview=${hasPywebview} canHttp=${canHttp} source=${affinityGraphSource}`);
  try {
    // CRITICAL: When in N-color mode, maskValues contains N-color group IDs (1,2,3,4), NOT instance labels.
    // We must send the saved instance mask (nColorInstanceMask) so the backend can properly relabel using affinity.
    const sourceMask = (nColorActive && nColorInstanceMask && nColorInstanceMask.length === maskValues.length)
      ? nColorInstanceMask
      : maskValues;
    const buf = sourceMask.buffer.slice(sourceMask.byteOffset, sourceMask.byteOffset + sourceMask.byteLength);
    const bytes = new Uint8Array(buf);
    const b64 = base64FromUint8(bytes);
    // Attach current affinity graph (required); prefer remote graph when available
    let graphPayload = null;
    if (affinityGraphSource === 'remote' && affinityGraphInfo && affinityGraphInfo.values) {
      graphPayload = {
        width: affinityGraphInfo.width,
        height: affinityGraphInfo.height,
        steps: affinitySteps.map((p) => [p[0] | 0, p[1] | 0]),
        encoded: base64FromUint8(affinityGraphInfo.values),
      };
    } else if (savedAffinityGraphPayload && savedAffinityGraphPayload.encoded) {
      graphPayload = savedAffinityGraphPayload;
    } else if (affinityGraphInfo && affinityGraphInfo.values && affinityGraphInfo.stepCount) {
      graphPayload = {
        width: affinityGraphInfo.width,
        height: affinityGraphInfo.height,
        steps: affinitySteps.map((p) => [p[0] | 0, p[1] | 0]),
        encoded: base64FromUint8(affinityGraphInfo.values),
      };
    }
    if (!graphPayload || !graphPayload.encoded) {
      console.warn('No affinity graph available for relabel_from_affinity');
      log(`[ncolor] relabel_from_affinity missing graph values=${affinityGraphInfo && affinityGraphInfo.values ? affinityGraphInfo.values.length : 0} stepCount=${affinityGraphInfo ? affinityGraphInfo.stepCount : 0}`);
      return false;
    }
    const affinityGraph = {
      width: Number(graphPayload.width) || imgWidth,
      height: Number(graphPayload.height) || imgHeight,
      steps: Array.isArray(graphPayload.steps) ? graphPayload.steps : affinitySteps.map((p) => [p[0] | 0, p[1] | 0]),
      encoded: graphPayload.encoded,
    };
    const payload = { mask: b64, width: imgWidth, height: imgHeight, affinityGraph };
    if (hasPywebview) {
      const result = await window.pywebview.api.relabel_from_affinity(payload);
      if (result && result.error) {
        log(`[ncolor] relabel_from_affinity error (pywebview): ${result.error}`);
        return false;
      }
      if (result && result.mask) {
        applySegmentationMask(result, { forceInstanceMask: true });
        log('Applied relabel_from_affinity from backend (pywebview).');
        return true;
      }
      log(`[ncolor] relabel_from_affinity missing mask (pywebview); keys=${result ? Object.keys(result) : 'none'}`);
    }
    if (canHttp) {
      const res = await fetch('/api/relabel_from_affinity', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (res.ok) {
        const result = await res.json();
        if (result && result.error) {
          log(`[ncolor] relabel_from_affinity error (HTTP): ${result.error}`);
          return false;
        }
        if (result && result.mask) {
          applySegmentationMask(result, { forceInstanceMask: true });
          log('Applied relabel_from_affinity from backend (HTTP).');
          return true;
        }
        log(`[ncolor] relabel_from_affinity missing mask (HTTP); keys=${result ? Object.keys(result) : 'none'}`);
      } else {
        log(`[ncolor] relabel_from_affinity HTTP status ${res.status}`);
      }
    }
  } catch (err) {
    console.warn('relabel_from_affinity call failed', err);
  }
  return false;
}

function drawAffinityGraphOverlay() {
  if (!ctx || !showAffinityGraph || !affinityGraphInfo || !affinityGraphInfo.values) {
    return;
  }
  const alpha = computeAffinityAlpha();
  if (alpha <= 0) {
    return;
  }
  if (!affinityGraphInfo.segments) {
    buildAffinityGraphSegments();
  }
  const segments = affinityGraphInfo.segments;
  if (!segments) {
    return;
  }
  ctx.save();
  applyViewTransform(ctx, { includeDpr: true });
  ctx.globalAlpha = alpha;
  ctx.lineWidth = 1;
  ctx.strokeStyle = AFFINITY_OVERLAY_COLOR;
  ctx.beginPath();
  for (let i = 0; i < segments.length; i += 1) {
    const seg = segments[i];
    const indices = seg && seg.indices;
    if (!seg || !indices || indices.size === 0) continue;
    const [dyRaw, dxRaw] = affinitySteps[i];
    const dx = dxRaw | 0;
    const dy = dyRaw | 0;
    for (const index of indices) {
      const x = index % affinityGraphInfo.width;
      const y = (index / affinityGraphInfo.width) | 0;
      const cx1 = x + 0.5;
      const cy1 = y + 0.5;
      const cx2 = x + dx + 0.5;
      const cy2 = y + dy + 0.5;
      ctx.moveTo(cx1, cy1);
      ctx.lineTo(cx2, cy2);
    }
  }
  ctx.stroke();
  ctx.restore();
}

function getPointerPosition(evt) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: evt.clientX - rect.left,
    y: evt.clientY - rect.top,
  };
}

function screenToImage(point) {
  const cos = Math.cos(viewState.rotation);
  const sin = Math.sin(viewState.rotation);
  const scale = viewState.scale || 1;
  const dx = point.x - viewState.offsetX;
  const dy = point.y - viewState.offsetY;
  const localX = (dx * cos + dy * sin) / scale;
  const localY = (-dx * sin + dy * cos) / scale;
  return {
    x: localX,
    y: localY,
  };
}

function imageToScreen(point) {
  const cos = Math.cos(viewState.rotation);
  const sin = Math.sin(viewState.rotation);
  const scaledX = point.x * viewState.scale;
  const scaledY = point.y * viewState.scale;
  const rotatedX = scaledX * cos - scaledY * sin;
  const rotatedY = scaledX * sin + scaledY * cos;
  return {
    x: rotatedX + viewState.offsetX,
    y: rotatedY + viewState.offsetY,
  };
}


function showDebugFillOverlay(rect) {
  if (!debugFillOverlay || !rect) {
    return;
  }
  const topLeft = imageToScreen({ x: rect.x, y: rect.y });
  const bottomRight = imageToScreen({ x: rect.x + rect.width, y: rect.y + rect.height });
  const width = Math.max(1, bottomRight.x - topLeft.x);
  const height = Math.max(1, bottomRight.y - topLeft.y);
  debugFillOverlay.style.display = 'block';
  debugFillOverlay.style.opacity = '1';
  debugFillOverlay.style.transform = `translate(${topLeft.x}px, ${topLeft.y}px)`;
  debugFillOverlay.style.width = `${width}px`;
  debugFillOverlay.style.height = `${height}px`;
  if (debugFillOverlayHideTimer) {
    clearTimeout(debugFillOverlayHideTimer);
  }
  debugFillOverlayHideTimer = setTimeout(() => {
    debugFillOverlay.style.opacity = '0';
    debugFillOverlayHideTimer = setTimeout(() => {
      if (debugFillOverlay) {
        debugFillOverlay.style.display = 'none';
      }
    }, 220);
  }, 150);
}

function resolveGestureOrigin(evt) {
  if (!canvas) {
    return { x: 0, y: 0 };
  }
  const rect = canvas.getBoundingClientRect();
  let x = typeof evt.clientX === 'number'
    ? evt.clientX - rect.left
    : Number.NaN;
  let y = typeof evt.clientY === 'number'
    ? evt.clientY - rect.top
    : Number.NaN;
  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    x = rect.width / 2;
    y = rect.height / 2;
  }
  return { x, y };
}

function resizeCanvas() {
  if (shuttingDown) {
    return;
  }
  const v = getViewportSize();
  const viewerRect = viewer ? viewer.getBoundingClientRect() : null;
  if (v.width <= 0 || v.height <= 0) {
    if (viewerRect) {
      log('resize skipped: viewer size ' + viewerRect.width.toFixed(1) + 'x' + viewerRect.height.toFixed(1));
    }
    if (!shuttingDown) {
      requestAnimationFrame(resizeCanvas);
    }
    return;
  }
  const renderWidth = Math.max(1, v.width);
  canvas.width = Math.max(1, Math.round(renderWidth * dpr));
  canvas.height = Math.max(1, Math.round(v.height * dpr));
  canvas.style.width = renderWidth + 'px';
  canvas.style.height = v.height + 'px';
  canvas.style.transform = '';
  resizeWebglOverlay();
  resizePointsOverlay();
  resizeVectorOverlay();
  resizePreviewCanvas();
  // Skip fit/recenter if viewState was restored from saved state
  // viewStateRestored stays true until user explicitly resets view
  if (!viewStateRestored && !fitViewToWindow(v)) {
    recenterView(v);
  }
  draw();
  drawBrushPreview(getHoverPoint());
}

function recenterView(bounds) {
  const metrics = bounds && typeof bounds.width === 'number'
    ? bounds
    : getViewportSize();
  const visibleWidth = metrics.visibleWidth || Math.max(1, metrics.width - (metrics.leftInset || 0));
  const height = metrics.height || 0;
  const leftInset = metrics.leftInset || getLeftPanelWidthPx();
  const imageCenter = { x: imgWidth / 2, y: imgHeight / 2 };
  const target = { x: leftInset + (visibleWidth / 2), y: height / 2 };
  setOffsetForImagePoint(imageCenter, target);
  log('recenter to ' + viewState.offsetX.toFixed(1) + ',' + viewState.offsetY.toFixed(1));
}

function fitViewToWindow(bounds) {
  if (!viewer) {
    return false;
  }
  if (userAdjustedScale && !autoFitPending) {
    return false;
  }
  const metrics = bounds && typeof bounds.width === 'number'
    ? bounds
    : getViewportSize();
  const visibleWidth = metrics.visibleWidth || Math.max(1, metrics.width - (metrics.leftInset || 0));
  if (visibleWidth <= 0 || metrics.height <= 0 || imgWidth === 0 || imgHeight === 0) {
    return false;
  }
  const marginPx = 2; // ensure fully visible by a small pixel margin
  // Fit based on the unobstructed viewport width (excluding the left tool rail)
  const scaleX = Math.max(0.0001, (visibleWidth - marginPx) / imgWidth);
  const scaleY = Math.max(0.0001, (metrics.height - marginPx) / imgHeight);
  const nextScale = Math.max(0.0001, Math.min(scaleX, scaleY));
  if (!Number.isFinite(nextScale) || nextScale <= 0) {
    return false;
  }
  viewState.scale = nextScale;
  viewState.rotation = 0;
  autoFitPending = false;
  recenterView(metrics);
  return true;
}

function resetView() {
  const metrics = getViewportSize();
  autoFitPending = true;
  userAdjustedScale = false;
  viewStateRestored = false;
  viewState.rotation = 0;
  if (!fitViewToWindow(metrics)) {
    viewState.rotation = 0;
    recenterView(metrics);
  }
  draw();
  updateHoverInfo(getHoverPoint() || null);
  renderHoverPreview();
  viewStateDirty = true;
  scheduleStateSave();
}

function clampGammaValue(value) {
  if (Number.isNaN(value)) {
    return currentGamma;
  }
  return Math.min(MAX_GAMMA, Math.max(MIN_GAMMA, value));
}

function updateGammaLabel() {
  if (gammaValue) {
    gammaValue.textContent = 'Gamma: ' + currentGamma.toFixed(2);
  }
}

function syncGammaControls() {
  if (gammaSlider) {
    const sliderValue = Math.round(currentGamma * 100);
    gammaSlider.value = String(Math.min(600, Math.max(10, sliderValue)));
    updateNativeRangeFill(gammaSlider);
    refreshSlider('gamma');
  }
  if (gammaInput) {
    gammaInput.value = currentGamma.toFixed(2);
  }
  updateGammaLabel();
}

function setGamma(gamma, { emit = true } = {}) {
  currentGamma = clampGammaValue(gamma);
  syncGammaControls();
  if (emit) {
    applyImageAdjustments();
  } else {
    renderHistogram();
  }
}

function applyGamma(gamma) {
  setGamma(gamma, { emit: true });
}

function sinebowColor(t) {
  const angle = 2 * Math.PI * (t - Math.floor(t));
  const r = Math.sin(angle) * 0.5 + 0.5;
  const g = Math.sin(angle + (2 * Math.PI) / 3) * 0.5 + 0.5;
  const b = Math.sin(angle + (4 * Math.PI) / 3) * 0.5 + 0.5;
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255), 200];
}

function rgbToHex(rgb) {
  const [r, g, b] = rgb;
  return '#' + [r, g, b]
    .map((v) => Math.max(0, Math.min(255, v)).toString(16).padStart(2, '0'))
    .join('');
}

function hexToRgb(hex) {
  if (!hex) return [0, 0, 0];
  const value = hex.replace('#', '');
  if (value.length !== 6) return [0, 0, 0];
  const r = parseInt(value.slice(0, 2), 16);
  const g = parseInt(value.slice(2, 4), 16);
  const b = parseInt(value.slice(4, 6), 16);
  return [r, g, b];
}

function hslToRgb(h, s, l) {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const hp = h * 6;
  const x = c * (1 - Math.abs((hp % 2) - 1));
  let r = 0;
  let g = 0;
  let b = 0;
  if (hp >= 0 && hp < 1) {
    r = c; g = x; b = 0;
  } else if (hp < 2) {
    r = x; g = c; b = 0;
  } else if (hp < 3) {
    r = 0; g = c; b = x;
  } else if (hp < 4) {
    r = 0; g = x; b = c;
  } else if (hp < 5) {
    r = x; g = 0; b = c;
  } else {
    r = c; g = 0; b = x;
  }
  const m = l - c / 2;
  return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
}

function ensureNColorPaletteLength(targetCount) {
  const target = Math.max(2, targetCount | 0);
  const base = (nColorPaletteColors && nColorPaletteColors.length)
    ? nColorPaletteColors.slice()
    : generateNColorSwatches(DEFAULT_NCOLOR_COUNT, 0.35);
  if (base.length >= target) {
    return base;
  }
  const next = base.slice();
  for (let i = next.length; i < target; i += 1) {
    const t = (0.35 + i / Math.max(target, 2)) % 1;
    const rgb = sinebowColor(t);
    next.push([rgb[0], rgb[1], rgb[2]]);
  }
  return next;
}

/**
 * Get a color from the current labelColormap at position t (0-1).
 * Used for generating N-color swatches with any colormap.
 */
function getColormapColorAtT(t, cmap = null) {
  const cmapName = cmap || labelColormap;
  if (cmapName === 'gray') {
    const v = Math.round(t * 255);
    return [v, v, v];
  }
  if (cmapName === 'pastel') {
    return hslToRgb(t, 0.55, 0.72);
  }
  if (cmapName === 'vivid') {
    return hslToRgb(t, 0.9, 0.5);
  }
  if (cmapName === 'sinebow') {
    return sinebowColor(t);
  }
  const stops = COLORMAP_STOPS[cmapName];
  if (stops) {
    return interpolateStops(stops, t);
  }
  // Fallback to sinebow
  return sinebowColor(t);
}

function generateNColorSwatches(count, offset = null) {
  const swatches = [];
  const total = Math.max(2, count);
  // Use global nColorHueOffset if no offset specified (only for cyclic colormaps)
  const hueOffset = offset !== null ? offset : nColorHueOffset;
  const hasCyclicOffset = colormapHasOffset(labelColormap);

  for (let i = 0; i < total; i += 1) {
    // For cyclic colormaps, apply hue offset; for linear, just use even spacing
    const t = hasCyclicOffset
      ? (hueOffset + i / total) % 1
      : i / (total - 1 || 1);  // Linear colormaps: 0 to 1
    const [r, g, b] = getColormapColorAtT(t);
    swatches.push([r, g, b]);
  }
  return swatches;
}

function seededRandom(seed) {
  let t = seed + 0x6D2B79F5;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

function getLabelShuffleKey(label) {
  if (!labelShuffle) {
    return label;
  }
  const seed = (labelShuffleSeed | 0) + 1;
  const mix = label ^ (seed * 0x9e3779b9);
  return Math.floor(seededRandom(mix) * 1e9);
}

// Colormap stops from matplotlib/cmap package (pypi cmap)
// Using 16 evenly-spaced stops for accurate interpolation
const COLORMAP_STOPS = {
  viridis: [
    '#440154', '#481a6c', '#472f7d', '#414487', '#39568c',
    '#31688e', '#2a788e', '#23888e', '#1f988b', '#22a884',
    '#35b779', '#54c568', '#7ad151', '#a5db36', '#d2e21b', '#fde725'
  ],
  magma: [
    '#000004', '#0c0926', '#1b0c41', '#2f0f60', '#4a0c6b',
    '#65156e', '#7e2482', '#982d80', '#b73779', '#d5446d',
    '#ed6059', '#f88a5f', '#feb078', '#fed799', '#fcfdbf'
  ],
  plasma: [
    '#0d0887', '#3a049a', '#5c01a6', '#7e03a8', '#9c179e',
    '#b52f8c', '#cc4778', '#de5f65', '#ed7953', '#f89540',
    '#fdb42f', '#fbd524', '#f0f921'
  ],
  inferno: [
    '#000004', '#0d0829', '#1b0c41', '#320a5e', '#4a0c6b',
    '#61136e', '#78206c', '#932667', '#ad305e', '#c73e53',
    '#df5543', '#f17336', '#f9932e', '#fbb535', '#fad948', '#fcffa4'
  ],
  cividis: [
    '#00204c', '#00336c', '#2a4858', '#43598e', '#5a6c8a',
    '#6e7f8e', '#808f8a', '#93a08a', '#a8b08c', '#bdc18d',
    '#d3d291', '#e8e395', '#fdea45'
  ],
  turbo: [
    '#30123b', '#4145ab', '#4675ed', '#39a2fc', '#1bcfd4',
    '#24eca6', '#61fc6c', '#a4fc3c', '#d1e834', '#f3c63a',
    '#fe9b2d', '#f56516', '#d93806', '#b11901', '#7a0402'
  ],
  gist_ncar: [
    '#000080', '#0000d4', '#0044ff', '#0099ff', '#00eeff',
    '#00ff99', '#00ff00', '#66ff00', '#ccff00', '#ffcc00',
    '#ff6600', '#ff0000', '#cc0000', '#800000'
  ],
  hot: [
    '#000000', '#230000', '#460000', '#690000', '#8c0000',
    '#af0000', '#d20000', '#f50000', '#ff1800', '#ff3b00',
    '#ff5e00', '#ff8100', '#ffa400', '#ffc700', '#ffea00',
    '#ffff0d', '#ffff4d', '#ffff8d', '#ffffcd', '#ffffff'
  ],
};

// Image colormap LUT size (256 entries for full 8-bit range)
const IMAGE_CMAP_LUT_SIZE = 256;

/**
 * Get the numeric colormap type value for the shader.
 * 0 = grayscale (passthrough)
 * 1 = grayscale with red clipping
 * 2+ = use LUT texture
 */
function getImageCmapTypeValue() {
  if (imageColormap === 'gray') return 0;
  if (imageColormap === 'gray-clip') return 1;
  return 2; // LUT-based
}

/**
 * Generate LUT data for an image colormap.
 * Returns Uint8Array of RGBA values (size * 4 bytes).
 */
function generateImageCmapLut(cmapName) {
  const data = new Uint8Array(IMAGE_CMAP_LUT_SIZE * 4);
  const stops = COLORMAP_STOPS[cmapName];

  for (let i = 0; i < IMAGE_CMAP_LUT_SIZE; i++) {
    const t = i / (IMAGE_CMAP_LUT_SIZE - 1);
    let rgb;

    if (stops) {
      rgb = interpolateStops(stops, t);
    } else {
      // Fallback to grayscale
      const v = Math.round(t * 255);
      rgb = [v, v, v];
    }

    const offset = i * 4;
    data[offset] = rgb[0];
    data[offset + 1] = rgb[1];
    data[offset + 2] = rgb[2];
    data[offset + 3] = 255;
  }

  return data;
}

/**
 * Create or update the image colormap LUT texture.
 */
function updateImageCmapTexture() {
  if (!webglPipeline || !webglPipeline.gl) return;

  const gl = webglPipeline.gl;

  // Only create LUT for non-grayscale colormaps
  if (imageColormap === 'gray' || imageColormap === 'gray-clip') {
    return;
  }

  // Create texture if needed
  if (!webglPipeline.imageCmapTexture) {
    webglPipeline.imageCmapTexture = gl.createTexture();
  }

  const lutData = generateImageCmapLut(imageColormap);

  gl.bindTexture(gl.TEXTURE_2D, webglPipeline.imageCmapTexture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, IMAGE_CMAP_LUT_SIZE, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, lutData);
  gl.bindTexture(gl.TEXTURE_2D, null);

  imageCmapLutDirty = false;
}

function interpolateStops(stops, t) {
  if (!stops || !stops.length) {
    return [0, 0, 0];
  }
  if (stops.length === 1) {
    return hexToRgb(stops[0]);
  }
  const clamped = Math.min(Math.max(t, 0), 0.999999);
  const scaled = clamped * (stops.length - 1);
  const idx = Math.floor(scaled);
  const frac = scaled - idx;
  const a = hexToRgb(stops[idx]);
  const b = hexToRgb(stops[Math.min(idx + 1, stops.length - 1)]);
  return [
    Math.round(a[0] + (b[0] - a[0]) * frac),
    Math.round(a[1] + (b[1] - a[1]) * frac),
    Math.round(a[2] + (b[2] - a[2]) * frac),
  ];
}

/**
 * Get palette index for a label.
 * Shuffle OFF: Labels map sequentially to palette (label 1 -> palette[1], etc.)
 * Shuffle ON: Labels offset by seed for different color assignment
 */
function getLabelOrderValue(label, paletteSize = 256) {
  // palette[0] is background (black/transparent), valid indices are 1 to paletteSize-1
  const effectiveSize = paletteSize - 1;

  if (!labelShuffle) {
    // Shuffle OFF: sequential mapping - label 1 -> palette[1], label 2 -> palette[2], etc.
    // The palette is golden-ratio spaced, so consecutive labels get well-separated colors
    return ((label - 1) % effectiveSize) + 1;
  }

  // Shuffle ON: offset labels by seed to get different color assignment
  // The palette itself provides the golden-ratio spreading, so just shift by seed
  const seed = labelShuffleSeed | 0;

  // Simple offset: (label + seed * prime) mod size
  // Using a prime multiplier on seed ensures different seeds give very different offsets
  const seedOffset = seed * 97;
  const idx = ((label - 1 + seedOffset) % effectiveSize);

  // Map to valid palette range [1, paletteSize-1]
  return idx + 1;
}

function getLabelColorFraction(label) {
  // Use currentMaxLabel to spread colors across full colormap range
  const maxLabel = Math.max(currentMaxLabel, 2);
  if (!labelShuffle) {
    // Sequential: labels 1..maxLabel map to t=0..1
    return ((label - 1) % maxLabel) / (maxLabel - 1);
  }
  // Shuffled: use seeded random but still within 0..1 range
  return seededRandom(getLabelShuffleKey(label));
}

function getColormapColor(label) {
  if (label <= 0) return null;
  const t = getLabelColorFraction(label);
  if (labelColormap === 'gray') {
    const v = Math.round(t * 255);
    return [v, v, v];
  }
  if (labelColormap === 'pastel') {
    return hslToRgb(t, 0.55, 0.72);
  }
  if (labelColormap === 'vivid') {
    return hslToRgb(t, 0.9, 0.5);
  }
  if (labelColormap === 'sinebow') {
    const [r, g, b] = sinebowColor(t);
    return [r, g, b];
  }
  const stops = COLORMAP_STOPS[labelColormap];
  if (stops) {
    return interpolateStops(stops, t);
  }
  const [r, g, b] = sinebowColor(t);
  return [r, g, b];
}

function generateSinebowPalette(size, offset = 0, sequential = false) {
  const count = Math.max(size, 2);
  const palette = new Array(count);
  palette[0] = [0, 0, 0, 0];
  const golden = 0.61803398875;
  for (let i = 1; i < count; i += 1) {
    // Sequential: evenly spaced around the color wheel (rainbow order)
    // Golden ratio: maximally separated colors (scrambled but distinct)
    const t = sequential
      ? (offset + (i - 1) / (count - 1)) % 1
      : (offset + i * golden) % 1;
    palette[i] = sinebowColor(t);
  }
  return palette;
}

/**
 * Update max label tracking.
 * Call this after any mask modification (segmentation, painting, loading).
 */
function updateMaxLabelFromMask() {
  if (!maskValues || maskValues.length === 0) return;
  let maxL = 0;
  for (let i = 0; i < maskValues.length; i++) {
    if (maskValues[i] > maxL) maxL = maskValues[i];
  }
  if (maxL > 0 && maxL !== currentMaxLabel) {
    currentMaxLabel = maxL;
    // Invalidate shuffle permutation cache
    shufflePermutation = null;
    paletteTextureDirty = true;
    clearColorCaches();
  }
}

/**
 * Build shuffle permutation for current max label count.
 * Creates a bijection [1..N] -> [1..N] using golden ratio for optimal visual separation.
 */
function buildShufflePermutation() {
  const N = currentMaxLabel;
  const seed = labelShuffleSeed | 0;
  if (shufflePermutation && shufflePermutationSize === N && shufflePermutationSeed === seed) {
    return shufflePermutation;
  }
  const golden = 0.61803398875;
  // Create array of {label, sortKey} and sort by sortKey
  const items = [];
  for (let i = 1; i <= N; i++) {
    const seedOffset = seed * 0.1;
    const sortKey = ((i + seedOffset) * golden) % 1;
    items.push({ label: i, sortKey });
  }
  items.sort((a, b) => a.sortKey - b.sortKey);
  // Build permutation: perm[label] = rank (1-based)
  const perm = new Array(N + 1);
  perm[0] = 0; // background stays 0
  for (let rank = 0; rank < items.length; rank++) {
    perm[items[rank].label] = rank + 1; // 1-based rank
  }
  shufflePermutation = perm;
  shufflePermutationSize = N;
  shufflePermutationSeed = seed;
  return perm;
}
// N-color palette is large enough; groups come from backend ncolor.label
nColorPalette = generateSinebowPalette(Math.max(colorTable.length || 0, 1024), nColorHueOffset);
nColorPaletteColors = generateNColorSwatches(DEFAULT_NCOLOR_COUNT);

let nColorActive = true;
let nColorValues = null; // per-pixel group IDs for N-color display only
let nColorInstanceMask = null;
renderNColorSwatches();
updateNColorPanel();
updateLabelShuffleControls();
// Initialize colormap preview after DOM is ready
setTimeout(() => {
  updateInstanceColormapPreview();
  if (ncolorHueOffsetSlider) {
    ncolorHueOffsetSlider.value = Math.round(nColorHueOffset * 100);
  }
}, 0);
// Single authoritative mask buffer: maskValues always holds instance labels.
const rawColorMap = new Map();
const nColorColorMap = new Map();
const nColorLabelToGroup = new Map();
// Legacy assignment structures retained but unused with single-buffer model
const nColorAssignments = new Map();
const nColorColorToLabel = new Map();
let nColorMaxColorId = 0;
let lastLabelBeforeNColor = null;

function hashColorForLabel(label, offset = 0.0) {
  const golden = 0.61803398875;
  const t = ((label * golden + offset) % 1 + 1) % 1;
  const base = sinebowColor(t);
  return [base[0], base[1], base[2]];
}

function clearColorCaches() {
  rawColorMap.clear();
  nColorColorMap.clear();
  // Also clear shuffle permutation cache to force regeneration
  shufflePermutationCache = null;
  shufflePermutationCacheSeed = null;
  shufflePermutationCacheSize = null;
}

function getColorFromMap(label, map, offset = 0.0) {
  if (label <= 0) {
    return null;
  }
  let rgb = map.get(label);
  if (!rgb) {
    rgb = hashColorForLabel(label, offset);
    map.set(label, rgb);
  }
  return rgb;
}

function getRawLabelColor(label) {
  if (label <= 0) {
    return null;
  }
  let rgb = rawColorMap.get(label);
  if (!rgb) {
    rgb = getColormapColor(label);
    rawColorMap.set(label, rgb);
  }
  return rgb;
}

function getNColorLabelColor(label) {
  if (label <= 0) {
    return null;
  }
  const palette = nColorPaletteColors.length ? nColorPaletteColors : generateNColorSwatches(DEFAULT_NCOLOR_COUNT, 0.35);
  const idx = (label - 1) % palette.length;
  return palette[idx];
}

function getDisplayColor(index) {
  const label = maskValues[index] | 0;
  if (label <= 0) return null;
  return nColorActive ? getNColorLabelColor(label) : getRawLabelColor(label);
}

function collectLabelsFromMask(sourceMask) {
  const seen = new Set();
  for (let i = 0; i < sourceMask.length; i += 1) {
    const value = sourceMask[i];
    if (value > 0) {
      seen.add(value);
    }
  }
  return Array.from(seen).sort((a, b) => a - b);
}

function rebuildNColorLabelMap() {
  nColorLabelToGroup.clear();
  if (!nColorValues || nColorValues.length !== maskValues.length) {
    return;
  }
  for (let i = 0; i < maskValues.length; i += 1) {
    const label = maskValues[i] | 0;
    const group = nColorValues[i] | 0;
    if (label > 0 && group > 0 && !nColorLabelToGroup.has(label)) {
      nColorLabelToGroup.set(label, group);
    }
  }
}

function resetNColorAssignments() {
  nColorAssignments.clear();
  nColorColorToLabel.clear();
  nColorLabelToGroup.clear();
  nColorMaxColorId = 0;
}

// Legacy N-color mapping helpers removed under single-buffer model

function applyImageAdjustments() {
  if (!originalImageData) {
    return;
  }
  const source = originalImageData.data;
  const img = offCtx.createImageData(imgWidth, imgHeight);
  const target = img.data;
  const low = windowLow / 255;
  const high = windowHigh / 255;
  const range = Math.max(high - low, 1 / 255);
  for (let i = 0; i < source.length; i += 4) {
    let value = source[i] / 255;
    value = Math.min(Math.max((value - low) / range, 0), 1);
    value = Math.pow(value, currentGamma);
    const v = Math.round(value * 255);
    target[i] = v;
    target[i + 1] = v;
    target[i + 2] = v;
    target[i + 3] = source[i + 3];
  }
  offCtx.putImageData(img, 0, 0);
  if (isWebglPipelineActive()) {
    uploadBaseTextureFromCanvas();
  }
  draw();
  renderHistogram();
}

function computeHistogram() {
  if (!originalImageData) {
    histogramData = null;
    return;
  }
  histogramData = new Uint32Array(256);
  const data = originalImageData.data;
  for (let i = 0; i < data.length; i += 4) {
    histogramData[data[i]] += 1;
  }
}


function histogramQuantile(q) {
  if (!histogramData) {
    return 0;
  }
  const total = histogramData.reduce((acc, v) => acc + v, 0);
  if (!total) {
    return 0;
  }
  const target = total * q;
  let cumulative = 0;
  for (let i = 0; i < histogramData.length; i += 1) {
    cumulative += histogramData[i];
    if (cumulative >= target) {
      return i;
    }
  }
  return histogramData.length - 1;
}

function renderHistogram() {
  if (!histogramCanvas || !histogramData) {
    return;
  }
  const ctx = histogramCanvas.getContext('2d');
  if (!ctx) {
    return;
  }
  const width = histogramCanvas.width;
  const height = histogramCanvas.height;
  ctx.clearRect(0, 0, width, height);
  const maxCount = Math.max(...histogramData);
  if (maxCount > 0) {
    ctx.fillStyle = accentColor;
    const binWidth = Math.max(width / 256, 1);
    for (let i = 0; i < 256; i += 1) {
      const value = histogramData[i] / maxCount;
      const barHeight = Math.max(1, Math.round(value * (height - 4)));
      const x = Math.floor(i * binWidth);
      ctx.fillRect(x, height - barHeight, Math.ceil(binWidth), barHeight);
    }
  }
  const lowX = (windowLow / 255) * width;
  const highX = (windowHigh / 255) * width;
  ctx.fillStyle = histogramWindowColor;
  ctx.fillRect(lowX, 0, Math.max(highX - lowX, 1), height);
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(lowX, 0);
  ctx.lineTo(lowX, height);
  ctx.moveTo(highX, 0);
  ctx.lineTo(highX, height);
  ctx.stroke();
  const gammaCurveColor = panelTextColor || accentColor;
  if (windowHigh > windowLow) {
    ctx.strokeStyle = gammaCurveColor;
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    const startX = Math.max(0, Math.floor(lowX));
    const endX = Math.min(width, Math.ceil(highX));
    for (let x = startX; x <= endX; x += 1) {
      const intensity = (x / width) * 255;
      const y = gammaCurveY(intensity, width, height);
      if (x === startX) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
  }
  updateHistogramCursor();
}

function updateHistogramUI() {
  renderHistogram();
}

function histogramValueFromEvent(evt) {
  const rect = histogramCanvas.getBoundingClientRect();
  const x = Math.min(Math.max(evt.clientX - rect.left, 0), rect.width);
  return Math.round((x / rect.width) * 255);
}

function gammaCurveY(intensity, width, height) {
  if (windowHigh <= windowLow) {
    return height - 2;
  }
  const clampedIntensity = Math.min(Math.max(intensity, windowLow), windowHigh);
  let t = (clampedIntensity - windowLow) / (windowHigh - windowLow);
  t = Math.min(Math.max(t, 0.0001), 0.9999);
  const mapped = Math.pow(t, 1 / currentGamma);
  const y = height - (mapped * (height - 4)) - 2;
  return Math.min(height - 2, Math.max(2, y));
}

function updateHistogramCursor(evt) {
  if (!histogramCanvas) {
    return;
  }
  if (histDragTarget) {
    histogramCanvas.style.cursor = (histDragTarget === 'range' || histDragTarget === 'gamma') ? 'grabbing' : 'ew-resize';
    if (histDragTarget === 'low') {
      histogramCanvas.dataset.tooltip = 'Low: ' + windowLow;
    } else if (histDragTarget === 'high') {
      histogramCanvas.dataset.tooltip = 'High: ' + windowHigh;
    } else {
      delete histogramCanvas.dataset.tooltip;
    }
    return;
  }
  const rect = histogramCanvas.getBoundingClientRect();
  const x = evt ? evt.clientX - rect.left : NaN;
  const width = rect.width;
  const height = rect.height;
  const lowX = (windowLow / 255) * width;
  const highX = (windowHigh / 255) * width;
  const threshold = HIST_HANDLE_THRESHOLD;
  let cursor = 'crosshair';
  if (!Number.isNaN(x)) {
    if (Math.abs(x - lowX) < threshold || Math.abs(x - highX) < threshold) {
      cursor = 'ew-resize';
    } else if (x > lowX && x < highX) {
      cursor = 'grab';
      if (evt && windowHigh > windowLow) {
        const intensity = histogramValueFromEvent(evt);
        const y = evt.clientY - rect.top;
        const curveY = gammaCurveY(intensity, width, height);
        if (Math.abs(y - curveY) < threshold) {
          cursor = 'grab';
        }
      }
    }
  }
  histogramCanvas.style.cursor = cursor;
  if (!Number.isNaN(x)) {
    if (Math.abs(x - lowX) < threshold) {
      histogramCanvas.dataset.tooltip = 'Low: ' + windowLow;
    } else if (Math.abs(x - highX) < threshold) {
      histogramCanvas.dataset.tooltip = 'High: ' + windowHigh;
    } else {
      delete histogramCanvas.dataset.tooltip;
    }
  } else {
    delete histogramCanvas.dataset.tooltip;
  }
}

function setWindowBounds(low, high, { emit = true } = {}) {
  let clampedLow = Math.round(low);
  let clampedHigh = Math.round(high);
  if (Number.isNaN(clampedLow)) clampedLow = windowLow;
  if (Number.isNaN(clampedHigh)) clampedHigh = windowHigh;
  clampedLow = Math.max(0, Math.min(255, clampedLow));
  clampedHigh = Math.max(0, Math.min(255, clampedHigh));
  if (clampedHigh <= clampedLow) {
    if (histDragTarget === 'low') {
      clampedLow = Math.max(0, Math.min(254, clampedHigh - 1));
    } else if (histDragTarget === 'high') {
      clampedHigh = Math.min(255, Math.max(1, clampedLow + 1));
    } else {
      clampedHigh = Math.min(255, Math.max(1, clampedLow + 1));
    }
  }
  windowLow = clampedLow;
  windowHigh = clampedHigh;
  updateHistogramUI();
  if (emit) {
    applyImageAdjustments();
  }
}

function handleHistogramPointerDown(evt) {
  if (!histogramCanvas) {
    return;
  }
  evt.preventDefault();
  histogramCanvas.setPointerCapture(evt.pointerId);
  const rect = histogramCanvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  const x = Math.min(Math.max(evt.clientX - rect.left, 0), width);
  const y = Math.min(Math.max(evt.clientY - rect.top, 0), height);
  const intensity = (x / width) * 255;
  const lowX = (windowLow / 255) * width;
  const highX = (windowHigh / 255) * width;
  const threshold = HIST_HANDLE_THRESHOLD;
  histDragTarget = null;
  if (windowHigh > windowLow) {
    const curveY = gammaCurveY(intensity, width, height);
    if (intensity >= windowLow && intensity <= windowHigh && Math.abs(y - curveY) <= threshold) {
      histDragTarget = 'gamma';
    }
  }
  if (!histDragTarget && Math.abs(x - lowX) <= threshold) {
    histDragTarget = 'low';
  } else if (!histDragTarget && Math.abs(x - highX) <= threshold) {
    histDragTarget = 'high';
  } else if (!histDragTarget && x > lowX && x < highX) {
    histDragTarget = 'range';
    histDragOffset = histogramValueFromEvent(evt) - windowLow;
  } else if (!histDragTarget) {
    histDragTarget = Math.abs(x - lowX) < Math.abs(x - highX) ? 'low' : 'high';
  }
  if (histDragTarget !== 'range') {
    histDragOffset = 0;
  }
  updateHistogramCursor(evt);
  handleHistogramPointerMove(evt);
}

function handleHistogramPointerMove(evt) {
  if (!histogramCanvas) {
    return;
  }
  if (!histDragTarget) {
    updateHistogramCursor(evt);
    return;
  }
  evt.preventDefault();
  const value = histogramValueFromEvent(evt);
  if (histDragTarget === 'low') {
    setWindowBounds(Math.min(value, windowHigh - 1), windowHigh);
  } else if (histDragTarget === 'high') {
    setWindowBounds(windowLow, Math.max(value, windowLow + 1));
  } else if (histDragTarget === 'range') {
    const span = windowHigh - windowLow;
    let newLow = value - histDragOffset;
    newLow = Math.max(0, Math.min(255 - span, newLow));
    setWindowBounds(newLow, newLow + span);
  } else if (histDragTarget === 'gamma') {
    if (windowHigh > windowLow) {
      const rect = histogramCanvas.getBoundingClientRect();
      const height = rect.height;
      const width = rect.width;
      const clampedValue = Math.min(Math.max(value, windowLow + 0.5), windowHigh - 0.5);
      let t = (clampedValue - windowLow) / (windowHigh - windowLow);
      t = Math.min(Math.max(t, 0.0001), 0.9999);
      const yRatio = 1 - Math.min(Math.max((evt.clientY - rect.top) / height, 0.0001), 0.9999);
      let newGamma = Math.log(t) / Math.log(yRatio);
      if (!Number.isFinite(newGamma) || newGamma <= 0) {
        newGamma = currentGamma;
      }
      setGamma(newGamma);
    }
  }
  updateHistogramCursor(evt);
}

function handleHistogramPointerUp(evt) {
  if (!histogramCanvas) {
    return;
  }
  evt.preventDefault();
  histogramCanvas.releasePointerCapture(evt.pointerId);
  histDragTarget = null;
  histDragOffset = 0;
  updateHistogramCursor(evt);
}

function updateHoverInfo(point) {
  const valueTarget = hoverValueDisplay || hoverInfo;
  const coordTarget = hoverCoordDisplay || hoverInfo;
  if (!point || !originalImageData) {
    cursorInsideImage = false;
    if (valueTarget) {
      valueTarget.textContent = 'Y: --, X: --, Val: --';
    }
    if (coordTarget && coordTarget !== valueTarget) {
      coordTarget.textContent = '';
    }
    if (histogramValueMarker) {
      histogramValueMarker.style.opacity = '0';
    }
    updateCursor();
    return;
  }
  const rawX = point.x;
  const rawY = point.y;
  if (rawX < 0 || rawY < 0 || rawX >= imgWidth || rawY >= imgHeight) {
    cursorInsideImage = false;
    if (valueTarget) {
      valueTarget.textContent = 'Y: --, X: --, Val: --';
    }
    if (coordTarget && coordTarget !== valueTarget) {
      coordTarget.textContent = '';
    }
    clearHoverPreview();
    if (histogramValueMarker) {
      histogramValueMarker.style.opacity = '0';
    }
    updateCursor();
    return;
  }
  const x = Math.min(imgWidth - 1, Math.max(0, Math.round(rawX)));
  const y = Math.min(imgHeight - 1, Math.max(0, Math.round(rawY)));
  const idx = (y * imgWidth + x) * 4;
  const r = originalImageData.data[idx];
  const g = originalImageData.data[idx + 1];
  const b = originalImageData.data[idx + 2];
  const value = CONFIG.isRgb ? `${r}, ${g}, ${b}` : r;
  cursorInsideImage = true;
  if (valueTarget) {
    valueTarget.textContent = 'Y: ' + y + ', X: ' + x + ', Val: ' + value;
  }
  if (coordTarget && coordTarget !== valueTarget) {
    coordTarget.textContent = '';
  }
  if (histogramValueMarker && Number.isFinite(value)) {
    const v = Math.max(0, Math.min(255, Number(value)));
    const frac = v / 255;
    histogramValueMarker.style.left = `calc(${(frac * 100).toFixed(2)}% - 1px)`;
    histogramValueMarker.style.opacity = '1';
  } else if (histogramValueMarker) {
    histogramValueMarker.style.opacity = '0';
  }
  updateCursor();
}

function handleBrushDoubleTapOnDown(evt, pointer, world) {
  if (!evt || tool !== 'brush' || eraseActive) {
    return null;
  }
  const type = evt.pointerType || (pointerState.isStylusPointer(evt) ? 'pen' : 'mouse');
  if (type !== 'pen' && type !== 'mouse') {
    return null;
  }
  const maxInterval = type === 'pen'
    ? BRUSH_DOUBLE_TAP_MAX_INTERVAL_STYLUS_MS
    : BRUSH_DOUBLE_TAP_MAX_INTERVAL_MOUSE_MS;
  const now = perfNow();
  const lastTime = brushTapHistory[type] || 0;
  const lastPos = brushTapLastPos[type];
  const pointerPos = { x: pointer.x, y: pointer.y };
  brushTapHistory[type] = now;
  brushTapLastPos[type] = pointerPos;
  if (!lastTime || (now - lastTime) > maxInterval) {
    brushTapUndoBaseline = typeof OmniHistory.getUndoCount === 'function'
      ? OmniHistory.getUndoCount()
      : null;
    return null;
  }
  if (lastPos) {
    const dp = Math.hypot(pointerPos.x - lastPos.x, pointerPos.y - lastPos.y);
    if (dp > BRUSH_DOUBLE_TAP_MAX_POINTER_DISTANCE) {
      brushTapUndoBaseline = typeof OmniHistory.getUndoCount === 'function'
        ? OmniHistory.getUndoCount()
        : null;
      return null;
    }
  }
  brushTapHistory[type] = 0;
  brushTapLastPos[type] = null;
  return {
    pointerType: type,
    pointerId: evt.pointerId,
    world: { x: world.x, y: world.y },
    undoBaseline: brushTapUndoBaseline,
  };
}

function updateColorModeLabel() {
  if (!colorMode) {
    return;
  }
  const mode = nColorActive ? 'N-Color' : 'Palette';
  colorMode.textContent = 'Mask Colors: ' + mode + " (toggle with 'N')";
}


function updateNColorPanel() {
  if (!cmapPanel || typeof nColorActive === 'undefined') {
    return;
  }
  cmapPanel.classList.toggle('ncolor-active', Boolean(nColorActive));
  // Also update the subsection visibility
  if (ncolorSubsection) {
    ncolorSubsection.classList.toggle('ncolor-active', Boolean(nColorActive));
  }
}

function renderNColorSwatches() {
  if (!ncolorSwatches) {
    return;
  }
  ncolorSwatches.innerHTML = '';
  const palette = nColorPaletteColors.length ? nColorPaletteColors : generateNColorSwatches(DEFAULT_NCOLOR_COUNT);
  palette.forEach((rgb, idx) => {
    const swatch = document.createElement('button');
    swatch.type = 'button';
    swatch.className = 'ncolor-swatch';
    swatch.style.setProperty('--swatch-color', rgbToHex(rgb));
    const input = document.createElement('input');
    input.type = 'color';
    input.value = rgbToHex(rgb);
    input.addEventListener('input', (evt) => {
      const next = hexToRgb(evt.target.value);
      const updated = palette.slice();
      updated[idx] = next;
      setNColorPaletteColors(updated, { render: true });
    });
    swatch.appendChild(input);
    swatch.addEventListener('click', () => input.click());
    ncolorSwatches.appendChild(swatch);
  });
  // Update colormap preview
  updateNColorColormapPreview();
}

/**
 * Update the N-color colormap preview pill with current palette colors.
 */
function updateNColorColormapPreview() {
  // Now handled by updateCmapPanelUI for the unified preview
  updateCmapPanelUI();
}

/**
 * Update the instance colormap preview pill with current colormap.
 */
function updateInstanceColormapPreview() {
  // Now handled by updateCmapPanelUI for the unified preview
  updateCmapPanelUI();
}

function setNColorPaletteColors(colors, { render = true, schedule = true } = {}) {
  nColorPaletteColors = Array.isArray(colors) ? colors.map((c) => c.slice(0, 3)) : [];
  if (render) {
    renderNColorSwatches();
  }
  if (nColorActive) {
    clearColorCaches();
    paletteTextureDirty = true;
    if (isWebglPipelineActive()) {
      markMaskTextureFullDirty();
      markOutlineTextureFullDirty();
    } else {
      redrawMaskCanvas();
    }
    draw();
  }
  if (schedule) {
    scheduleStateSave();
  }
}

function buildPaletteTextureData() {
  const size = PALETTE_TEXTURE_SIZE;
  const data = new Uint8Array(size * 4);
  if (nColorActive) {
    const palette = nColorPaletteColors.length ? nColorPaletteColors : generateNColorSwatches(DEFAULT_NCOLOR_COUNT, 0.35);
    const count = palette.length || 1;
    for (let i = 0; i < size; i += 1) {
      // Use same indexing as getNColorLabelColor: (label - 1) % count
      // For i=0 (background), use black. For i>=1, use (i-1) % count
      const rgb = i === 0 ? [0, 0, 0] : (palette[(i - 1) % count] || [0, 0, 0]);
      const base = i * 4;
      data[base] = rgb[0] || 0;
      data[base + 1] = rgb[1] || 0;
      data[base + 2] = rgb[2] || 0;
      data[base + 3] = 255;
    }
    return data;
  }
  // Instance mode: generate unique colors for each label
  // CRITICAL: The shader accesses palette[label] directly, so palette[i] must contain the color for label i.
  // palette[0] = background (label 0), palette[1] = color for label 1, etc.
  for (let i = 0; i < size; i += 1) {
    const rgb = i === 0 ? [0, 0, 0] : (getColormapColor(i) || [0, 0, 0]);
    const base = i * 4;
    data[base] = rgb[0] || 0;
    data[base + 1] = rgb[1] || 0;
    data[base + 2] = rgb[2] || 0;
    data[base + 3] = 255;
  }
  return data;
}

function updatePaletteTextureIfNeeded() {
  if (!paletteTextureDirty) return;
  if (!isWebglPipelineActive() || !webglPipeline || !webglPipeline.paletteTexture) return;
  const { gl } = webglPipeline;
  const data = buildPaletteTextureData();
  gl.bindTexture(gl.TEXTURE_2D, webglPipeline.paletteTexture);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, PALETTE_TEXTURE_SIZE, 1, gl.RGBA, gl.UNSIGNED_BYTE, data);
  gl.bindTexture(gl.TEXTURE_2D, null);
  paletteTextureDirty = false;
}

function updateLabelShuffleControls() {
  if (!labelShuffleToggle) {
    return;
  }
  if (labelShuffleSeedInput) {
    labelShuffleSeedInput.disabled = !labelShuffle;
  }
}

function formatLabelsFromCurrentMask() {
  return new Promise((resolve) => {
    try {
      const buf = maskValues.buffer.slice(maskValues.byteOffset, maskValues.byteOffset + maskValues.byteLength);
      const bytes = new Uint8Array(buf);
      const b64 = base64FromUint8(bytes);
      const payload = { mask: b64, width: imgWidth, height: imgHeight };
      const applyMapping = (obj) => {
        try {
          if (!obj || !obj.mask) { resolve(false); return; }
          const bin = atob(obj.mask);
          if (bin.length !== maskValues.length * 4) { resolve(false); return; }
          const buffer = new ArrayBuffer(bin.length);
          const arr = new Uint8Array(buffer);
          for (let i = 0; i < bin.length; i += 1) arr[i] = bin.charCodeAt(i);
          const labels = new Uint32Array(buffer);
          let hasNonZero = false;
          for (let i = 0; i < labels.length; i += 1) {
            maskValues[i] = labels[i];
            if (labels[i] > 0) {
              hasNonZero = true;
            }
          }
          maskHasNonZero = hasNonZero;
          clearColorCaches();
          paletteTextureDirty = true;
          if (isWebglPipelineActive()) {
            markMaskTextureFullDirty();
            markOutlineTextureFullDirty();
          } else {
            redrawMaskCanvas();
          }
          draw();
          resolve(true);
        } catch (e) { resolve(false); }
      };
      if (window.pywebview && window.pywebview.api && typeof window.pywebview.api.format_labels === 'function') {
        window.pywebview.api.format_labels(payload).then(applyMapping).catch(() => resolve(false));
        return;
      }
      if (typeof fetch === 'function') {
        fetch('/api/format_labels', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
          .then((res) => (res.ok ? res.json() : Promise.reject(new Error('HTTP ' + res.status))))
          .then(applyMapping)
          .catch(() => resolve(false));
        return;
      }
      resolve(false);
    } catch (_) {
      resolve(false);
    }
  });
}

function toggleColorMode() {
  if (!nColorActive) {
    // ON: compute groups from current mask and write into maskValues.
    lastLabelBeforeNColor = currentLabel;
    const prevLabel = currentLabel | 0;
    recomputeNColorFromCurrentMask(true).then((ok) => {
      if (!ok) console.warn('N-color mapping failed');
      // Repaint outlines with new palette without changing the graph
      rebuildOutlineFromAffinity();
      // Preserve currentLabel if valid in group space; otherwise default to 1
      try {
        let maxGroup = 0;
        for (let i = 0, n = maskValues.length; i < n; i += Math.max(1, Math.floor(n / 2048))) {
          const g = maskValues[i] | 0; if (g > maxGroup) maxGroup = g;
        }
        // Fallback full scan if sample returned 0
        if (maxGroup === 0) {
          for (let i = 0; i < maskValues.length; i += 1) { const g = maskValues[i] | 0; if (g > maxGroup) maxGroup = g; }
        }
        if (!(prevLabel >= 1 && prevLabel <= maxGroup)) {
          currentLabel = maxGroup >= 1 ? 1 : 0;
        }
      } catch (_) { currentLabel = 1; }
      updateMaskLabel();
      updateColorModeLabel();
      paletteTextureDirty = true;
      if (autoNColorToggle) autoNColorToggle.checked = nColorActive;
      updateNColorPanel();
      draw();
      scheduleStateSave();
    });
    return;
  }
  // OFF: restore saved instance mask directly (don't recompute from affinity graph)
  // The nColorInstanceMask was saved when N-color was turned ON - use it directly
  const prevLabel = currentLabel | 0;
  const savedInstance = (nColorInstanceMask && nColorInstanceMask.length === maskValues.length)
    ? nColorInstanceMask
    : null;

  if (savedInstance) {
    // Restore directly from saved instance mask
    // Count unique labels to verify
    const uniqueLabels = new Set();
    for (let i = 0; i < savedInstance.length; i++) {
      if (savedInstance[i] > 0) uniqueLabels.add(savedInstance[i]);
    }
    log(`[ncolor] OFF: restoring saved instance mask directly (${uniqueLabels.size} unique labels)`);
    maskValues.set(savedInstance);
    maskHasNonZero = true;
    nColorActive = false;
    nColorInstanceMask = null;
  } else {
    // No saved mask - fall back to relabeling from affinity (legacy behavior)
    log('[ncolor] OFF: no saved instance mask, falling back to relabelFromAffinity');
    if (!affinityGraphInfo || !affinityGraphInfo.values) {
      if (savedAffinityGraphPayload && savedAffinityGraphPayload.encoded) {
        applyAffinityGraphPayload(savedAffinityGraphPayload);
      } else {
        rebuildLocalAffinityGraph();
      }
    }
    relabelFromAffinity()
      .then((ok) => {
        if (!ok) {
          console.warn('relabel_from_affinity failed during N-color OFF');
        }
        nColorActive = false;
        nColorInstanceMask = null;
        finishNColorOff(prevLabel);
      })
      .catch((err) => {
        console.warn('N-color OFF relabel failed', err);
        nColorActive = false;
        nColorInstanceMask = null;
        finishNColorOff(prevLabel);
      });
    return; // async path
  }

  // Synchronous path - finish immediately
  finishNColorOff(prevLabel);

  function finishNColorOff(prevLabel) {
    // CRITICAL: Update currentMaxLabel to reflect actual labels in restored mask
    updateMaxLabelFromMask();
    clearColorCaches();
    paletteTextureDirty = true;
    if (isWebglPipelineActive()) {
      markMaskTextureFullDirty();
      markOutlineTextureFullDirty();
    } else {
      redrawMaskCanvas();
    }
    // Preserve currentLabel if still present; else default to 1
    try {
      let found = false;
      if (prevLabel > 0) {
        for (let i = 0; i < maskValues.length; i += Math.max(1, Math.floor(maskValues.length / 2048))) {
          if ((maskValues[i] | 0) === prevLabel) { found = true; break; }
        }
        if (!found) {
          for (let i = 0; i < maskValues.length; i += 1) { if ((maskValues[i] | 0) === prevLabel) { found = true; break; } }
        }
      }
      if (!found) {
        currentLabel = 1;
      }
    } catch (_) { currentLabel = 1; }
    updateMaskLabel();
    updateColorModeLabel();
    paletteTextureDirty = true;
    if (autoNColorToggle) autoNColorToggle.checked = nColorActive;
    updateNColorPanel();
    draw();
    scheduleStateSave();
  }
}



// buildLinksFromCurrentGraph removed: we keep a single connectivity source based on raw labels only

function getSegmentationSettingsPayload() {
  const payload = {
    mask_threshold: Number(maskThreshold.toFixed(2)),
    flow_threshold: Number(flowThreshold.toFixed(2)),
    cluster: Boolean(clusterEnabled),
    affinity_seg: Boolean(affinitySegEnabled),
    niter: niterAuto ? null : (niter | 0),
    model: segmentationModel,
    use_gpu: useGpuToggle ? Boolean(useGpuToggle.checked) : false,
  };
  if (customSegmentationModelPath) {
    payload.model_path = customSegmentationModelPath;
  }
  if (sessionId) {
    payload.sessionId = sessionId;
  }
  if (currentImagePath) {
    payload.image_path = currentImagePath;
  }
  return payload;
}


let tooltipState = null;
function showConfirmDialog(message, { confirmText = 'OK', cancelText = 'Cancel' } = {}) {
  return new Promise((resolve) => {
    const backdrop = document.createElement('div');
    backdrop.className = 'omni-confirm-backdrop';
    const dialog = document.createElement('div');
    dialog.className = 'omni-confirm';
    const msg = document.createElement('div');
    msg.className = 'omni-confirm-message';
    msg.textContent = message;
    const actions = document.createElement('div');
    actions.className = 'omni-confirm-actions';
    const cancelBtn = document.createElement('button');
    cancelBtn.type = 'button';
    cancelBtn.className = 'omni-confirm-button';
    cancelBtn.textContent = cancelText;
    const okBtn = document.createElement('button');
    okBtn.type = 'button';
    okBtn.className = 'omni-confirm-button omni-confirm-primary';
    okBtn.textContent = confirmText;
    actions.appendChild(cancelBtn);
    actions.appendChild(okBtn);
    dialog.appendChild(msg);
    dialog.appendChild(actions);
    backdrop.appendChild(dialog);
    document.body.appendChild(backdrop);
    const cleanup = (result) => {
      backdrop.remove();
      document.removeEventListener('keydown', onKey);
      resolve(result);
    };
    const onKey = (evt) => {
      if (evt.key === 'Enter') {
        evt.preventDefault();
        cleanup(true);
      } else if (evt.key === 'Escape') {
        evt.preventDefault();
        cleanup(false);
      }
    };
    document.addEventListener('keydown', onKey);
    cancelBtn.addEventListener('click', () => cleanup(false));
    okBtn.addEventListener('click', () => cleanup(true));
    backdrop.addEventListener('click', (evt) => {
      if (evt.target === backdrop) cleanup(false);
    });
    setTimeout(() => okBtn.focus(), 0);
  });
}

function initTooltips() {
  if (tooltipState) {
    return;
  }
  const tooltip = document.createElement('div');
  tooltip.className = 'omni-tooltip';
  tooltip.setAttribute('role', 'tooltip');
  document.body.appendChild(tooltip);
  tooltipState = {
    tooltip,
    timer: null,
    target: null,
    lastPoint: null,
  };

  document.querySelectorAll('[title]').forEach((el) => {
    const title = el.getAttribute('title');
    if (title) {
      el.dataset.tooltip = title;
      el.removeAttribute('title');
    }
  });

  const getTooltipText = (el) => {
    if (!el) return '';
    const existing = el.dataset.tooltip;
    if (existing) return existing;
    const title = el.getAttribute('title');
    if (title) {
      el.dataset.tooltip = title;
      el.removeAttribute('title');
      return title;
    }
    return '';
  };

  const positionTooltip = (x, y, target) => {
    const tip = tooltipState.tooltip;
    if (!tip) return;
    const padding = 12;
    const offset = 14;
    let left = x + offset;
    let top = y + offset;
    const rect = tip.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    if (left + rect.width + padding > vw) {
      left = Math.max(padding, x - rect.width - offset);
    }
    if (top + rect.height + padding > vh) {
      top = Math.max(padding, y - rect.height - offset);
    }
    if (Number.isFinite(left) && Number.isFinite(top)) {
      tip.style.left = `${left}px`;
      tip.style.top = `${top}px`;
    } else if (target) {
      const trect = target.getBoundingClientRect();
      tip.style.left = `${Math.min(vw - rect.width - padding, Math.max(padding, trect.left))}px`;
      tip.style.top = `${Math.min(vh - rect.height - padding, Math.max(padding, trect.bottom + offset))}px`;
    }
  };

  const showTooltip = (el, point) => {
    const text = getTooltipText(el);
    if (!text) return;
    const tip = tooltipState.tooltip;
    tip.textContent = text;
    tip.classList.add('visible');
    const { x, y } = point || { x: 0, y: 0 };
    positionTooltip(x, y, el);
  };

  const hideTooltip = () => {
    if (!tooltipState) return;
    const tip = tooltipState.tooltip;
    if (tooltipState.timer) {
      clearTimeout(tooltipState.timer);
      tooltipState.timer = null;
    }
    tooltipState.target = null;
    tip.classList.remove('visible');
  };

  const scheduleTooltip = (target, point) => {
    if (!target) return;
    if (tooltipState.timer) {
      clearTimeout(tooltipState.timer);
      tooltipState.timer = null;
    }
    tooltipState.target = target;
    tooltipState.lastPoint = point;
    tooltipState.timer = setTimeout(() => {
      if (tooltipState && tooltipState.target === target) {
        showTooltip(target, tooltipState.lastPoint);
      }
    }, 200);
  };

  const refreshDynamicTooltip = (target, point) => {
    if (!target || target.dataset.tooltipDynamic !== 'true') {
      return false;
    }
    if (target.closest('[data-tooltip-disabled="true"]')) {
      return false;
    }
    const text = getTooltipText(target);
    if (!text) {
      if (tooltipState.target === target) {
        hideTooltip();
      }
      return true;
    }
    if (tooltipState.tooltip.classList.contains('visible') && tooltipState.target === target) {
      tooltipState.tooltip.textContent = text;
      positionTooltip(point.x, point.y, target);
      return true;
    }
    scheduleTooltip(target, point);
    return true;
  };

  document.addEventListener('pointerover', (evt) => {
    const target = evt.target.closest('[data-tooltip], [title], [data-tooltip-dynamic="true"]');
    if (!target) return;
    if (target.closest('[data-tooltip-disabled="true"]')) return;
    scheduleTooltip(target, { x: evt.clientX, y: evt.clientY });
  });

  document.addEventListener('pointermove', (evt) => {
    if (!tooltipState) return;
    const point = { x: evt.clientX, y: evt.clientY };
    tooltipState.lastPoint = point;
    const dynamicTarget = evt.target.closest('[data-tooltip-dynamic="true"]');
    if (dynamicTarget) {
      refreshDynamicTooltip(dynamicTarget, point);
    }
    if (tooltipState.tooltip.classList.contains('visible')) {
      positionTooltip(point.x, point.y, tooltipState.target);
    }
  });

  document.addEventListener('pointerout', (evt) => {
    if (!tooltipState) return;
    const target = evt.target.closest('[data-tooltip], [title], [data-tooltip-dynamic="true"]');
    if (!target) return;
    if (target.closest('[data-tooltip-disabled="true"]')) return;

    if (tooltipState.target === target) {
      hideTooltip();
    }
  });

  document.addEventListener('focusin', (evt) => {
    const target = evt.target.closest('[data-tooltip], [title], [data-tooltip-dynamic="true"]');
    if (!target) return;
    if (target.closest('[data-tooltip-disabled="true"]')) return;

    if (tooltipState.timer) clearTimeout(tooltipState.timer);
    tooltipState.target = target;
    const rect = target.getBoundingClientRect();
    tooltipState.lastPoint = { x: rect.left + rect.width / 2, y: rect.top };
    tooltipState.timer = setTimeout(() => {
      if (tooltipState && tooltipState.target === target) {
        showTooltip(target, tooltipState.lastPoint);
      }
    }, 200);
  });

  document.addEventListener('focusout', () => {
    hideTooltip();
  });
}

const overlayUpdateThrottleMs = 140;
const DEBUG_AFFINITY = (typeof window !== 'undefined' && Boolean(window.__DEBUG_AFFINITY));

function debugAffinity(message) {
  if (!DEBUG_AFFINITY || typeof log !== 'function') {
    return;
  }
  log(message);
}
let lastRebuildTime = 0;
let INTERACTIVE_REBUILD_INTERVAL = 120;
let lastRebuildDuration = 120;

// Incremental painting pipeline
// immediate redraw path (no incremental mask pipeline)

function scheduleMaskRebuild({ interactive = false } = {}) {
  // Avoid kicking off expensive recomputes while the user is painting.
  if (isPainting) {
    pendingMaskRebuild = true;
    return;
  }
  if (!canRebuildMask) {
    if (!isSegmenting) {
      runSegmentation();
    }
    return;
  }
  const now = Date.now();
  if (interactive && !isSegmenting && (now - lastRebuildTime) >= INTERACTIVE_REBUILD_INTERVAL) {
    pendingMaskRebuild = false;
    triggerMaskRebuild(true);
    return;
  }
  pendingMaskRebuild = true;
  if (segmentationUpdateTimer !== null) {
    clearTimeout(segmentationUpdateTimer);
  }
  let delay = SEGMENTATION_UPDATE_DELAY;
  if (interactive) {
    const elapsed = now - lastRebuildTime;
    const remaining = Math.max(10, INTERACTIVE_REBUILD_INTERVAL - elapsed);
    delay = Math.min(Math.max(10, remaining), SEGMENTATION_UPDATE_DELAY);
  }
  segmentationUpdateTimer = setTimeout(() => {
    segmentationUpdateTimer = null;
    if (isSegmenting) {
      return;
    }
    pendingMaskRebuild = false;
    triggerMaskRebuild(interactive);
  }, delay);
}

async function requestMaskRebuild() {
  const payload = getSegmentationSettingsPayload();
  payload.mode = 'recompute';
  if (window.pywebview && window.pywebview.api) {
    const api = window.pywebview.api;
    if (typeof api.resegment === 'function') {
      return api.resegment(payload);
    }
    if (typeof api.segment === 'function') {
      return api.segment(payload);
    }
  }
  const requestInit = {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  };
  let response = await fetch('/api/resegment', requestInit);
  if (!response.ok) {
    if (response.status === 404) {
      response = await fetch('/api/segment', requestInit);
    }
    if (!response.ok) {
      throw new Error('HTTP ' + response.status);
    }
  }
  return response.json();
}

async function triggerMaskRebuild(interactive = false) {
  // Defer rebuilds during paint to keep interactions smooth and to avoid
  // overwriting in-progress edits with async responses.
  if (isPainting) {
    pendingMaskRebuild = true;
    return;
  }
  if (!canRebuildMask) {
    return;
  }
  if (isSegmenting) {
    pendingMaskRebuild = true;
    return;
  }
  pendingMaskRebuild = false;
  isSegmenting = true;
  const startTime = Date.now();
  setSegmentStatus('Updating mask…');
  try {
    const raw = await requestMaskRebuild();
    const payload = typeof raw === 'string' ? JSON.parse(raw) : raw;
    if (payload && payload.error) {
      throw new Error(payload.error);
    }
    applySegmentationMask(payload);
    setSegmentStatus('Masks updated.');
    const endTime = Date.now();
    lastRebuildTime = endTime;
    const duration = Math.max(1, endTime - startTime);
    lastRebuildDuration = duration;
    if (interactive) {
      INTERACTIVE_REBUILD_INTERVAL = Math.max(10, Math.min(200, duration));
    } else {
      INTERACTIVE_REBUILD_INTERVAL = Math.max(20, Math.min(300, duration));
    }
  } catch (err) {
    const message = err && err.message ? err.message : err;
    setSegmentStatus('Mask update failed: ' + message, true);
    if (typeof message === 'string' && message.toLowerCase().includes('no cached segmentation data available')) {
      canRebuildMask = false;
    }
  } finally {
    isSegmenting = false;
    if (pendingMaskRebuild && segmentationUpdateTimer === null) {
      const shouldRetry = pendingMaskRebuild;
      pendingMaskRebuild = false;
      if (shouldRetry) {
        scheduleMaskRebuild();
      }
    }
  }
}

function updateOverlayImages(payload) {
  if (!payload) {
    return;
  }
  if (Object.prototype.hasOwnProperty.call(payload, 'flowOverlay')) {
    setOverlayImage('flow', payload.flowOverlay);
  }
  if (Object.prototype.hasOwnProperty.call(payload, 'distanceOverlay')) {
    setOverlayImage('distance', payload.distanceOverlay);
  }
}

function setOverlayImage(kind, dataUrl) {
  const isFlow = kind === 'flow';
  const isDistance = kind === 'distance';
  if (!isFlow && !isDistance) {
    return;
  }
  const toggle = isFlow ? flowOverlayToggle : distanceOverlayToggle;
  const currentSource = isFlow ? flowOverlaySource : distanceOverlaySource;
  const currentImage = isFlow ? flowOverlayImage : distanceOverlayImage;
  if (!toggle) {
    return;
  }
  if (!dataUrl) {
    toggle.checked = false;
    toggle.disabled = true;
    if (isFlow) {
      flowOverlayImage = null;
      flowOverlaySource = null;
      showFlowOverlay = false;
    } else if (isDistance) {
      distanceOverlayImage = null;
      distanceOverlaySource = null;
      showDistanceOverlay = false;
    }
    if (isWebglPipelineActive()) {
      updateOverlayTexture(kind, null);
    }
    draw();
    return;
  }
  const url = typeof dataUrl === 'string' && dataUrl.startsWith('data:')
    ? dataUrl
    : 'data:image/png;base64,' + dataUrl;
  if (url === currentSource && currentImage && currentImage.complete) {
    if (isWebglPipelineActive()) {
      updateOverlayTexture(kind, currentImage);
    }
    toggle.disabled = false;
    if (toggle.checked) {
      if (isFlow) {
        showFlowOverlay = true;
      } else if (isDistance) {
        showDistanceOverlay = true;
      }
      draw();
    }
    return;
  }
  const image = new Image();
  image.onload = () => {
    if (isFlow) {
      flowOverlayImage = image;
      flowOverlaySource = url;
      if (toggle.checked) {
        showFlowOverlay = true;
      }
    } else if (isDistance) {
      distanceOverlayImage = image;
      distanceOverlaySource = url;
      if (toggle.checked) {
        showDistanceOverlay = true;
      }
    }
    if (isWebglPipelineActive()) {
      updateOverlayTexture(kind, image);
    }
    toggle.disabled = false;
    draw();
  };
  image.onerror = () => {
    if (isFlow) {
      flowOverlayImage = null;
      flowOverlaySource = null;
      showFlowOverlay = false;
    } else if (isDistance) {
      distanceOverlayImage = null;
      distanceOverlaySource = null;
      showDistanceOverlay = false;
    }
    if (isWebglPipelineActive()) {
      updateOverlayTexture(kind, null);
    }
    toggle.checked = false;
    toggle.disabled = true;
    draw();
  };
  toggle.disabled = true;
  image.src = url;
}

function setSegmentStatus(message, isError = false) {
  if (!segmentStatus) {
    return;
  }
  segmentStatus.textContent = message || '';
  segmentStatus.style.color = isError ? '#ff8a8a' : '#9aa';
}



function updatePointsOverlayBuffers() {
  if (!pointsOverlay || !pointsOverlay.enabled) {
    return;
  }
  const gl = pointsOverlay.gl;
  if (!pointsOverlay.positionBuffer || !pointsOverlay.colorBuffer) {
    return;
  }
  if (!pointsOverlay || !pointsOverlay.pointsPositions || !pointsOverlay.pointsColors || !pointsOverlay.pointsCount) {
    pointsOverlay.pointsCount = 0;
    gl.bindBuffer(gl.ARRAY_BUFFER, pointsOverlay.positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, pointsOverlay.colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(), gl.STATIC_DRAW);
    return;
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, pointsOverlay.positionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, pointsOverlay.pointsPositions, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, pointsOverlay.colorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, pointsOverlay.pointsColors, gl.STATIC_DRAW);
  pointsOverlay.pointsCount = pointsOverlay.pointsCount | 0;
  pointsOverlay.width = pointsOverlayData ? (pointsOverlayData.width | 0) : 0;
  pointsOverlay.height = pointsOverlayData ? (pointsOverlayData.height | 0) : 0;
}


function buildVectorOverlayData(pointsPositions, count, width, height) {
  if (!pointsPositions || !count || !maskValues || maskValues.length !== width * height) {
    return null;
  }
  const startPositions = new Float32Array(count * 2);
  let idx = 0;
  for (let y = 0; y < height; y += 1) {
    const rowOffset = y * width;
    for (let x = 0; x < width; x += 1) {
      if (maskValues[rowOffset + x] > 0) {
        if (idx >= count) {
          break;
        }
        startPositions[idx * 2] = x + 0.5;
        startPositions[idx * 2 + 1] = y + 0.5;
        idx += 1;
      }
    }
    if (idx >= count) {
      break;
    }
  }
  const actual = Math.min(idx, count);
  if (actual <= 0) {
    return null;
  }
  const segmentsPer = 3; // main + 2 arrow wings
  const vertexCount = actual * segmentsPer * 2;
  const positions = new Float32Array(vertexCount * 2);
  const colors = new Uint8Array(vertexCount * 4);
  let v = 0;
  const alpha = 200;
  for (let i = 0; i < actual; i += 1) {
    const sx = startPositions[i * 2];
    const sy = startPositions[i * 2 + 1];
    const ex = pointsPositions[i * 2];
    const ey = pointsPositions[i * 2 + 1];
    const dx = ex - sx;
    const dy = ey - sy;
    const len = Math.hypot(dx, dy) || 1;
    const ux = dx / len;
    const uy = dy / len;
    const arrowLen = Math.min(VECTOR_ARROW_LENGTH, len * 0.5);
    const arrowWidth = VECTOR_ARROW_WIDTH;
    const baseX = ex - ux * arrowLen;
    const baseY = ey - uy * arrowLen;
    const px = -uy;
    const py = ux;
    const lx = baseX + px * arrowWidth;
    const ly = baseY + py * arrowWidth;
    const rx = baseX - px * arrowWidth;
    const ry = baseY - py * arrowWidth;

    // main line
    positions[v] = sx; positions[v + 1] = sy; v += 2;
    positions[v] = ex; positions[v + 1] = ey; v += 2;
    // left wing
    positions[v] = ex; positions[v + 1] = ey; v += 2;
    positions[v] = lx; positions[v + 1] = ly; v += 2;
    // right wing
    positions[v] = ex; positions[v + 1] = ey; v += 2;
    positions[v] = rx; positions[v + 1] = ry; v += 2;
  }
  for (let i = 0; i < vertexCount; i += 1) {
    const o = i * 4;
    colors[o] = 255;
    colors[o + 1] = 255;
    colors[o + 2] = 255;
    colors[o + 3] = alpha;
  }
  return {
    positions,
    colors,
    vertexCount,
    width,
    height,
  };
}

function findNearestPointIndex(world, maxDist = POINT_PICK_RADIUS) {
  if (!pointsOverlay || !pointsOverlay.pointsPositions || !pointsOverlay.pointsCount) {
    return -1;
  }
  const positions = pointsOverlay.pointsPositions;
  let best = -1;
  let bestDist = maxDist;
  for (let i = 0; i < pointsOverlay.pointsCount; i += 1) {
    const dx = positions[i * 2] - world.x;
    const dy = positions[i * 2 + 1] - world.y;
    const d = Math.hypot(dx, dy);
    if (d <= bestDist) {
      bestDist = d;
      best = i;
    }
  }
  return best;
}

function setPointColor(index, color) {
  if (!pointsOverlay || !pointsOverlay.pointsColors || index < 0) {
    return;
  }
  const colors = pointsOverlay.pointsColors;
  const offset = index * 4;
  colors[offset] = color[0];
  colors[offset + 1] = color[1];
  colors[offset + 2] = color[2];
  colors[offset + 3] = color[3];
  if (webglOverlay && webglOverlay.pointsColors) {
    webglOverlay.pointsColors.set(colors);
  }
  updatePointsOverlayBuffers();
}

function restoreSelectedPoints() {
  if (!selectedPointCoords.length || !pointsOverlay || !pointsOverlay.pointsPositions) {
    return;
  }
  selectedPointIndices = new Set();
  const positions = pointsOverlay.pointsPositions;
  selectedPointCoords.forEach((pt) => {
    let best = -1;
    let bestDist = POINT_PICK_RADIUS;
    for (let i = 0; i < pointsOverlay.pointsCount; i += 1) {
      const dx = positions[i * 2] - pt.x;
      const dy = positions[i * 2 + 1] - pt.y;
      const d = Math.hypot(dx, dy);
      if (d <= bestDist) {
        bestDist = d;
        best = i;
      }
    }
    if (best >= 0) {
      selectedPointIndices.add(best);
      setPointColor(best, POINT_SELECT_COLOR);
    }
  });
}

function togglePointSelectionAt(world) {
  const index = findNearestPointIndex(world);
  if (index < 0) {
    return false;
  }
  const positions = pointsOverlay.pointsPositions;
  const x = positions[index * 2];
  const y = positions[index * 2 + 1];
  if (selectedPointIndices.has(index)) {
    selectedPointIndices.delete(index);
    selectedPointCoords = selectedPointCoords.filter((pt) => Math.hypot(pt.x - x, pt.y - y) > 0.5);
    setPointColor(index, POINT_DEFAULT_COLOR);
  } else {
    selectedPointIndices.add(index);
    selectedPointCoords.push({ x, y });
    setPointColor(index, POINT_SELECT_COLOR);
  }
  return true;
}

function applyPointsPayload(pointsPayload) {
  const gl = webglOverlay && webglOverlay.gl ? webglOverlay.gl : null;
  if (!pointsPayload || !pointsPayload.encoded || !pointsPayload.count) {
    pointsOverlayInfo = null;
    pointsOverlayData = null;
    vectorOverlayInfo = null;
    vectorOverlayData = null;
    selectedPointCoords = [];
    selectedPointIndices = new Set();
    if (pointsOverlay) {
      pointsOverlay.pointsCount = 0;
      pointsOverlay.pointsPositions = null;
      pointsOverlay.pointsColors = null;
      if (vectorOverlay) {
        vectorOverlay.vertexCount = 0;
        vectorOverlay.positions = null;
        vectorOverlay.colors = null;
      }
      if (pointsOverlay.gl && pointsOverlay.positionBuffer) {
        pointsOverlay.gl.bindBuffer(pointsOverlay.gl.ARRAY_BUFFER, pointsOverlay.positionBuffer);
        pointsOverlay.gl.bufferData(pointsOverlay.gl.ARRAY_BUFFER, new Float32Array(), pointsOverlay.gl.STATIC_DRAW);
      }
      if (pointsOverlay.gl && pointsOverlay.colorBuffer) {
        pointsOverlay.gl.bindBuffer(pointsOverlay.gl.ARRAY_BUFFER, pointsOverlay.colorBuffer);
        pointsOverlay.gl.bufferData(pointsOverlay.gl.ARRAY_BUFFER, new Uint8Array(), pointsOverlay.gl.STATIC_DRAW);
      }
      if (pointsOverlay.canvas) {
        pointsOverlay.canvas.style.display = 'none';
        pointsOverlay.canvas.style.opacity = '0';
      }
    }
    if (webglOverlay) { webglOverlay.pointsCount = 0; }
    if (gl && webglOverlay && webglOverlay.pointsPositionBuffer) {
      gl.bindBuffer(gl.ARRAY_BUFFER, webglOverlay.pointsPositionBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(), gl.DYNAMIC_DRAW);
    }
    if (gl && webglOverlay && webglOverlay.pointsColorBuffer) {
      gl.bindBuffer(gl.ARRAY_BUFFER, webglOverlay.pointsColorBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Uint8Array(), gl.DYNAMIC_DRAW);
    }
    if (pointsOverlayToggle) {
      pointsOverlayToggle.checked = false;
      pointsOverlayToggle.disabled = true;
    }
    if (vectorOverlayToggle) {
      vectorOverlayToggle.checked = false;
      vectorOverlayToggle.disabled = true;
    }
    if (vectorOverlayRow) {
      vectorOverlayRow.style.display = 'none';
    }
    updateOverlayVisibility();
    return;
  }
  const binary = atob(pointsPayload.encoded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  const coords = new Float32Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength / 4));
  const count = pointsPayload.count || Math.floor(coords.length / 2);

  const positions = new Float32Array(count * 2);
  const colors = new Uint8Array(count * 4);
  for (let i = 0; i < count; i += 1) {
    const y = coords[i * 2];
    const x = coords[i * 2 + 1];
    positions[i * 2] = x + 0.5;
    positions[i * 2 + 1] = y + 0.5;
    colors[i * 4] = POINT_DEFAULT_COLOR[0];
    colors[i * 4 + 1] = POINT_DEFAULT_COLOR[1];
    colors[i * 4 + 2] = POINT_DEFAULT_COLOR[2];
    colors[i * 4 + 3] = POINT_DEFAULT_COLOR[3];
  }
  pointsOverlayInfo = {
    width: pointsPayload.width,
    height: pointsPayload.height,
    count,
  };
  pointsOverlayData = pointsPayload;
  if (pointsOverlay) { pointsOverlay.pointsPositions = positions; pointsOverlay.pointsColors = colors; pointsOverlay.pointsCount = count; }
  restoreSelectedPoints();
  if (webglOverlay) { webglOverlay.pointsCount = count; }
  if (webglOverlay) { webglOverlay.pointsPositions = positions; }
  if (webglOverlay) { webglOverlay.pointsColors = colors; }

  const vectorData = buildVectorOverlayData(positions, count, pointsPayload.width, pointsPayload.height);
  vectorOverlayData = vectorData;
  if (vectorOverlay) {
    vectorOverlay.positions = vectorData ? vectorData.positions : null;
    vectorOverlay.colors = vectorData ? vectorData.colors : null;
    vectorOverlay.vertexCount = vectorData ? vectorData.vertexCount : 0;
  }
  if (gl && webglOverlay && webglOverlay.pointsPositionBuffer) {
    gl.bindBuffer(gl.ARRAY_BUFFER, webglOverlay.pointsPositionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);
  }
  if (gl && webglOverlay && webglOverlay.pointsColorBuffer) {
    gl.bindBuffer(gl.ARRAY_BUFFER, webglOverlay.pointsColorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, colors, gl.DYNAMIC_DRAW);
  }
  if (pointsOverlayToggle) {
    pointsOverlayToggle.disabled = false;
  }
  if (vectorOverlayToggle) {
    vectorOverlayToggle.disabled = !affinitySegEnabled;
  }
  if (vectorOverlayRow) {
    vectorOverlayRow.style.display = affinitySegEnabled ? '' : 'none';
  }
  updatePointsOverlayBuffers();
  updateVectorOverlayBuffers();
  updateVectorOverlayVisibility();
  updateOverlayVisibility();
}

function applySegmentationMask(payload, options = {}) {
  // If a segmentation update comes in while painting, apply it after the stroke
  if (isPainting) {
    pendingSegmentationPayload = payload;
    return;
  }
  if (!payload || !payload.mask) {
    throw new Error('segment payload missing mask');
  }
  if (payload && Object.prototype.hasOwnProperty.call(payload, 'canRebuild')) {
    canRebuildMask = Boolean(payload.canRebuild);
  } else if (!canRebuildMask) {
    canRebuildMask = true;
  }
  const forceInstanceMask = Boolean(options.forceInstanceMask);
  let maskPayload = payload.mask;
  // Debug: log what conditions are met
  log(`[applySegmentationMask] forceInstanceMask=${forceInstanceMask} nColorActive=${nColorActive} hasNColorMask=${Boolean(payload.nColorMask)}`);
  if (!forceInstanceMask && nColorActive && payload.nColorMask) {
    maskPayload = payload.nColorMask;
    try {
      const instBin = atob(payload.mask);
      log(`[applySegmentationMask] saving instance mask: instBin.length=${instBin.length} expected=${maskValues.length * 4}`);
      if (instBin.length === maskValues.length * 4) {
        const instBuf = new ArrayBuffer(instBin.length);
        const instArr = new Uint8Array(instBuf);
        for (let i = 0; i < instBin.length; i += 1) instArr[i] = instBin.charCodeAt(i);
        nColorInstanceMask = new Uint32Array(instBuf);
        // Count unique labels in saved instance mask
        const uniqueInst = new Set();
        for (let i = 0; i < nColorInstanceMask.length; i++) {
          if (nColorInstanceMask[i] > 0) uniqueInst.add(nColorInstanceMask[i]);
        }
        log(`[applySegmentationMask] saved instance mask with ${uniqueInst.size} unique labels`);
      }
    } catch (e) {
      log(`[applySegmentationMask] error saving instance mask: ${e.message}`);
    }
  }
  const binary = atob(maskPayload);
  const byteLength = binary.length;
  const expectedBytes = maskValues.length * 4;
  let changed = false;
  let hasNonZero = false;
  if (byteLength === expectedBytes) {
    const buffer = new ArrayBuffer(byteLength);
    const bytes = new Uint8Array(buffer);
    for (let i = 0; i < byteLength; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    const labels = new Uint32Array(buffer);
    for (let i = 0; i < labels.length; i += 1) {
      const value = labels[i];
      if (maskValues[i] !== value) {
        changed = true;
      }
      maskValues[i] = value;
      if (value > 0) {
        hasNonZero = true;
      }
    }
  } else if (byteLength === maskValues.length) {
    for (let i = 0; i < byteLength; i += 1) {
      const value = binary.charCodeAt(i);
      if (maskValues[i] !== value) {
        changed = true;
      }
      maskValues[i] = value;
      if (value > 0) {
        hasNonZero = true;
      }
    }
  } else {
    throw new Error('mask size mismatch (' + byteLength + ' bytes vs expected ' + expectedBytes + ')');
  }
  if (paintingApi && typeof paintingApi.rebuildComponents === 'function') {
    paintingApi.rebuildComponents();
  }
  maskHasNonZero = hasNonZero;
  // Update max label for dynamic palette sizing
  updateMaxLabelFromMask();
  // Switching to a new mask implicitly leaves current color mode as-is; caller controls nColorActive.
  resetNColorAssignments();
  clearColorCaches();
  // CRITICAL: Must rebuild palette texture to reflect current nColorActive state
  paletteTextureDirty = true;
  if (isWebglPipelineActive()) {
    markMaskTextureFullDirty();
    markOutlineTextureFullDirty();
  } else {
    redrawMaskCanvas();
  }
  updateOverlayImages(payload);
  const hasRemoteAffinity = affinitySegEnabled
    && payload.affinityGraph
    && payload.affinityGraph.encoded;
  if (affinitySegEnabled && !hasRemoteAffinity) {
    console.warn('[affinity] missing affinityGraph in payload', {
      hasAffinitySeg: affinitySegEnabled,
      hasPayload: Boolean(payload.affinityGraph),
      source: affinityGraphSource,
    });
  }
  if (hasRemoteAffinity) {
    savedAffinityGraphPayload = payload.affinityGraph;
    applyAffinityGraphPayload(payload.affinityGraph);
    if (affinityGraphSource === 'remote' && affinityGraphInfo && affinityGraphInfo.values) {
      affinityGraphNeedsLocalRebuild = false;
    }
  } else if (!affinitySegEnabled) {
    savedAffinityGraphPayload = null;
    affinityGraphSource = 'local';
    affinityGraphNeedsLocalRebuild = true;
    rebuildLocalAffinityGraph();
  } else if (!savedAffinityGraphPayload && (!affinityGraphInfo || !affinityGraphInfo.values)) {
    // Cold-start fallback: do not build local graph in affinity mode.
    if (!affinitySegEnabled) {
      affinityGraphSource = 'local';
      affinityGraphNeedsLocalRebuild = true;
      rebuildLocalAffinityGraph();
    }
  }
  applyPointsPayload(payload.points);
  scheduleStateSave();
  draw();
  // If a relabel was requested from a stroke, set currentLabel to the mode over the edited indices
  // If a relabel was requested from a stroke, set currentLabel to the mode over the edited indices
  try {
    const pending = window.__pendingRelabelSelection;
    if (pending && pending.length) {
      const counts = new Map();
      for (let i = 0; i < pending.length; i += 1) {
        const idx = pending[i] | 0;
        if (idx < 0 || idx >= maskValues.length) continue;
        const v = maskValues[idx] | 0;
        if (v > 0) counts.set(v, (counts.get(v) || 0) + 1);
      }
      let best = 0; let bestC = -1;
      counts.forEach((c, v) => { if (c > bestC) { bestC = c; best = v; } });
      if (best > 0) {
        currentLabel = best;
      }
      window.__pendingRelabelSelection = null;
    }
  } catch (_) { /* ignore */ }
  // In single-buffer mode, currentLabel remains valid across updates.
  updateMaskLabel();
  if (!nColorActive) {
    updateColorModeLabel();
  }
  if (!changed) {
    log('mask update returned identical data');
  }
  scheduleStateSave();
}

async function requestSegmentation() {
  const settings = getSegmentationSettingsPayload();
  if (window.pywebview && window.pywebview.api && window.pywebview.api.segment) {
    return window.pywebview.api.segment(settings);
  }
  const response = await fetch('/api/segment', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(settings),
  });
  if (!response.ok) {
    throw new Error('HTTP ' + response.status);
  }
  return response.json();
}


function setSegmentButtonLoading(active) {
  if (!segmentButton) {
    return;
  }
  if (active) {
    segmentButton.setAttribute('data-loading', 'true');
  } else {
    segmentButton.removeAttribute('data-loading');
  }
}

async function runSegmentation() {
  if (!segmentButton || isSegmenting) {
    return;
  }
  isSegmenting = true;
  segmentButton.disabled = true;
  setSegmentButtonLoading(true);
  setSegmentStatus('');
  try {
    const raw = await requestSegmentation();
    const payload = typeof raw === 'string' ? JSON.parse(raw) : raw;
    if (payload.error) {
      throw new Error(payload.error);
    }
    applySegmentationMask(payload);
    if (nColorActive) {
      // Only recompute N-color if payload didn't already include pre-computed nColorMask
      // (applySegmentationMask already applied it and saved the instance mask correctly)
      if (!payload.nColorMask) {
        const ok = await recomputeNColorFromCurrentMask(true);
        if (!ok) {
          console.warn('Auto N-color mapping failed');
        }
      }
      updateColorModeLabel();
      updateMaskLabel();
      scheduleStateSave();
    }
    setSegmentStatus('Segmentation complete.');
  } catch (err) {
    console.error(err);
    const msg = err && err.message ? err.message : err;
    setSegmentStatus('Segmentation failed: ' + msg, true);
  } finally {
    isSegmenting = false;
    segmentButton.disabled = false;
    setSegmentButtonLoading(false);
    if (pendingMaskRebuild && segmentationUpdateTimer === null && canRebuildMask) {
      triggerMaskRebuild().catch((error) => {
        console.error(error);
      });
    }
  }
}

canvas.addEventListener('wheel', (evt) => {
  evt.preventDefault();
  const pointer = getPointerPosition(evt);
  const imagePoint = screenToImage(pointer);
  const deltaY = Number.isFinite(evt.deltaY) ? evt.deltaY : 0;
  const deltaZ = Number.isFinite(evt.deltaZ) ? evt.deltaZ : 0;

  let scaleChanged = false;
  if (deltaY !== 0) {
    const factor = deltaY < 0 ? 1.1 : 0.9;
    const nextScale = Math.min(Math.max(viewState.scale * factor, 0.1), 40);
    if (nextScale !== viewState.scale) {
      viewState.scale = nextScale;
      scaleChanged = true;
    }
  }

  let rotationApplied = false;
  if (pointerState.options.touch.enableRotate && deltaZ !== 0) {
    wheelRotationBuffer += deltaZ;
    log('wheel gesture deltaY=' + deltaY.toFixed(2) + ' deltaZ=' + deltaZ.toFixed(2) + ' buffer=' + wheelRotationBuffer.toFixed(2));
    const threshold = Math.max(pointerState.options.touch.rotationDeadzoneDegrees || 0, 0);
    if (Math.abs(wheelRotationBuffer) >= threshold) {
      const appliedDegrees = wheelRotationBuffer;
      viewState.rotation = normalizeAngle(viewState.rotation + (appliedDegrees * Math.PI / 180));
      wheelRotationBuffer = 0;
      log('wheel rotation applied ' + appliedDegrees.toFixed(2) + ' deg');
      rotationApplied = true;
      setCursorTemporary(dotCursorCss, 500);
    }
  } else if (deltaZ === 0) {
    wheelRotationBuffer = 0;
  }

  if (scaleChanged || rotationApplied) {
    setOffsetForImagePoint(imagePoint, pointer);
    userAdjustedScale = true;
    autoFitPending = false;
    viewStateRestored = false;
    scheduleDraw();
    scheduleStateSave();
  }
  if (!rotationApplied && deltaY !== 0) {
    setCursorTemporary(dotCursorCss, 350);
  }
  // Show simple dot cursor during wheel interactions
  setCursorTemporary(dotCursorCss, 350);
}, { passive: false });

function startPointerPan(evt) {
  isPainting = false;
  if (typeof paintingApi.cancelStroke === 'function') {
    paintingApi.cancelStroke();
  }
  isPanning = true;
  lastPoint = getPointerPosition(evt);
  setCursorHold(dotCursorCss);
  wheelRotationBuffer = 0;
  try {
    canvas.setPointerCapture(evt.pointerId);
    activePointerId = evt.pointerId;
  } catch (_) {
    /* ignore */
  }
  const worldPoint = screenToImage(lastPoint);
  setHoverState(worldPoint, { x: lastPoint.x, y: lastPoint.y });
  updateHoverInfo(worldPoint);
  renderHoverPreview();
}

function beginBrushStroke(evt, worldPoint) {
  let result = null;
  if (typeof paintingApi.beginStroke === 'function') {
    result = paintingApi.beginStroke(worldPoint) || null;
  }
  isPainting = true;
  canvas.classList.add('painting');
  updateCursor();
  try {
    canvas.setPointerCapture(evt.pointerId);
    activePointerId = evt.pointerId;
  } catch (_) {
    /* ignore */
  }
  const startPoint = result && result.lastPoint ? result.lastPoint : worldPoint;
  let initialPoint = startPoint;
  if (typeof paintingApi.processPaintQueue === 'function') {
    const processed = paintingApi.processPaintQueue();
    if (processed) {
      initialPoint = processed;
    }
  }
  if (initialPoint) {
    const pointer = getPointerPosition(evt);
    const worldHover = { x: initialPoint.x, y: initialPoint.y };
    const screenHover = pointer ? { x: pointer.x, y: pointer.y } : null;
    setHoverState(worldHover, screenHover);
    drawBrushPreview(worldHover);
    updateHoverInfo(worldHover);
  } else {
    clearHoverState();
    drawBrushPreview(null);
    updateHoverInfo(null);
  }
}

canvas.addEventListener('pointerdown', (evt) => {
  cursorInsideCanvas = true;
  gestureState = null;
  pointerState.registerPointerDown(evt);
  const pointer = getPointerPosition(evt);
  lastPoint = pointer;
  const world = screenToImage(pointer);
  updateHoverInfo(world);
  const pointerType = evt.pointerType || (pointerState.isStylusPointer(evt) ? 'pen' : 'mouse');
  if (!pendingBrushDoubleTap && tool === 'brush' && (pointerType === 'pen' || pointerType === 'mouse')) {
    const tapTarget = handleBrushDoubleTapOnDown(evt, pointer, world);
    if (tapTarget) {
      pendingBrushDoubleTap = tapTarget;
      return;
    }
  }
  const isStylus = pointerState.isStylusPointer(evt);
  if (isStylus) {
    if (!pointerState.options.stylus.allowSimultaneousTouchGestures) {
      if (panPointerId !== null) {
        try {
          canvas.releasePointerCapture(panPointerId);
        } catch (_) {
          /* ignore */
        }
      }
      panPointerId = null;
      touchPointers.clear();
      pinchState = null;
      isPanning = false;
      spacePan = false;
    }
    if (pointerState.isBarrelPanActive()) {
      startPointerPan(evt);
      return;
    }
    const mode = getActiveToolMode();
    if (mode === 'fill') {
      floodFill(world);
      scheduleStateSave();
      return;
    }
    if (mode === 'picker') {
      if (showPointsOverlay && pointsOverlayInfo && togglePointSelectionAt(world)) {
        scheduleStateSave();
        return;
      }
      pickColor(world);
      scheduleStateSave();
      return;
    }
    if (mode === 'erase') {
      startEraseOverride();
    } else if (mode === 'draw' && eraseActive) {
      stopEraseOverride();
    }
    beginBrushStroke(evt, world);
    return;
  }
  if (evt.pointerType === 'touch') {
    handleTouchPointerDown(evt, pointer);
    return;
  }
  const isSecondaryButton = evt.button === 2
    || (evt.buttons & 2) !== 0
    || (evt.ctrlKey && evt.button === 0);
  if (isSecondaryButton && pointerState.options.mouse.secondaryPan) {
    if (typeof evt.preventDefault === 'function') {
      evt.preventDefault();
    }
    startPointerPan(evt);
    return;
  }
  if (evt.button === 0) {
    if (spacePan) {
      startPointerPan(evt);
      return;
    }
    if (!pointerState.options.mouse.primaryDraw) {
      startPointerPan(evt);
      return;
    }
    if (tool === 'fill') {
      floodFill(world);
      return;
    }
    if (tool === 'picker') {
      if (showPointsOverlay && pointsOverlayInfo && togglePointSelectionAt(world)) {
        scheduleStateSave();
        return;
      }
      pickColor(world);
      return;
    }
    beginBrushStroke(evt, world);
    return;
  }
  startPointerPan(evt);
});

canvas.addEventListener('pointermove', (evt) => {
  pointerState.registerPointerMove(evt);
  cursorInsideCanvas = true;
  const pointer = getPointerPosition(evt);
  const world = screenToImage(pointer);
  if (!isPanning && !isPainting && pointerState.isStylusPointer(evt) && pointerState.isBarrelPanActive()) {
    startPointerPan(evt);
    return;
  }
  if (!isPanning && evt.pointerType !== 'touch' && (evt.buttons & 2) !== 0) {
    if (typeof evt.preventDefault === 'function') {
      evt.preventDefault();
    }
    startPointerPan(evt);
    return;
  }
  if (evt.pointerType === 'touch') {
    if (pointerState.shouldIgnoreTouch(evt)) {
      evt.preventDefault();
      return;
    }
    if (touchPointers.has(evt.pointerId)) {
      touchPointers.set(evt.pointerId, { x: evt.clientX, y: evt.clientY });
    }
    if (pinchState && pinchState.pointers.every((id) => touchPointers.has(id))) {
      const rect = canvas.getBoundingClientRect();
      const [idA, idB] = pinchState.pointers;
      const ptA = touchPointers.get(idA);
      const ptB = touchPointers.get(idB);
      if (ptA && ptB) {
        const dx = ptA.x - ptB.x;
        const dy = ptA.y - ptB.y;
        const distance = Math.hypot(dx, dy);
        if (distance > 0.0 && pinchState.startDistance > 0.0) {
          let nextScale = viewState.scale;
          if (pointerState.options.touch.enablePinchZoom) {
            const ratio = distance / pinchState.startDistance;
            nextScale = Math.min(Math.max(pinchState.startScale * ratio, 0.1), 40);
          } else {
            nextScale = pinchState.startScale;
          }
          viewState.scale = nextScale;
          viewStateDirty = true;
          if (pointerState.options.touch.enableRotate) {
            const angle = Math.atan2(dy, dx);
            const delta = normalizeAngleDelta(angle - pinchState.startAngle);
            const rotationDegrees = delta * RAD_TO_DEG;
            const deadzone = Math.max(pointerState.options.touch.rotationDeadzoneDegrees || 0, 0);
            if (pinchState.lastLoggedRotation === undefined || Math.abs(rotationDegrees - pinchState.lastLoggedRotation) >= Math.max(1, deadzone * 2)) {
              if (DEBUG_AFFINITY) {
                log('pinch rotation raw=' + rotationDegrees.toFixed(2) + ' deg');
                pinchState.lastLoggedRotation = rotationDegrees;
              }
            }
            const applyRotation = Math.abs(rotationDegrees) >= deadzone;
            viewState.rotation = applyRotation
              ? normalizeAngle(pinchState.startRotation + delta)
              : pinchState.startRotation;
            if (applyRotation) {
              if (DEBUG_AFFINITY) {
                log('pinch rotation applied delta=' + rotationDegrees.toFixed(2) + ' deg total=' + (viewState.rotation * RAD_TO_DEG).toFixed(2));
              }
              viewStateDirty = true;
            }
          }
          const midpoint = {
            x: (ptA.x + ptB.x) / 2,
            y: (ptA.y + ptB.y) / 2,
          };
          const midCanvas = {
            x: midpoint.x - rect.left,
            y: midpoint.y - rect.top,
          };
          setOffsetForImagePoint(pinchState.imageMid, midCanvas);
          userAdjustedScale = true;
          autoFitPending = false;
          scheduleDraw();
          const worldPoint = screenToImage(midCanvas);
          setHoverState(worldPoint, midCanvas);
          updateHoverInfo(worldPoint);
          renderHoverPreview();
          clearInteractionPending();
        }
      }
      evt.preventDefault();
      return;
    }
    if (isPanning && evt.pointerId === panPointerId) {
      queuePanPointer({ x: pointer.x, y: pointer.y });
      schedulePointerFrame();
      if (typeof evt.preventDefault === 'function') {
        evt.preventDefault();
      }
      return;
    }
  }
  if (isPanning && evt.pointerId === panPointerId) {
    queuePanPointer({ x: pointer.x, y: pointer.y });
    schedulePointerFrame();
    if (typeof evt.preventDefault === 'function') {
      evt.preventDefault();
    }
    return;
  }
  if (isPainting) {
    if (USE_COALESCED_EVENTS && typeof evt.getCoalescedEvents === 'function') {
      const events = evt.getCoalescedEvents();
      if (events && events.length) {
        const rect = canvas.getBoundingClientRect();
        for (let i = 0; i < events.length; i += 1) {
          const e = events[i];
          const local = { x: e.clientX - rect.left, y: e.clientY - rect.top };
          const w = screenToImage(local);
          queuePaintPoint(w);
        }
      } else {
        queuePaintPoint(world);
      }
    } else {
      queuePaintPoint(world);
    }
    setHoverState(world, pointer);
    drawBrushPreview(world);
    updateHoverInfo(world);
    clearInteractionPending();
    schedulePointerFrame();
    return;
  }
  if (!isPanning && !spacePan) {
    queueHoverUpdate({ x: pointer.x, y: pointer.y }, PREVIEW_TOOL_TYPES.has(tool));
    schedulePointerFrame();
  }
  if (isPanning) {
    queuePanPointer({ x: pointer.x, y: pointer.y });
    schedulePointerFrame();
    if (typeof evt.preventDefault === 'function') {
      evt.preventDefault();
    }
  }
});

canvas.addEventListener('dblclick', (evt) => {
  if (spacePan) {
    evt.preventDefault();
    resetView();
  }
});

if (supportsGestureEvents && canvas) {
  canvas.addEventListener('gesturestart', handleGestureStart, { passive: false });
  canvas.addEventListener('gesturechange', handleGestureChange, { passive: false });
  canvas.addEventListener('gestureend', handleGestureEnd, { passive: false });
}
if (supportsGestureEvents && viewer && viewer !== canvas) {
  viewer.addEventListener('gesturestart', handleGestureStart, { passive: false });
  viewer.addEventListener('gesturechange', handleGestureChange, { passive: false });
  viewer.addEventListener('gestureend', handleGestureEnd, { passive: false });
}

function stopInteraction(evt) {
  if (evt) {
    cursorInsideCanvas = evt.type !== 'pointerleave' && evt.type !== 'pointercancel';
    if (typeof evt.clientX === 'number' && typeof evt.clientY === 'number') {
      const pointer = getPointerPosition(evt);
      const screenPoint = { x: pointer.x, y: pointer.y };
      if (CROSSHAIR_TOOL_TYPES.has(tool) || PREVIEW_TOOL_TYPES.has(tool)) {
        const world = screenToImage(pointer);
        setHoverState(world, screenPoint);
      } else {
        setHoverState(getHoverPoint(), screenPoint);
      }
    }
  }
  if (evt && (evt.type === 'pointerup' || evt.type === 'pointercancel')) {
    pointerState.registerPointerUp(evt);
  }
  if (evt && evt.button === 2 && typeof evt.preventDefault === 'function') {
    evt.preventDefault();
  }
  if (evt && evt.pointerType === 'touch') {
    touchPointers.delete(evt.pointerId);
    if (pinchState && pinchState.pointers.includes(evt.pointerId)) {
      pinchState = null;
    }
    if (evt.pointerId === panPointerId) {
      panPointerId = null;
    }
  }
  const wasPainting = isPainting;
  if (wasPainting) {
    canvas.classList.remove('painting');
  }
  // Mark painting complete before finalize so deferred overlay rebuild can run
  isPainting = false;
  if (wasPainting) {
    if (typeof paintingApi.finalizeStroke === 'function') {
      paintingApi.finalizeStroke();
    }
  }
  if (typeof paintingApi.cancelStroke === 'function') {
    paintingApi.cancelStroke();
  }
  isPanning = false;
  spacePan = false;
  updateCursor();
  clearInteractionPending();
  if (evt && evt.type === 'pointerleave') {
    clearHoverPreview();
  } else {
    updateHoverInfo(getHoverPoint() || null);
    renderHoverPreview();
  }
  if (evt && evt.pointerId !== undefined) {
    try {
      canvas.releasePointerCapture(evt.pointerId);
    } catch (_) {
      /* ignore */
    }
  } else if (activePointerId !== null) {
    try {
      canvas.releasePointerCapture(activePointerId);
    } catch (_) {
      /* ignore */
    }
  }
  activePointerId = null;
  wheelRotationBuffer = 0;
  clearCursorOverride();
}

let contextMenuEl = null;
let contextMenuBackdrop = null;

function hideContextMenu() {
  if (contextMenuEl) contextMenuEl.remove();
  if (contextMenuBackdrop) contextMenuBackdrop.remove();
  contextMenuEl = null;
  contextMenuBackdrop = null;
}

function showContextMenu(x, y) {
  hideContextMenu();
  const backdrop = document.createElement('div');
  backdrop.className = 'omni-context-backdrop';
  backdrop.addEventListener('pointerdown', hideContextMenu);
  const menu = document.createElement('div');
  menu.className = 'omni-context-menu';
  const item = document.createElement('button');
  item.type = 'button';
  item.className = 'omni-context-item';
  item.textContent = 'Clear Cache & Reload';
  item.addEventListener('click', () => {
    hideContextMenu();
    clearCacheAndReload();
  });
  menu.appendChild(item);
  document.body.appendChild(backdrop);
  document.body.appendChild(menu);
  const rect = menu.getBoundingClientRect();
  const left = Math.min(x, window.innerWidth - rect.width - 8);
  const top = Math.min(y, window.innerHeight - rect.height - 8);
  menu.style.left = `${Math.max(8, left)}px`;
  menu.style.top = `${Math.max(8, top)}px`;
  contextMenuEl = menu;
  contextMenuBackdrop = backdrop;
}

function handleContextMenuEvent(evt) {
  const target = evt && evt.target ? evt.target : null;
  const isPanelContext = target && typeof target.closest === 'function'
    ? Boolean(target.closest('#leftPanel') || target.closest('#sidebar'))
    : false;
  if (!isPanelContext) {
    return;
  }
  if (evt && typeof evt.preventDefault === 'function') {
    evt.preventDefault();
  }
  if (document.querySelector('.omni-confirm-backdrop')) {
    return;
  }
  if (evt) {
    showContextMenu(evt.clientX || 0, evt.clientY || 0);
  }
  if (activePointerId !== null) {
    try {
      canvas.releasePointerCapture(activePointerId);
    } catch (_) {
      /* ignore */
    }
    activePointerId = null;
  }
  if (isPainting) {
    if (typeof paintingApi.finalizeStroke === 'function') {
      paintingApi.finalizeStroke();
    }
  }
  if (typeof paintingApi.cancelStroke === 'function') {
    paintingApi.cancelStroke();
  }
  isPainting = false;
  isPanning = false;
  spacePan = false;
  updateCursor();
}


function maybeUndoLastStrokeAtPoint(world, label) {
  if (!world || !Number.isFinite(world.x) || !Number.isFinite(world.y)) {
    return false;
  }
  const stack = (typeof OmniHistory.getUndoStack === 'function') ? OmniHistory.getUndoStack() : null;
  if (!stack || !stack.length) {
    return false;
  }
  const entry = stack[stack.length - 1];
  if (!entry || !entry.indices || !entry.after) {
    return false;
  }
  const x = Math.min(imgWidth - 1, Math.max(0, Math.round(world.x)));
  const y = Math.min(imgHeight - 1, Math.max(0, Math.round(world.y)));
  const idx = (y * imgWidth + x) | 0;
  const indices = entry.indices;
  const after = entry.after;
  const want = typeof label === 'number' ? (label | 0) : null;
  for (let i = 0; i < indices.length; i += 1) {
    if ((indices[i] | 0) === idx) {
      if (want !== null && (after[i] | 0) !== want) {
        return false;
      }
      const popped = (typeof OmniHistory.undo === 'function') ? OmniHistory.undo() : null;
      if (!popped) {
        return false;
      }
      applyHistoryEntry(popped, false);
      updateHistoryButtons();
      return true;
    }
  }
  return false;
}

function handlePointerUp(evt) {
  const type = evt.pointerType || 'mouse';
  if (pendingBrushDoubleTap
    && pendingBrushDoubleTap.pointerType === type
    && (pendingBrushDoubleTap.pointerId === undefined || pendingBrushDoubleTap.pointerId === evt.pointerId)) {
    const target = pendingBrushDoubleTap;
    pendingBrushDoubleTap = null;
    if (typeof paintingApi.cancelStroke === 'function') {
      paintingApi.cancelStroke();
    }
    brushTapHistory[type] = 0;
    brushTapLastPos[type] = null;
    maybeUndoLastStrokeAtPoint(target.world, currentLabel);
    floodFill(target.world);
    stopInteraction(evt);
    scheduleStateSave();
    return;
  }
  pendingBrushDoubleTap = null;
  stopInteraction(evt);
}

function handlePointerCancel(evt) {
  pendingBrushDoubleTap = null;
  stopInteraction(evt);
}

canvas.addEventListener('pointerup', handlePointerUp);
canvas.addEventListener('pointerleave', (evt) => {
  stopInteraction(evt);
  clearHoverPreview();
  clearHoverState();
});
canvas.addEventListener('pointercancel', handlePointerCancel);
canvas.addEventListener('mouseenter', () => {
  cursorInsideCanvas = true;
  updateCursor();
});
canvas.addEventListener('mouseleave', () => {
  cursorInsideCanvas = false;
  cursorInsideImage = false;
  clearHoverPreview();
  updateHoverInfo(null);
  updateCursor();
});
canvas.addEventListener('contextmenu', (evt) => {
  if (evt && typeof evt.preventDefault === 'function') {
    evt.preventDefault();
  }
});
if (viewer) {
  viewer.addEventListener('contextmenu', (evt) => {
    if (evt && typeof evt.preventDefault === 'function') {
      evt.preventDefault();
    }
  });
}
if (leftPanelEl) {
  leftPanelEl.addEventListener('contextmenu', handleContextMenuEvent);
}
if (sidebarEl) {
  sidebarEl.addEventListener('contextmenu', handleContextMenuEvent);
}

window.addEventListener('keydown', (evt) => {
  const tag = evt.target && evt.target.tagName ? evt.target.tagName.toLowerCase() : '';
  if (tag === 'input') {
    return;
  }
  const key = evt.key.toLowerCase();
  const modifier = evt.ctrlKey || evt.metaKey;
  if (key === 'escape' && dropdownOpenId) {
    const entry = dropdownRegistry.get(dropdownOpenId);
    if (entry) {
      closeDropdown(entry);
    } else {
      dropdownOpenId = null;
    }
    evt.preventDefault();
    return;
  }
  if (modifier && !evt.altKey) {
    if (key >= '1' && key <= '4') {
      const idx = parseInt(key, 10) - 1;
      const mode = TOOL_MODE_ORDER[idx];
      if (mode) {
        selectToolMode(mode);
        scheduleStateSave();
      }
      evt.preventDefault();
      return;
    }
  }
  if (!modifier && !evt.altKey && !evt.repeat) {
    if (key === 'w' && hasPrevImage) {
      evt.preventDefault();
      navigateDirectory(-1);
      return;
    }
    if (key === 's' && hasNextImage) {
      evt.preventDefault();
      navigateDirectory(1);
      return;
    }
  }
  if (!modifier && !evt.altKey && key === 'e') {
    startEraseOverride();
    evt.preventDefault();
  }
  if (!modifier && !evt.altKey && key === 'h') {
    resetView();
    evt.preventDefault();
    return;
  }
  if (modifier && key === 'z') {
    if (evt.shiftKey) {
      redo();
    } else {
      undo();
    }
    evt.preventDefault();
    return;
  }
  if (modifier && key === 'y') {
    redo();
    evt.preventDefault();
    return;
  }
  if (modifier && !evt.altKey && key === 'x') {
    evt.preventDefault();
    promptClearMasks();
    return;
  }
  if (!modifier && !evt.altKey) {
    if (key === 'b') {
      selectToolMode('draw');
      scheduleStateSave();
      evt.preventDefault();
      return;
    }
    if (key === 'g') {
      selectToolMode('fill');
      scheduleStateSave();
      evt.preventDefault();
      return;
    }
    if (key === 'i') {
      selectToolMode('picker');
      scheduleStateSave();
      evt.preventDefault();
      return;
    }
    if (key === 'r') {
      const direction = evt.shiftKey ? -1 : 1;
      rotateView(direction * (Math.PI / 4));
      evt.preventDefault();
      return;
    }
    if (key === '[') {
      setBrushDiameter(brushDiameter - 1, true);
      evt.preventDefault();
      return;
    }
    if (key === ']') {
      setBrushDiameter(brushDiameter + 1, true);
      evt.preventDefault();
      return;
    }
    if (key === ' ') {
        spacePan = true;
      updateCursor();
      evt.preventDefault();
      return;
    }
  }
  if (key === 'm') {
    maskVisible = !maskVisible;
    if (maskVisible) {
      setMaskDisplayMode(MASK_DISPLAY_MODES.OUTLINED, { silent: true });
    } else {
      setMaskDisplayMode(MASK_DISPLAY_MODES.HIDDEN, { silent: true });
    }
    updateMaskVisibilityLabel();
    if (typeof saved.segMode === "string") {
      segMode = saved.segMode;
    }
    setSegMode(segMode, { silent: true });
    if (maskVisibilityToggle) {
      maskVisibilityToggle.checked = maskVisible;
    }
    draw();
    evt.preventDefault();
    scheduleStateSave();
    return;
  }
  if (!modifier && !evt.altKey && key === 'n') {
    toggleColorMode();
    evt.preventDefault();
    return;
  }
  if (!modifier && key >= '0' && key <= '9') {
    const next = parseInt(key, 10);
    if (eraseActive) {
      erasePreviousLabel = next;
    } else {
      // Digits set the current paint value directly (group ID in N-color mode, instance label otherwise)
      currentLabel = next;
      updateMaskLabel();
      scheduleStateSave();
    }
    evt.preventDefault();
  }
});

window.addEventListener('keyup', (evt) => {
  if (evt.key === ' ') {
    spacePan = false;
    updateCursor();
  }
  if (evt.key && evt.key.toLowerCase() === 'e') {
    stopEraseOverride();
  }
});

// Always save to localStorage on page unload (server save requires sessionId)
window.addEventListener('beforeunload', () => {
  try {
    saveViewerState({ immediate: true, seq: stateDirtySeq });
  } catch (_) {
    /* ignore */
  }
});

function initialize() {
  log('initialize');
  refreshOppositeStepMapping();
  clearAffinityGraphData();
  showAffinityGraph = true;
  if (affinityGraphToggle) {
    affinityGraphToggle.checked = true;
  }
  autoFitPending = true;
  userAdjustedScale = false;
  viewStateRestored = false;
  const img = new Image();
  img.onload = () => {
    log('image loaded: ' + imgWidth + 'x' + imgHeight);
    offCtx.drawImage(img, 0, 0);
    if (gl && webglPipelineRequested && !webglPipelineReady) {
      initializeWebglPipelineResources(img);
    }
    originalImageData = offCtx.getImageData(0, 0, imgWidth, imgHeight);
    windowLow = 0;
    windowHigh = 255;
    currentGamma = DEFAULT_GAMMA;
    computeHistogram();
    if (histogramData) {
      const lowQ = histogramQuantile(0.01);
      const highQ = histogramQuantile(0.99);
      setWindowBounds(lowQ, highQ, { emit: false });
    }
    setGamma(currentGamma, { emit: false });
    updateHistogramUI();
    applyImageAdjustments();
    let restored = false;
    if (savedViewerState) {
      try {
        restoreViewerState(savedViewerState);
        restored = true;
      } catch (err) {
        console.warn('Failed to restore viewer state', err);
      }
    }
    if (!restored) {
      maskValues.fill(0);
      outlineState.fill(0);
      maskHasNonZero = false;
      if (typeof OmniHistory.clear === 'function') {
        OmniHistory.clear();
      }
      needsMaskRedraw = true;
      updateHistoryButtons();
      applyMaskRedrawImmediate();
      draw();
      scheduleAffinityRebuildIfStale('initialize');
    }
    resizeCanvas();
    updateBrushControls();
    updateToolButtons();
    updateImageInfo();
    updateHistoryButtons();
  };
  img.onerror = (evt) => {
    const detail = evt?.message || 'unknown error';
    log('image load failed: ' + detail);
    setLoadingOverlay('Failed to load image', true);
  };
  img.src = imageDataUrl;
  updateCursor();
  ensureWebglOverlayReady();
  setupDragAndDrop();
  updateImageInfo();
}

window.addEventListener('resize', resizeCanvas);
let orientationResizePending = false;
window.addEventListener('orientationchange', () => {
  if (orientationResizePending) {
    return;
  }
  orientationResizePending = true;
  setTimeout(() => {
    orientationResizePending = false;
    resizeCanvas();
  }, 120);
});

if (gammaSlider) {
  gammaSlider.addEventListener('input', (evt) => {
    const value = parseInt(evt.target.value, 10);
    const gamma = Number.isNaN(value) ? currentGamma : value / 100.0;
    setGamma(gamma);
    refreshSlider('gamma');
  });
}

if (gammaInput) {
  gammaInput.addEventListener('change', () => {
    const value = parseFloat(gammaInput.value);
    if (Number.isNaN(value)) {
      gammaInput.value = currentGamma.toFixed(2);
      return;
    }
    setGamma(value);
  });
}

brushSizeSlider.addEventListener('input', (evt) => {
  const value = parseInt(evt.target.value, 10);
  if (!Number.isNaN(value)) {
    setBrushDiameter(value, true);
    refreshSlider('brushSizeSlider');
  }
});

brushSizeInput.addEventListener('change', (evt) => {
  let value = parseInt(evt.target.value, 10);
  if (Number.isNaN(value)) {
    brushSizeInput.value = String(brushDiameter);
    return;
  }
  setBrushDiameter(value, true);
});

if (brushKernelModeSelect) {
  brushKernelModeSelect.addEventListener('change', (evt) => {
    setBrushKernelMode(evt.target.value);
  });
}
if (brushKernelToggle) {
  brushKernelToggle.addEventListener('change', (evt) => {
    const nextMode = evt.target.checked ? BRUSH_KERNEL_MODES.SNAPPED : BRUSH_KERNEL_MODES.SMOOTH;
    setBrushKernelMode(nextMode);
  });
}

if (!savedViewerState) {
  updateMaskLabel();
  updateToolButtons();
} else {
  // Avoid flashing a default label/tool before restore completes.
  if (labelValueInput) {
    labelValueInput.value = '';
  }
}
updateMaskVisibilityLabel();
updateToolInfo();
updateBrushControls();
updateColorModeLabel();
if (autoNColorToggle) autoNColorToggle.checked = nColorActive;
updateHoverInfo(null);
if (segmentButton) {
  segmentButton.addEventListener('click', () => {
    runSegmentation();
  });
}
if (clearMasksButton) {
  clearMasksButton.addEventListener('click', (evt) => {
    const skip = Boolean(evt && (evt.metaKey || evt.ctrlKey));
    promptClearMasks({ skipConfirm: skip });
  });
}
if (clearCacheButton) {
  clearCacheButton.addEventListener('click', async () => {
    if (!confirm('Clear cached viewer state and reload?')) {
      return;
    }
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        const storageKeys = Object.keys(window.localStorage);
        storageKeys
          .filter((key) => key.startsWith('OMNI') || key.includes('omnipose'))
          .forEach((key) => {
            try {
              window.localStorage.removeItem(key);
            } catch (_) {
              /* ignore */
            }
          });
      }
    } catch (_) {
      // ignore storage access errors
    }
    try {
      const response = await fetch('/api/clear_cache', { method: 'POST', keepalive: true });
      if (!response.ok) {
        console.warn('clear_cache request failed', response.status);
      }
    } catch (_) {
      // ignore network errors
    }
    window.location.reload();
  });
}
if (toolStopButtons.length) {
  toolStopButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const mode = btn.getAttribute('data-mode') || 'draw';
      selectToolMode(mode);
      scheduleStateSave();
      if (viewer && typeof viewer.focus === 'function') {
        try {
          viewer.focus({ preventScroll: true });
        } catch (_) {
          viewer.focus();
        }
      }
    });
  });
}
updateToolButtons();
updateToolOptions();
initTooltips();
if (histogramCanvas) {
  histogramCanvas.addEventListener('pointerdown', handleHistogramPointerDown);
  histogramCanvas.addEventListener('pointermove', handleHistogramPointerMove);
  histogramCanvas.addEventListener('pointerup', handleHistogramPointerUp);
  histogramCanvas.addEventListener('pointercancel', handleHistogramPointerUp);
  histogramCanvas.addEventListener('pointerleave', (evt) => {
    if (!histDragTarget) {
      updateHistogramCursor(evt);
    }
  });
  updateHistogramCursor();
}
if (maskThresholdSlider) {
  maskThresholdSlider.addEventListener('input', (evt) => {
    setMaskThreshold(evt.target.value);
  });
  maskThresholdSlider.addEventListener('change', (evt) => {
    setMaskThreshold(evt.target.value);
  });
}
if (maskThresholdInput) {
  maskThresholdInput.addEventListener('input', (evt) => {
    if (evt.target.value === '') {
      return;
    }
    setMaskThreshold(evt.target.value);
  });
  maskThresholdInput.addEventListener('change', (evt) => {
    setMaskThreshold(evt.target.value);
  });
  attachNumberInputStepper(maskThresholdInput, (delta) => {
    setMaskThreshold(maskThreshold + delta);
  });
}
if (flowThresholdSlider) {
  flowThresholdSlider.addEventListener('input', (evt) => {
    setFlowThreshold(evt.target.value);
  });
  flowThresholdSlider.addEventListener('change', (evt) => {
    setFlowThreshold(evt.target.value);
  });
}
if (flowThresholdInput) {
  flowThresholdInput.addEventListener('input', (evt) => {
    if (evt.target.value === '') {
      return;
    }
    setFlowThreshold(evt.target.value);
  });
  flowThresholdInput.addEventListener('change', (evt) => {
    setFlowThreshold(evt.target.value);
  });
  attachNumberInputStepper(flowThresholdInput, (delta) => {
    setFlowThreshold(flowThreshold + delta);
  });
}
if (niterSlider) {
  const handleNiterSlider = (evt) => {
    const target = evt.target;
    if (target.min !== String(NITER_MIN)) {
      target.min = String(NITER_MIN);
    }
    const minVal = Number(target.min);
    const value = Number(target.value);
    if (Number.isFinite(minVal) && value <= minVal) {
      setNiter('auto');
      return;
    }
    setNiter(target.value);
  };
  niterSlider.addEventListener('input', handleNiterSlider);
  niterSlider.addEventListener('change', handleNiterSlider);
}
if (niterInput) {
  niterInput.addEventListener('input', (evt) => {
    if (evt.target.value === '') {
      return;
    }
    setNiter(evt.target.value);
  });
  niterInput.addEventListener('change', (evt) => {
    setNiter(evt.target.value);
  });
  attachNumberInputStepper(niterInput, (delta) => {
    const raw = String(niterInput.value || '').trim().toLowerCase();
    const base = raw === 'auto' ? 0 : niter;
    setNiter(base + delta);
  });
}
if (clusterToggle) {
  clusterToggle.addEventListener('change', (evt) => {
    setClusterEnabled(evt.target.checked);
    segMode = clusterEnabled ? (affinitySegEnabled ? 'none' : 'cluster') : (affinitySegEnabled ? 'affinity' : 'none');
    updateSegModeControls();
    scheduleStateSave();
  });
}
if (affinityToggle) {
  affinityToggle.addEventListener('change', (evt) => {
    setAffinitySegEnabled(evt.target.checked);
    segMode = clusterEnabled ? (affinitySegEnabled ? 'none' : 'cluster') : (affinitySegEnabled ? 'affinity' : 'none');
    updateSegModeControls();
    scheduleStateSave();
  });
}
if (affinityGraphToggle) {
  affinityGraphToggle.addEventListener('change', (evt) => {
    showAffinityGraph = Boolean(evt.target.checked);
    debugAffinity('[affinity] toggle ' + (showAffinityGraph ? 'on' : 'off'));
    if (showAffinityGraph && affinityGraphNeedsLocalRebuild) {
      rebuildLocalAffinityGraph();
    }
    if (webglOverlay && webglOverlay.enabled) {
      const needsBuild = showAffinityGraph
        && webglOverlay.needsGeometryRebuild
        && affinityGraphInfo
        && affinityGraphInfo.width > 0
        && affinityGraphInfo.height > 0;
      if (needsBuild) {
        ensureWebglGeometry(affinityGraphInfo.width, affinityGraphInfo.height);
      }
      updateOverlayVisibility();
    }
    scheduleDraw();
    scheduleStateSave();
  });
}
if (flowOverlayToggle) {
  flowOverlayToggle.addEventListener('change', (evt) => {
    if (!flowOverlayImage || !flowOverlayImage.complete) {
      flowOverlayToggle.checked = false;
      showFlowOverlay = false;
      return;
    }
    showFlowOverlay = evt.target.checked;
    draw();
    scheduleStateSave();
  });
}
if (distanceOverlayToggle) {
  distanceOverlayToggle.addEventListener('change', (evt) => {
    if (!distanceOverlayImage || !distanceOverlayImage.complete) {
      distanceOverlayToggle.checked = false;
      showDistanceOverlay = false;
      return;
    }
    showDistanceOverlay = evt.target.checked;
    draw();
    scheduleStateSave();
  });
}
if (pointsOverlayToggle) {
  pointsOverlayToggle.addEventListener('change', (evt) => {
    if (!pointsOverlayInfo || !webglOverlay || !webglOverlay.pointsCount) {
      pointsOverlayToggle.checked = false;
      showPointsOverlay = false;
      updateOverlayVisibility();
      return;
    }
    showPointsOverlay = evt.target.checked;
    updateOverlayVisibility();
    draw();
    scheduleStateSave();
  });
}
if (vectorOverlayToggle) {
  vectorOverlayToggle.addEventListener('change', (evt) => {
    if (!vectorOverlayData || !vectorOverlayData.vertexCount || !affinitySegEnabled) {
      showVectorOverlay = false;
      updateVectorOverlayVisibility();
      return;
    }
    vectorOverlayPreferred = Boolean(evt.target.checked);
    showVectorOverlay = vectorOverlayPreferred;
    updateVectorOverlayVisibility();
    draw();
    scheduleStateSave();
  });
}
if (imageVisibilityToggle) {
  imageVisibilityToggle.addEventListener('change', (evt) => {
    const visible = Boolean(evt.target.checked);
    if (visible === imageVisible) {
      return;
    }
    setImageVisible(visible);
  });
}
if (maskVisibilityToggle) {
  maskVisibilityToggle.addEventListener('change', (evt) => {
    const visible = Boolean(evt.target.checked);
    if (visible === maskVisible) {
      return;
    }
    maskVisible = visible;
    updateMaskVisibilityLabel();
    setPanelCollapsed(labelStylePanel, !maskVisible);
    if (typeof saved.segMode === "string") {
      segMode = saved.segMode;
    }
    setSegMode(segMode, { silent: true });
    draw();
    scheduleStateSave();
  });
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) {
    return '--';
  }
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = Math.max(0, bytes);
  let idx = 0;
  while (value >= 1024 && idx < units.length - 1) {
    value /= 1024;
    idx += 1;
  }
  return value.toFixed(idx >= 2 ? 1 : 0) + ' ' + units[idx];
}

function updateSystemInfo(info) {
  if (!info) {
    return;
  }
  if (systemRamEl) {
    const used = formatBytes(info.ram_used);
    const total = formatBytes(info.ram_total);
    systemRamEl.textContent = used !== '--' && total !== '--'
      ? `${used} / ${total}`
      : '--';
  }
  if (systemCpuEl) {
    const cores = Number(info.cpu_cores) || 0;
    systemCpuEl.textContent = cores > 0 ? `${cores} cores` : '--';
  }
  if (systemGpuEl) {
    if (info.gpu_available) {
      const label = info.gpu_label
        || (info.gpu_backend ? String(info.gpu_backend).toUpperCase() : null)
        || info.gpu_name
        || 'GPU available';
      systemGpuEl.textContent = label;
    } else {
      systemGpuEl.textContent = 'Not available';
    }
  }
  if (useGpuToggle) {
    useGpuToggle.disabled = !info.gpu_available;
    useGpuToggle.checked = Boolean(info.use_gpu && info.gpu_available);
  }
}

async function fetchSystemInfo() {
  if (!systemRamEl && !systemCpuEl && !systemGpuEl && !useGpuToggle) {
    return;
  }
  try {
    const res = await fetch('/api/system_info', { method: 'GET' });
    if (!res.ok) {
      return;
    }
    const info = await res.json();
    updateSystemInfo(info);
  } catch (err) {
    // silent
  }
}

/**
 * Check if a colormap supports hue offset (cyclic colormaps).
 */
function colormapHasOffset(cmapValue) {
  const entry = LABEL_COLORMAPS.find(c => c.value === cmapValue);
  return entry ? entry.hasOffset : false;
}

/**
 * Generate CSS gradient string for a colormap.
 */
function generateColormapGradient(cmapValue, numStops = 32) {
  const stops = [];
  for (let i = 0; i < numStops; i++) {
    const t = i / (numStops - 1);
    let rgb;
    if (cmapValue === 'sinebow') {
      rgb = sinebowColor(t);
    } else if (cmapValue === 'vivid') {
      rgb = hslToRgb(t, 0.9, 0.5);
    } else if (cmapValue === 'pastel') {
      rgb = hslToRgb(t, 0.55, 0.72);
    } else if (cmapValue === 'gray') {
      const v = Math.round(t * 255);
      rgb = [v, v, v];
    } else if (COLORMAP_STOPS[cmapValue]) {
      rgb = interpolateStops(COLORMAP_STOPS[cmapValue], t);
    } else {
      rgb = sinebowColor(t);
    }
    const pct = (t * 100).toFixed(1);
    stops.push(`rgb(${rgb[0]},${rgb[1]},${rgb[2]}) ${pct}%`);
  }
  return `linear-gradient(to right, ${stops.join(', ')})`;
}

/**
 * Update the cmap panel UI based on current colormap selection.
 */
function updateCmapPanelUI() {
  if (!cmapPanel) return;
  const hasOffset = colormapHasOffset(labelColormap);
  const gradient = generateColormapGradient(labelColormap);
  // Update panel class for CSS targeting
  cmapPanel.classList.toggle('cmap-no-offset', !hasOffset);
  // Update slider track gradient and knob visibility
  if (cmapHueOffsetWrapper) {
    const track = cmapHueOffsetWrapper.querySelector('.slider-track');
    if (track) {
      // Set gradient via CSS custom property - ::before uses clip-path for rounded inset
      track.style.setProperty('--cmap-gradient', gradient);
    }
    cmapHueOffsetWrapper.classList.toggle('no-offset', !hasOffset);
  }
  // Update preview pill gradient (shown for non-cyclic colormaps)
  if (cmapPreviewPill) {
    cmapPreviewPill.style.backgroundImage = gradient;
  }
}

function initCmapSelect() {
  if (!cmapSelect) {
    return;
  }
  cmapSelect.innerHTML = '';
  LABEL_COLORMAPS.forEach((entry) => {
    const option = document.createElement('option');
    option.value = entry.value;
    option.textContent = entry.label;
    cmapSelect.appendChild(option);
  });
  cmapSelect.value = labelColormap;
  // Update dropdown entry's options array (it was empty at registration time)
  const dropdownEntry = dropdownRegistry.get('cmapSelect');
  if (dropdownEntry) {
    dropdownEntry.options = LABEL_COLORMAPS.map(e => ({ value: e.value, label: e.label }));
  }
  refreshDropdown('cmapSelect');
  updateCmapPanelUI();
  cmapSelect.addEventListener('change', () => {
    labelColormap = cmapSelect.value || 'sinebow';
    clearColorCaches();
    paletteTextureDirty = true;
    updateCmapPanelUI();
    // Regenerate N-color palette with new colormap
    const count = nColorPaletteColors.length || DEFAULT_NCOLOR_COUNT;
    nColorPaletteColors = generateNColorSwatches(count);
    renderNColorSwatches();
    // Update active label color to reflect new colormap
    updateLabelControls();
    if (isWebglPipelineActive()) {
      markMaskTextureFullDirty();
      markOutlineTextureFullDirty();
    } else {
      redrawMaskCanvas();
    }
    draw();
    scheduleStateSave();
  });
}

// Legacy alias
const initLabelColormapSelect = initCmapSelect;

/**
 * Compute relative luminance from RGB values (0-255).
 * Returns value 0-1 where 0 is black, 1 is white.
 */
function getLuminance(r, g, b) {
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255;
}

/**
 * Update the image colormap panel UI (gradient preview on dropdown).
 */
function updateImageCmapPanelUI() {
  const hasGradient = imageColormap !== 'gray' && imageColormap !== 'gray-clip';
  const dropdownWrapper = imageCmapSelect ? imageCmapSelect.closest('.dropdown--gradient-preview') : null;
  const toggle = dropdownWrapper ? dropdownWrapper.querySelector('.dropdown-toggle') : null;

  if (dropdownWrapper) {
    dropdownWrapper.classList.toggle('has-gradient', hasGradient);

    // Get display label for tooltip
    const cmapEntry = IMAGE_COLORMAPS.find(c => c.value === imageColormap);
    const cmapLabel = cmapEntry ? cmapEntry.label : imageColormap;

    if (hasGradient) {
      const stops = COLORMAP_STOPS[imageColormap];
      if (stops && stops.length) {
        // Build colormap gradient
        const gradientStops = stops.map((hex, i) => {
          const pct = (i / (stops.length - 1)) * 100;
          return `${hex} ${pct.toFixed(1)}%`;
        });
        dropdownWrapper.style.setProperty('--cmap-gradient', `linear-gradient(to right, ${gradientStops.join(', ')})`);

        // Arrow color based on rightmost stop
        const lastRgb = hexToRgb(stops[stops.length - 1]);
        const lastLum = getLuminance(lastRgb[0], lastRgb[1], lastRgb[2]);
        dropdownWrapper.style.setProperty('--cmap-arrow-color', lastLum < 0.5 ? '#fff' : '#000');
      }
      // Set tooltip to show colormap name on hover
      if (toggle) toggle.title = cmapLabel;
    } else {
      dropdownWrapper.style.removeProperty('--cmap-gradient');
      dropdownWrapper.style.removeProperty('--cmap-arrow-color');
      // Clear tooltip for grayscale options (text is visible)
      if (toggle) toggle.removeAttribute('title');
    }
  }
}

/**
 * Initialize the image colormap dropdown.
 */
function initImageCmapSelect() {
  if (!imageCmapSelect) {
    return;
  }
  imageCmapSelect.innerHTML = '';
  IMAGE_COLORMAPS.forEach((entry) => {
    const option = document.createElement('option');
    option.value = entry.value;
    option.textContent = entry.label;
    imageCmapSelect.appendChild(option);
  });
  imageCmapSelect.value = imageColormap;
  // Update dropdown entry's options array
  const dropdownEntry = dropdownRegistry.get('imageCmapSelect');
  if (dropdownEntry) {
    dropdownEntry.options = IMAGE_COLORMAPS.map(e => ({ value: e.value, label: e.label }));
  }
  refreshDropdown('imageCmapSelect');
  updateImageCmapPanelUI();
  imageCmapSelect.addEventListener('change', () => {
    imageColormap = imageCmapSelect.value || 'gray';
    imageCmapLutDirty = true;
    updateImageCmapTexture();
    updateImageCmapPanelUI();
    draw();
    scheduleStateSave();
  });
}

function initSegmentationModelSelect() {
  if (!segmentationModelSelect) {
    return;
  }
  const options = Array.isArray(CONFIG.models)
    ? CONFIG.models
    : (Array.isArray(CONFIG.modelOptions) ? CONFIG.modelOptions : SEGMENTATION_MODELS);
  if (options && options.length) {
    segmentationModelSelect.innerHTML = '';
    for (const opt of options) {
      const option = document.createElement('option');
      if (typeof opt === 'string') {
        option.value = opt;
        option.textContent = opt;
      } else if (opt && typeof opt === 'object') {
        option.value = String(opt.value ?? opt.name ?? '');
        option.textContent = String(opt.label ?? opt.name ?? opt.value ?? '');
      } else {
        continue;
      }
      segmentationModelSelect.appendChild(option);
    }
    const addOption = document.createElement('option');
    addOption.value = '__add__';
    addOption.textContent = 'Add model...';
    segmentationModelSelect.appendChild(addOption);
  }
  const dropdownEntry = dropdownRegistry.get('segmentationModel');
  if (dropdownEntry) {
    dropdownEntry.options = Array.from(segmentationModelSelect.options).map((opt) => ({
      value: opt.value,
      label: opt.textContent || opt.value,
      disabled: opt.disabled,
    }));
    if (typeof dropdownEntry.buildMenu === 'function') {
      dropdownEntry.buildMenu();
    }
  }
  if (segmentationModel && segmentationModel.startsWith('file:')) {
    const label = segmentationModel.replace(/^file:/, '') || 'Custom Model';
    let option = null;
    for (const opt of segmentationModelSelect.options) {
      if (opt.value === segmentationModel) {
        option = opt;
        break;
      }
    }
    if (!option) {
      option = document.createElement('option');
      option.value = segmentationModel;
      option.textContent = label;
      segmentationModelSelect.insertBefore(option, segmentationModelSelect.lastElementChild);
    }
  }
  segmentationModelSelect.value = segmentationModel || DEFAULT_SEGMENTATION_MODEL;
  refreshDropdown('segmentationModel');
}

initSegmentationModelSelect();

initLabelColormapSelect();

initImageCmapSelect();

if (segmentationModelSelect) {
  segmentationModelSelect.addEventListener('change', (evt) => {
    const value = String(evt.target.value || '');
    if (value === '__add__') {
      if (segmentationModelFile) {
        segmentationModelFile.click();
      }
      segmentationModelSelect.value = segmentationModel || DEFAULT_SEGMENTATION_MODEL;
      return;
    }
    segmentationModel = value || DEFAULT_SEGMENTATION_MODEL;
    customSegmentationModelPath = null;
    scheduleStateSave();
  });
}

if (segmentationModelFile) {
  segmentationModelFile.addEventListener('change', (evt) => {
    const files = evt.target.files;
    if (!files || !files.length) {
      return;
    }
    const file = files[0];
    const label = file.name || 'Custom Model';
    const value = 'file:' + label;
    let option = null;
    if (segmentationModelSelect) {
      for (const opt of segmentationModelSelect.options) {
        if (opt.value === value) {
          option = opt;
          break;
        }
      }
    }
    if (!option && segmentationModelSelect) {
      option = document.createElement('option');
      option.value = value;
      option.textContent = label;
      segmentationModelSelect.insertBefore(option, segmentationModelSelect.lastElementChild);
    }
    segmentationModel = value;
    customSegmentationModelPath = file.path || file.name || null;
    if (segmentationModelSelect) {
      segmentationModelSelect.value = value;
    }
    scheduleStateSave();
  });
}

if (autoNColorToggle) {
  autoNColorToggle.addEventListener('change', (evt) => {
    const wantsNColor = Boolean(evt.target.checked);
    if (wantsNColor !== nColorActive) {
      toggleColorMode();
    } else {
      autoNColorToggle.checked = nColorActive;
      scheduleStateSave();
    }
  });
}
if (ncolorAddColor) {
  ncolorAddColor.addEventListener('click', () => {
    const currentCount = (nColorPaletteColors && nColorPaletteColors.length)
      ? nColorPaletteColors.length
      : DEFAULT_NCOLOR_COUNT;
    // Regenerate entire palette with new count for even spacing
    const next = generateNColorSwatches(currentCount + 1);
    setNColorPaletteColors(next);
  });
}
if (ncolorRemoveColor) {
  ncolorRemoveColor.addEventListener('click', () => {
    const currentCount = (nColorPaletteColors && nColorPaletteColors.length)
      ? nColorPaletteColors.length
      : DEFAULT_NCOLOR_COUNT;
    if (currentCount <= 2) {
      return;
    }
    // Regenerate entire palette with new count for even spacing
    const next = generateNColorSwatches(currentCount - 1);
    setNColorPaletteColors(next);
  });
}
if (cmapHueOffsetSlider) {
  // Initialize slider to current value and gradient
  cmapHueOffsetSlider.value = Math.round(nColorHueOffset * 100);
  refreshSlider('cmapHueOffset');
  updateCmapPanelUI();
  cmapHueOffsetSlider.addEventListener('input', (evt) => {
    nColorHueOffset = parseInt(evt.target.value, 10) / 100;
    // Regenerate N-color palette with new hue offset
    const count = nColorPaletteColors.length || DEFAULT_NCOLOR_COUNT;
    nColorPaletteColors = generateNColorSwatches(count);
    renderNColorSwatches();
    // Always update display (affects both N-color and instance mode for cyclic colormaps)
    clearColorCaches();
    paletteTextureDirty = true;
    if (isWebglPipelineActive()) {
      markMaskTextureFullDirty();
      markOutlineTextureFullDirty();
    } else {
      redrawMaskCanvas();
    }
    draw();
    scheduleStateSave();
  });
}
if (labelShuffleToggle) {
  labelShuffleToggle.addEventListener('change', (evt) => {
    labelShuffle = Boolean(evt.target.checked);
    // Invalidate shuffle permutation cache
    shufflePermutation = null;
    paletteTextureDirty = true;
    updateLabelShuffleControls();
    clearColorCaches();
    paletteTextureDirty = true;
    updateInstanceColormapPreview();
    if (!nColorActive) {
      if (isWebglPipelineActive()) {
        markMaskTextureFullDirty();
        markOutlineTextureFullDirty();
      } else {
        redrawMaskCanvas();
      }
      draw();
    }
    scheduleStateSave();
  });
}
if (labelShuffleSeedInput) {
  const applySeed = (evt) => {
    const value = parseInt(evt.target.value, 10);
    labelShuffleSeed = Number.isFinite(value) ? value : 0;
    // Invalidate shuffle permutation cache
    shufflePermutation = null;
    paletteTextureDirty = true;
    evt.target.value = String(labelShuffleSeed);
    clearColorCaches();
    paletteTextureDirty = true;
    updateInstanceColormapPreview();
    if (!nColorActive) {
      if (isWebglPipelineActive()) {
        markMaskTextureFullDirty();
        markOutlineTextureFullDirty();
      } else {
        redrawMaskCanvas();
      }
      draw();
    }
    scheduleStateSave();
  };
  labelShuffleSeedInput.addEventListener('change', applySeed);
  labelShuffleSeedInput.addEventListener('keydown', (evt) => {
    if (evt.key === 'Enter') {
      applySeed(evt);
    }
  });
}
if (useGpuToggle) {
  useGpuToggle.addEventListener('change', async (evt) => {
    const wantsGpu = Boolean(evt.target.checked);
    try {
      const res = await fetch('/api/use_gpu', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ use_gpu: wantsGpu }),
      });
      if (!res.ok) {
        throw new Error('request failed');
      }
      const info = await res.json();
      updateSystemInfo(info);
    } catch (err) {
      useGpuToggle.checked = !wantsGpu;
    }
  });
}
if (maskStyleButtons && maskStyleButtons.length) {
  maskStyleButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const mode = btn.getAttribute('data-mask-style');
      setMaskDisplayMode(mode);
    });
  });
}
if (segModeButtons && segModeButtons.length) {
  segModeButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const mode = btn.getAttribute('data-seg-mode');
      setSegMode(mode);
    });
  });
}
if (maskOpacitySlider) {
  maskOpacitySlider.addEventListener('input', (evt) => {
    setMaskOpacity(evt.target.value);
  });
  maskOpacitySlider.addEventListener('change', (evt) => {
    setMaskOpacity(evt.target.value);
  });
}
if (maskOpacityInput) {
  maskOpacityInput.addEventListener('input', (evt) => {
    if (evt.target.value === '') {
      return;
    }
    setMaskOpacity(evt.target.value);
  });
  maskOpacityInput.addEventListener('change', (evt) => {
    setMaskOpacity(evt.target.value);
  });
  attachNumberInputStepper(maskOpacityInput, (delta) => {
    setMaskOpacity(maskOpacity + delta);
  });
}
syncMaskThresholdControls();
syncFlowThresholdControls();
syncNiterControls();
setClusterEnabled(clusterEnabled, { silent: true });
setAffinitySegEnabled(affinitySegEnabled, { silent: true });
if (maskVisibilityToggle) {
  maskVisibilityToggle.checked = maskVisible;
}
setImageVisible(imageVisible, { silent: true });
if (imageVisibilityToggle) {
  imageVisibilityToggle.checked = imageVisible;
}
setPanelCollapsed(intensityPanel, !imageVisible);
setPanelCollapsed(labelStylePanel, !maskVisible);
syncMaskOpacityControls();
updateMaskStyleControls();
updateSegModeControls();
fetchSystemInfo();

let bootstrapped = false;
function boot() {
  if (bootstrapped) {
    return;
  }
  bootstrapped = true;
  log('boot (readyState=' + document.readyState + ')');
  initialize();
  setTimeout(resizeCanvas, 100);
  flushLogs();
}

if (document.readyState === 'complete' || document.readyState === 'interactive') {
  boot();
} else {
  window.addEventListener('load', boot);
}
window.addEventListener('pywebviewready', () => {
  setPywebviewReady(true);
  log('pywebview ready event');
  flushLogs();
  boot();
});
window.addEventListener('beforeunload', () => {
  shuttingDown = true;
  clearLogQueue();
});

if (typeof window !== 'undefined') {
  window.__OMNI_DEBUG__ = window.__OMNI_DEBUG__ || {};
  window.__OMNI_DEBUG__.getCounters = () => ({
    draw: DEBUG_COUNTERS.draw,
    requestPaintFrame: DEBUG_COUNTERS.requestPaintFrame,
    flushMaskTextureUpdates: DEBUG_COUNTERS.flushMaskTextureUpdates,
    applyMaskRedrawImmediate: DEBUG_COUNTERS.applyMaskRedrawImmediate,
    affinityUpdateLastMs: DEBUG_COUNTERS.affinityUpdateLastMs,
  });
  window.__OMNI_DEBUG__.resetCounters = () => {
    DEBUG_COUNTERS.draw = 0;
    DEBUG_COUNTERS.requestPaintFrame = 0;
    DEBUG_COUNTERS.flushMaskTextureUpdates = 0;
    DEBUG_COUNTERS.applyMaskRedrawImmediate = 0;
    DEBUG_COUNTERS.affinityUpdateLastMs = 0;
  };
  window.__OMNI_DEBUG__.initializeWebglPipelineResources = (imageSource = null) => {
    initializeWebglPipelineResources(imageSource);
  };
  window.__OMNI_DEBUG__.markMaskTextureFullDirty = () => {
    markMaskTextureFullDirty();
  };
  window.__OMNI_DEBUG__.markOutlineTextureFullDirty = () => {
    markOutlineTextureFullDirty();
  };
  window.__OMNI_DEBUG__.applyMaskRedrawImmediate = () => {
    applyMaskRedrawImmediate();
  };
  window.__OMNI_DEBUG__.requestPaintFrame = () => {
    requestPaintFrame();
  };
  window.__OMNI_DEBUG__.draw = () => {
    draw();
  };
  window.__OMNI_DEBUG__.isWebglPipelineActive = () => isWebglPipelineActive();
  window.__OMNI_DEBUG__.getMaskUploadState = () => ({
    isWebglPipelineActive: isWebglPipelineActive(),
    webglPipelineReady,
    needsMaskRedraw,
    maskTextureFullDirty,
    pendingMaskTextureFull,
    maskDirtyRegionCount: maskDirtyRegions.length,
    maskDirtyRegionsPreview: maskDirtyRegions.slice(0, 6),
    outlineTextureFullDirty,
    pendingOutlineTextureFull,
    outlineDirtyRegionCount: outlineDirtyRegions.length,
    outlineDirtyRegionsPreview: outlineDirtyRegions.slice(0, 6),
  });
  window.__OMNI_DEBUG__.collectFillDiagnostics = (index = null) => {
    const painting = window.OmniPainting;
    const state = painting && typeof painting.__debugGetState === 'function'
      ? painting.__debugGetState()
      : null;
    const ctx = state ? state.ctx : null;
    const startIdx = Number.isFinite(index)
      ? (index | 0)
      : (state && state.lastFillResult ? state.lastFillResult.startIdx | 0 : null);
    const maskValue = (ctx && Number.isFinite(startIdx) && startIdx >= 0 && startIdx < ctx.maskValues.length)
      ? ctx.maskValues[startIdx] | 0
      : null;
    const undoStack = window.OmniHistory && typeof window.OmniHistory.getUndoStack === 'function'
      ? window.OmniHistory.getUndoStack()
      : [];
    const latestUndo = undoStack && undoStack.length ? undoStack[undoStack.length - 1] : null;
    const report = {
      index: startIdx,
      maskValue,
      lastFillResult: state ? state.lastFillResult || null : null,
      latestUndo: latestUndo
        ? {
          indicesLength: latestUndo.indices ? latestUndo.indices.length : 0,
          beforeSample: latestUndo.before ? latestUndo.before[0] : null,
          afterSample: latestUndo.after ? latestUndo.after[0] : null,
        }
        : null,
      counters: window.__OMNI_DEBUG__.getCounters ? window.__OMNI_DEBUG__.getCounters() : null,
      uploadState: window.__OMNI_DEBUG__.getMaskUploadState ? window.__OMNI_DEBUG__.getMaskUploadState() : null,
      forceGridMask: Boolean(window.__OMNI_FORCE_GRID_MASK__),
    dirtyTracker: Object.assign({}, DEBUG_DIRTY_TRACKER),
    ctxFlags: ctx
      ? {
        hasMarkMaskTextureFullDirty: typeof ctx.markMaskTextureFullDirty === 'function',
        hasMarkOutlineTextureFullDirty: typeof ctx.markOutlineTextureFullDirty === 'function',
        hasMarkNeedsMaskRedraw: typeof ctx.markNeedsMaskRedraw === 'function',
        hasRequestPaintFrame: typeof ctx.requestPaintFrame === 'function',
        hasMarkMaskIndicesDirty: typeof ctx.markMaskIndicesDirty === 'function',
        hasMarkOutlineIndicesDirty: typeof ctx.markOutlineIndicesDirty === 'function',
        isWebglPipelineActive: typeof ctx.isWebglPipelineActive === 'function'
          ? Boolean(ctx.isWebglPipelineActive())
          : null,
      }
      : null,
      viewerState: {
        maskVisible,
        maskOpacity,
        outlinesVisible,
        imageVisible,
        showFlowOverlay,
        showDistanceOverlay,
        scale: viewState.scale,
        offsetX: viewState.offsetX,
        offsetY: viewState.offsetY,
        rotation: viewState.rotation,
      },
    paintingDiagnostics: {
      missingMaskDirty: (typeof globalThis === 'object' && globalThis.__OMNI_DEBUG__)
        ? (globalThis.__OMNI_DEBUG__.missingMaskDirty || 0)
        : 0,
      missingOutlineDirty: (typeof globalThis === 'object' && globalThis.__OMNI_DEBUG__)
        ? (globalThis.__OMNI_DEBUG__.missingOutlineDirty || 0)
        : 0,
    },
      finalizeCallCount: state && typeof state.finalizeCallCount === 'number'
        ? state.finalizeCallCount
        : 0,
      finalizeHistory: (typeof globalThis === 'object' && globalThis.__OMNI_DEBUG__ && globalThis.__OMNI_DEBUG__.finalizeHistory)
        ? globalThis.__OMNI_DEBUG__.finalizeHistory.slice(-8)
        : null,
  };
    console.log('[debug-fill-report]', report);
    return report;
  };
  window.__OMNI_DEBUG__.summarizeDirtyRegions = () => ({
    maskDirtyRegionCount: maskDirtyRegions.length,
    maskDirtyRegions: maskDirtyRegions.map((rect) => Object.assign({}, rect)),
    outlineDirtyRegionCount: outlineDirtyRegions.length,
    outlineDirtyRegions: outlineDirtyRegions.map((rect) => Object.assign({}, rect)),
    needsMaskRedraw,
    maskTextureFullDirty,
    outlineTextureFullDirty,
    dirtyTracker: Object.assign({}, DEBUG_DIRTY_TRACKER),
  });
  window.__OMNI_DEBUG__.getDirtyTracker = () => Object.assign({}, DEBUG_DIRTY_TRACKER);
  window.__OMNI_DEBUG__.summarizeAffinityGraph = () => {
    if (!affinityGraphInfo || !affinityGraphInfo.values) {
      return {
        hasGraph: false,
        showAffinityGraph,
      };
    }
    const { values, stepCount, width, height } = affinityGraphInfo;
    const planeStride = width * height;
    const counts = new Array(stepCount).fill(0);
    for (let step = 0; step < stepCount; step += 1) {
      const offset = step * planeStride;
      let count = 0;
      for (let i = 0; i < planeStride; i += 1) {
        if (values[offset + i]) {
          count += 1;
        }
      }
      counts[step] = count;
    }
    const totalEdges = counts.reduce((sum, c) => sum + c, 0);
    return {
      hasGraph: true,
      showAffinityGraph,
      width,
      height,
      stepCount,
      totalEdges,
      counts,
    };
  };
  window.__OMNI_DEBUG__.highlightFillRect = (rect) => {
    showDebugFillOverlay(rect);
  };
  window.__OMNI_DEBUG__.hashColor = (label) => {
    const golden = 0.61803398875;
    const t = ((label * golden) % 1 + 1) % 1;
    const angle = Math.PI * 2 * t;
    const r = Math.sin(angle) * 0.5 + 0.5;
    const g = Math.sin(angle + 2.09439510239) * 0.5 + 0.5;
    const b = Math.sin(angle + 4.18879020479) * 0.5 + 0.5;
    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
  };
  window.__OMNI_DEBUG__.resetWebglPipeline = () => {
    if (webglPipeline && webglPipeline.gl) {
      try {
        webglPipeline.gl.bindVertexArray(null);
      } catch (_) {
        /* ignore */
      }
    }
    webglPipeline = null;
    webglPipelineReady = false;
    pendingMaskTextureFull = false;
    pendingOutlineTextureFull = false;
    maskTextureFullDirty = false;
    outlineTextureFullDirty = false;
    maskDirtyRegions.length = 0;
    outlineDirtyRegions.length = 0;
    needsMaskRedraw = false;
  };
  window.__OMNI_DEBUG__.getAffinityInfo = () => ({
    hasGraph: Boolean(affinityGraphInfo && affinityGraphInfo.values && affinityGraphInfo.values.length),
    stepCount: affinityGraphInfo ? affinityGraphInfo.stepCount | 0 : 0,
    nonZeroEdges: affinityGraphInfo && affinityGraphInfo.values ? affinityGraphInfo.values.some((v) => v !== 0) : false,
    showAffinityGraph,
    lastLocalAffinityBuild: window.__OMNI_DEBUG__.lastLocalAffinityBuild || null,
  });
  window.__OMNI_DEBUG__.setAffinitySentinel = (value = 255) => {
    if (!affinityGraphInfo || !affinityGraphInfo.values || !affinityGraphInfo.values.length) {
      console.warn('[affinity] no graph to mark');
      return false;
    }
    affinityGraphInfo.values[0] = value & 0xff;
    if (webglOverlay && webglOverlay.enabled) {
      webglOverlay.needsGeometryRebuild = true;
    }
    scheduleStateSave();
    return true;
  };
  window.__OMNI_DEBUG__.getAffinityPersistSnapshot = () => ({
    source: affinityGraphSource,
    needsLocal: affinityGraphNeedsLocalRebuild,
    hasGraph: Boolean(affinityGraphInfo && affinityGraphInfo.values && affinityGraphInfo.values.length),
    firstByte: affinityGraphInfo && affinityGraphInfo.values ? affinityGraphInfo.values[0] : null,
    savedGraph: Boolean(savedViewerState && savedViewerState.affinityGraph && savedViewerState.affinityGraph.encoded),
  });
  // Debug function to get full affinity state for testing
  window.__OMNI_DEBUG__.getAffinityState = () => ({
    showAffinityGraph,
    affinitySegEnabled,
    affinityGraphSource,
    affinityGraphNeedsLocalRebuild,
    hasAffinityGraphInfo: Boolean(affinityGraphInfo),
    hasValues: Boolean(affinityGraphInfo && affinityGraphInfo.values),
    hasSegments: Boolean(affinityGraphInfo && affinityGraphInfo.segments),
    stepCount: affinityGraphInfo ? affinityGraphInfo.stepCount : 0,
    width: affinityGraphInfo ? affinityGraphInfo.width : 0,
    height: affinityGraphInfo ? affinityGraphInfo.height : 0,
    valuesLength: affinityGraphInfo && affinityGraphInfo.values ? affinityGraphInfo.values.length : 0,
    webglOverlayEnabled: Boolean(webglOverlay && webglOverlay.enabled),
    nextSlot: webglOverlay ? webglOverlay.nextSlot : 0,
    maxUsedSlotIndex: webglOverlay ? webglOverlay.maxUsedSlotIndex : -1,
    savedAffinityGraphPayload: Boolean(savedAffinityGraphPayload),
  });
  // Debug function to access maskValues
  window.__OMNI_DEBUG__.getMaskValues = () => maskValues;
  // Debug function to access nColorInstanceMask
  window.__OMNI_DEBUG__.getNColorInstanceMask = () => nColorInstanceMask;
  // Debug function to check nColor state
  window.__OMNI_DEBUG__.getNColorState = () => ({
    nColorActive,
    hasNColorInstanceMask: Boolean(nColorInstanceMask),
    instanceMaskLength: nColorInstanceMask ? nColorInstanceMask.length : 0,
    maskValuesLength: maskValues ? maskValues.length : 0,
  });
  // Debug function to update affinity for indices
  window.__OMNI_DEBUG__.updateAffinityForIndices = (indices) => {
    if (!indices || !indices.length) return { error: 'no indices' };
    const before = window.__OMNI_DEBUG__.getAffinityState();
    updateAffinityGraphForIndices(indices);
    const after = window.__OMNI_DEBUG__.getAffinityState();
    return { before, after, indicesCount: indices.length };
  };
  // Debug function to check affinity edges in a region
  window.__OMNI_DEBUG__.checkAffinityInRegion = (x, y, w, h) => {
    if (!affinityGraphInfo || !affinityGraphInfo.values) {
      return { error: 'no affinity graph' };
    }
    const info = affinityGraphInfo;
    const planeStride = info.width * info.height;
    let edgeCount = 0;
    for (let dy = 0; dy < h; dy++) {
      for (let dx = 0; dx < w; dx++) {
        const px = x + dx;
        const py = y + dy;
        const idx = py * info.width + px;
        if (idx < 0 || idx >= planeStride) continue;
        for (let s = 0; s < info.stepCount; s++) {
          if (info.values[s * planeStride + idx]) {
            edgeCount++;
          }
        }
      }
    }
    return {
      region: { x, y, w, h },
      edgeCount,
      stepCount: info.stepCount,
    };
  };
  // Debug function to paint a region and update affinity
  window.__OMNI_DEBUG__.paintAndUpdateAffinity = (x, y, w, h, label) => {
    if (!maskValues) return { error: 'no maskValues' };
    const width = imgWidth;
    const height = imgHeight;
    const indices = [];
    for (let dy = 0; dy < h; dy++) {
      for (let dx = 0; dx < w; dx++) {
        const px = x + dx;
        const py = y + dy;
        const idx = py * width + px;
        if (idx >= 0 && idx < maskValues.length) {
          maskValues[idx] = label;
          indices.push(idx);
        }
      }
    }
    const stateBefore = window.__OMNI_DEBUG__.getAffinityState();
    updateAffinityGraphForIndices(indices);
    const stateAfter = window.__OMNI_DEBUG__.getAffinityState();
    const affinityInRegion = window.__OMNI_DEBUG__.checkAffinityInRegion(x, y, w, h);
    return {
      paintedCount: indices.length,
      label,
      stateBefore,
      stateAfter,
      affinityInRegion,
    };
  };
}
// no incremental rebuild method; geometry rebuilds in ensureWebglGeometry

async function clearCacheAndReload() {
  const confirmed = await showConfirmDialog('Clear cached viewer state and reload?', { confirmText: 'Clear', cancelText: 'Cancel' });
  if (!confirmed) {
    return;
  }
  try {
    if (typeof window !== 'undefined' && window.localStorage) {
      const storageKeys = Object.keys(window.localStorage);
      storageKeys
        .filter((key) => key.startsWith('OMNI') || key.includes('omnipose'))
        .forEach((key) => {
          try {
            window.localStorage.removeItem(key);
          } catch (_) {
            /* ignore */
          }
        });
    }
  } catch (_) {
    // ignore storage access errors
  }
  try {
    await fetch('/api/clear_state', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sessionId }),
    });
  } catch (_) {
    // ignore errors
  }
  window.location.reload();
}

// Clear cache button removed; now available via context menu
