// Extracted from app.js touch and pointer controls

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
  hoverPoint = null;
  drawBrushPreview(null);
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
  strokeChanges = null;
  isPanning = true;
  panPointerId = evt.pointerId;
  try {
    canvas.setPointerCapture(evt.pointerId);
  } catch (_) {
    /* ignore */
  }
  hoverPoint = null;
  hoverScreenPoint = null;
  drawBrushPreview(null);
  updateHoverInfo(null);
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
  pendingGestureUpdate = null;
  gestureFrameScheduled = false;
  wheelRotationBuffer = 0;
  hoverPoint = null;
  hoverScreenPoint = null;
  drawBrushPreview(null);
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
  pendingGestureUpdate = {
    origin: resolveGestureOrigin(evt),
    scale: Number.isFinite(evt.scale) ? evt.scale : 1,
    rotation: Number.isFinite(evt.rotation) ? evt.rotation : 0,
  };
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
  pendingGestureUpdate = null;
  gestureFrameScheduled = false;
  clearCursorOverride();
  renderHoverPreview();
  updateHoverInfo(hoverPoint || null);
}

const MIN_BRUSH_DIAMETER = 1;
const MAX_BRUSH_DIAMETER = 25;

const BRUSH_KERNEL_MODES = {
  SMOOTH: 'smooth',
  SNAPPED: 'snapped',
};
let brushKernelMode = BRUSH_KERNEL_MODES.SMOOTH;
const snappedKernelCache = new Map();
let pendingGestureUpdate = null;
let gestureFrameScheduled = false;

function clampBrushDiameter(value) {
  const numeric = Number(value);
  if (Number.isNaN(numeric)) {
    return MIN_BRUSH_DIAMETER;
  }
  return Math.max(MIN_BRUSH_DIAMETER, Math.min(MAX_BRUSH_DIAMETER, Math.round(numeric)));
}

const defaultDiameter = clampBrushDiameter(initialBrushRadius * 2 + 1);
let brushDiameter = defaultDiameter;

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
  registerSlider(root);
});
  registerDropdown(root);
});
updateBrushControls();

window.addEventListener('resize', () => {
  if (dropdownOpenId) {
    const entry = dropdownRegistry.get(dropdownOpenId);
    positionDropdown(entry);
  }
});


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

attachNumberInputStepper(brushSizeInput, (delta) => {
  setBrushDiameter(brushDiameter + delta, true);
});

attachNumberInputStepper(gammaInput, (delta) => {
  setGamma(currentGamma + delta);
});



  typeof CONFIG.maskThreshold === 'number' ? CONFIG.maskThreshold : -2,
  MASK_THRESHOLD_MIN,
  MASK_THRESHOLD_MAX,
);
  typeof CONFIG.flowThreshold === 'number' ? CONFIG.flowThreshold : 0,
  FLOW_THRESHOLD_MIN,
  FLOW_THRESHOLD_MAX,
);

// Interaction dot overlay removed; dot is now the cursor itself


function resizePreviewCanvas() {
  previewCanvas.width = canvas.width;
  previewCanvas.height = canvas.height;
  // The preview canvas shares the same drawing buffer dimensions as the primary canvas,
  // but we MUST also mirror the CSS width/height so the 2D overlay aligns with the main image.
  // If we leave the CSS size at the default (device pixels), browsers up-scale/down-scale
  // the overlay independently of the viewer canvas. On HiDPI screens this creates a 2x stretch,
  // which was the root cause of the brush preview circle appearing offset and squashed.
  // Always keep the element's CSS size in sync with the primary canvas so coordinate spaces match.
  if (canvas.style.width) {
    previewCanvas.style.width = canvas.style.width;
  } else {
    const rect = viewer ? viewer.getBoundingClientRect() : null;
      if (rect) {
        previewCanvas.style.width = rect.width + 'px';
      }
  }
  if (canvas.style.height) {
    previewCanvas.style.height = canvas.style.height;
  } else {
    const rect = viewer ? viewer.getBoundingClientRect() : null;
      if (rect) {
        previewCanvas.style.height = rect.height + 'px';
      }
  }
  previewCanvas.style.transform = canvas.style.transform || '';
}

function clearPreview() {
  previewCtx.setTransform(1, 0, 0, 1, 0, 0);
  previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
}

function drawTouchOverlay() {
  if (!debugTouchOverlay) {
    return;
  }
  const rect = canvas.getBoundingClientRect();
  previewCtx.save();
  previewCtx.setTransform(1, 0, 0, 1, 0, 0);
  previewCtx.lineWidth = 2;
  previewCtx.strokeStyle = 'rgba(0, 200, 255, 0.9)';
  previewCtx.fillStyle = 'rgba(0, 200, 255, 0.2)';
  let rendered = false;
  touchPointers.forEach((data) => {
    const x = data.x - rect.left;
    const y = data.y - rect.top;
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      return;
    }
    rendered = true;
    previewCtx.beginPath();
    previewCtx.arc(x, y, 16, 0, Math.PI * 2);
    previewCtx.fill();
    previewCtx.stroke();
  });
  previewCtx.restore();
}

// Interaction dot overlay removed

function drawBrushPreview(point, { crosshairOnly = false } = {}) {
  clearPreview();

  if (point) {
    let circleCenter = null;
    if (!crosshairOnly && PREVIEW_TOOL_TYPES.has(tool)) {
      const pixels = enumerateBrushPixels(point.x, point.y);
      if (pixels.length) {
        const kernelCenter = getBrushKernelCenter(point.x, point.y);
        circleCenter = (brushKernelMode === BRUSH_KERNEL_MODES.SNAPPED)
          ? { x: kernelCenter.x + 0.5, y: kernelCenter.y + 0.5 }
          : { x: kernelCenter.x, y: kernelCenter.y };
        const radius = (brushDiameter - 1) / 2;
        previewCtx.save();
        applyViewTransform(previewCtx, { includeDpr: true });
        previewCtx.imageSmoothingEnabled = false;
        previewCtx.fillStyle = 'rgba(255, 255, 255, 0.24)';
        const size = 1;
        for (const pixel of pixels) {
          previewCtx.fillRect(pixel.x, pixel.y, size, size);
        }
        previewCtx.restore();
        if (radius >= 0) {
          previewCtx.save();
          applyViewTransform(previewCtx, { includeDpr: true });
          previewCtx.lineWidth = 1 / Math.max(viewState.scale * dpr, 1);
          previewCtx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
          previewCtx.beginPath();
          const radiusPixels = radius + 0.5;
          previewCtx.arc(circleCenter.x, circleCenter.y, radiusPixels, 0, Math.PI * 2);
          previewCtx.stroke();
          previewCtx.restore();
        }
      }
    }
    const wantCrosshair = BRUSH_CROSSHAIR_ENABLED || crosshairOnly;
    if (wantCrosshair) {
      const crosshairHalf = 0.25;
      const crossPoint = circleCenter && !crosshairOnly ? circleCenter : { x: point.x, y: point.y };
      previewCtx.save();
      applyViewTransform(previewCtx, { includeDpr: true });
      previewCtx.lineWidth = 1 / Math.max(viewState.scale * dpr, 1);
      previewCtx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
      previewCtx.beginPath();
      previewCtx.moveTo(crossPoint.x - crosshairHalf, crossPoint.y);
      previewCtx.lineTo(crossPoint.x + crosshairHalf, crossPoint.y);
      previewCtx.moveTo(crossPoint.x, crossPoint.y - crosshairHalf);
      previewCtx.lineTo(crossPoint.x, crossPoint.y + crosshairHalf);
      previewCtx.stroke();
      previewCtx.restore();
    }
  }
  drawTouchOverlay();
}

function scheduleGestureUpdate() {
  if (gestureFrameScheduled) {
    return;
  }
  gestureFrameScheduled = true;
  // Throttle pinch updates so we stay synced with the browser compositor and avoid the iPad stutter regression.
  requestAnimationFrame(() => {
    gestureFrameScheduled = false;
    applyPendingGestureUpdate();
  });
}

function applyPendingGestureUpdate() {
  if (!gestureState || !pendingGestureUpdate) {
    return;
  }
  const { origin, scale, rotation } = pendingGestureUpdate;
  pendingGestureUpdate = null;
  const currentOrigin = origin;
  gestureState.origin = currentOrigin;
  const imagePoint = screenToImage(currentOrigin);
  gestureState.imagePoint = imagePoint;
  hoverPoint = { x: imagePoint.x, y: imagePoint.y };
  hoverScreenPoint = { x: currentOrigin.x, y: currentOrigin.y };
  setCursorHold(dotCursorCss);
  if (pointerState.options.touch.enablePinchZoom) {
    const nextScale = gestureState.startScale * scale;
    if (Number.isFinite(nextScale) && nextScale > 0) {
      viewState.scale = Math.min(Math.max(nextScale, 0.1), 40);
    }
  }
  if (pointerState.options.touch.enableRotate) {
    const deadzone = Math.max(pointerState.options.touch.rotationDeadzoneDegrees || 0, 0);
    if (Math.abs(rotation) >= deadzone) {
      viewState.rotation = normalizeAngle(gestureState.startRotation + (rotation * Math.PI / 180));
      setCursorHold(dotCursorCss);
    } else {
      viewState.rotation = gestureState.startRotation;
    }
  }
  setOffsetForImagePoint(gestureState.imagePoint, gestureState.origin);
  userAdjustedScale = true;
  autoFitPending = false;
  draw();
  drawBrushPreview(null);
}

function renderHoverPreview() {
  if (hoverPoint) {
    if (PREVIEW_TOOL_TYPES.has(tool)) {
      drawBrushPreview(hoverPoint);
    } else if (CROSSHAIR_TOOL_TYPES.has(tool) && cursorInsideImage) {
      drawBrushPreview(hoverPoint, { crosshairOnly: true });
    } else {
      drawBrushPreview(null);
    }
  } else {
    drawBrushPreview(null);
  }
}

let pointerFrameScheduled = false;
let paintStrokeQueue = [];
let pendingPanPointer = null;
let hoverUpdatePending = false;
let pendingHoverScreenPoint = null;
let hoverScreenPoint = null;
let pendingHoverHasPreview = false;
let pendingAffinityIndexSet = null;
function schedulePointerFrame() {
  if (pointerFrameScheduled) {
    return;
  }
  pointerFrameScheduled = true;
  requestAnimationFrame(processPointerFrame);
}

function processPointerFrame() {
  pointerFrameScheduled = false;
  let panUpdated = false;
  if (isPanning && pendingPanPointer) {
    const dx = pendingPanPointer.x - lastPoint.x;
    const dy = pendingPanPointer.y - lastPoint.y;
    if (dx !== 0 || dy !== 0) {
      viewState.offsetX += dx;
      viewState.offsetY += dy;
      viewStateDirty = true;
      panUpdated = true;
    }
    lastPoint = pendingPanPointer;
    pendingPanPointer = null;
    hoverPoint = null;
    drawBrushPreview(null);
    updateHoverInfo(screenToImage(lastPoint));
    hoverUpdatePending = false;
    pendingHoverScreenPoint = null;
    pendingHoverHasPreview = false;
    hoverScreenPoint = null;
    setCursorHold(dotCursorCss);
  }

  if (isPainting && paintStrokeQueue.length) {
    for (let i = 0; i < paintStrokeQueue.length; i += 1) {
      paintStroke(paintStrokeQueue[i]);
    }
    const lastWorld = paintStrokeQueue[paintStrokeQueue.length - 1];
    hoverPoint = lastWorld;
    hoverScreenPoint = null;
    drawBrushPreview(hoverPoint);
    updateHoverInfo(hoverPoint);
    paintStrokeQueue.length = 0;
    hoverUpdatePending = false;
    pendingHoverScreenPoint = null;
    pendingHoverHasPreview = false;
  } else if (hoverUpdatePending) {
    if (pendingHoverScreenPoint) {
      const world = screenToImage(pendingHoverScreenPoint);
      if (pendingHoverHasPreview) {
        hoverPoint = world;
        hoverScreenPoint = { x: pendingHoverScreenPoint.x, y: pendingHoverScreenPoint.y };
        drawBrushPreview(hoverPoint);
      } else if (CROSSHAIR_TOOL_TYPES.has(tool)) {
        hoverPoint = world;
        hoverScreenPoint = { x: pendingHoverScreenPoint.x, y: pendingHoverScreenPoint.y };
        drawBrushPreview(hoverPoint, { crosshairOnly: true });
      } else {
        hoverPoint = null;
        hoverScreenPoint = null;
        drawBrushPreview(null);
      }
      updateHoverInfo(world);
    } else {
      hoverPoint = null;
      hoverScreenPoint = null;
      drawBrushPreview(null);
      updateHoverInfo(null);
    }
    hoverUpdatePending = false;
    pendingHoverScreenPoint = null;
    pendingHoverHasPreview = false;
  }

  if (panUpdated) {
    draw();
  }
}

function queuePaintPoint(world) {
  paintStrokeQueue.push({ x: world.x, y: world.y });
}

function updateCursor() {
  if (cursorOverride) {
    canvas.style.cursor = cursorOverride;
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
  if (!maskLabel) return;
  const prefix = nColorActive ? 'Mask Group' : 'Mask Label';
  maskLabel.textContent = prefix + ': ' + currentLabel;
}

function updateHistoryButtons() {
  if (undoButton) {
    undoButton.disabled = undoStack.length === 0;
  }
  if (redoButton) {
    redoButton.disabled = redoStack.length === 0;
  }
}

function updateLabelControls() {
  if (!labelValueInput) {
    return;
  }
  labelValueInput.value = String(currentLabel);
  if (currentLabel > 0) {
    const rgb = hashColorForLabel(currentLabel);
    if (Array.isArray(rgb) && rgb.length >= 3) {
      const [r, g, b] = rgb;
      const color = 'rgb(' + (r | 0) + ', ' + (g | 0) + ', ' + (b | 0) + ')';
      const luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
      const textColor = luminance > 0.62 ? 'rgba(20, 20, 20, 0.86)' : '#f6f6f6';
      labelValueInput.style.setProperty('background', color);
      labelValueInput.style.setProperty('border-color', 'rgba(0, 0, 0, 0.18)');
      labelValueInput.style.setProperty('color', textColor);
      labelValueInput.style.setProperty('caret-color', textColor);
      return;
    }
  }
  labelValueInput.style.removeProperty('background');
  labelValueInput.style.removeProperty('border-color');
  labelValueInput.style.removeProperty('color');
  labelValueInput.style.removeProperty('caret-color');
}

function updateMaskVisibilityLabel() {
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
    description = 'Fill (F)';
  } else if (mode === 'picker') {
    description = 'Eyedropper (I)';
  }
  toolInfo.textContent = 'Tool: ' + description;
}

function applyCurrentLabel(nextLabel, { scheduleSave = true } = {}) {
  const parsed = Number(nextLabel);
  if (!Number.isFinite(parsed)) {
    return;
  }
  const normalized = Math.max(0, Math.floor(parsed));
  if (normalized === currentLabel) {
    return;
  }
  if (eraseActive && normalized !== 0) {
    stopEraseOverride();
  }
  currentLabel = normalized;
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
  drawBrushPreview(hoverPoint);
}

function syncMaskOpacityControls() {
  if (maskOpacitySlider) {
    maskOpacitySlider.value = maskOpacity.toFixed(2);
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
    draw();
    scheduleStateSave();
  }
}

function updateMaskThresholdLabel() {
}

function syncMaskThresholdControls() {
  if (maskThresholdSlider) {
    maskThresholdSlider.value = maskThreshold.toFixed(1);
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

function syncFlowThresholdControls() {
  if (flowThresholdSlider) {
    flowThresholdSlider.value = flowThreshold.toFixed(1);
    refreshSlider('flowThreshold');
  }
  if (flowThresholdInput && document.activeElement !== flowThresholdInput) {
    flowThresholdInput.value = flowThreshold.toFixed(1);
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
  if (!silent) {
    scheduleMaskRebuild();
    scheduleStateSave();
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
  if (!silent) {
    draw();
    scheduleStateSave();
  }
}

function pushHistory(indices, before, after) {
  undoStack.push({ indices, before, after });
  if (undoStack.length > HISTORY_LIMIT) {
    undoStack.shift();
  }
  redoStack.length = 0;
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
}

function undo() {
  if (!undoStack.length) {
    return;
  }
  const entry = undoStack.pop();
  applyHistoryEntry(entry, false);
  redoStack.push(entry);
  // Update affinity graph incrementally for the edited indices
  try { updateAffinityGraphForIndices(entry.indices); } catch (_) {}
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
  if (!redoStack.length) {
    return;
  }
  const entry = redoStack.pop();
  applyHistoryEntry(entry, true);
  undoStack.push(entry);
  // Update affinity graph incrementally for the edited indices
  try { updateAffinityGraphForIndices(entry.indices); } catch (_) {}
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
  const hadPixels = hasAnyMaskPixels();
  const shouldRecord = Boolean(recordHistory && hadPixels);
  if (shouldRecord) {
    const before = maskValues.slice();
    const after = new Uint32Array(before.length);
    pushHistory(null, before, after);
  }
  maskValues.fill(0);
  maskHasNonZero = false;
  outlineState.fill(0);
  nColorActive = false;
  nColorValues = null;
  floodVisited = null;
  floodStack = null;
  floodOutput = null;
  clearColorCaches();
  clearAffinityGraphData();
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
  drawBrushPreview(hoverPoint);
  updateMaskLabel();
  updateColorModeLabel();
  log('Masks cleared');
  scheduleStateSave();
  return true;
}

function promptClearMasks({ skipConfirm = false } = {}) {
  if (!skipConfirm && !hasAnyMaskPixels() && !nColorActive) {
    return false;
  }
  if (!skipConfirm) {
    const confirmed = window.confirm(CLEAR_MASK_CONFIRM_MESSAGE);
    if (!confirmed) {
      return false;
    }
  }
  return performClearMasks({ recordHistory: true });
}

function collectBrushIndices(target, centerX, centerY) {
  const pixels = enumerateBrushPixels(centerX, centerY);
  for (const pixel of pixels) {
    target.add(pixel.y * imgWidth + pixel.x);
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
  log('brush kernel mode set to ' + brushKernelMode);
  drawBrushPreview(hoverPoint);
}

function getBrushKernelCenter(x, y) {
  if (brushKernelMode === BRUSH_KERNEL_MODES.SNAPPED) {
    return { x: Math.round(x), y: Math.round(y) };
  }
  return { x, y };
}

function enumerateBrushPixels(rawCenterX, rawCenterY) {
  const centerX = rawCenterX - 0.5;
  const centerY = rawCenterY - 0.5;
  if (brushKernelMode === BRUSH_KERNEL_MODES.SNAPPED) {
    return enumerateSnappedBrushPixels(centerX, centerY);
  }
  return enumerateSmoothBrushPixels(centerX, centerY);
}

function enumerateSmoothBrushPixels(centerX, centerY) {
  const pixels = [];
  const radius = (brushDiameter - 1) / 2;
  if (radius <= 0) {
    const x = Math.round(centerX);
    const y = Math.round(centerY);
    if (x >= 0 && x < imgWidth && y >= 0 && y < imgHeight) {
      pixels.push({ x, y });
    }
    return pixels;
  }
  const threshold = (radius + 0.35) * (radius + 0.35);
  const minX = Math.max(0, Math.floor(centerX - radius - 1));
  const maxX = Math.min(imgWidth - 1, Math.ceil(centerX + radius + 1));
  for (let x = minX; x <= maxX; x += 1) {
    const dx = x - centerX;
    if (dx * dx > threshold) {
      continue;
    }
    const dy = Math.sqrt(Math.max(0, threshold - dx * dx));
    let yMin = Math.ceil(centerY - dy);
    let yMax = Math.floor(centerY + dy);
    if (yMax < yMin) {
      continue;
    }
    if (yMin < 0) {
      yMin = 0;
    }
    if (yMax >= imgHeight) {
      yMax = imgHeight - 1;
    }
    for (let y = yMin; y <= yMax; y += 1) {
      pixels.push({ x, y });
    }
  }
  return pixels;
}

function enumerateSnappedBrushPixels(centerX, centerY) {
  const center = getBrushKernelCenter(centerX, centerY);
  const offsets = getSnappedKernelOffsets(brushDiameter);
  const pixels = [];
  for (const offset of offsets) {
    const x = center.x + offset.x;
    const y = center.y + offset.y;
    if (x >= 0 && x < imgWidth && y >= 0 && y < imgHeight) {
      pixels.push({ x, y });
    }
  }
  return pixels;
}

function getSnappedKernelOffsets(diameter) {
  if (snappedKernelCache.has(diameter)) {
    return snappedKernelCache.get(diameter);
  }
  const radius = (diameter - 1) / 2;
  const threshold = (radius + 0.35) * (radius + 0.35);
  const span = Math.ceil(radius + 1);
  const offsets = [];
  for (let y = -span; y <= span; y += 1) {
    for (let x = -span; x <= span; x += 1) {
      if (x * x + y * y <= threshold) {
        offsets.push({ x, y });
      }
    }
  }
  snappedKernelCache.set(diameter, offsets);
  return offsets;
}

function traverseLine(x0, y0, x1, y1, callback) {
  let ix0 = Math.round(x0);
  let iy0 = Math.round(y0);
  const ix1 = Math.round(x1);
  const iy1 = Math.round(y1);
  const dx = Math.abs(ix1 - ix0);
  const sx = ix0 < ix1 ? 1 : -1;
  const dy = -Math.abs(iy1 - iy0);
  const sy = iy0 < iy1 ? 1 : -1;
  let err = dx + dy;
  while (true) {
    callback(ix0, iy0);
    if (ix0 === ix1 && iy0 === iy1) break;
    const e2 = 2 * err;
    if (e2 >= dy) {
      err += dy;
      ix0 += sx;
    }
    if (e2 <= dx) {
      err += dx;
      iy0 += sy;
    }
  }
}

function paintStroke(point) {
  if (!strokeChanges) {
    strokeChanges = new Map();
  }
  const start = lastPaintPoint ? { x: lastPaintPoint.x, y: lastPaintPoint.y } : { x: point.x, y: point.y };
  const local = new Set();
  // Sample along the segment at scale-adjusted spacing to balance smoothness and performance
  const dx = point.x - start.x;
  const dy = point.y - start.y;
  const dist = Math.hypot(dx, dy);
  const spacing = Math.max(0.15, 0.5 / Math.max(viewState.scale, 0.0001));
  const steps = Math.max(1, Math.ceil(dist / spacing));
  for (let i = 0; i <= steps; i += 1) {
    const t = steps === 0 ? 1 : i / steps;
    const px = start.x + dx * t;
    const py = start.y + dy * t;
    collectBrushIndices(local, px, py);
  }
  // Single-buffer model: paint the current label value (instance or group ID depending on mode)
  const paintLabel = currentLabel;
  try {
    if (hasNColor && nColorActive) {
      window.__lastPaintRawLabel = paintLabel | 0;
      window.__lastPaintGroupId = (currentLabel | 0);
    } else {
      window.__lastPaintRawLabel = paintLabel | 0;
      window.__lastPaintGroupId = 0;
    }
  } catch (_) { /* noop */ }
  let changed = false;
  const changedIndices = [];
  local.forEach((idx) => {
    if (!strokeChanges.has(idx)) {
      const original = maskValues[idx];
      if (original === paintLabel) {
        return;
      }
      strokeChanges.set(idx, original);
    }
    if (maskValues[idx] !== paintLabel) {
      maskValues[idx] = paintLabel;
      changedIndices.push(idx);
      changed = true;
      if (paintLabel > 0) {
        maskHasNonZero = true;
      }
    }
  });
  if (changed) {
    if (changedIndices.length) {
      if (isPainting) {
        if (!pendingAffinityIndexSet) {
          pendingAffinityIndexSet = new Set();
        }
        for (let i = 0; i < changedIndices.length; i += 1) {
          pendingAffinityIndexSet.add(changedIndices[i]);
        }
      } else {
        updateAffinityGraphForIndices(changedIndices);
      }
    }
    if (isWebglPipelineActive() && changedIndices.length) {
      markMaskIndicesDirty(changedIndices);
    }
    clearColorCaches();
    needsMaskRedraw = true;
    requestPaintFrame();
    scheduleStateSave();
    // Keep currentLabel unchanged; maskValues already reflects the selected mode's label space.
  }
  lastPaintPoint = { x: point.x, y: point.y };
}

function finalizeStroke() {
  if (!strokeChanges || strokeChanges.size === 0) {
    strokeChanges = null;
    return;
  }
  const keys = Array.from(strokeChanges.keys()).sort((a, b) => a - b);
  const count = keys.length;
  const indices = new Uint32Array(count);
  const before = new Uint32Array(count);
  const after = new Uint32Array(count);
  for (let i = 0; i < count; i += 1) {
    const idx = keys[i];
    indices[i] = idx;
    before[i] = strokeChanges.get(idx);
    after[i] = currentLabel;
  }
  pushHistory(indices, before, after);
  try { window.__pendingRelabelSelection = indices; } catch (_) { /* noop */ }
  strokeChanges = null;
  const affinityFlushed = flushPendingAffinityUpdates();
  if (!affinityFlushed) {
    updateAffinityGraphForIndices(indices);
  }
  if (webglOverlay && webglOverlay.enabled) {
    webglOverlay.needsGeometryRebuild = true;
  }
  markAffinityGeometryDirty();
  if (isWebglPipelineActive()) {
    markMaskIndicesDirty(indices);
  }
  draw();
  scheduleStateSave();
  // Apply any queued segmentation result that arrived during the stroke, otherwise run any pending rebuild
  if (pendingSegmentationPayload) {
    const queued = pendingSegmentationPayload;
    pendingSegmentationPayload = null;
    applySegmentationMask(queued);
    pendingMaskRebuild = false;
  } else if (pendingMaskRebuild && segmentationUpdateTimer === null && canRebuildMask) {
    triggerMaskRebuild();
  }
}


function ensureFloodBuffers() {
  const size = maskValues.length | 0;
  if (!floodVisited || floodVisited.length !== size) {
    floodVisited = new Uint32Array(size);
    floodVisitStamp = 1;
  } else if (floodVisitStamp >= 0xffffffff) {
    floodVisited.fill(0);
    floodVisitStamp = 1;
  }
  if (!floodStack || floodStack.length !== size) {
    floodStack = new Uint32Array(size);
  }
  if (!floodOutput || floodOutput.length !== size) {
    floodOutput = new Uint32Array(size);
  }
}

function floodFill(point) {
  const sx = Math.round(point.x);
  const sy = Math.round(point.y);
  if (sx < 0 || sy < 0 || sx >= imgWidth || sy >= imgHeight) {
    return;
  }
  const startIdx = sy * imgWidth + sx;
  const targetLabel = maskValues[startIdx] | 0;
  const paintLabel = currentLabel | 0;
  if (targetLabel === paintLabel) {
    return;
  }

  if (!maskHasNonZero && targetLabel === 0 && paintLabel > 0) {
    const total = maskValues.length;
    const idxArr = new Uint32Array(total);
    const before = new Uint32Array(total);
    const after = new Uint32Array(total);
    after.fill(paintLabel);
    for (let i = 0; i < total; i += 1) {
      idxArr[i] = i;
      maskValues[i] = paintLabel;
    }
    maskHasNonZero = true;
    pushHistory(idxArr, before, after);
    rebuildLocalAffinityGraph();
    if (webglOverlay && webglOverlay.enabled) {
      webglOverlay.needsGeometryRebuild = true;
    }
    clearColorCaches();
    if (isWebglPipelineActive()) {
      markMaskTextureFullDirty();
      markOutlineTextureFullDirty();
    }
    needsMaskRedraw = true;
    requestPaintFrame();
    return;
  }

  ensureFloodBuffers();
  const width = imgWidth;
  const height = imgHeight;
  const totalPixels = maskValues.length;
  const stack = floodStack;
  const visited = floodVisited;
  const output = floodOutput;
  let top = 0;
  let count = 0;

  let stamp = floodVisitStamp++;
  if (stamp >= 0xffffffff) {
    visited.fill(0);
    floodVisitStamp = 1;
    stamp = floodVisitStamp++;
  }

  stack[top++] = startIdx;

  while (top > 0) {
    const idx = stack[--top];
    if (idx < 0 || idx >= totalPixels) {
      continue;
    }
    if (visited[idx] === stamp) {
      continue;
    }
    if ((maskValues[idx] | 0) !== targetLabel) {
      continue;
    }

    const row = (idx / width) | 0;
    let left = idx;
    while (left % width !== 0) {
      const candidate = left - 1;
      if ((maskValues[candidate] | 0) !== targetLabel || visited[candidate] === stamp) {
        break;
      }
      left = candidate;
    }
    let right = idx;
    while (right % width !== width - 1) {
      const candidate = right + 1;
      if ((maskValues[candidate] | 0) !== targetLabel || visited[candidate] === stamp) {
        break;
      }
      right = candidate;
    }

    const leftX = left % width;
    const rightX = right % width;
    const rowOffset = row * width;

    for (let xi = leftX; xi <= rightX; xi += 1) {
      const fillIdx = rowOffset + xi;
      if (visited[fillIdx] !== stamp) {
        visited[fillIdx] = stamp;
        output[count++] = fillIdx;
      }
    }

    const yAbove = row - 1;
    if (yAbove >= 0) {
      const aboveOffset = yAbove * width;
      let xi = leftX;
      while (xi <= rightX) {
        const idxAbove = aboveOffset + xi;
        if (visited[idxAbove] === stamp || (maskValues[idxAbove] | 0) !== targetLabel) {
          xi += 1;
        } else {
          stack[top++] = idxAbove;
          xi += 1;
          while (xi <= rightX) {
            const checkIdx = aboveOffset + xi;
            if (visited[checkIdx] === stamp || (maskValues[checkIdx] | 0) !== targetLabel) {
              break;
            }
            xi += 1;
          }
        }
      }
    }

    const yBelow = row + 1;
    if (yBelow < height) {
      const belowOffset = yBelow * width;
      let xi = leftX;
      while (xi <= rightX) {
        const idxBelow = belowOffset + xi;
        if (visited[idxBelow] === stamp || (maskValues[idxBelow] | 0) !== targetLabel) {
          xi += 1;
        } else {
          stack[top++] = idxBelow;
          xi += 1;
          while (xi <= rightX) {
            const checkIdx = belowOffset + xi;
            if (visited[checkIdx] === stamp || (maskValues[checkIdx] | 0) !== targetLabel) {
              break;
            }
            xi += 1;
          }
        }
      }
    }
  }

  if (count === 0) {
    return;
  }

  const idxArr = new Uint32Array(count);
  const before = new Uint32Array(count);
  if (targetLabel !== 0) {
    before.fill(targetLabel);
  }
  const after = new Uint32Array(count);
  if (paintLabel !== 0) {
    after.fill(paintLabel);
  }
  const fillsAll = count === totalPixels;
  if (fillsAll) {
    for (let i = 0; i < count; i += 1) {
      idxArr[i] = i;
      maskValues[i] = paintLabel;
    }
  } else {
    for (let i = 0; i < count; i += 1) {
      const fillIdx = output[i];
      idxArr[i] = fillIdx;
      maskValues[fillIdx] = paintLabel;
    }
  }
  if (paintLabel > 0) {
    maskHasNonZero = true;
  } else if (paintLabel === 0 && fillsAll) {
    maskHasNonZero = false;
  } else if (paintLabel === 0 && maskHasNonZero) {
    maskHasNonZero = false;
    for (let i = 0; i < maskValues.length; i += 1) {
      if ((maskValues[i] | 0) > 0) {
        maskHasNonZero = true;
        break;
      }
    }
  }

  pushHistory(idxArr, before, after);
  const largeFill = fillsAll || count >= totalPixels * FLOOD_REBUILD_THRESHOLD;
  if (largeFill) {
    rebuildLocalAffinityGraph();
    if (webglOverlay && webglOverlay.enabled) {
      webglOverlay.needsGeometryRebuild = true;
    }
  } else {
    updateAffinityGraphForIndices(idxArr);
  }
  if (fillsAll) {
    markMaskTextureFullDirty();
    markOutlineTextureFullDirty();
  } else {
    markMaskIndicesDirty(idxArr);
  }
  clearColorCaches();
  needsMaskRedraw = true;
  applyMaskRedrawImmediate();
  draw();
  scheduleStateSave();
  // Do not change currentLabel here; display coloring comes from nColor.
}

function pickColor(point) {
  const sx = Math.round(point.x);
  const sy = Math.round(point.y);
  if (sx < 0 || sy < 0 || sx >= imgWidth || sy >= imgHeight) {
    return;
  }
  const idx = sy * imgWidth + sx;
  // Single-buffer model: pick the current pixel's label (group ID or instance label)
  currentLabel = maskValues[idx] | 0;
  updateMaskLabel();
  log('picker set ' + (nColorActive ? 'color group ' : 'raw label ') + currentLabel);
}

function labelAtPoint(point) {
  if (!point) return 0;
  const x = Math.round(point.x);
  const y = Math.round(point.y);
  if (x < 0 || y < 0 || x >= imgWidth || y >= imgHeight) {
    return 0;
  }
  return maskValues[y * imgWidth + x] | 0;
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
    // Alpha policy:
    // - If outlinesVisible: outline pixels use max alpha; interior uses half alpha
    // - Else: uniform alpha for all mask pixels
    const maxAlpha = Math.round(Math.max(0, Math.min(1, maskOpacity)) * 255);
    let alpha = maxAlpha;
    if (outlinesVisible) {
      alpha = outlineState[i] ? maxAlpha : Math.max(0, Math.min(255, (maxAlpha >> 1))); // half alpha for interior
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

function serializeHistoryEntry(entry) {
  if (!entry) return null;
  return {
    indices: base64FromUint32(entry.indices || new Uint32Array()),
    before: base64FromUint32(entry.before || new Uint32Array()),
    after: base64FromUint32(entry.after || new Uint32Array()),
  };
}

function deserializeHistoryEntry(serialized, expectedLength) {
  if (!serialized) return null;
  return {
    indices: uint32FromBase64(serialized.indices || '', expectedLength),
    before: uint32FromBase64(serialized.before || ''),
    after: uint32FromBase64(serialized.after || ''),
  };
}

function serializeHistoryStack(stack, limit = HISTORY_LIMIT) {
  if (!Array.isArray(stack)) {
    return [];
  }
  const start = Math.max(0, stack.length - limit);
  const result = [];
  for (let i = start; i < stack.length; i += 1) {
    const serialized = serializeHistoryEntry(stack[i]);
    if (serialized) {
      result.push(serialized);
    }
  }
  return result;
}

function deserializeHistoryStack(serialized, target, expectedLength = maskValues.length) {
  if (!Array.isArray(serialized) || !target) {
    updateHistoryButtons();
    return;
  }
  target.length = 0;
  const start = Math.max(0, serialized.length - HISTORY_LIMIT);
  for (let i = start; i < serialized.length; i += 1) {
    const entry = deserializeHistoryEntry(serialized[i], expectedLength);
    if (entry) {
      target.push(entry);
    }
  }
  updateHistoryButtons();
}

function collectViewerState() {
  return {
    width: imgWidth,
    height: imgHeight,
    mask: base64FromUint32(maskValues),
    maskHasNonZero: Boolean(maskHasNonZero),
    outline: base64FromUint8Array(outlineState),
    undoStack: serializeHistoryStack(undoStack),
    redoStack: serializeHistoryStack(redoStack),
    currentLabel,
    maskOpacity,
    maskThreshold,
    flowThreshold,
    viewState: {
      scale: viewState.scale,
      offsetX: viewState.offsetX,
      offsetY: viewState.offsetY,
      rotation: viewState.rotation,
    },
    tool,
    brushDiameter,
    maskVisible,
    imageVisible,
    nColorActive,
    nColorValues: nColorActive && nColorValues ? base64FromUint32(nColorValues) : null,
    clusterEnabled,
    affinitySegEnabled,
    showFlowOverlay,
    showDistanceOverlay,
    showAffinityGraph: Boolean(showAffinityGraph),
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
    } else {
      maskValues.fill(0);
      maskHasNonZero = false;
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
    deserializeHistoryStack(saved.undoStack, undoStack, expectedLength);
    deserializeHistoryStack(saved.redoStack, redoStack, expectedLength);
    if (typeof saved.currentLabel === 'number') {
      currentLabel = saved.currentLabel;
      updateMaskLabel();
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
    if (saved.viewState) {
      const vs = saved.viewState;
      if (typeof vs.scale === 'number') viewState.scale = vs.scale;
      if (typeof vs.offsetX === 'number') viewState.offsetX = vs.offsetX;
      if (typeof vs.offsetY === 'number') viewState.offsetY = vs.offsetY;
      if (typeof vs.rotation === 'number') viewState.rotation = vs.rotation;
    }
    if (typeof saved.brushDiameter === 'number') {
      setBrushDiameter(saved.brushDiameter, false);
    }
    if (typeof saved.tool === 'string') {
      setTool(saved.tool);
    }
    if (typeof saved.maskVisible === 'boolean') {
      maskVisible = saved.maskVisible;
      if (maskVisibilityToggle) {
        maskVisibilityToggle.checked = maskVisible;
      }
      updateMaskVisibilityLabel();
    }
    if (typeof saved.imageVisible === 'boolean') {
      setImageVisible(saved.imageVisible, { silent: true });
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
    if (typeof saved.showAffinityGraph === 'boolean') {
      showAffinityGraph = saved.showAffinityGraph;
      if (affinityGraphToggle) {
        affinityGraphToggle.checked = showAffinityGraph;
      }
      if (!showAffinityGraph) {
        clearWebglOverlaySurface();
      }
    }
    if (saved.nColorActive) {
      try {
        const decoded = saved.nColorValues ? uint32FromBase64(saved.nColorValues, maskValues.length) : null;
        if (decoded && decoded.length === maskValues.length) {
          nColorValues = decoded;
          nColorActive = true;
        } else {
          nColorActive = false;
          nColorValues = null;
        }
      } catch (err) {
        console.warn('Failed to restore nColor state', err);
        nColorActive = false;
        nColorValues = null;
      }
    } else {
      nColorActive = false;
      nColorValues = null;
    }
    updateColorModeLabel();
    updateMaskLabel();
    updateMaskVisibilityLabel();
    updateToolInfo();
    updateBrushControls();
    if (flowOverlayToggle) flowOverlayToggle.checked = showFlowOverlay;
    if (distanceOverlayToggle) distanceOverlayToggle.checked = showDistanceOverlay;
    if (isWebglPipelineActive()) {
      markMaskTextureFullDirty();
      markOutlineTextureFullDirty();
    } else {
      redrawMaskCanvas();
    }
    markAffinityGeometryDirty();
  } finally {
    isRestoringState = false;
  }
  rebuildLocalAffinityGraph();
  if (webglOverlay && webglOverlay.enabled) {
    webglOverlay.needsGeometryRebuild = true;
  }
  if (!showAffinityGraph) {
    clearWebglOverlaySurface();
  }
  needsMaskRedraw = true;
  applyMaskRedrawImmediate();
  draw();
  stateDirty = false;
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
out vec4 outColor;

void main() {
  outColor = v_color;
}
`;
  const program = createWebglProgram(gl, vertexSource, fragmentSource);
  if (!program) {
    webglOverlay = { enabled: false, failed: true };
    return false;
  }
  const positionBuffer = gl.createBuffer();
  const colorBuffer = gl.createBuffer();
  gl.useProgram(program);
  gl.clearColor(0, 0, 0, 0);
  gl.lineWidth(1);
  const attribs = {
    position: gl.getAttribLocation(program, 'a_position'),
    color: gl.getAttribLocation(program, 'a_color'),
  };
  const uniforms = {
    matrix: gl.getUniformLocation(program, 'u_matrix'),
  };
  webglOverlay = {
    enabled: true,
    failed: false,
    shared: true,
    canvas: null,
    gl,
    program,
    attribs,
    uniforms,
    positionBuffer,
    colorBuffer,
    positionsArray: null,
    colorsArray: null,
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
  const newColors = new Float32Array(newCap * 8);
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
  // Two vertices per edge
  webglOverlay.colorsArray[baseCol] = r;
  webglOverlay.colorsArray[baseCol + 1] = g;
  webglOverlay.colorsArray[baseCol + 2] = b;
  webglOverlay.colorsArray[baseCol + 3] = a;
  webglOverlay.colorsArray[baseCol + 4] = r;
  webglOverlay.colorsArray[baseCol + 5] = g;
  webglOverlay.colorsArray[baseCol + 6] = b;
  webglOverlay.colorsArray[baseCol + 7] = a;
  if (BATCH_LIVE_OVERLAY_UPDATES) {
    webglOverlay.dirtyColSlots.add(slot);
  } else {
    const { gl } = webglOverlay;
    gl.bindBuffer(gl.ARRAY_BUFFER, webglOverlay.colorBuffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, baseCol * 4, webglOverlay.colorsArray.subarray(baseCol, baseCol + 8));
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
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(), gl.DYNAMIC_DRAW);
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
    if (seg && seg.map && seg.map.size) {
      totalEdges += seg.map.size;
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
    if (!seg || !seg.map || seg.map.size === 0) continue;
    const rgba = seg.rgba || parseCssColorToRgba(seg.color, AFFINITY_LINE_ALPHA);
    seg.map.forEach((coords, index) => {
      const slot = cursor;
      // write positions
      const basePos = slot * 4;
      webglOverlay.positionsArray[basePos] = coords[0];
      webglOverlay.positionsArray[basePos + 1] = coords[1];
      webglOverlay.positionsArray[basePos + 2] = coords[2];
      webglOverlay.positionsArray[basePos + 3] = coords[3];
      // write colors
      const baseCol = slot * 8;
      webglOverlay.colorsArray[baseCol] = rgba[0];
      webglOverlay.colorsArray[baseCol + 1] = rgba[1];
      webglOverlay.colorsArray[baseCol + 2] = rgba[2];
      webglOverlay.colorsArray[baseCol + 3] = rgba[3];
      webglOverlay.colorsArray[baseCol + 4] = rgba[0];
      webglOverlay.colorsArray[baseCol + 5] = rgba[1];
      webglOverlay.colorsArray[baseCol + 6] = rgba[2];
      webglOverlay.colorsArray[baseCol + 7] = rgba[3];
      seg.slots.set(index, slot);
      cursor += 1;
    });
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

function drawAffinityGraphWebgl() {
  if (!ensureWebglOverlayReady() || !webglOverlay || !webglOverlay.enabled) {
    return false;
  }
  if (webglOverlay.shared) {
    const matrix = computeWebglMatrix(webglOverlay.matrixCache, canvas.width, canvas.height);
    drawAffinityGraphShared(matrix);
    return true;
  }
  if (webglOverlay.shared) {
    const matrix = computeWebglMatrix(webglOverlay.matrixCache, canvas.width, canvas.height);
    drawAffinityGraphShared(matrix);
    return true;
  }
  const {
    gl, canvas: glCanvas, program, attribs, uniforms, matrixCache, msaa, resolve,
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
  if (!showAffinityGraph || !affinityGraphInfo || !affinityGraphInfo.values) {
    if (webglOverlay) {
      webglOverlay.displayAlpha = 0;
    }
    finalizeOverlayFrame(false);
    return true;
  }
  // Build geometry if marked dirty, or if buffers are empty/mismatched
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
  gl.useProgram(program);
  const matrix = computeWebglMatrix(matrixCache, glCanvas.width, glCanvas.height);
  gl.uniformMatrix3fv(uniforms.matrix, false, matrix);
  // Compute a global alpha based on the projected pixel size of the shortest affinity edge
  const s = Math.max(0.0001, Number(viewState && viewState.scale ? viewState.scale : 1.0));
  const minStep = minAffinityStepLength > 0 ? minAffinityStepLength : 1.0;
  const minEdgePx = Math.max(0, s * minStep);
  const dprSafe = Number.isFinite(dpr) && dpr > 0 ? dpr : 1;
  const cutoff = OVERLAY_PIXEL_FADE_CUTOFF * dprSafe;
  const t = cutoff <= 0 ? 1 : Math.max(0, Math.min(1, minEdgePx / cutoff));
  const alphaScale = t * t * (3 - 2 * t);
  const clampedAlpha = Math.max(0, Math.min(1, alphaScale));
  webglOverlay.displayAlpha = clampedAlpha;
  if (typeof window !== 'undefined') {
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
        gl.bufferSubData(gl.ARRAY_BUFFER, baseCol * 4, webglOverlay.colorsArray.subarray(baseCol, baseCol + 8));
      }
      webglOverlay.dirtyColSlots.clear();
    }
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, webglOverlay.positionBuffer);
  gl.enableVertexAttribArray(attribs.position);
  gl.vertexAttribPointer(attribs.position, 2, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, webglOverlay.colorBuffer);
  gl.enableVertexAttribArray(attribs.color);
  gl.vertexAttribPointer(attribs.color, 4, gl.FLOAT, false, 0, 0);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  const edgesToDraw = Math.max(webglOverlay.edgeCount | 0, (webglOverlay.maxUsedSlotIndex | 0) + 1);
  const verticesToDraw = Math.max(0, edgesToDraw) * 2;
  if (verticesToDraw > 0) {
    hasContent = true;
    gl.drawArrays(gl.LINES, 0, verticesToDraw);
  }
  gl.disableVertexAttribArray(attribs.position);
  gl.disableVertexAttribArray(attribs.color);
  gl.disable(gl.BLEND);
  finalizeOverlayFrame(hasContent);
  return true;
}

// Affinity graph rendering is handled exclusively by WebGL.

function clearAffinityGraphData() {
  affinityGraphInfo = null;
  affinityGraphNeedsLocalRebuild = true;
  affinityGraphSource = 'none';
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
  markAffinityGeometryDirty();
}

function applyAffinityGraphPayload(payload) {
  // If the backend did not provide an affinity graph, rebuild locally from the current mask
  // so that outlines and (optionally) the overlay remain consistent after parameter changes.
  if (!payload || !payload.encoded || !payload.steps || !payload.steps.length) {
    // Reset to defaults and clear previous buffers
    clearAffinityGraphData();
    // Rebuild a local graph from maskValues to keep outlines up-to-date
    rebuildLocalAffinityGraph();
    if (webglOverlay && webglOverlay.enabled) {
      webglOverlay.needsGeometryRebuild = true;
    }
    // Build segments for overlay if the user toggle is on
    if (showAffinityGraph && affinityGraphInfo && affinityGraphInfo.values) {
      buildAffinityGraphSegments();
    }
    // Always refresh outlines from the (local) graph
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
  refreshOppositeStepMapping();
  if (!maskValues || maskValues.length === 0) {
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
    const rawLabel = (maskValues[index] | 0);
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
        const neighborRaw = (maskValues[neighborIndex] | 0);
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
    return;
  }
  const planeStride = width * height;
  const segments = new Array(stepCount);
  for (let s = 0; s < stepCount; s += 1) {
    const [dyRaw, dxRaw] = affinitySteps[s];
    const dy = dyRaw | 0;
    const dx = dxRaw | 0;
    const planeOffset = s * planeStride;
    const map = new Map();
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
      map.set(idx, new Float32Array([x + 0.5, y + 0.5, nx + 0.5, ny + 0.5]));
    }
    segments[s] = {
      color: AFFINITY_OVERLAY_COLOR,
      map,
    };
  }
  affinityGraphInfo.segments = segments;
  markAffinityGeometryDirty();
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
  let changed = false;
  indicesSet.forEach((idx) => {
    const before = outlineState[idx];
    updateOutlineForIndex(idx);
    if (outlineState[idx] !== before) {
      changed = true;
    }
  });
  if (changed) {
    if (isWebglPipelineActive()) {
      const rectIndices = Array.from(indicesSet);
      markOutlineIndicesDirty(rectIndices);
    } else {
      // Refresh mask rendering to apply per-pixel alpha changes only if outline changed
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
  if (!indices || !indices.length) {
    rebuildLocalAffinityGraph();
    return;
  }
  if (!affinityGraphInfo || !affinityGraphInfo.values || !affinityGraphInfo.stepCount) {
    // Initialize the local affinity graph from the current mask so we can incrementally update it
    rebuildLocalAffinityGraph();
    if (!affinityGraphInfo || !affinityGraphInfo.values || !affinityGraphInfo.stepCount) {
      return;
    }
  }
  const info = affinityGraphInfo;
  if (!info.segments) {
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
  const affected = new Set();
  for (const value of indices) {
    const index = Number(value) | 0;
    if (index < 0 || index >= planeStride) continue;
    affected.add(index);
    const x = index % width;
    const y = (index / width) | 0;
    for (let s = 0; s < stepCount; s += 1) {
      const [dyRaw, dxRaw] = affinitySteps[s];
      const dx = dxRaw | 0;
      const dy = dyRaw | 0;
      const nx = x + dx;
      const ny = y + dy;
      if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
        affected.add(ny * width + nx);
      }
      const px = x - dx;
      const py = y - dy;
      if (px >= 0 && px < width && py >= 0 && py < height) {
        affected.add(py * width + px);
      }
    }
  }
  if (!affected.size) return;
  const { values, segments } = info;
  const outlineUpdateSet = new Set(affected);
  for (const index of affected) {
    const label = (maskValues[index] | 0);
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
        const neighborRaw = (maskValues[neighborIndex] | 0);
        if (label > 0 && neighborRaw > 0 && neighborRaw === label) {
          value = 1;
        }
        outlineUpdateSet.add(neighborIndex);
      }
      values[planeOffset] = value;
      // Maintain symmetric edge in the opposite step direction
      if (neighborIndex >= 0) {
        const oppIdx = affinityOppositeSteps[s] | 0;
        if (oppIdx >= 0) {
          values[oppIdx * planeStride + neighborIndex] = value;
        }
      }
      const segment = segments[s];
      if (segment) {
        if (!segment.rgba) segment.rgba = parseCssColorToRgba(segment.color, AFFINITY_LINE_ALPHA);
        if (!segment.map) segment.map = new Map();
        if (neighborIndex >= 0) {
          const cx1 = x + 0.5;
          const cy1 = y + 0.5;
          const cx2 = (x + (dx | 0)) + 0.5;
          const cy2 = (y + (dy | 0)) + 0.5;
          if (value) {
            let coords = segment.map.get(index);
            if (coords) {
              coords[0] = cx1; coords[1] = cy1; coords[2] = cx2; coords[3] = cy2;
            } else {
              segment.map.set(index, new Float32Array([cx1, cy1, cx2, cy2]));
            }
          } else {
            segment.map.delete(index);
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
        const oppSegment = segments[oppIdx];
        if (oppSegment) {
          if (!oppSegment.rgba) oppSegment.rgba = parseCssColorToRgba(oppSegment.color, AFFINITY_LINE_ALPHA);
          if (!oppSegment.map) oppSegment.map = new Map();
          const cx1o = (x + (dx | 0)) + 0.5;
          const cy1o = (y + (dy | 0)) + 0.5;
          const cx2o = x + 0.5;
          const cy2o = y + 0.5;
          if (value) {
            let coords2 = oppSegment.map.get(neighborIndex);
            if (coords2) {
              coords2[0] = cx1o; coords2[1] = cy1o; coords2[2] = cx2o; coords2[3] = cy2o;
            } else {
              oppSegment.map.set(neighborIndex, new Float32Array([cx1o, cy1o, cx2o, cy2o]));
            }
          } else {
            oppSegment.map.delete(neighborIndex);
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
  affinityGraphNeedsLocalRebuild = false;
  updateOutlineForIndices(outlineUpdateSet);
  markAffinityGeometryDirty();
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
  draw();
  drawBrushPreview(hoverPoint);
  viewStateDirty = true;
  // Briefly show dot cursor for keyboard rotation
  setCursorTemporary(dotCursorCss, 700);
  // Show a simple dot cursor briefly
  setCursorTemporary(dotCursorCss, 600);
}

function draw() {
  if (shuttingDown) {
    return;
  }
  updateFps(typeof performance !== 'undefined' ? performance.now() : Date.now());
  if (shouldLogDraw()) {
    log('draw start scale=' + viewState.scale.toFixed(3) + ' offset=' + viewState.offsetX.toFixed(1) + ',' + viewState.offsetY.toFixed(1));
  }
  if (isWebglPipelineActive()) {
    if (needsMaskRedraw) {
      flushMaskTextureUpdates();
      needsMaskRedraw = false;
    }
    drawWebglFrame();
    drawBrushPreview(hoverPoint);
    if (!loggedPixelSample && canvas.width > 0 && canvas.height > 0) {
      loggedPixelSample = true;
      hideLoadingOverlay();
    }
    return;
  }
  if (!ctx) {
    return;
  }
  // Do not rebuild affinity graph locally; graph remains fixed
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.restore();
  ctx.save();
  applyViewTransform(ctx, { includeDpr: true });
  const smooth = viewState.scale < 1;
  ctx.imageSmoothingEnabled = smooth;
  ctx.imageSmoothingQuality = smooth ? 'high' : 'low';
  if (imageVisible) {
    ctx.drawImage(offscreen, 0, 0);
  }
  if (maskVisible) {
    // Always render mask overlay with nearest-neighbor for crisp pixel edges
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(maskCanvas, 0, 0);
    // Outline is integrated via per-pixel alpha; no separate overlay draw
  }
  if (showFlowOverlay && flowOverlayImage && flowOverlayImage.complete) {
    ctx.save();
    ctx.globalAlpha = 0.7;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(flowOverlayImage, 0, 0);
    ctx.restore();
  }
  if (showDistanceOverlay && distanceOverlayImage && distanceOverlayImage.complete) {
    ctx.save();
    ctx.globalAlpha = 0.6;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(distanceOverlayImage, 0, 0);
    ctx.restore();
  }
  ctx.restore();
  drawAffinityGraphOverlay();
  drawBrushPreview(hoverPoint);
  if (!loggedPixelSample && canvas.width > 0 && canvas.height > 0) {
    try {
      const cx = Math.floor(canvas.width / 2);
      const cy = Math.floor(canvas.height / 2);
      const sample = ctx.getImageData(cx, cy, 1, 1).data;
      log('center pixel rgba=' + Array.from(sample).join(','));
    } catch (err) {
      log('center pixel read failed: ' + (err && err.message ? err.message : err));
    }
    loggedPixelSample = true;
    hideLoadingOverlay();
  }

  if (viewStateDirty) {
    viewStateDirty = false;
    scheduleStateSave(800);
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

// Recompute N-color mapping from the CURRENT client mask and overwrite labels with group IDs.
function recomputeNColorFromCurrentMask(forceActive = false) {
  return new Promise((resolve) => {
    try {
      if (!nColorActive && !forceActive) { resolve(false); return; }
      const buf = maskValues.buffer.slice(maskValues.byteOffset, maskValues.byteOffset + maskValues.byteLength);
      const bytes = new Uint8Array(buf);
      const b64 = base64FromUint8(bytes);
      const payload = { mask: b64, width: imgWidth, height: imgHeight };
      const applyMapping = (obj) => {
        try {
          if (!obj || !obj.nColorMask) { resolve(false); return; }
          const bin = atob(obj.nColorMask);
          if (bin.length !== maskValues.length * 4) { resolve(false); return; }
          const buffer = new ArrayBuffer(bin.length);
          const arr = new Uint8Array(buffer);
          for (let i = 0; i < bin.length; i += 1) arr[i] = bin.charCodeAt(i);
          const groups = new Uint32Array(buffer);
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
          clearColorCaches();
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
  try {
    const buf = maskValues.buffer.slice(maskValues.byteOffset, maskValues.byteOffset + maskValues.byteLength);
    const bytes = new Uint8Array(buf);
    const b64 = base64FromUint8(bytes);
    // Attach current affinity graph (required); do not rebuild locally
    if (!affinityGraphInfo || !affinityGraphInfo.values || !affinityGraphInfo.stepCount) {
      console.warn('No affinity graph available for relabel_from_affinity');
      return false;
    }
    const enc = base64FromUint8(affinityGraphInfo.values);
    const affinityGraph = {
      width: affinityGraphInfo.width,
      height: affinityGraphInfo.height,
      steps: affinitySteps.map((p) => [p[0] | 0, p[1] | 0]),
      encoded: enc,
    };
    const payload = { mask: b64, width: imgWidth, height: imgHeight, affinityGraph };
    if (hasPywebview) {
      const result = await window.pywebview.api.relabel_from_affinity(payload);
      if (result && !result.error && result.mask) {
        applySegmentationMask(result);
        log('Applied relabel_from_affinity from backend (pywebview).');
        return true;
      }
    }
    if (canHttp) {
      const res = await fetch('/api/relabel_from_affinity', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (res.ok) {
        const result = await res.json();
        if (result && !result.error && result.mask) {
          applySegmentationMask(result);
          log('Applied relabel_from_affinity from backend (HTTP).');
          return true;
        }
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
    if (!seg || !seg.map) continue;
    seg.map.forEach((coords) => {
      ctx.moveTo(coords[0], coords[1]);
      ctx.lineTo(coords[2], coords[3]);
    });
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
  resizePreviewCanvas();
  if (!fitViewToWindow(v)) {
    recenterView(v);
  }
  draw();
  drawBrushPreview(hoverPoint);
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
  viewState.rotation = 0;
  if (!fitViewToWindow(metrics)) {
    viewState.rotation = 0;
    recenterView(metrics);
  }
  draw();
  renderHoverPreview();
  updateHoverInfo(hoverPoint || null);
  viewStateDirty = true;
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

function generateSinebowPalette(size, offset = 0) {
  const count = Math.max(size, 2);
  const palette = new Array(count);
  palette[0] = [0, 0, 0, 0];
  const golden = 0.61803398875;
  for (let i = 1; i < count; i += 1) {
    const t = (offset + i * golden) % 1;
    palette[i] = sinebowColor(t);
  }
  return palette;
}

defaultPalette = generateSinebowPalette(Math.max(colorTable.length || 0, 256), 0.0);
// N-color palette is large enough; groups come from backend ncolor.label
nColorPalette = generateSinebowPalette(Math.max(colorTable.length || 0, 1024), 0.35);

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

function getColorFromMap(label, map) {
  if (label <= 0) {
    return null;
  }
  let rgb = map.get(label);
  if (!rgb) {
    rgb = hashColorForLabel(label);
    map.set(label, rgb);
  }
  return rgb;
}

function getDisplayColor(index) {
  const rawLabel = maskValues[index] | 0;
  if (rawLabel <= 0) return null;
  if (nColorActive && nColorValues && nColorValues.length === maskValues.length) {
    const groupId = nColorValues[index] | 0;
    if (groupId > 0) return getColorFromMap(groupId, nColorColorMap);
  }
  return getColorFromMap(rawLabel, rawColorMap);
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


function resetNColorAssignments() {
  nColorAssignments.clear();
  nColorColorToLabel.clear();
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
  if (histRangeLabel) {
    histRangeLabel.textContent = 'Window: [' + windowLow + ', ' + windowHigh + ']';
  }
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
  if (!hoverInfo) {
    cursorInsideImage = false;
    updateCursor();
    return;
  }
  if (!point || !originalImageData) {
    cursorInsideImage = false;
    hoverInfo.textContent = 'Y: --, X: --, Val: --';
    updateCursor();
    return;
  }
  const x = Math.round(point.x);
  const y = Math.round(point.y);
  if (x < 0 || y < 0 || x >= imgWidth || y >= imgHeight) {
    cursorInsideImage = false;
    hoverInfo.textContent = 'Y: --, X: --, Val: --';
    updateCursor();
    return;
  }
  const idx = (y * imgWidth + x) * 4;
  const value = originalImageData.data[idx];
  cursorInsideImage = true;
  hoverInfo.textContent = 'Y: ' + y + ', X: ' + x + ', Val: ' + value;
  updateCursor();
}

function updateColorModeLabel() {
  if (!colorMode) {
    return;
  }
  const mode = nColorActive ? 'N-Color' : 'Palette';
  colorMode.textContent = 'Mask Colors: ' + mode + " (toggle with 'N')";
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
      draw();
      scheduleStateSave();
    });
    return;
  }
  // OFF: relabel by affinity using current groups; result is instance labels
  try { window.__pendingRelabelSelection = null; } catch (_) {}
  const prevLabel = currentLabel | 0;
  relabelFromAffinity()
    .then((ok) => {
      if (!ok) console.warn('relabel_from_affinity failed during N-color OFF');
      nColorActive = false;
      clearColorCaches();
      if (isWebglPipelineActive()) {
        markMaskTextureFullDirty();
        markOutlineTextureFullDirty();
      } else {
        redrawMaskCanvas();
      }
      // Do not modify the affinity graph when toggling OFF
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
      draw();
      scheduleStateSave();
    })
    .catch((err) => {
      console.warn('N-color OFF relabel failed', err);
    });
}

// buildLinksFromCurrentGraph removed: we keep a single connectivity source based on raw labels only

function getSegmentationSettingsPayload() {
  const payload = {
    mask_threshold: Number(maskThreshold.toFixed(2)),
    flow_threshold: Number(flowThreshold.toFixed(2)),
    cluster: Boolean(clusterEnabled),
    affinity_seg: Boolean(affinitySegEnabled),
  };
  if (sessionId) {
    payload.sessionId = sessionId;
  }
  if (currentImagePath) {
    payload.image_path = currentImagePath;
  }
  return payload;
}

const overlayUpdateThrottleMs = 140;
let lastRebuildTime = 0;
let INTERACTIVE_REBUILD_INTERVAL = 120;
let lastRebuildDuration = 120;
let nColorActive = false;
let nColorValues = null; // per-pixel group IDs for N-color display only
// Single authoritative mask buffer: maskValues always holds instance labels.
const rawColorMap = new Map();
const nColorColorMap = new Map();
// Legacy assignment structures retained but unused with single-buffer model
const nColorAssignments = new Map();
const nColorColorToLabel = new Map();
let nColorMaxColorId = 0;
let lastLabelBeforeNColor = null;

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
    } else {
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
      } else {
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
    } else {
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
    } else {
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

function applySegmentationMask(payload) {
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
  const binary = atob(payload.mask);
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
  maskHasNonZero = hasNonZero;
  // Switching to a new mask implicitly leaves current color mode as-is; caller controls nColorActive.
  resetNColorAssignments();
  clearColorCaches();
  if (isWebglPipelineActive()) {
    markMaskTextureFullDirty();
    markOutlineTextureFullDirty();
  } else {
    redrawMaskCanvas();
  }
  updateOverlayImages(payload);
  applyAffinityGraphPayload(payload.affinityGraph);
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

async function runSegmentation() {
  if (!segmentButton || isSegmenting) {
    return;
  }
  isSegmenting = true;
  segmentButton.disabled = true;
  setSegmentStatus('Running segmentation…');
  try {
    const raw = await requestSegmentation();
    const payload = typeof raw === 'string' ? JSON.parse(raw) : raw;
    if (payload.error) {
      throw new Error(payload.error);
    }
    applySegmentationMask(payload);
    setSegmentStatus('Segmentation complete.');
  } catch (err) {
    console.error(err);
    const msg = err && err.message ? err.message : err;
    setSegmentStatus('Segmentation failed: ' + msg, true);
  } finally {
    isSegmenting = false;
    segmentButton.disabled = false;
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
    draw();
  }
  if (!rotationApplied && deltaY !== 0) {
    setCursorTemporary(dotCursorCss, 350);
  }
  // Show simple dot cursor during wheel interactions
  setCursorTemporary(dotCursorCss, 350);
}, { passive: false });

function startPointerPan(evt) {
  isPainting = false;
  strokeChanges = null;
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
  hoverPoint = null;
  hoverScreenPoint = null;
  drawBrushPreview(null);
  updateHoverInfo(null);
}

function beginBrushStroke(evt, worldPoint) {
  strokeChanges = new Map();
  isPainting = true;
  canvas.classList.add('painting');
  updateCursor();
  try {
    canvas.setPointerCapture(evt.pointerId);
    activePointerId = evt.pointerId;
  } catch (_) {
    /* ignore */
  }
  paintStrokeQueue.length = 0;
  lastPaintPoint = null;
  paintStroke(worldPoint);
  hoverPoint = worldPoint ? { x: worldPoint.x, y: worldPoint.y } : null;
  if (hoverPoint) {
    const pointer = getPointerPosition(evt);
    hoverScreenPoint = { x: pointer.x, y: pointer.y };
  } else {
    hoverScreenPoint = null;
  }
  drawBrushPreview(hoverPoint);
  updateHoverInfo(hoverPoint);
}

canvas.addEventListener('pointerdown', (evt) => {
  cursorInsideCanvas = true;
  gestureState = null;
  pointerState.registerPointerDown(evt);
  const pointer = getPointerPosition(evt);
  lastPoint = pointer;
  const world = screenToImage(pointer);
  updateHoverInfo(world);
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
              log('pinch rotation raw=' + rotationDegrees.toFixed(2) + ' deg');
              pinchState.lastLoggedRotation = rotationDegrees;
            }
            const applyRotation = Math.abs(rotationDegrees) >= deadzone;
            viewState.rotation = applyRotation
              ? normalizeAngle(pinchState.startRotation + delta)
              : pinchState.startRotation;
            if (applyRotation) {
              log('pinch rotation applied delta=' + rotationDegrees.toFixed(2) + ' deg total=' + (viewState.rotation * RAD_TO_DEG).toFixed(2));
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
          draw();
          drawBrushPreview(null);
        }
      }
      evt.preventDefault();
      return;
    }
    if (isPanning && evt.pointerId === panPointerId) {
      const dx = pointer.x - lastPoint.x;
      const dy = pointer.y - lastPoint.y;
      viewState.offsetX += dx;
      viewState.offsetY += dy;
      lastPoint = pointer;
      draw();
      updateHoverInfo(null);
      evt.preventDefault();
      setCursorHold(dotCursorCss);
      viewStateDirty = true;
      return;
    }
  }
  if (isPanning && evt.pointerId === panPointerId) {
    pendingPanPointer = { x: pointer.x, y: pointer.y };
    hoverUpdatePending = true;
    pendingHoverScreenPoint = null;
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
    hoverPoint = { x: world.x, y: world.y };
    hoverScreenPoint = { x: pointer.x, y: pointer.y };
    drawBrushPreview(hoverPoint);
    updateHoverInfo(hoverPoint);
    hoverUpdatePending = false;
    pendingHoverScreenPoint = null;
    pendingHoverHasPreview = false;
    schedulePointerFrame();
    return;
  }
  if (!isPanning && !spacePan) {
    pendingHoverScreenPoint = { x: pointer.x, y: pointer.y };
    pendingHoverHasPreview = PREVIEW_TOOL_TYPES.has(tool);
    hoverUpdatePending = true;
    schedulePointerFrame();
  }
  if (isPanning) {
    pendingPanPointer = { x: pointer.x, y: pointer.y };
    hoverUpdatePending = true;
    pendingHoverScreenPoint = null;
    pendingHoverHasPreview = false;
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
      hoverScreenPoint = { x: pointer.x, y: pointer.y };
      if (CROSSHAIR_TOOL_TYPES.has(tool) || PREVIEW_TOOL_TYPES.has(tool)) {
        const world = screenToImage(pointer);
        hoverPoint = { x: world.x, y: world.y };
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
    finalizeStroke();
  }
  isPanning = false;
  spacePan = false;
  updateCursor();
  lastPaintPoint = null;
  paintStrokeQueue.length = 0;
  pendingPanPointer = null;
  hoverUpdatePending = false;
  pendingHoverScreenPoint = null;
  pendingHoverHasPreview = false;
  if (evt && evt.type === 'pointerleave') {
    clearHoverPreview();
  } else {
    renderHoverPreview();
    updateHoverInfo(hoverPoint || null);
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

function handleContextMenuEvent(evt) {
  if (evt && typeof evt.preventDefault === 'function') {
    evt.preventDefault();
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
    finalizeStroke();
  }
  isPainting = false;
  isPanning = false;
  spacePan = false;
  updateCursor();
}

canvas.addEventListener('pointerup', stopInteraction);
canvas.addEventListener('pointerleave', stopInteraction);
canvas.addEventListener('pointercancel', stopInteraction);
canvas.addEventListener('mouseenter', () => {
  cursorInsideCanvas = true;
  updateCursor();
});
canvas.addEventListener('mouseleave', () => {
  cursorInsideCanvas = false;
  cursorInsideImage = false;
  updateCursor();
});
canvas.addEventListener('contextmenu', handleContextMenuEvent);
if (viewer) {
  viewer.addEventListener('contextmenu', handleContextMenuEvent);
}

window.addEventListener('keydown', (evt) => {
  const tag = evt.target && evt.target.tagName ? evt.target.tagName.toLowerCase() : '';
  if (tag === 'input') {
    return;
  }
  const key = evt.key.toLowerCase();
  const modifier = evt.ctrlKey || evt.metaKey;
  if (key === 'escape' && dropdownOpenId) {
    closeDropdown(dropdownRegistry.get(dropdownOpenId));
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
    updateMaskVisibilityLabel();
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

if (sessionId) {
  window.addEventListener('beforeunload', () => {
    try {
      saveViewerState({ immediate: true });
    } catch (_) {
      /* ignore */
    }
  });
}
