// Extracted from app.js lines 2731-3526

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
  updateLabelControls();
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

updateLabelControls();
updateToolButtons();
updateBrushControls();

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

document.querySelectorAll('[data-slider-id]').forEach((root) => {
  registerSlider(root);
});
document.querySelectorAll('[data-dropdown-id]').forEach((root) => {
  registerDropdown(root);
});

updateLabelControls();
updateToolButtons();
updateBrushControls();
