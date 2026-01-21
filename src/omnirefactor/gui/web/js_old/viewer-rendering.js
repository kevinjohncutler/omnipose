// Extracted from app.js lines 5270-6469

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
  if (typeof window !== 'undefined') {
    window.__OMNI_LAST_DRAW_TS = Date.now();
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

function isImageVisibleInViewport({ margin = 24 } = {}) {
  if (!canvas) {
    return true;
  }
  const rect = canvas.getBoundingClientRect();
  if (!rect || rect.width <= 0 || rect.height <= 0) {
    return false;
  }
  const marginPx = Math.max(0, margin);
  const bounds = {
    left: -marginPx,
    right: rect.width + marginPx,
    top: -marginPx,
    bottom: rect.height + marginPx,
  };
  const testPoints = [
    { x: 0, y: 0 },
    { x: imgWidth, y: 0 },
    { x: 0, y: imgHeight },
    { x: imgWidth, y: imgHeight },
    { x: imgWidth / 2, y: imgHeight / 2 },
  ];
  for (let i = 0; i < testPoints.length; i += 1) {
    const pt = imageToScreen(testPoints[i]);
    if (!pt) {
      continue;
    }
    if (pt.x >= bounds.left && pt.x <= bounds.right && pt.y >= bounds.top && pt.y <= bounds.bottom) {
      return true;
    }
  }
  return false;
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
    log('computeHistogram skipped: no original image data');
    return;
  }
  histogramData = new Uint32Array(256);
  const data = originalImageData.data;
  for (let i = 0; i < data.length; i += 4) {
    histogramData[data[i]] += 1;
  }
  log('histogram stats bins[0]=' + histogramData[0] + ' bins[128]=' + histogramData[128] + ' bins[255]=' + histogramData[255]);
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
