// Extracted from app.js lines 3527-5269

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
  setImageVisible(true, { silent: true });
  if (imageVisibilityToggle) {
    imageVisibilityToggle.checked = true;
  }
  needsMaskRedraw = true;
  applyMaskRedrawImmediate();
  if (typeof isImageVisibleInViewport === 'function' && !isImageVisibleInViewport()) {
    resetView();
  } else {
    draw();
  }
  stateDirty = false;
  updateImageInfo();
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

