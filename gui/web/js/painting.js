(function initOmniPainting(global) {
  'use strict';

  const state = {
    ctx: null,
    strokeChanges: null,
    paintQueue: [],
    pendingAffinitySet: null,
    floodVisited: null,
    floodVisitStamp: 1,
    floodStack: null,
    floodOutput: null,
    lastPaintPoint: null,
    isPainting: false,
  };

  const brushApi = global.OmniBrush || {};

  function ensureCtx() {
    if (!state.ctx) {
      throw new Error('OmniPainting.init must be called before use');
    }
    return state.ctx;
  }

  function init(options) {
    state.ctx = Object.assign({
      maskValues: null,
      outlineState: null,
      viewState: null,
      getImageDimensions: () => ({ width: 0, height: 0 }),
      getCurrentLabel: () => 0,
      setCurrentLabel: () => {},
      hasNColor: false,
      isNColorActive: () => false,
      getMaskHasNonZero: () => false,
      setMaskHasNonZero: () => {},
      markMaskIndicesDirty: () => {},
      markMaskTextureFullDirty: () => {},
      markOutlineTextureFullDirty: () => {},
      updateAffinityGraphForIndices: () => {},
      rebuildLocalAffinityGraph: () => {},
      markAffinityGeometryDirty: () => {},
      isWebglPipelineActive: () => false,
      clearColorCaches: () => {},
      requestPaintFrame: () => {},
      scheduleStateSave: () => {},
      pushHistory: () => {},
      log: () => {},
      draw: () => {},
      redrawMaskCanvas: () => {},
      markNeedsMaskRedraw: () => {},
      applySegmentationMask: () => {},
      getPendingSegmentationPayload: () => null,
      setPendingSegmentationPayload: () => {},
      getPendingMaskRebuild: () => false,
      setPendingMaskRebuild: () => {},
      getSegmentationTimer: () => null,
      setSegmentationTimer: () => {},
      canRebuildMask: () => false,
      triggerMaskRebuild: () => {},
      applyMaskRedrawImmediate: () => {},
      collectBrushIndices: null,
      floodRebuildThreshold: 0.35,
    }, options || {});
    state.strokeChanges = null;
    state.paintQueue.length = 0;
    state.pendingAffinitySet = null;
    state.floodVisited = null;
    state.floodStack = null;
    state.floodOutput = null;
    state.floodVisitStamp = 1;
    state.lastPaintPoint = null;
    state.isPainting = false;
  }

  function beginStroke(point) {
    ensureCtx();
    state.strokeChanges = new Map();
    state.paintQueue.length = 0;
    state.pendingAffinitySet = null;
    state.lastPaintPoint = null;
    state.isPainting = true;
    if (point && Number.isFinite(point.x) && Number.isFinite(point.y)) {
      internalPaintStroke(point);
    }
    return { lastPoint: state.lastPaintPoint };
  }

  function queuePaintPoint(point) {
    if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y)) {
      return;
    }
    state.paintQueue.push({ x: point.x, y: point.y });
  }

  function processPaintQueue() {
    let lastWorld = null;
    if (!state.paintQueue.length) {
      return lastWorld;
    }
    while (state.paintQueue.length) {
      const point = state.paintQueue.shift();
      lastWorld = internalPaintStroke(point);
    }
    return lastWorld;
  }

  function cancelStroke() {
    state.strokeChanges = null;
    state.paintQueue.length = 0;
    state.pendingAffinitySet = null;
    state.lastPaintPoint = null;
    state.isPainting = false;
  }

  function flushPendingAffinityUpdates() {
    const ctx = ensureCtx();
    if (!state.pendingAffinitySet || state.pendingAffinitySet.size === 0) {
      return false;
    }
    const indices = Array.from(state.pendingAffinitySet);
    state.pendingAffinitySet.clear();
    state.pendingAffinitySet = null;
    if (indices.length && typeof ctx.updateAffinityGraphForIndices === 'function') {
      ctx.updateAffinityGraphForIndices(indices);
    }
    return true;
  }

  function finalizeStroke() {
    const ctx = ensureCtx();
    state.isPainting = false;
    const changes = state.strokeChanges;
    state.strokeChanges = null;
    state.lastPaintPoint = null;
    state.paintQueue.length = 0;
    const maskValues = ctx.maskValues;
    if (!changes || !changes.size || !maskValues) {
      flushPendingAffinityUpdates();
      return;
    }
    const keys = Array.from(changes.keys()).sort((a, b) => a - b);
    const count = keys.length;
    const indices = new Uint32Array(count);
    const before = new Uint32Array(count);
    const after = new Uint32Array(count);
    for (let i = 0; i < count; i += 1) {
      const idx = keys[i];
      indices[i] = idx;
      before[i] = changes.get(idx);
      after[i] = maskValues[idx] | 0;
    }
    if (typeof ctx.pushHistory === 'function') {
      ctx.pushHistory(indices, before, after);
    }
    try {
      global.__pendingRelabelSelection = indices;
    } catch (err) {
      /* ignore */
    }
    const affinityFlushed = flushPendingAffinityUpdates();
    if (!affinityFlushed && typeof ctx.updateAffinityGraphForIndices === 'function') {
      ctx.updateAffinityGraphForIndices(indices);
    }
    if (typeof ctx.markAffinityGeometryDirty === 'function') {
      ctx.markAffinityGeometryDirty();
    }
    if (typeof ctx.isWebglPipelineActive === 'function' && ctx.isWebglPipelineActive()) {
      if (typeof ctx.markMaskIndicesDirty === 'function') {
        ctx.markMaskIndicesDirty(indices);
      }
    } else if (typeof ctx.redrawMaskCanvas === 'function') {
      ctx.redrawMaskCanvas();
    }
    if (typeof ctx.draw === 'function') {
      ctx.draw();
    }
    if (typeof ctx.scheduleStateSave === 'function') {
      ctx.scheduleStateSave();
    }
    const pendingPayload = typeof ctx.getPendingSegmentationPayload === 'function'
      ? ctx.getPendingSegmentationPayload()
      : null;
    if (pendingPayload) {
      if (typeof ctx.setPendingSegmentationPayload === 'function') {
        ctx.setPendingSegmentationPayload(null);
      }
      if (typeof ctx.applySegmentationMask === 'function') {
        ctx.applySegmentationMask(pendingPayload);
      }
      if (typeof ctx.setPendingMaskRebuild === 'function') {
        ctx.setPendingMaskRebuild(false);
      }
    } else {
      const pendingMask = typeof ctx.getPendingMaskRebuild === 'function'
        ? ctx.getPendingMaskRebuild()
        : false;
      const timer = typeof ctx.getSegmentationTimer === 'function'
        ? ctx.getSegmentationTimer()
        : null;
      const canRebuild = typeof ctx.canRebuildMask === 'function'
        ? ctx.canRebuildMask()
        : false;
      if (pendingMask && timer === null && canRebuild) {
        if (typeof ctx.triggerMaskRebuild === 'function') {
          ctx.triggerMaskRebuild();
        }
      }
    }
  }

  function internalPaintStroke(point) {
    const ctx = ensureCtx();
    const maskValues = ctx.maskValues;
    const viewState = ctx.viewState || {};
    if (!maskValues || !Number.isFinite(point.x) || !Number.isFinite(point.y)) {
      return null;
    }
    const start = state.lastPaintPoint
      ? { x: state.lastPaintPoint.x, y: state.lastPaintPoint.y }
      : { x: point.x, y: point.y };
    const dx = point.x - start.x;
    const dy = point.y - start.y;
    const dist = Math.hypot(dx, dy);
    const scale = Number.isFinite(viewState.scale) && viewState.scale > 0 ? viewState.scale : 1;
    const spacing = Math.max(0.15, 0.5 / Math.max(scale, 0.0001));
    const steps = Math.max(1, Math.ceil(dist / spacing));
    const local = new Set();
    const collect = ctx.collectBrushIndices
      || (brushApi && typeof brushApi.collectBrushIndices === 'function'
        ? brushApi.collectBrushIndices
        : null);
    if (!collect) {
      state.lastPaintPoint = { x: point.x, y: point.y };
      return state.lastPaintPoint;
    }
    for (let i = 0; i <= steps; i += 1) {
      const t = steps === 0 ? 1 : i / steps;
      const px = start.x + dx * t;
      const py = start.y + dy * t;
      collect(local, px, py);
    }
    const paintLabel = ctx.getCurrentLabel ? ctx.getCurrentLabel() : 0;
    try {
      if (ctx.hasNColor && typeof ctx.isNColorActive === 'function' && ctx.isNColorActive()) {
        global.__lastPaintRawLabel = paintLabel | 0;
        global.__lastPaintGroupId = paintLabel | 0;
      } else {
        global.__lastPaintRawLabel = paintLabel | 0;
        global.__lastPaintGroupId = 0;
      }
    } catch (err) {
      /* ignore */
    }
    let maskHasNonZero = ctx.getMaskHasNonZero ? ctx.getMaskHasNonZero() : false;
    let changed = false;
    const changedIndices = [];
    local.forEach((idx) => {
      if (idx < 0 || idx >= maskValues.length) {
        return;
      }
      if (!state.strokeChanges) {
        state.strokeChanges = new Map();
      }
      if (!state.strokeChanges.has(idx)) {
        state.strokeChanges.set(idx, maskValues[idx] | 0);
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
      if (state.isPainting) {
        if (!state.pendingAffinitySet) {
          state.pendingAffinitySet = new Set();
        }
        for (let i = 0; i < changedIndices.length; i += 1) {
          state.pendingAffinitySet.add(changedIndices[i]);
        }
      } else if (changedIndices.length && typeof ctx.updateAffinityGraphForIndices === 'function') {
        ctx.updateAffinityGraphForIndices(changedIndices);
      }
      if (typeof ctx.setMaskHasNonZero === 'function') {
        ctx.setMaskHasNonZero(maskHasNonZero);
      }
      if (changedIndices.length) {
        if (typeof ctx.isWebglPipelineActive === 'function' && ctx.isWebglPipelineActive()) {
          if (typeof ctx.markMaskIndicesDirty === 'function') {
            ctx.markMaskIndicesDirty(changedIndices);
          }
        } else if (typeof ctx.markMaskIndicesDirty === 'function') {
          ctx.markMaskIndicesDirty(changedIndices);
        }
      }
      if (typeof ctx.clearColorCaches === 'function') {
        ctx.clearColorCaches();
      }
      if (typeof ctx.markNeedsMaskRedraw === 'function') {
        ctx.markNeedsMaskRedraw();
      }
      if (typeof ctx.requestPaintFrame === 'function') {
        ctx.requestPaintFrame();
      }
      if (typeof ctx.scheduleStateSave === 'function') {
        ctx.scheduleStateSave();
      }
    }
    state.lastPaintPoint = { x: point.x, y: point.y };
    return state.lastPaintPoint;
  }

  function ensureFloodBuffers() {
    const ctx = ensureCtx();
    const maskValues = ctx.maskValues;
    if (!maskValues) {
      return;
    }
    const size = maskValues.length | 0;
    if (!state.floodVisited || state.floodVisited.length !== size) {
      state.floodVisited = new Uint32Array(size);
      state.floodVisitStamp = 1;
    } else if (state.floodVisitStamp >= 0xffffffff) {
      state.floodVisited.fill(0);
      state.floodVisitStamp = 1;
    }
    if (!state.floodStack || state.floodStack.length !== size) {
      state.floodStack = new Uint32Array(size);
    }
    if (!state.floodOutput || state.floodOutput.length !== size) {
      state.floodOutput = new Uint32Array(size);
    }
  }

  function floodFill(point) {
    const ctx = ensureCtx();
    const maskValues = ctx.maskValues;
    if (!maskValues || !point) {
      return;
    }
    const dims = ctx.getImageDimensions ? ctx.getImageDimensions() : { width: 0, height: 0 };
    const width = dims.width | 0;
    const height = dims.height | 0;
    const sx = Math.round(point.x);
    const sy = Math.round(point.y);
    if (sx < 0 || sy < 0 || sx >= width || sy >= height) {
      return;
    }
    const startIdx = sy * width + sx;
    const targetLabel = maskValues[startIdx] | 0;
    const paintLabel = ctx.getCurrentLabel ? ctx.getCurrentLabel() : 0;
    if (targetLabel === paintLabel) {
      return;
    }

    let maskHasNonZero = ctx.getMaskHasNonZero ? ctx.getMaskHasNonZero() : false;
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
      if (typeof ctx.setMaskHasNonZero === 'function') {
        ctx.setMaskHasNonZero(true);
      }
      if (typeof ctx.pushHistory === 'function') {
        ctx.pushHistory(idxArr, before, after);
      }
      if (typeof ctx.rebuildLocalAffinityGraph === 'function') {
        ctx.rebuildLocalAffinityGraph();
      }
      if (typeof ctx.markAffinityGeometryDirty === 'function') {
        ctx.markAffinityGeometryDirty();
      }
      if (typeof ctx.markMaskTextureFullDirty === 'function') {
        ctx.markMaskTextureFullDirty();
      }
      if (typeof ctx.markOutlineTextureFullDirty === 'function') {
        ctx.markOutlineTextureFullDirty();
      }
      if (typeof ctx.clearColorCaches === 'function') {
        ctx.clearColorCaches();
      }
      if (typeof ctx.markNeedsMaskRedraw === 'function') {
        ctx.markNeedsMaskRedraw();
      }
      if (typeof ctx.applyMaskRedrawImmediate === 'function') {
        ctx.applyMaskRedrawImmediate();
      }
      if (typeof ctx.draw === 'function') {
        ctx.draw();
      }
      if (typeof ctx.scheduleStateSave === 'function') {
        ctx.scheduleStateSave();
      }
      return;
    }

    ensureFloodBuffers();
    const totalPixels = maskValues.length;
    const stack = state.floodStack;
    const visited = state.floodVisited;
    const output = state.floodOutput;
    let top = 0;
    let count = 0;
    let stamp = state.floodVisitStamp++;
    if (stamp >= 0xffffffff) {
      visited.fill(0);
      state.floodVisitStamp = 1;
      stamp = state.floodVisitStamp++;
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
    const after = new Uint32Array(count);
    if (targetLabel !== 0) {
      before.fill(targetLabel);
    }
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
    if (typeof ctx.setMaskHasNonZero === 'function') {
      ctx.setMaskHasNonZero(maskHasNonZero);
    }
    if (typeof ctx.pushHistory === 'function') {
      ctx.pushHistory(idxArr, before, after);
    }
    const thresholdRatio = typeof ctx.floodRebuildThreshold === 'number'
      ? Math.min(1, Math.max(0, ctx.floodRebuildThreshold))
      : 0.35;
    const largeFill = fillsAll || count >= Math.ceil(maskValues.length * thresholdRatio);
    if (largeFill) {
      if (typeof ctx.rebuildLocalAffinityGraph === 'function') {
        ctx.rebuildLocalAffinityGraph();
      }
      if (typeof ctx.markAffinityGeometryDirty === 'function') {
        ctx.markAffinityGeometryDirty();
      }
    } else if (typeof ctx.updateAffinityGraphForIndices === 'function') {
      ctx.updateAffinityGraphForIndices(idxArr);
    }
    if (fillsAll) {
      if (typeof ctx.markMaskTextureFullDirty === 'function') {
        ctx.markMaskTextureFullDirty();
      }
      if (typeof ctx.markOutlineTextureFullDirty === 'function') {
        ctx.markOutlineTextureFullDirty();
      }
    } else if (typeof ctx.markMaskIndicesDirty === 'function') {
      ctx.markMaskIndicesDirty(idxArr);
    }
    if (typeof ctx.clearColorCaches === 'function') {
      ctx.clearColorCaches();
    }
    if (typeof ctx.markNeedsMaskRedraw === 'function') {
      ctx.markNeedsMaskRedraw();
    }
    if (typeof ctx.applyMaskRedrawImmediate === 'function') {
      ctx.applyMaskRedrawImmediate();
    }
    if (typeof ctx.draw === 'function') {
      ctx.draw();
    }
    if (typeof ctx.scheduleStateSave === 'function') {
      ctx.scheduleStateSave();
    }
  }

  function pickColor(point) {
    const ctx = ensureCtx();
    const dims = ctx.getImageDimensions ? ctx.getImageDimensions() : { width: 0, height: 0 };
    const maskValues = ctx.maskValues;
    if (!maskValues || !point) {
      return;
    }
    const x = Math.round(point.x);
    const y = Math.round(point.y);
    if (x < 0 || y < 0 || x >= dims.width || y >= dims.height) {
      return;
    }
    const idx = y * dims.width + x;
    const label = maskValues[idx] | 0;
    if (typeof ctx.setCurrentLabel === 'function') {
      ctx.setCurrentLabel(label);
    }
    if (typeof ctx.log === 'function') {
      ctx.log('picker set ' + label);
    }
  }

  function labelAtPoint(point) {
    const ctx = ensureCtx();
    const dims = ctx.getImageDimensions ? ctx.getImageDimensions() : { width: 0, height: 0 };
    const maskValues = ctx.maskValues;
    if (!maskValues || !point) {
      return 0;
    }
    const x = Math.round(point.x);
    const y = Math.round(point.y);
    if (x < 0 || y < 0 || x >= dims.width || y >= dims.height) {
      return 0;
    }
    return maskValues[y * dims.width + x] | 0;
  }

  const api = global.OmniPainting || {};
  Object.assign(api, {
    init,
    beginStroke,
    queuePaintPoint,
    processPaintQueue,
    finalizeStroke,
    cancelStroke,
    flushPendingAffinityUpdates,
    floodFill,
    pickColor,
    labelAtPoint,
  });
  global.OmniPainting = api;
})(typeof window !== 'undefined' ? window : globalThis);
