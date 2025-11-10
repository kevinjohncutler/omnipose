(function initOmniPainting(global) {
  'use strict';

  const state = {
    ctx: null,
    strokeChanges: null,
    paintQueue: [],
    pendingAffinitySet: null,
    lastPaintPoint: null,
    isPainting: false,
    pendingMaskFlush: false,
    componentTracker: null,
    originalOptions: null,
    maskPipeline: null,
    maskPipelineEnabled: false,
    debugGridApplied: false,
    debugGridPending: false,
    finalizeCallCount: 0,
  };

  function isGridLoggingEnabled() {
    if (typeof globalThis !== 'object' || !globalThis.__OMNI_DEBUG__) {
      return false;
    }
    return Boolean(globalThis.__OMNI_DEBUG__.gridLogs);
  }

  function logDebugGrid(ctx, message, payload) {
    if (!isGridLoggingEnabled()) {
      return;
    }
    if (ctx && typeof ctx.log === 'function') {
      ctx.log(message, payload);
    } else if (typeof console !== 'undefined' && typeof console.debug === 'function') {
      console.debug(message, payload);
    } else if (typeof console !== 'undefined' && typeof console.log === 'function') {
      console.log(message, payload);
    }
  }

  const ENABLE_DEBUG_GRID_MASK = (() => {
    if (typeof globalThis === 'undefined') {
      return true;
    }
    if (Object.prototype.hasOwnProperty.call(globalThis, '__OMNI_FORCE_GRID_MASK__')) {
      return Boolean(globalThis.__OMNI_FORCE_GRID_MASK__);
    }
    globalThis.__OMNI_FORCE_GRID_MASK__ = true;
    return true;
  })();

  state.debugGridPending = ENABLE_DEBUG_GRID_MASK;

  const wasmFillBase64 = 'AGFzbQEAAAABCAFgBH9/f38AAwIBAAUDAQABBxkCBm1lbW9yeQIADGZpbGxfaW5kaWNlcwAACjcBNQEDfwNAIAQgAk8NASABIARBAnRqKAIAIQUgACAFQQJ0aiEGIAYgAzYCACAEQQFqIQQMAAsL';

  function decodeBase64ToUint8Array(str) {
    if (typeof atob === 'function') {
      const binary = atob(str);
      const len = binary.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i += 1) {
        bytes[i] = binary.charCodeAt(i);
      }
      return bytes;
    }
    if (typeof Buffer === 'function') {
      return Uint8Array.from(Buffer.from(str, 'base64'));
    }
    if (typeof globalThis !== 'undefined' && typeof globalThis.Buffer === 'function') {
      return Uint8Array.from(globalThis.Buffer.from(str, 'base64'));
    }
    throw new Error('Base64 decoding not supported in this environment');
  }

  function createWasmtimeFillState() {
    const binary = decodeBase64ToUint8Array(wasmFillBase64);
    const module = new WebAssembly.Module(binary);
    const instance = new WebAssembly.Instance(module, {});
    const memory = instance.exports.memory;
    const fillFn = instance.exports.fill_indices;
    const state = {
      instance,
      memory,
      fillFn,
      maskPtr: 0,
      maskBytes: 0,
      maskLength: 0,
      maskArray: null,
      heapOffset: 0,
      heapLimit: 0,
      scratchPtr: 0,
      scratchLength: 0,
    };
    return state;
  }

  const fillWasmState = createWasmtimeFillState();

  function updateMaskReference(array, forceNotify = false) {
    if (!array) {
      return;
    }
    const ctx = state.ctx;
    const prev = ctx ? ctx.maskValues : null;
    if (ctx) {
      ctx.maskValues = array;
    }
    const bufferChanged = prev !== array;
    if (ENABLE_DEBUG_GRID_MASK && bufferChanged) {
      state.debugGridApplied = false;
      state.debugGridPending = true;
    }
    if (state.originalOptions) {
      state.originalOptions.maskValues = array;
    }
    const shouldNotify = forceNotify || prev !== array;
    if (!shouldNotify) {
      return;
    }
    const notify = (ctx && typeof ctx.onMaskBufferReplaced === 'function')
      ? ctx.onMaskBufferReplaced
      : (state.originalOptions && typeof state.originalOptions.onMaskBufferReplaced === 'function'
        ? state.originalOptions.onMaskBufferReplaced
        : null);
    if (typeof notify === 'function') {
      try {
        notify(array);
      } catch (err) {
        /* ignore */
      }
    }
    if (state.maskPipeline && typeof state.maskPipeline.onMaskBufferReplaced === 'function') {
      try {
        state.maskPipeline.onMaskBufferReplaced();
      } catch (err) {
        /* ignore */
      }
    }
    if (ENABLE_DEBUG_GRID_MASK && (bufferChanged || forceNotify)) {
      applyDebugGridIfNeeded(false);
    }
  }

  function ensureWasmMemoryBytes(targetBytes) {
    const pageSize = 65536;
    const memory = fillWasmState.memory;
    let grew = false;
    while (targetBytes > memory.buffer.byteLength) {
      const currentBytes = memory.buffer.byteLength;
      const currentPages = currentBytes / pageSize;
      const requiredPages = Math.ceil(targetBytes / pageSize);
      const growBy = Math.max(0, requiredPages - currentPages);
      if (growBy <= 0) {
        break;
      }
      memory.grow(growBy);
      grew = true;
    }
    fillWasmState.heapLimit = memory.buffer.byteLength;
    if (fillWasmState.maskLength > 0) {
      fillWasmState.maskArray = new Uint32Array(memory.buffer, fillWasmState.maskPtr, fillWasmState.maskLength);
      updateMaskReference(fillWasmState.maskArray);
    }
    if (grew) {
      refreshComponentViews();
    }
  }

  function alignTo(ptr, alignment) {
    const mask = alignment - 1;
    return (ptr + mask) & ~mask;
  }

  function ensureMaskBuffer(totalPixels) {
    const bytesPerPixel = 4;
    const maskBytes = totalPixels * bytesPerPixel;
    if (fillWasmState.maskArray && fillWasmState.maskLength === totalPixels) {
      return fillWasmState.maskArray;
    }
    const reserveFactor = 6;
    const reserveBytes = maskBytes * reserveFactor;
    const totalBytes = alignTo(maskBytes + reserveBytes, 16);
    ensureWasmMemoryBytes(totalBytes);
    let offset = 0;
    const maskPtr = alignTo(offset, 4);
    offset = maskPtr + maskBytes;
    fillWasmState.maskPtr = maskPtr;
    fillWasmState.maskBytes = maskBytes;
    fillWasmState.maskLength = totalPixels;
    fillWasmState.maskArray = new Uint32Array(fillWasmState.memory.buffer, maskPtr, totalPixels);
    fillWasmState.heapOffset = alignTo(offset, 8);
    fillWasmState.scratchPtr = fillWasmState.heapOffset;
    fillWasmState.scratchLength = totalPixels;
    const scratchBytes = totalPixels * 4;
    fillWasmState.heapOffset = fillWasmState.scratchPtr + scratchBytes;
    fillWasmState.heapOffset = alignTo(fillWasmState.heapOffset, 16);
    fillWasmState.heapBase = fillWasmState.heapOffset;
    fillWasmState.heapLimit = fillWasmState.memory.buffer.byteLength;
    fillWasmState.componentPool = [];
    updateMaskReference(fillWasmState.maskArray, true);
    if (state.maskPipeline && typeof state.maskPipeline.onMaskBufferReplaced === 'function') {
      state.maskPipeline.onMaskBufferReplaced();
    }
    return fillWasmState.maskArray;
  }

  function ensureScratchCapacity(length) {
    if (length <= fillWasmState.scratchLength) {
      return new Uint32Array(fillWasmState.memory.buffer, fillWasmState.scratchPtr, fillWasmState.scratchLength);
    }
    fillWasmState.scratchLength = length;
    const requiredBytes = fillWasmState.scratchPtr + length * 4;
    ensureWasmMemoryBytes(requiredBytes);
    fillWasmState.heapOffset = alignTo(requiredBytes, 16);
    if (fillWasmState.heapOffset < fillWasmState.heapBase) {
      fillWasmState.heapBase = fillWasmState.heapOffset;
    }
    return new Uint32Array(fillWasmState.memory.buffer, fillWasmState.scratchPtr, fillWasmState.scratchLength);
  }

  function tagWasmArray(array, ptr, capacity) {
    Object.defineProperty(array, '__wasmPtr', { value: ptr, configurable: true, enumerable: false });
    Object.defineProperty(array, '__capacity', { value: capacity, configurable: true, enumerable: false });
    Object.defineProperty(array, '__length', { value: array.length | 0, configurable: true, enumerable: false, writable: true });
    return array;
  }

  function refreshComponentViews() {
    if (!state.componentTracker || !state.componentTracker.components) {
      return;
    }
    const memory = fillWasmState.memory ? fillWasmState.memory.buffer : null;
    if (!memory) {
      return;
    }
    state.componentTracker.components.forEach((entry) => {
      if (!entry || !entry.indices || typeof entry.indices.__wasmPtr !== 'number') {
        return;
      }
      const ptr = entry.indices.__wasmPtr | 0;
      const len = entry.indices.length | 0;
      const capacity = entry.indices.__capacity ? (entry.indices.__capacity | 0) : len;
      const view = new Uint32Array(memory, ptr, len);
      tagWasmArray(view, ptr, capacity);
      entry.indices = view;
    });
  }

  function seedDebugGridMask(ctx, width, height) {
    if (!ctx || !ctx.maskValues || width <= 0 || height <= 0) {
      return;
    }
    if (typeof globalThis !== 'undefined') {
      globalThis.__OMNI_DEBUG__ = globalThis.__OMNI_DEBUG__ || {};
      globalThis.__OMNI_DEBUG__.gridSeedAttempts = (globalThis.__OMNI_DEBUG__.gridSeedAttempts || 0) + 1;
    }
    const mask = ctx.maskValues;
    const total = mask.length | 0;
    if (total !== (width * height)) {
      return;
    }
    const targetDivisionsX = 7;
    const targetDivisionsY = 7;
    const lineThickness = Math.max(1, Math.round(Math.min(width, height) / 120));
    const columnStride = Math.max(lineThickness, Math.floor(width / targetDivisionsX));
    const rowStride = Math.max(lineThickness, Math.floor(height / targetDivisionsY));
    const indexSet = new Set();
    const markIndex = (idx) => {
      if (idx < 0 || idx >= total) {
        return;
      }
      if ((mask[idx] | 0) === 1) {
        return;
      }
      indexSet.add(idx);
    };
    const paintVerticalBand = (x) => {
      for (let t = 0; t < lineThickness && (x + t) < width; t += 1) {
        const col = x + t;
        for (let y = 0; y < height; y += 1) {
          markIndex(y * width + col);
        }
      }
    };
    const paintHorizontalBand = (y) => {
      for (let t = 0; t < lineThickness && (y + t) < height; t += 1) {
        const rowIndex = (y + t) * width;
        for (let x = 0; x < width; x += 1) {
          markIndex(rowIndex + x);
        }
      }
    };
    paintVerticalBand(0);
    paintVerticalBand(width - lineThickness);
    paintHorizontalBand(0);
    paintHorizontalBand(height - lineThickness);
    for (let x = columnStride; x < width; x += columnStride) {
      paintVerticalBand(x);
    }
    for (let y = rowStride; y < height; y += rowStride) {
      paintHorizontalBand(y);
    }
    if (!indexSet.size) {
      return;
    }
    const count = indexSet.size | 0;
    const indices = new Uint32Array(count);
    const before = new Uint32Array(count);
    const after = new Uint32Array(count);
    let cursor = 0;
    indexSet.forEach((idx) => {
      const prev = mask[idx] | 0;
      indices[cursor] = idx | 0;
      before[cursor] = prev;
      after[cursor] = 1;
      mask[idx] = 1;
      cursor += 1;
    });
    if (typeof ctx.setMaskHasNonZero === 'function') {
      ctx.setMaskHasNonZero(true);
    }
    if (typeof ctx.markMaskTextureFullDirty === 'function') {
      ctx.markMaskTextureFullDirty();
    }
    if (typeof ctx.markOutlineTextureFullDirty === 'function') {
      ctx.markOutlineTextureFullDirty();
    }
    if (typeof ctx.markNeedsMaskRedraw === 'function') {
      ctx.markNeedsMaskRedraw();
    }
    if (typeof ctx.applyMaskRedrawImmediate === 'function') {
      try {
        ctx.applyMaskRedrawImmediate();
      } catch (_) {
        /* ignore */
      }
    } else if (typeof ctx.requestPaintFrame === 'function') {
      ctx.requestPaintFrame();
    }
    const finalizeSeed = () => {
      if (!ctx || !ctx.maskValues) {
        return;
      }
      try {
        fillMaskWithIndices(indices, count, 1);
      } catch (err) {
        if (typeof ctx.log === 'function') {
          ctx.log('debug grid fillMaskWithIndices failed: ' + err);
        }
      }
      if (state.componentTracker) {
        try {
          state.componentTracker.rebuild(ctx);
        } catch (err) {
          if (typeof ctx.log === 'function') {
            ctx.log('debug grid tracker rebuild failed: ' + err);
          }
        }
      }
      finalizeComponentFill(ctx, indices, 1, before, after, mask.length, {
        seedKind: 'debug-grid',
        paintLabel: 1,
        count,
        imageWidth: width,
        imageHeight: height,
      }, null);
        logDebugGrid(ctx, '[debug-grid] finalizeSeed applied', { count });
      if (typeof ctx.applyMaskRedrawImmediate === 'function') {
        try {
          ctx.applyMaskRedrawImmediate();
        } catch (err) {
          if (typeof ctx.log === 'function') {
            ctx.log('debug grid applyMaskRedrawImmediate failed: ' + err);
          }
        }
      } else if (typeof ctx.requestPaintFrame === 'function') {
        ctx.requestPaintFrame();
      }
    };
    if (typeof setTimeout === 'function') {
      setTimeout(finalizeSeed, 0);
    } else if (typeof requestAnimationFrame === 'function') {
      requestAnimationFrame(() => finalizeSeed());
    } else {
      finalizeSeed();
    }
    state.debugGridApplied = true;
    state.debugGridPending = false;
    return count;
  }

  function applyDebugGridIfNeeded(force = false) {
    if (!ENABLE_DEBUG_GRID_MASK) {
      return false;
    }
    if (!state.ctx || !state.ctx.maskValues || !state.ctx.getImageDimensions) {
      return false;
    }
    if (!force && !state.debugGridPending) {
      return false;
    }
    const dims = state.ctx.getImageDimensions();
    const width = dims && Number.isFinite(dims.width) ? (dims.width | 0) : 0;
    const height = dims && Number.isFinite(dims.height) ? (dims.height | 0) : 0;
    if (width <= 0 || height <= 0) {
      return false;
    }
    const mask = state.ctx.maskValues;
    if (!force) {
      let hasNonZero = false;
      for (let i = 0; i < mask.length; i += 1) {
        if ((mask[i] | 0) > 0) {
          hasNonZero = true;
          break;
        }
      }
      if (hasNonZero) {
        logDebugGrid(state.ctx, '[debug-grid] mask already non-zero; skipping seed');
        state.debugGridPending = false;
        state.debugGridApplied = false;
        return false;
      }
    }
    const appliedCount = seedDebugGridMask(state.ctx, width, height) | 0;
    state.debugGridApplied = appliedCount > 0;
    state.debugGridPending = false;
    if (appliedCount > 0) {
      logDebugGrid(state.ctx, '[debug-grid] seeded grid', { width, height, count: appliedCount });
    }
    scheduleGridPersistenceCheck();
    return true;
  }

  function scheduleGridPersistenceCheck() {
    if (!ENABLE_DEBUG_GRID_MASK) {
      return;
    }
    if (!state.ctx || !state.ctx.maskValues) {
      return;
    }
    let remainingChecks = 10;
    const check = () => {
      if (!state.ctx || !state.ctx.maskValues) {
        return;
      }
      const mask = state.ctx.maskValues;
      const step = Math.max(1, Math.floor(mask.length / 4096));
      let hasNonZero = false;
      for (let i = 0; i < mask.length; i += step) {
        if ((mask[i] | 0) > 0) {
          hasNonZero = true;
          break;
        }
      }
      if (!hasNonZero && state.debugGridApplied) {
        state.debugGridApplied = false;
        state.debugGridPending = true;
        if (typeof state.ctx.log === 'function') {
        logDebugGrid(state.ctx, '[debug-grid] detected cleared mask; reapplying seed');
        }
        applyDebugGridIfNeeded(true);
        return;
      }
      if (remainingChecks > 0) {
        remainingChecks -= 1;
        setTimeout(check, 500);
      }
    };
    setTimeout(check, 500);
  }

  function maskPipelineActive() {
    return Boolean(state.maskPipelineEnabled && state.maskPipeline);
  }

  function ensureMaskPipeline(ctx) {
    if (!state.maskPipelineEnabled) {
      return;
    }
    if (!state.maskPipeline) {
      const factory = global.OmniMaskPipeline && typeof global.OmniMaskPipeline.createMaskPipeline === 'function'
        ? global.OmniMaskPipeline.createMaskPipeline
        : null;
      if (factory) {
        state.maskPipeline = factory({
          logger: ctx && typeof ctx.log === 'function' ? ctx.log : null,
          onAnomaly: ctx && typeof ctx.log === 'function'
            ? (detail) => ctx.log('mask pipeline divergence', detail)
            : null,
        });
      }
    }
    if (state.maskPipeline && ctx) {
      state.maskPipeline.attachCtx(ctx);
    }
  }

  function recordMaskPipelineMutation(kind, indices, beforeLabels, afterLabels, meta) {
    if (!maskPipelineActive()) {
      return;
    }
    try {
      state.maskPipeline.applyDiff(kind, indices, beforeLabels, afterLabels, meta || null);
    } catch (err) {
      if (state.ctx && typeof state.ctx.log === 'function') {
        state.ctx.log('mask pipeline mutation error', err);
      }
    }
  }

  function recordMaskPipelineNoop(meta) {
    if (!maskPipelineActive()) {
      return;
    }
    try {
      state.maskPipeline.recordNoopFill(meta || null);
    } catch (err) {
      if (state.ctx && typeof state.ctx.log === 'function') {
        state.ctx.log('mask pipeline noop error', err);
      }
    }
  }

  function ensureComponentArrayView(array) {
    if (!array || typeof array.__wasmPtr !== 'number') {
      return array;
    }
    const buffer = fillWasmState.memory ? fillWasmState.memory.buffer : null;
    if (!buffer) {
      return array;
    }
    if (array.buffer === buffer && array.byteLength !== 0) {
      return array;
    }
    const ptr = array.__wasmPtr | 0;
    const capacity = array.__capacity ? (array.__capacity | 0) : (array.length | 0);
    const recorded = array.__length ? (array.__length | 0) : (array.length | 0);
    const len = array.length && array.length > 0 ? (array.length | 0) : recorded || capacity;
    if (!len || len < 0) {
      return array;
    }
    const refreshed = new Uint32Array(buffer, ptr, len);
    const tagged = tagWasmArray(refreshed, ptr, capacity);
    tagged.__length = len;
    return tagged;
  }

  function resetComponentStorage() {
    fillWasmState.componentPool = [];
    if (typeof fillWasmState.heapBase === 'number') {
      fillWasmState.heapOffset = fillWasmState.heapBase;
    }
  }

  function borrowComponentBuffer(length) {
    const pool = fillWasmState.componentPool || (fillWasmState.componentPool = []);
    for (let i = 0; i < pool.length; i += 1) {
      const entry = pool[i];
      if ((entry.capacity | 0) >= (length | 0)) {
        pool.splice(i, 1);
        const arr = new Uint32Array(fillWasmState.memory.buffer, entry.ptr, length);
        return tagWasmArray(arr, entry.ptr, entry.capacity);
      }
    }
    const capacity = Math.max(1, length | 0);
    const ptr = alignTo(fillWasmState.heapOffset, 16);
    const requiredBytes = ptr + capacity * 4;
    if (requiredBytes > fillWasmState.heapLimit) {
      ensureWasmMemoryBytes(requiredBytes);
    }
    fillWasmState.heapOffset = requiredBytes;
    if (fillWasmState.heapOffset > fillWasmState.heapLimit) {
      fillWasmState.heapLimit = fillWasmState.heapOffset;
    }
    const arr = new Uint32Array(fillWasmState.memory.buffer, ptr, length);
    return tagWasmArray(arr, ptr, capacity);
  }

  function releaseComponentArray(array) {
    if (!array || typeof array.__wasmPtr !== 'number') {
      return;
    }
    const ptr = array.__wasmPtr | 0;
    const capacity = array.__capacity ? (array.__capacity | 0) : (array.length | 0);
    if (!fillWasmState.componentPool) {
      fillWasmState.componentPool = [];
    }
    fillWasmState.componentPool.push({ ptr, capacity });
  }

  function fillMaskWithIndices(indicesArray, count, paintLabel) {
    if (!indicesArray || typeof indicesArray.__wasmPtr !== 'number') {
      const scratch = ensureScratchCapacity(count);
      scratch.set(indicesArray.subarray(0, count));
      fillWasmState.fillFn(fillWasmState.maskPtr, fillWasmState.scratchPtr, count, paintLabel);
      return;
    }
    fillWasmState.fillFn(fillWasmState.maskPtr, indicesArray.__wasmPtr, count, paintLabel);
  }
  const brushApi = global.OmniBrush || {};
  const BACKGROUND_STEPS = [
    [0, -1],
    [0, 1],
    [-1, 0],
    [1, 0],
  ];
  const DEFAULT_FOREGROUND_STEPS = [
    [-1, -1],
    [-1, 0],
    [-1, 1],
    [0, -1],
    [0, 1],
    [1, -1],
    [1, 0],
    [1, 1],
  ];

  function normalizeSteps(steps) {
    if (!Array.isArray(steps) || !steps.length) {
      return DEFAULT_FOREGROUND_STEPS.map((pair) => [pair[0], pair[1]]);
    }
    const normalized = [];
    for (let i = 0; i < steps.length; i += 1) {
      const entry = steps[i];
      if (!entry || entry.length < 2) {
        continue;
      }
      const dy = entry[0] | 0;
      const dx = entry[1] | 0;
      normalized.push([dy, dx]);
    }
    if (!normalized.length) {
      return DEFAULT_FOREGROUND_STEPS.map((pair) => [pair[0], pair[1]]);
    }
    return normalized;
  }

  function createComponentTracker() {
    const tracker = {
      ids: null,
      components: new Map(),
      labelIndex: new Map(),
      pendingMask: null,
      pendingList: [],
      pendingHead: 0,
      visit: null,
      visitStamp: 1,
      queue: null,
      collect: null,
      nextComponentId: 1,
      size: 0,
      width: 0,
      height: 0,
      foregroundSteps: normalizeSteps(DEFAULT_FOREGROUND_STEPS),
      registerComponent(id, label, indices, entry) {
        const ensured = ensureComponentArrayView(indices);
        const target = entry || { id, label, indices: ensured };
        target.id = id;
        target.label = label;
        target.indices = ensured;
        target.size = ensured.length | 0;
        this.components.set(id, target);
        let bucket = this.labelIndex.get(label);
        if (!bucket) {
          bucket = new Set();
          this.labelIndex.set(label, bucket);
        }
        bucket.add(id);
        return target;
      },
      unregisterComponent(entry) {
        if (!entry || !entry.id) {
          return;
        }
        this.components.delete(entry.id);
        const bucket = this.labelIndex.get(entry.label);
        if (bucket) {
          bucket.delete(entry.id);
          if (bucket.size === 0) {
            this.labelIndex.delete(entry.label);
          }
        }
        releaseComponentArray(entry.indices);
        entry.indices = null;
      },
      labelHasOtherComponents(label, excludeId) {
        const bucket = this.labelIndex.get(label);
        if (!bucket || bucket.size === 0) {
          return false;
        }
        if (!excludeId) {
          return bucket.size > 0;
        }
        if (bucket.size > 1) {
          return true;
        }
        return !bucket.has(excludeId);
      },
      ensure(ctx) {
        if (!ctx || !ctx.maskValues) {
          this.clear();
          return false;
        }
        const mask = ctx.maskValues;
        const total = mask.length | 0;
        if (!Number.isFinite(total) || total <= 0) {
          this.clear();
          return false;
        }
        const dims = ctx.getImageDimensions ? ctx.getImageDimensions() : { width: 0, height: 0 };
        const width = dims.width | 0;
        const height = dims.height | 0;
        if (width <= 0 || height <= 0 || width * height !== total) {
          this.clear();
          return false;
        }
        if (this.ids && this.size === total && this.width === width && this.height === height) {
          return true;
        }
        this.size = total;
        this.width = width;
        this.height = height;
        this.ids = new Uint32Array(total);
        this.pendingMask = new Uint8Array(total);
        this.pendingList = [];
        this.pendingHead = 0;
        this.queue = new Uint32Array(total);
        this.collect = new Uint32Array(total);
        this.visit = new Uint32Array(total);
        this.visitStamp = 1;
        this.nextComponentId = 1;
        this.components.clear();
        for (let i = 0; i < total; i += 1) {
          this.pendingMask[i] = 0;
          this.ids[i] = 0;
          this.pendingList.push(i);
        }
        this.processPending(ctx);
        return true;
      },
      clear() {
        this.ids = null;
        this.pendingMask = null;
        this.pendingList = [];
        this.pendingHead = 0;
        this.queue = null;
        this.collect = null;
        this.visit = null;
        this.visitStamp = 1;
        this.nextComponentId = 1;
        this.size = 0;
        this.width = 0;
        this.height = 0;
        this.components.clear();
        resetComponentStorage();
      },
      ensureSteps(ctx) {
        const source = ctx && typeof ctx.getAffinitySteps === 'function'
          ? ctx.getAffinitySteps()
          : DEFAULT_FOREGROUND_STEPS;
        this.foregroundSteps = normalizeSteps(source);
      },
      queuePendingIndex(idx) {
        if (!this.pendingMask || idx < 0 || idx >= this.size) {
          return;
        }
        if (this.pendingMask[idx]) {
          return;
        }
        this.pendingMask[idx] = 1;
        this.pendingList.push(idx);
      },
      dequeuePendingIndex() {
        if (!this.pendingList) {
          return -1;
        }
        while (this.pendingHead < this.pendingList.length) {
          const idx = this.pendingList[this.pendingHead++];
          if (idx < 0 || idx >= this.size) {
            continue;
          }
          if (!this.pendingMask[idx]) {
            continue;
          }
          this.pendingMask[idx] = 0;
          return idx;
        }
        this.pendingList.length = 0;
        this.pendingHead = 0;
        return -1;
      },
      removeComponent(id) {
        if (!id || !this.components.has(id)) {
          return;
        }
        const entry = this.components.get(id);
        const retained = entry ? entry.indices : null;
        this.unregisterComponent(entry);
        const arr = retained ? ensureComponentArrayView(retained) : null;
        if (!arr || !arr.length) {
          throw new Error('Component tracker removed entry without indices');
        }
        for (let i = 0; i < arr.length; i += 1) {
          const idx = arr[i] | 0;
          if (idx < 0 || idx >= this.size) {
            continue;
          }
          this.ids[idx] = 0;
          this.queuePendingIndex(idx);
        }
      },
      processPending(ctx) {
        if (!ctx || !ctx.maskValues || !this.ids) {
          return;
        }
        this.ensureSteps(ctx);
        const mask = ctx.maskValues;
        let idx = this.dequeuePendingIndex();
        while (idx !== -1) {
          if (this.ids[idx] === 0) {
            const label = mask[idx] | 0;
            const component = this.buildComponent(idx, label, ctx);
            if (component && component.indices.length) {
              const newId = this.nextComponentId++;
              const indices = component.indices;
              for (let i = 0; i < indices.length; i += 1) {
                const index = indices[i] | 0;
                if (index < 0 || index >= this.size) {
                  continue;
                }
                this.ids[index] = newId;
                if (this.pendingMask[index]) {
                  this.pendingMask[index] = 0;
                }
              }
              component.id = newId;
              component.label = label;
              this.registerComponent(newId, label, indices, component);
            }
          }
          idx = this.dequeuePendingIndex();
        }
      },
      buildComponent(startIdx, label, ctx) {
        if (!ctx || !ctx.maskValues || startIdx < 0 || startIdx >= this.size) {
          return null;
        }
        const mask = ctx.maskValues;
        const targetLabel = label | 0;
        const steps = targetLabel > 0 ? this.foregroundSteps : BACKGROUND_STEPS;
        const queue = this.queue;
        const collect = this.collect;
        if (!queue || !collect) {
          return null;
        }
        const visit = this.visit;
        if (!visit) {
          return null;
        }
        let stamp = this.visitStamp++;
        if (stamp >= 0xffffffff) {
          visit.fill(0);
          this.visitStamp = 2;
          stamp = 1;
        }
        let head = 0;
        let tail = 0;
        queue[tail++] = startIdx;
        visit[startIdx] = stamp;
        let count = 0;
        while (head < tail) {
          const idx = queue[head++] | 0;
          if ((mask[idx] | 0) !== targetLabel) {
            continue;
          }
          collect[count++] = idx;
          if (this.pendingMask && this.pendingMask[idx]) {
            this.pendingMask[idx] = 0;
          }
          const x = idx % this.width;
          const y = (idx / this.width) | 0;
          for (let s = 0; s < steps.length; s += 1) {
            const dy = steps[s][0] | 0;
            const dx = steps[s][1] | 0;
            const nx = x + dx;
            const ny = y + dy;
            if (nx < 0 || nx >= this.width || ny < 0 || ny >= this.height) {
              continue;
            }
            const neighbor = ny * this.width + nx;
            if (this.ids[neighbor] !== 0) {
              continue;
            }
            if ((mask[neighbor] | 0) !== targetLabel) {
              continue;
            }
            if (visit[neighbor] === stamp) {
              continue;
            }
            visit[neighbor] = stamp;
            queue[tail++] = neighbor;
          }
        }
        if (count === 0) {
          return null;
        }
        const result = borrowComponentBuffer(count);
        result.set(collect.subarray(0, count));
        return { id: 0, label: targetLabel, indices: result };
      },
      componentAt(index, ctx) {
        if (!this.ensure(ctx)) {
          return null;
        }
        this.processPending(ctx);
        if (!this.ids) {
          return null;
        }
        if (index < 0 || index >= this.size) {
          return null;
        }
        let id = this.ids[index] | 0;
        if (!id) {
          this.queuePendingIndex(index);
          this.processPending(ctx);
          id = this.ids[index] | 0;
        }
        if (!id) {
          return null;
        }
        let entry = this.components.get(id) || null;
        if (entry && entry.indices) {
          entry.indices = ensureComponentArrayView(entry.indices);
        }
        if (!entry || !entry.indices || entry.indices.length === 0) {
          throw new Error('Component tracker returned empty component');
        }
        if (entry.indices) {
          entry.indices = ensureComponentArrayView(entry.indices);
        }
        return entry;
      },
      collectNeighborComponents(index, label, targetSet, ctx) {
        if (!this.ids || !ctx || !ctx.maskValues) {
          return;
        }
        const mask = ctx.maskValues;
        const targetLabel = label | 0;
        if (targetLabel < 0) {
          return;
        }
        const steps = targetLabel > 0 ? this.foregroundSteps : BACKGROUND_STEPS;
        const x = index % this.width;
        const y = (index / this.width) | 0;
        for (let i = 0; i < steps.length; i += 1) {
          const dy = steps[i][0] | 0;
          const dx = steps[i][1] | 0;
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || nx >= this.width || ny < 0 || ny >= this.height) {
            continue;
          }
          const neighborIndex = ny * this.width + nx;
          if ((mask[neighborIndex] | 0) !== targetLabel) {
            continue;
          }
          const cid = this.ids[neighborIndex] | 0;
          if (cid) {
            targetSet.add(cid);
          }
        }
      },
      recordChanges(indices, beforeLabels, afterLabels, ctx) {
        if (!indices || !ctx || !ctx.maskValues) {
          return;
        }
        if (!this.ensure(ctx)) {
          return;
        }
        const len = indices.length | 0;
        if (len <= 0) {
          return;
        }
        const affected = new Set();
        for (let i = 0; i < len; i += 1) {
          const idx = Number(indices[i]) | 0;
          if (idx < 0 || idx >= this.size) {
            continue;
          }
          const before = beforeLabels ? (beforeLabels[i] | 0) : (this.ids[idx] ? this.components.get(this.ids[idx])?.label || 0 : 0);
          const after = afterLabels ? (afterLabels[i] | 0) : (ctx.maskValues[idx] | 0);
          if (before === after) {
            continue;
          }
          if (this.ids[idx]) {
            affected.add(this.ids[idx]);
          }
          this.ids[idx] = 0;
          this.queuePendingIndex(idx);
        }
        for (let i = 0; i < len; i += 1) {
          const idx = Number(indices[i]) | 0;
          if (idx < 0 || idx >= this.size) {
            continue;
          }
          const before = beforeLabels ? (beforeLabels[i] | 0) : 0;
          const after = afterLabels ? (afterLabels[i] | 0) : (ctx.maskValues[idx] | 0);
          if (before !== after && before > 0) {
            this.collectNeighborComponents(idx, before, affected, ctx);
          }
          if (before !== after) {
            this.collectNeighborComponents(idx, after, affected, ctx);
          }
        }
        affected.forEach((id) => {
          const existing = this.components.get(id);
          if (existing && existing.indices) {
            existing.indices = ensureComponentArrayView(existing.indices);
            existing.size = existing.indices.length | 0;
            if (!existing.indices.length) {
              throw new Error('component tracker empty component before removal');
            }
          }
          this.removeComponent(id);
        });
        this.processPending(ctx);
      },
      componentIdAt(index) {
        if (!this.ids || index < 0 || index >= this.size) {
          return 0;
        }
        return this.ids[index] | 0;
      },
      ensureComponentForIndex(index, ctx) {
        if (!this.ensure(ctx)) {
          return null;
        }
        if (index < 0 || index >= this.size) {
          return null;
        }
        let id = this.ids[index] | 0;
        if (id) {
          return this.components.get(id) || null;
        }
        this.queuePendingIndex(index);
        this.processPending(ctx);
        id = this.ids[index] | 0;
        return id ? this.components.get(id) || null : null;
      },
      mergeFilledComponent(component, neighborIds, newLabel, ctx) {
        if (!component || !this.ensure(ctx)) {
          return component;
        }
        const label = newLabel | 0;
        const mergeList = [];
        if (neighborIds && neighborIds.size) {
          neighborIds.forEach((cid) => {
            const norm = cid | 0;
            if (!norm || norm === component.id) {
              return;
            }
            const entry = this.components.get(norm);
            if (entry && entry.indices && entry.indices.length) {
              mergeList.push(entry);
            }
          });
        }
        const prevLabel = component.label;
        let prevIndices = ensureComponentArrayView(component.indices);
        if (!prevIndices || !prevIndices.length) {
          throw new Error('component tracker encountered empty component before merge id=' + component.id + ' label=' + component.label);
        }
        component.indices = prevIndices;
        let merged = prevIndices;
        let mergedLength = merged ? merged.length : 0;
        if (mergeList.length) {
          let total = mergedLength;
          for (let i = 0; i < mergeList.length; i += 1) {
            const entry = mergeList[i];
            const arr = ensureComponentArrayView(entry.indices);
            entry.indices = arr;
            entry.size = arr.length | 0;
            total += entry.size;
          }
          const next = borrowComponentBuffer(total);
          prevIndices = ensureComponentArrayView(prevIndices);
          if (!prevIndices || !prevIndices.length) {
            throw new Error('component tracker merge empty prevIndices id=' + component.id + ' label=' + component.label);
          }
          component.indices = prevIndices;
          if (mergedLength > 0 && prevIndices) {
            next.set(prevIndices.subarray(0, mergedLength), 0);
          }
          let offset = mergedLength;
          for (let i = 0; i < mergeList.length; i += 1) {
            const entry = mergeList[i];
            const arr = ensureComponentArrayView(entry.indices);
            entry.indices = arr;
            const copyLen = entry.size | 0;
            if (copyLen > 0) {
              if ((offset + copyLen) > next.length) {
                throw new Error('component tracker merge overflow id=' + component.id);
              }
              next.set(arr.subarray(0, copyLen), offset);
              offset += copyLen;
            }
          }
          merged = next;
          mergedLength = total;
        }
        let id = component.id;
        if (!id) {
          id = this.nextComponentId++;
        } else {
          this.components.delete(id);
          const prevBucket = this.labelIndex.get(prevLabel);
          if (prevBucket) {
            prevBucket.delete(id);
            if (prevBucket.size === 0) {
              this.labelIndex.delete(prevLabel);
            }
          }
        }
        for (let i = 0; i < mergedLength; i += 1) {
          const idx = merged[i] | 0;
          if (idx < 0 || idx >= this.size) {
            continue;
          }
          this.ids[idx] = id;
          if (this.pendingMask && this.pendingMask[idx]) {
            this.pendingMask[idx] = 0;
          }
        }
        if (prevIndices && merged !== prevIndices) {
          releaseComponentArray(prevIndices);
        }
        component.id = id;
        component.label = label;
        if (!merged || !merged.length) {
          throw new Error('component tracker merge created empty buffer id=' + component.id + ' label=' + label);
        }
        component.indices = merged;
        component.size = merged.length | 0;
        this.registerComponent(id, label, merged, component);
        for (let i = 0; i < mergeList.length; i += 1) {
          const entry = mergeList[i];
          if (entry && entry.id !== id) {
            this.unregisterComponent(entry);
          }
        }
        return component;
      },
      rebuild(ctx) {
        if (!this.ensure(ctx)) {
          return;
        }
        if (!ctx || !ctx.maskValues) {
          return;
        }
        const total = this.size;
        this.ids.fill(0);
        this.components.clear();
        this.labelIndex.clear();
        this.pendingList.length = 0;
        this.pendingHead = 0;
        for (let i = 0; i < total; i += 1) {
          this.pendingMask[i] = 0;
          this.pendingList.push(i);
        }
        resetComponentStorage();
        this.processPending(ctx);
      },
    };
    return tracker;
  }

  function collectComponentIndices(startIdx, targetLabel, tracker, ctx) {
    const mask = ctx && ctx.maskValues ? ctx.maskValues : null;
    if (!mask || startIdx < 0 || startIdx >= mask.length) {
      return null;
    }
    const dims = ctx.getImageDimensions ? ctx.getImageDimensions() : { width: 0, height: 0 };
    const width = dims.width | 0;
    const height = dims.height | 0;
    if (width <= 0 || height <= 0 || width * height !== mask.length) {
      return null;
    }
    const steps = (targetLabel > 0 ? tracker.foregroundSteps : BACKGROUND_STEPS) || BACKGROUND_STEPS;
    const total = mask.length | 0;
    const queue = new Uint32Array(total);
    const visited = new Uint8Array(total);
    const collected = new Uint32Array(total);
    let head = 0;
    let tail = 0;
    queue[tail++] = startIdx;
    visited[startIdx] = 1;
    let count = 0;
    while (head < tail) {
      const idx = queue[head++] | 0;
      if ((mask[idx] | 0) !== targetLabel) {
        continue;
      }
      collected[count++] = idx;
      const x = idx % width;
      const y = (idx / width) | 0;
      for (let s = 0; s < steps.length; s += 1) {
        const dy = steps[s][0] | 0;
        const dx = steps[s][1] | 0;
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
          continue;
        }
        const nidx = ny * width + nx;
        if (visited[nidx]) {
          continue;
        }
        visited[nidx] = 1;
        queue[tail++] = nidx;
      }
    }
    if (count === 0) {
      return null;
    }
    return collected.subarray(0, count);
  }

  function ensureCtx() {
    if (!state.ctx) {
      throw new Error('OmniPainting.init must be called before use');
    }
    return state.ctx;
  }

  function init(options) {
    const originalOptions = options || {};
    state.originalOptions = originalOptions;
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
      scheduleDraw: () => {},
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
      enqueueAffinityIndexBatch: null,
      onMaskBufferReplaced: null,
      floodRebuildThreshold: 0.35,
    }, options || {});
    const globalFlag = (typeof globalThis !== 'undefined' && Object.prototype.hasOwnProperty.call(globalThis, '__OMNI_MASK_PIPELINE_V2__'))
      ? Boolean(globalThis.__OMNI_MASK_PIPELINE_V2__)
      : false;
    state.maskPipelineEnabled = Boolean(originalOptions.enableMaskPipelineV2 || globalFlag);
    state.strokeChanges = null;
    state.paintQueue.length = 0;
    state.pendingAffinitySet = null;
    state.lastPaintPoint = null;
    state.isPainting = false;
    state.pendingMaskFlush = false;
    const dims = state.ctx.getImageDimensions ? state.ctx.getImageDimensions() : { width: 0, height: 0 };
    const width = dims.width | 0;
    const height = dims.height | 0;
    const totalPixels = width * height;
    if (totalPixels > 0) {
      const existingMask = state.ctx.maskValues;
      const wasmMask = ensureMaskBuffer(totalPixels);
      if (existingMask && existingMask !== wasmMask) {
        const minCopy = Math.min(existingMask.length | 0, totalPixels);
        for (let i = 0; i < minCopy; i += 1) {
          wasmMask[i] = existingMask[i] | 0;
        }
      }
      updateMaskReference(wasmMask, true);
    }
    if (ENABLE_DEBUG_GRID_MASK) {
      applyDebugGridIfNeeded(false);
    }
    if (!state.componentTracker) {
      state.componentTracker = createComponentTracker();
    }
    state.componentTracker.rebuild(state.ctx);
    ensureMaskPipeline(state.ctx);
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

  function scheduleDeferredMaskUpdate() {
    const ctx = ensureCtx();
    if (state.pendingMaskFlush) {
      return;
    }
    state.pendingMaskFlush = true;
    const scheduler = typeof requestAnimationFrame === 'function'
      ? requestAnimationFrame
      : (cb) => setTimeout(cb, 16);
    scheduler(() => {
      state.pendingMaskFlush = false;
      if (typeof ctx.requestPaintFrame === 'function') {
        ctx.requestPaintFrame();
      }
      if (typeof ctx.markNeedsMaskRedraw === 'function') {
        ctx.markNeedsMaskRedraw();
      }
      if (typeof ctx.applyMaskRedrawImmediate === 'function') {
        ctx.applyMaskRedrawImmediate();
      }
    });
  }


  function boundingRectFromIndices(indices, width, height) {
    if (!indices || !indices.length || !Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
      return null;
    }
    const total = width * height;
    let minX = width;
    let minY = height;
    let maxX = -1;
    let maxY = -1;
    for (let i = 0; i < indices.length; i += 1) {
      const idx = indices[i] | 0;
      if (idx < 0 || idx >= total) {
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

  function notifyMaskDirty(ctx, idxArr) {
    if (!ctx) {
      return;
    }
    if (typeof ctx.markMaskIndicesDirty === 'function') {
      try {
        ctx.markMaskIndicesDirty(idxArr);
      } catch (err) {
        if (typeof ctx.log === 'function') {
          ctx.log('debug grid markMaskIndicesDirty failed: ' + err);
        }
      }
    } else if (typeof globalThis === 'object' && globalThis.__OMNI_DEBUG__) {
      globalThis.__OMNI_DEBUG__.missingMaskDirty = (globalThis.__OMNI_DEBUG__.missingMaskDirty || 0) + 1;
    }
    if (typeof ctx.markMaskTextureFullDirty === 'function') {
      try {
        ctx.markMaskTextureFullDirty();
      } catch (err) {
        if (typeof ctx.log === 'function') {
          ctx.log('debug grid markMaskTextureFullDirty fallback failed: ' + err);
        }
      }
    }
    if (typeof ctx.markOutlineIndicesDirty === 'function') {
      try {
        ctx.markOutlineIndicesDirty(idxArr);
      } catch (err) {
        if (typeof ctx.log === 'function') {
          ctx.log('debug grid markOutlineIndicesDirty failed: ' + err);
        }
      }
    } else if (typeof globalThis === 'object' && globalThis.__OMNI_DEBUG__) {
      globalThis.__OMNI_DEBUG__.missingOutlineDirty = (globalThis.__OMNI_DEBUG__.missingOutlineDirty || 0) + 1;
    }
    if (typeof ctx.markOutlineTextureFullDirty === 'function') {
      try {
        ctx.markOutlineTextureFullDirty();
      } catch (err) {
        if (typeof ctx.log === 'function') {
          ctx.log('debug grid markOutlineTextureFullDirty fallback failed: ' + err);
        }
      }
    }
    if (typeof ctx.markNeedsMaskRedraw === 'function') {
      try {
        ctx.markNeedsMaskRedraw();
      } catch (err) {
        if (typeof ctx.log === 'function') {
          ctx.log('debug grid markNeedsMaskRedraw failed: ' + err);
        }
      }
    }
    if (typeof ctx.applyMaskRedrawImmediate === 'function') {
      ctx.applyMaskRedrawImmediate();
    } else if (typeof ctx.redrawMaskCanvas === 'function') {
      ctx.redrawMaskCanvas();
    }
    if (typeof ctx.requestPaintFrame === 'function') {
      try {
        ctx.requestPaintFrame();
      } catch (err) {
        if (typeof ctx.log === 'function') {
          ctx.log('debug grid requestPaintFrame failed: ' + err);
        }
      }
    }
  }


  function finalizeComponentFill(
    ctx,
    indices,
    paintLabel,
    beforeLabels,
    afterLabels,
    totalPixels,
    meta,
  ) {
    if (!ctx || !indices || !indices.length) {
      return false;
    }
    const maskValues = ctx.maskValues;
    const count = indices.length | 0;
    if (count <= 0) {
      return;
    }
    const sourceIndices = indices instanceof Uint32Array ? indices : (() => {
      const arr = new Uint32Array(count);
      for (let i = 0; i < count; i += 1) {
        arr[i] = indices[i] | 0;
      }
      return arr;
    })();
    const idxArr = new Uint32Array(sourceIndices.length);
    idxArr.set(sourceIndices);
    state.finalizeCallCount = (state.finalizeCallCount || 0) + 1;
    if (typeof global.__OMNI_DEBUG__ === 'object') {
      const history = global.__OMNI_DEBUG__.finalizeHistory || (global.__OMNI_DEBUG__.finalizeHistory = []);
      history.push({
        count: state.finalizeCallCount,
        size: idxArr.length,
        label: paintLabel | 0,
      });
      if (history.length > 64) {
        history.shift();
      }
    }
    const before = beforeLabels instanceof Uint32Array ? beforeLabels : (() => {
      const arr = new Uint32Array(count);
      for (let i = 0; i < count; i += 1) {
        arr[i] = beforeLabels ? (beforeLabels[i] | 0) : 0;
      }
      return arr;
    })();
    const after = afterLabels instanceof Uint32Array ? afterLabels : (() => {
      const arr = new Uint32Array(count);
      for (let i = 0; i < count; i += 1) {
        arr[i] = afterLabels ? (afterLabels[i] | 0) : paintLabel;
      }
      return arr;
    })();
    for (let i = 0; i < idxArr.length; i += 1) {
      const idx = idxArr[i] | 0;
      if (idx >= 0 && idx < maskValues.length) {
        maskValues[idx] = after[i] | 0;
      }
    }
    const debugHighlightRect = meta && Number.isFinite(meta.imageWidth) && Number.isFinite(meta.imageHeight)
      ? boundingRectFromIndices(idxArr, meta.imageWidth | 0, meta.imageHeight | 0)
      : null;
    const debugApi = typeof global.__OMNI_DEBUG__ === 'object' ? global.__OMNI_DEBUG__ : null;
    const highlightEnabled = debugApi && Object.prototype.hasOwnProperty.call(debugApi, 'enableFillHighlight')
      ? Boolean(debugApi.enableFillHighlight)
      : false;
    if (debugHighlightRect && highlightEnabled
      && debugApi && typeof debugApi.highlightFillRect === 'function') {
      try {
        debugApi.highlightFillRect(debugHighlightRect);
      } catch (_) {
        /* ignore */
      }
    }
    if (typeof ctx.pushHistory === 'function') {
      ctx.pushHistory(idxArr, before, after);
    }
    recordMaskPipelineMutation('fill', idxArr, before, after, {
      paintLabel: paintLabel | 0,
      count,
      totalPixels,
    });
    if (typeof ctx.log === 'function') {
      const sampleIdx = idxArr.length ? idxArr[0] : -1;
      const beforeSample = sampleIdx >= 0 ? before[0] : null;
      const afterSample = sampleIdx >= 0 ? after[0] : null;
      const maskSampleAfter = sampleIdx >= 0 ? (maskValues[sampleIdx] | 0) : null;
      ctx.log('[debug-fill]', {
        sampleIdx,
        before: beforeSample,
        after: afterSample,
        maskSampleAfter,
        paintLabel,
        count,
      });
      if (sampleIdx >= 0 && maskSampleAfter !== (paintLabel | 0)) {
        ctx.log('[debug-fill-mismatch]', {
          sampleIdx,
          maskSampleAfter,
          paintLabel,
          count,
        });
        for (let i = 0; i < idxArr.length; i += 1) {
          const idx = idxArr[i] | 0;
          maskValues[idx] = paintLabel | 0;
        }
      }
    }
    try {
      global.__pendingRelabelSelection = idxArr;
    } catch (err) {
      /* ignore */
    }
    const total = Number.isFinite(totalPixels) && totalPixels > 0
      ? totalPixels
      : (maskValues && maskValues.length ? maskValues.length : count);
    const fillsAll = total > 0 && count === total;
    let affinityUpdated = false;
    if (typeof ctx.updateAffinityGraphForIndices === 'function') {
      try {
        ctx.updateAffinityGraphForIndices(idxArr);
        affinityUpdated = true;
      } catch (err) {
        if (typeof ctx.log === 'function') {
          ctx.log('debug grid updateAffinityGraphForIndices failed: ' + err);
        }
      }
    }
    if (!affinityUpdated && typeof ctx.rebuildLocalAffinityGraph === 'function') {
      try {
        ctx.rebuildLocalAffinityGraph();
      } catch (err) {
        if (typeof ctx.log === 'function') {
          ctx.log('debug grid rebuildLocalAffinityGraph failed: ' + err);
        }
      }
    }
    if (typeof ctx.clearColorCaches === 'function') {
      try {
        ctx.clearColorCaches();
      } catch (err) {
        if (typeof ctx.log === 'function') {
          ctx.log('debug grid clearColorCaches failed: ' + err);
        }
      }
    }
    let currentHasNonZero = typeof ctx.getMaskHasNonZero === 'function'
      ? ctx.getMaskHasNonZero()
      : null;
    if (paintLabel > 0) {
      currentHasNonZero = true;
    } else if (paintLabel === 0) {
      if (fillsAll) {
        currentHasNonZero = false;
      } else if (currentHasNonZero) {
        let hasNonZero = false;
        if (maskValues) {
          for (let i = 0; i < maskValues.length; i += 1) {
            if ((maskValues[i] | 0) > 0) {
              hasNonZero = true;
              break;
            }
          }
        }
        currentHasNonZero = hasNonZero;
      }
    }
    if (currentHasNonZero === null) {
      let hasNonZero = false;
      if (maskValues) {
        for (let i = 0; i < maskValues.length; i += 1) {
          if ((maskValues[i] | 0) > 0) {
            hasNonZero = true;
            break;
          }
        }
      }
      currentHasNonZero = hasNonZero;
    }
    if (typeof ctx.setMaskHasNonZero === 'function') {
      try {
        ctx.setMaskHasNonZero(Boolean(currentHasNonZero));
      } catch (err) {
        if (typeof ctx.log === 'function') {
          ctx.log('debug grid setMaskHasNonZero failed: ' + err);
        }
      }
    }
    const skipViewerNotify = meta && meta.skipViewerNotify;
    if (!skipViewerNotify) {
      notifyMaskDirty(ctx, idxArr);
      if (typeof ctx.requestPaintFrame === 'function') {
        try {
          ctx.requestPaintFrame();
        } catch (err) {
          if (typeof ctx.log === 'function') {
            ctx.log('debug grid requestPaintFrame failed: ' + err);
          }
        }
      }
    }
    scheduleDeferredMaskUpdate();
    if (typeof ctx.scheduleStateSave === 'function') {
      try {
        ctx.scheduleStateSave();
      } catch (err) {
        if (typeof ctx.log === 'function') {
          ctx.log('debug grid scheduleStateSave failed: ' + err);
        }
      }
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
    return true;
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
    recordMaskPipelineMutation('brush-finalize', indices, before, after, {
      count: indices.length,
    });
    try {
      global.__pendingRelabelSelection = indices;
    } catch (err) {
      /* ignore */
    }
    const affinityFlushed = flushPendingAffinityUpdates();
    if (!affinityFlushed && typeof ctx.updateAffinityGraphForIndices === 'function') {
      ctx.updateAffinityGraphForIndices(indices);
    }
    const hasLiveOverlay = typeof ctx.hasLiveAffinityOverlay === 'function'
      ? ctx.hasLiveAffinityOverlay()
      : false;
    if (!hasLiveOverlay && typeof ctx.markAffinityGeometryDirty === 'function') {
      ctx.markAffinityGeometryDirty();
    }
    if (typeof ctx.isWebglPipelineActive === 'function' && ctx.isWebglPipelineActive()) {
      if (typeof ctx.markMaskIndicesDirty === 'function') {
        ctx.markMaskIndicesDirty(indices);
      }
    } else if (typeof ctx.redrawMaskCanvas === 'function') {
      ctx.redrawMaskCanvas();
    }
    if (typeof ctx.scheduleStateSave === 'function') {
      try {
        ctx.scheduleStateSave();
      } catch (err) {
        if (typeof ctx.log === 'function') {
          ctx.log('debug grid scheduleStateSave failed: ' + err);
        }
      }
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
      let beforeLabels = null;
      let afterLabels = null;
      const ensureLabelArrays = () => {
        if (beforeLabels && afterLabels) {
          return;
        }
        const count = changedIndices.length;
        beforeLabels = new Uint32Array(count);
        afterLabels = new Uint32Array(count);
        for (let i = 0; i < count; i += 1) {
          const idx = changedIndices[i];
          const beforeValue = state.strokeChanges && state.strokeChanges.has(idx)
            ? state.strokeChanges.get(idx)
            : paintLabel;
          beforeLabels[i] = beforeValue | 0;
          afterLabels[i] = maskValues[idx] | 0;
        }
      };
      if (state.componentTracker) {
        ensureLabelArrays();
        state.componentTracker.recordChanges(changedIndices, beforeLabels, afterLabels, ctx);
      }
      if (maskPipelineActive()) {
        ensureLabelArrays();
        recordMaskPipelineMutation('brush', changedIndices, beforeLabels, afterLabels, {
          tool: typeof ctx.getTool === 'function' ? ctx.getTool() : 'brush',
          paintLabel: paintLabel | 0,
          strokeActive: state.isPainting,
        });
      }
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
      scheduleDeferredMaskUpdate();
      if (typeof ctx.scheduleStateSave === 'function') {
        ctx.scheduleStateSave();
      }
    }
    state.lastPaintPoint = { x: point.x, y: point.y };
    return state.lastPaintPoint;
  }

  function floodFill(point) {
    const ctx = ensureCtx();
    const maskValues = ctx.maskValues;
    if (!maskValues || !point) {
      state.lastFillResult = {
        status: 'aborted',
        abortReason: 'no-mask-or-point',
      };
      return;
    }
    const dims = ctx.getImageDimensions ? ctx.getImageDimensions() : { width: 0, height: 0 };
    const width = dims.width | 0;
    const height = dims.height | 0;
    const sx = Math.round(point.x);
    const sy = Math.round(point.y);
    if (sx < 0 || sy < 0 || sx >= width || sy >= height) {
      state.lastFillResult = {
        status: 'aborted',
        abortReason: 'point-out-of-bounds',
      };
      return;
    }
    const startIdx = sy * width + sx;
    const targetLabel = maskValues[startIdx] | 0;
    const paintLabel = ctx.getCurrentLabel ? ctx.getCurrentLabel() : 0;
    const fillResult = {
      width,
      height,
      startIdx,
      paintLabel,
      targetLabel,
      status: 'aborted',
      fallbackTriggered: false,
      runCount: 0,
      runRows: 0,
      componentSize: 0,
      abortReason: null,
    };
    if (targetLabel === paintLabel) {
      fillResult.abortReason = 'same-label';
      fillResult.status = 'noop';
      state.lastFillResult = fillResult;
      recordMaskPipelineNoop({
        reason: 'same-label',
        label: paintLabel | 0,
      });
      return;
    }
    const tracker = state.componentTracker;
    if (!tracker) {
      fillResult.abortReason = 'no-tracker';
      state.lastFillResult = fillResult;
      return;
    }
    let component = tracker.componentAt(startIdx, ctx);
    if (!component || !component.indices || component.indices.length === 0) {
      fillResult.abortReason = 'empty-component';
      state.lastFillResult = fillResult;
      return;
    }
    tracker.ensure(ctx);
    tracker.ensureSteps(ctx);
    const bfsIndices = collectComponentIndices(startIdx, targetLabel, tracker, ctx);
    let indices = null;
    if (bfsIndices && bfsIndices.length) {
      const borrowed = borrowComponentBuffer(bfsIndices.length);
      borrowed.set(bfsIndices);
      if (component.indices && component.indices !== borrowed && typeof component.indices.__wasmPtr === 'number') {
        releaseComponentArray(component.indices);
      }
      component.indices = borrowed;
      component.size = borrowed.length | 0;
      indices = borrowed;
    } else {
      const ensured = ensureComponentArrayView(component.indices);
      component.indices = ensured;
      component.size = ensured ? (ensured.length | 0) : 0;
      indices = ensured;
    }
    const size = indices && indices.length ? (indices.length | 0) : 0;
    if (size === 0) {
      fillResult.abortReason = 'empty-indices';
      state.lastFillResult = fillResult;
      return;
    }
    fillResult.componentSize = size;
    const rowSet = new Set();
    let touchesBorder = false;
    for (let i = 0; i < size; i += 1) {
      const idx = indices[i] | 0;
      const row = width > 0 ? ((idx / width) | 0) : 0;
      rowSet.add(row);
      const col = width > 0 ? (idx % width) : 0;
      if (row === 0 || row === (height - 1) || col === 0 || col === (width - 1)) {
        touchesBorder = true;
      }
    }
    fillResult.runRows = rowSet.size;
    fillResult.touchesBorder = touchesBorder;
    fillResult.runCount = size;
    const before = new Uint32Array(size);
    before.fill(targetLabel);
    const after = new Uint32Array(size);
    after.fill(paintLabel);
    const finalizeIndices = new Uint32Array(indices);
    const componentId = component.id || (tracker.componentIdAt ? tracker.componentIdAt(indices[0] | 0) : 0);
    const shouldCheckNeighbors = tracker.labelHasOtherComponents
      ? tracker.labelHasOtherComponents(paintLabel, componentId)
      : true;
    const neighborIds = shouldCheckNeighbors ? new Set() : null;
    const idsRef = tracker.ids;
    const steps = paintLabel > 0 ? tracker.foregroundSteps : BACKGROUND_STEPS;
    if (shouldCheckNeighbors) {
      for (let i = 0; i < size; i += 1) {
        const idx = indices[i] | 0;
        if (idx < 0 || idx >= maskValues.length) {
          continue;
        }
        const x = idx % width;
        const y = (idx / width) | 0;
        for (let s = 0; s < steps.length; s += 1) {
          const dy = steps[s][0] | 0;
          const dx = steps[s][1] | 0;
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
            continue;
          }
          const neighborIndex = ny * width + nx;
          if ((maskValues[neighborIndex] | 0) !== paintLabel) {
            continue;
          }
          let neighborId = idsRef ? (idsRef[neighborIndex] | 0) : 0;
          if (!neighborId && tracker.ensureComponentForIndex) {
            const neighborComponent = tracker.ensureComponentForIndex(neighborIndex, ctx);
            neighborId = neighborComponent && neighborComponent.id ? neighborComponent.id | 0 : 0;
          }
          if (neighborId && neighborId !== componentId && neighborIds) {
            neighborIds.add(neighborId);
          }
        }
      }
    }
    // Use the WASM fill helper so every pixel in the component is updated,
    // even for very large “outside” regions that wrap the canvas.
    fillMaskWithIndices(indices, size, paintLabel);
    tracker.mergeFilledComponent(component, neighborIds, paintLabel, ctx);
    const finalizedPrimary = finalizeComponentFill(ctx, finalizeIndices, paintLabel, before, after, maskValues.length, {
      touchesBorder: fillResult.touchesBorder,
      componentSize: fillResult.componentSize,
      runCount: fillResult.runCount,
      imageWidth: width,
      imageHeight: height,
      skipViewerNotify: true,
    });
    if (finalizedPrimary) {
      notifyMaskDirty(ctx, finalizeIndices);
    } else {
      notifyMaskDirty(ctx, indices);
    }
    const verify = collectComponentIndices(startIdx, 0, tracker, ctx);
    if (verify && verify.length) {
      const verifyBefore = new Uint32Array(verify.length);
      const verifyAfter = new Uint32Array(verify.length);
      for (let i = 0; i < verify.length; i += 1) {
        const idx = verify[i] | 0;
        verifyBefore[i] = 0;
        verifyAfter[i] = paintLabel | 0;
        if (idx >= 0 && idx < maskValues.length) {
          maskValues[idx] = paintLabel | 0;
        }
      }
      const verifyClone = new Uint32Array(verify);
      const finalizedVerify = finalizeComponentFill(ctx, verifyClone, paintLabel, verifyBefore, verifyAfter, maskValues.length, {
        fallback: true,
        touchesBorder: fillResult.touchesBorder,
        componentSize: verify.length,
        imageWidth: width,
        imageHeight: height,
      });
      if (finalizedVerify) {
        notifyMaskDirty(ctx, verifyClone);
      } else {
        notifyMaskDirty(ctx, verify);
      }
      fillResult.fallbackTriggered = true;
    }
    fillResult.status = 'ok';
    state.lastFillResult = fillResult;
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

  function rebuildComponents() {
    const ctx = ensureCtx();
    if (!state.componentTracker) {
      state.componentTracker = createComponentTracker();
    }
    state.componentTracker.rebuild(ctx);
  }

  function debugForceMaskRefresh() {
    if (!state.ctx) {
      return false;
    }
    const ctx = state.ctx;
    if (typeof ctx.markMaskTextureFullDirty === 'function') {
      try {
        ctx.markMaskTextureFullDirty();
      } catch (err) {
        if (typeof ctx.log === 'function') {
          ctx.log('debug grid markMaskTextureFullDirty failed: ' + err);
        }
      }
    }
    if (typeof ctx.markOutlineTextureFullDirty === 'function') {
      try {
        ctx.markOutlineTextureFullDirty();
      } catch (err) {
        if (typeof ctx.log === 'function') {
          ctx.log('debug grid markOutlineTextureFullDirty failed: ' + err);
        }
      }
    }
    if (typeof ctx.applyMaskRedrawImmediate === 'function') {
      ctx.applyMaskRedrawImmediate();
    } else if (typeof ctx.redrawMaskCanvas === 'function') {
      ctx.redrawMaskCanvas();
    }
    if (typeof ctx.requestPaintFrame === 'function') {
      try {
        ctx.requestPaintFrame();
      } catch (err) {
        if (typeof ctx.log === 'function') {
          ctx.log('debug grid requestPaintFrame failed: ' + err);
        }
      }
    }
    return true;
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
    rebuildComponents,
    __debugGetState: () => state,
    __debugForceMaskRefresh: debugForceMaskRefresh,
    __debugApplyGridIfNeeded: (force = false) => applyDebugGridIfNeeded(Boolean(force)),
    __debugCollectComponent(startIdx, targetLabel) {
      const ctx = ensureCtx();
      const mask = ctx && ctx.maskValues ? ctx.maskValues : null;
      if (!mask || !mask.length || typeof startIdx !== 'number') {
        return null;
      }
      const dims = ctx.getImageDimensions ? ctx.getImageDimensions() : { width: 0, height: 0 };
      const width = dims.width | 0;
      const height = dims.height | 0;
      const total = mask.length | 0;
      const idx = startIdx | 0;
      if (idx < 0 || idx >= total) {
        return null;
      }
      const label = typeof targetLabel === 'number' ? (targetLabel | 0) : (mask[idx] | 0);
      const indices = collectComponentIndices(idx, label, state.componentTracker, ctx);
      if (!indices || !indices.length) {
        return {
          index: idx,
          label,
          size: 0,
        };
      }
      let minX = width;
      let maxX = -1;
      let minY = height;
      let maxY = -1;
      for (let i = 0; i < indices.length; i += 1) {
        const compIdx = indices[i] | 0;
        const y = (compIdx / width) | 0;
        const x = compIdx - y * width;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
      return {
        index: idx,
        label,
        size: indices.length,
        bounds: {
          minX,
          maxX,
          minY,
          maxY,
        },
        sample: Array.from(indices.slice(0, Math.min(indices.length, 16))),
      };
    },
  });
  global.OmniPainting = api;
  global.__OMNI_DEBUG__ = global.__OMNI_DEBUG__ || {};
  global.__OMNI_DEBUG__.setGridLoggingEnabled = (value) => {
    global.__OMNI_DEBUG__.gridLogs = Boolean(value);
  };
  global.__OMNI_DEBUG__.setForceGridMask = (value) => {
    const next = Boolean(value);
    global.__OMNI_FORCE_GRID_MASK__ = next;
    state.debugGridPending = next;
    if (next) {
      applyDebugGridIfNeeded(true);
    }
  };
  global.__OMNI_DEBUG__.setFillHighlightEnabled = (value) => {
    const next = Boolean(value);
    global.__OMNI_DEBUG__.enableFillHighlight = next;
  };
})(typeof window !== 'undefined' ? window : globalThis);
