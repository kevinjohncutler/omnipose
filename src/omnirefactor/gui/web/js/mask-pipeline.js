(function initMaskPipeline(global) {
  'use strict';

  function toUint32Array(source) {
    if (!source) {
      return new Uint32Array(0);
    }
    if (source instanceof Uint32Array) {
      return source;
    }
    if (Array.isArray(source)) {
      const arr = new Uint32Array(source.length);
      for (let i = 0; i < source.length; i += 1) {
        arr[i] = source[i] | 0;
      }
      return arr;
    }
    if (typeof source.length === 'number') {
      const arr = new Uint32Array(source.length);
      for (let i = 0; i < source.length; i += 1) {
        arr[i] = source[i] | 0;
      }
      return arr;
    }
    return new Uint32Array(0);
  }

  function copyMask(mask) {
    if (!mask || typeof mask.length !== 'number') {
      return null;
    }
    const clone = new Uint32Array(mask.length);
    clone.set(mask);
    return clone;
  }

  function createMaskPipeline(options = {}) {
    const state = {
      ctx: null,
      enabled: true,
      shadowMask: null,
      telemetry: {
        mutations: 0,
        brushMutations: 0,
        fillMutations: 0,
        noopFills: 0,
        divergence: 0,
        lastDivergenceSample: null,
      },
      logger: typeof options.logger === 'function' ? options.logger : null,
      onAnomaly: typeof options.onAnomaly === 'function' ? options.onAnomaly : null,
    };

    function logDebug(kind, data) {
      if (state.logger) {
        try {
          state.logger('[mask-pipeline] ' + kind, data);
        } catch (err) {
          /* ignore logging errors */
        }
      }
    }

    function ensureShadowMask() {
      if (!state.ctx) {
        return;
      }
      const mask = state.ctx.maskValues;
      if (!mask) {
        state.shadowMask = null;
        return;
      }
      if (!state.shadowMask || state.shadowMask.length !== mask.length) {
        state.shadowMask = copyMask(mask);
        return;
      }
      state.shadowMask.set(mask);
    }

    function attachCtx(ctx) {
      state.ctx = ctx || null;
      ensureShadowMask();
    }

    function detachCtx() {
      state.ctx = null;
      state.shadowMask = null;
    }

    function recordNoopFill(meta) {
      state.telemetry.noopFills += 1;
      if (options.verbose) {
        logDebug('noop-fill', meta || null);
      }
    }

    function applyDiff(kind, indices, beforeLabels, afterLabels, meta) {
      if (!state.enabled || !state.ctx || !state.shadowMask || !indices || !indices.length) {
        return;
      }
      const idxArr = toUint32Array(indices);
      const before = toUint32Array(beforeLabels);
      const after = toUint32Array(afterLabels);
      const mask = state.shadowMask;
      const len = idxArr.length;
      state.telemetry.mutations += 1;
      if (kind === 'brush') {
        state.telemetry.brushMutations += 1;
      } else if (kind === 'fill') {
        state.telemetry.fillMutations += 1;
      }
      for (let i = 0; i < len; i += 1) {
        const idx = idxArr[i] | 0;
        if (idx < 0 || idx >= mask.length) {
          continue;
        }
        mask[idx] = after[i] | 0;
      }
      if (options.detectDivergence !== false) {
        sampleDivergence(idxArr, mask, meta);
      }
    }

    function sampleDivergence(indices, shadow, meta) {
      const ctxMask = state.ctx && state.ctx.maskValues ? state.ctx.maskValues : null;
      if (!ctxMask || ctxMask.length !== shadow.length) {
        ensureShadowMask();
        return;
      }
      for (let i = 0; i < indices.length; i += 1) {
        const idx = indices[i] | 0;
        if (idx < 0 || idx >= shadow.length) {
          continue;
        }
        if (shadow[idx] !== ctxMask[idx]) {
          state.telemetry.divergence += 1;
          state.telemetry.lastDivergenceSample = {
            index: idx,
            expected: shadow[idx],
            actual: ctxMask[idx],
            meta: meta || null,
          };
          if (state.onAnomaly) {
            try {
              state.onAnomaly(state.telemetry.lastDivergenceSample);
            } catch (err) {
              /* ignore */
            }
          }
          break;
        }
      }
    }

    function onMaskBufferReplaced() {
      ensureShadowMask();
    }

    function getTelemetry() {
      return Object.assign({}, state.telemetry);
    }

    return {
      attachCtx,
      detachCtx,
      recordNoopFill,
      applyDiff,
      onMaskBufferReplaced,
      getTelemetry,
    };
  }

  const api = global.OmniMaskPipeline || {};
  api.createMaskPipeline = createMaskPipeline;
  global.OmniMaskPipeline = api;
})(typeof window !== 'undefined' ? window : globalThis);
