const fs = require('fs');
const path = require('path');
const vm = require('vm');

function loadPainting() {
  const paintingPath = path.resolve(__dirname, '../../gui/web/js/painting.js');
  const code = fs.readFileSync(paintingPath, 'utf8');
  const sandbox = {
    console,
    performance: { now: () => 0 },
    requestAnimationFrame: (cb) => {
      if (typeof cb === 'function') {
        setTimeout(cb, 0);
      }
      return 0;
    },
    cancelAnimationFrame: () => {},
    setTimeout: (cb, delay = 0) => setTimeout(cb, delay),
    clearTimeout: (id) => clearTimeout(id),
    atob: (str) => Buffer.from(str, 'base64').toString('binary'),
    btoa: (str) => Buffer.from(str, 'binary').toString('base64'),
    Uint8Array,
    Uint16Array,
    Uint32Array,
    Uint8ClampedArray,
    Float32Array,
    Array,
    Math,
    Date,
    Buffer,
  };
  sandbox.global = sandbox;
  sandbox.globalThis = sandbox;
  sandbox.window = sandbox;
  sandbox.__OMNI_FORCE_GRID_MASK__ = true;
  sandbox.__OMNI_SKIP_GRID_REPAINT__ = false;
  vm.createContext(sandbox);
  new vm.Script(code, { filename: 'painting.js' }).runInContext(sandbox);
  return sandbox.window.OmniPainting;
}

function createCtx(width, height) {
  const maskValues = new Uint32Array(width * height);
  const outlineState = new Uint8Array(width * height);
  let currentLabel = 1;
  const maskDirtyRects = [];
  const outlineDirtyRects = [];
  const affinityLogs = [];
  let maskHasNonZero = false;
  const ctx = {
    maskValues,
    outlineState,
    viewState: {},
    log: (...parts) => {
      console.log('[ctx]', ...parts);
    },
    getImageDimensions: () => ({ width, height }),
    getCurrentLabel: () => currentLabel,
    setCurrentLabel: (value) => {
      currentLabel = value | 0;
    },
    setMaskHasNonZero: (value) => {
      maskHasNonZero = Boolean(value);
    },
    getMaskHasNonZero: () => maskHasNonZero,
    markMaskIndicesDirty: (indices) => {
      maskDirtyRects.push(Array.from(indices));
    },
    markMaskTextureFullDirty: () => {},
    markOutlineTextureFullDirty: () => {},
    markOutlineIndicesDirty: (indices) => {
      outlineDirtyRects.push(Array.from(indices));
    },
    markNeedsMaskRedraw: () => {},
    applyMaskRedrawImmediate: () => {},
    requestPaintFrame: () => {},
    scheduleStateSave: () => {},
    updateAffinityGraphForIndices: (indices) => {
      affinityLogs.push(Array.from(indices));
    },
    rebuildLocalAffinityGraph: () => {},
    markAffinityGeometryDirty: () => {},
    isWebglPipelineActive: () => false,
    clearColorCaches: () => {},
    scheduleDraw: () => {},
    draw: () => {},
    pushHistory: () => {},
    collectBrushIndices: null,
    getPendingSegmentationPayload: () => null,
    setPendingSegmentationPayload: () => {},
    getPendingMaskRebuild: () => false,
    setPendingMaskRebuild: () => {},
    getSegmentationTimer: () => null,
    setSegmentationTimer: () => {},
    canRebuildMask: () => false,
    triggerMaskRebuild: () => {},
  };
  return {
    ctx,
    maskDirtyRects,
    outlineDirtyRects,
    affinityLogs,
  };
}

async function main() {
  const painting = loadPainting();
  const width = 392;
  const height = 384;
  const { ctx } = createCtx(width, height);
  painting.init(ctx);
  if (typeof painting.__debugApplyGridIfNeeded === 'function') {
    painting.__debugApplyGridIfNeeded(true);
  }
  await new Promise((resolve) => setTimeout(resolve, 10));

  const maxTiles = 256;
  let tilesFilled = 0;
  const visited = new Set();
  const queue = [];

  function findNextZero() {
    for (let idx = 0; idx < ctx.maskValues.length; idx += 1) {
      if ((ctx.maskValues[idx] | 0) === 0 && !visited.has(idx)) {
        return idx;
      }
    }
    return -1;
  }

  function collectComponent(startIdx) {
    const arr = ctx.maskValues;
    const comp = [];
    const seen = new Set();
    queue.length = 0;
    queue.push(startIdx);
    seen.add(startIdx);
    while (queue.length) {
      const idx = queue.pop();
      if ((arr[idx] | 0) !== 0) {
        continue;
      }
      comp.push(idx);
      const x = idx % width;
      const y = (idx / width) | 0;
      const neighbors = [
        idx - width,
        idx + width,
        idx - 1,
        idx + 1,
      ];
      if (y > 0) neighbors[0] = idx - width;
      if (y >= height - 1) neighbors[1] = -1;
      if (x <= 0) neighbors[2] = -1;
      if (x >= width - 1) neighbors[3] = -1;
      for (const n of neighbors) {
        if (n >= 0 && !seen.has(n)) {
          seen.add(n);
          queue.push(n);
        }
      }
    }
    return comp;
  }

  while (tilesFilled < maxTiles) {
    const idx = findNextZero();
    if (idx === -1) {
      break;
    }
    const component = collectComponent(idx);
    component.forEach((i) => visited.add(i));
    const x = idx % width;
    const y = (idx / width) | 0;
    ctx.setCurrentLabel(1);
    painting.floodFill({ x, y });
    await new Promise((resolve) => setTimeout(resolve, 0));
    let failures = 0;
    for (const pixel of component) {
      if ((ctx.maskValues[pixel] | 0) !== 1) {
        failures += 1;
      }
    }
    if (failures > 0) {
      console.error('Fill failure', {
        tileIndex: tilesFilled,
        start: { x, y, idx },
        failures,
      });
      process.exit(1);
    }
    tilesFilled += 1;
  }

  console.log(JSON.stringify({ tilesFilled }));
}

main().catch((err) => {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
