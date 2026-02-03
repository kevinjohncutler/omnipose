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
        cb();
      }
      return 0;
    },
    setTimeout: (cb) => {
      if (typeof cb === 'function') {
        cb();
      }
      return 0;
    },
    clearTimeout: () => {},
    Buffer,
  };
  sandbox.global = sandbox;
  sandbox.globalThis = sandbox;
  sandbox.window = sandbox;
  sandbox.__OMNI_FORCE_GRID_MASK__ = false;
  sandbox.window.__OMNI_FORCE_GRID_MASK__ = false;
  sandbox.__OMNI_SKIP_GRID_REPAINT__ = true;
  sandbox.window.__OMNI_SKIP_GRID_REPAINT__ = true;
  vm.createContext(sandbox);
  const script = new vm.Script(code, { filename: 'painting.js' });
  script.runInContext(sandbox);
  return sandbox.window.OmniPainting;
}

function createCtx(width, height, options = {}) {
  let maskValues = options.maskValues || new Uint32Array(width * height);
  let currentLabel = options.currentLabel ?? 0;
  let hasNonZero = false;
  for (let i = 0; i < maskValues.length; i += 1) {
    if ((maskValues[i] | 0) > 0) {
      hasNonZero = true;
      break;
    }
  }
  const history = options.history || [];
  const affinity = options.affinity || [];
  const affinityRebuilds = options.affinityRebuilds || [];
  const maskDirty = options.maskDirty || [];
  const maskFullDirty = options.maskFullDirty || { count: 0 };
  const outlineFullDirty = options.outlineFullDirty || { count: 0 };
  const outlineDirty = options.outlineDirty || [];
  const redrawStats = options.redrawStats || { count: 0 };
  const requestPaintStats = options.requestPaintStats || { count: 0 };
  const markNeedsStats = options.markNeedsStats || { count: 0 };
  const ctx = {
    maskValues,
    outlineState: new Uint8Array(width * height),
    viewState: {},
    getImageDimensions: () => ({ width, height }),
    getCurrentLabel: () => currentLabel,
    setCurrentLabel: (value) => {
      currentLabel = value;
    },
    getMaskHasNonZero: () => hasNonZero,
    setMaskHasNonZero: (value) => {
      hasNonZero = Boolean(value);
    },
    markMaskIndicesDirty: (indices) => {
      maskDirty.push(Array.from(indices));
    },
    updateAffinityGraphForIndices: (indices) => {
      const len = indices && typeof indices.length === 'number' ? indices.length : 0;
      affinity.push(len);
    },
    hasNColor: false,
    isNColorActive: () => false,
    markMaskTextureFullDirty: () => {
      maskFullDirty.count += 1;
    },
    markOutlineTextureFullDirty: () => {
      outlineFullDirty.count += 1;
    },
    markOutlineIndicesDirty: (indices) => {
      outlineDirty.push(Array.from(indices));
    },
    clearColorCaches: () => {},
    requestPaintFrame: () => {
      requestPaintStats.count += 1;
    },
    markNeedsMaskRedraw: () => {
      markNeedsStats.count += 1;
    },
    applyMaskRedrawImmediate: () => {
      redrawStats.count += 1;
    },
    scheduleStateSave: () => {},
    scheduleDraw: () => {},
    draw: () => {},
    rebuildLocalAffinityGraph: () => {
      affinityRebuilds.push(maskValues.length);
    },
    getAffinityGraphInfo: () => null,
    getAffinitySteps: () => [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]],
    markAffinityGeometryDirty: () => {},
    enqueueAffinityIndexBatch: () => {},
    onMaskBufferReplaced: (next) => {
      maskValues = next;
      ctx.maskValues = next;
    },
    collectBrushIndices: null,
    getPendingSegmentationPayload: () => null,
    setPendingSegmentationPayload: () => {},
    getPendingMaskRebuild: () => false,
    setPendingMaskRebuild: () => {},
    getSegmentationTimer: () => null,
    setSegmentationTimer: () => {},
    canRebuildMask: () => false,
    triggerMaskRebuild: () => {},
    applySegmentationMask: () => {},
    isWebglPipelineActive: () => true,
    redrawMaskCanvas: () => {},
    hasLiveAffinityOverlay: () => false,
    pushHistory: (indices, before, after) => {
      history.push({
        indices: Array.from(indices),
        before: Array.from(before),
        after: Array.from(after),
      });
    },
    log: () => {},
  };
  if (options.overrides && typeof options.overrides === 'object') {
    Object.assign(ctx, options.overrides);
  }
  return {
    ctx,
    history,
    affinity,
    affinityRebuilds,
    maskDirty,
    maskFullDirty,
    outlineFullDirty,
    outlineDirty,
    redrawStats,
    requestPaintStats,
    markNeedsStats,
    getMaskValues: () => maskValues,
  };
}

function everyValue(array, expected) {
  for (let i = 0; i < array.length; i += 1) {
    if (array[i] !== expected) {
      return false;
    }
  }
  return true;
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function runBackgroundFillTest(painting) {
  const width = 4;
  const height = 4;
  const history = [];
  const affinity = [];
  const maskDirty = [];
  const ctxPayload = createCtx(width, height, {
    currentLabel: 5,
    history,
    affinity,
    maskDirty,
  });
  painting.init(ctxPayload.ctx);
  painting.floodFill({ x: 0, y: 0 });
  const maskValues = ctxPayload.getMaskValues();
  assert(everyValue(maskValues, 5), 'background fill should recolor entire mask');
  assert(history.length === 1, 'background fill should push history entry');
  assert(history[0].indices.length === width * height, 'history indices cover entire mask');
  const sawRebuild = ctxPayload.affinityRebuilds.length > 0;
  assert(sawRebuild, 'affinity rebuild should run for full mask');
  assert(maskDirty.length === 0, 'background fill should not rely on incremental mask dirty regions');
  assert(ctxPayload.maskFullDirty.count > 0, 'background fill should mark mask texture fully dirty');
  assert(ctxPayload.outlineFullDirty.count > 0, 'background fill should refresh outline texture');
}

function runForegroundFillTest(painting) {
  const width = 6;
  const height = 6;
  const maskValues = new Uint32Array(width * height);
  for (let i = 0; i < maskValues.length; i += 1) {
    maskValues[i] = 1;
  }
  // insert a separating zero row to create two components
  for (let x = 0; x < width; x += 1) {
    maskValues[3 * width + x] = 0;
  }
  const history = [];
  const affinity = [];
  const maskDirty = [];
  const ctxPayload = createCtx(width, height, {
    maskValues,
    currentLabel: 3,
    history,
    affinity,
    maskDirty,
  });
  painting.init(ctxPayload.ctx);
  painting.floodFill({ x: 1, y: 1 });
  const updated = ctxPayload.getMaskValues();
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = y * width + x;
      if (y < 3) {
        assert(updated[idx] === 3, 'foreground fill should recolor target component');
      } else if (y === 3) {
        assert(updated[idx] === 0, 'separator row remains zero');
      } else {
        assert(updated[idx] === 1, 'disconnected component remains original label');
      }
    }
  }
  assert(history.length === 1, 'foreground fill pushes history once');
  assert(history[0].indices.length === width * 3, 'history indices match foreground component size');
  const sawForegroundRebuild = ctxPayload.affinityRebuilds.length > 0;
  assert(sawForegroundRebuild, 'affinity rebuild should cover filled component');
  assert(maskDirty.length === 0, 'foreground fill should not rely on incremental mask dirty regions');
  assert(ctxPayload.maskFullDirty.count > 0, 'foreground fill should mark mask texture fully dirty');
  assert(ctxPayload.outlineFullDirty.count > 0, 'foreground fill should mark outline texture fully dirty');
}

function runBarrierRegressionTest(painting) {
  const width = 128;
  const height = 64;
  const maskValues = new Uint32Array(width * height);
  maskValues.fill(1);
  const barrierRow = height / 2;
  for (let x = 0; x < width; x += 1) {
    maskValues[barrierRow * width + x] = 0;
  }
  const ctxPayload = createCtx(width, height, {
    maskValues,
    currentLabel: 2,
  });
  painting.init(ctxPayload.ctx);
  painting.floodFill({ x: 10, y: 10 });
  const updated = ctxPayload.getMaskValues();
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = y * width + x;
      if (y < barrierRow) {
        assert(updated[idx] === 2, 'barrier regression: upper half should be paint label');
      } else if (y === barrierRow) {
        assert(updated[idx] === 0, 'barrier regression: barrier row should stay zero');
      } else {
        assert(updated[idx] === 1, 'barrier regression: lower half should remain original label');
      }
    }
  }
  const state = painting.__debugGetState();
  const result = state && state.lastFillResult ? state.lastFillResult : null;
  assert(result && result.status === 'ok', 'barrier regression: fill should succeed');
  assert(!result.fallbackTriggered, 'barrier regression: wasm path should not fall back');
  assert(result.runCount > 0, 'barrier regression: wasm run emitter should record runs');
}

function runTallStripeFillTest(painting) {
  const width = 96;
  const height = 96;
  const barrierColumnStart = (width / 2) - 2;
  const barrierWidth = 4;
  const maskValues = new Uint32Array(width * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = y * width + x;
      if (x >= barrierColumnStart && x < barrierColumnStart + barrierWidth) {
        maskValues[idx] = 0;
      } else {
        maskValues[idx] = 1;
      }
    }
  }
  const ctxPayload = createCtx(width, height, {
    maskValues,
    currentLabel: 5,
  });
  painting.init(ctxPayload.ctx);
  painting.floodFill({ x: barrierColumnStart - 2, y: height - 1 });
  const updated = ctxPayload.getMaskValues();
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = y * width + x;
      if (x < barrierColumnStart) {
        assert(updated[idx] === 5, 'tall stripe: left stripe should adopt paint label');
      } else if (x >= barrierColumnStart && x < barrierColumnStart + barrierWidth) {
        assert(updated[idx] === 0, 'tall stripe: barrier column remains zero');
      } else {
        assert(updated[idx] === 1, 'tall stripe: right stripe stays original label');
      }
    }
  }
  const state = painting.__debugGetState();
  const result = state && state.lastFillResult ? state.lastFillResult : null;
  assert(result && result.status === 'ok', 'tall stripe: fill should succeed');
  assert(!result.fallbackTriggered, 'tall stripe: wasm path should not fall back');
  assert(result.runCount > 0, 'tall stripe: wasm run emitter should record runs');
  assert(result.runRows === height, 'tall stripe: runs should cover every row of the component');
}

function runVoidFillNoopTest(painting) {
  const width = 8;
  const height = 4;
  const maskValues = new Uint32Array(width * height);
  maskValues.fill(7);
  const ctxPayload = createCtx(width, height, {
    maskValues,
    currentLabel: 7,
  });
  painting.init(ctxPayload.ctx);
  painting.floodFill({ x: 1, y: 1 });
  assert(ctxPayload.history.length === 0, 'void fill should not push history entries');
  assert(ctxPayload.maskDirty.length === 0, 'void fill should not mark mask indices dirty');
  assert(ctxPayload.maskFullDirty.count === 0, 'void fill should not request full mask redraw');
  assert(ctxPayload.outlineFullDirty.count === 0, 'void fill should not request outline redraw');
  assert(ctxPayload.affinity.length === 0, 'void fill should not enqueue affinity updates');
  const state = painting.__debugGetState();
  assert(state && state.lastFillResult && state.lastFillResult.status === 'noop', 'void fill should report noop status');
}

function runAdjacentVoidFillTest(painting) {
  const width = 6;
  const height = 3;
  const maskValues = new Uint32Array(width * height);
  maskValues.fill(9);
  const voidIndices = [
    { x: 2, y: 1 },
    { x: 3, y: 1 },
    { x: 2, y: 0 },
  ];
  voidIndices.forEach(({ x, y }) => {
    maskValues[y * width + x] = 0;
  });
  const ctxPayload = createCtx(width, height, {
    maskValues,
    currentLabel: 4,
  });
  painting.init(ctxPayload.ctx);
  painting.floodFill({ x: 2, y: 1 });
  const updated = ctxPayload.getMaskValues();
  voidIndices.forEach(({ x, y }) => {
    const idx = y * width + x;
    assert(updated[idx] === 4, 'adjacent void pixels should adopt paint label');
  });
  for (let idx = 0; idx < updated.length; idx += 1) {
    if (!voidIndices.some(({ x, y }) => y * width + x === idx)) {
      assert(updated[idx] === 9, 'non-void pixels should remain original label');
    }
  }
}

function runMultiRefreshVoidFillTest(painting) {
  const width = 8;
  const height = 8;
  const maskValues = new Uint32Array(width * height);
  maskValues.fill(6);
  const voidA = { x: 2, y: 2 };
  const voidB = { x: 5, y: 5 };
  maskValues[voidA.y * width + voidA.x] = 0;
  maskValues[voidB.y * width + voidB.x] = 0;
  const ctxPayload = createCtx(width, height, {
    maskValues,
    currentLabel: 9,
  });
  painting.init(ctxPayload.ctx);
  painting.floodFill(voidA);
  let updated = ctxPayload.getMaskValues();
  assert(updated[voidA.y * width + voidA.x] === 9, 'first void should fill before refresh');
  assert(updated[voidB.y * width + voidB.x] === 0, 'second void untouched before refresh');
  const refreshedMask = new Uint32Array(updated);
  const ctxPayload2 = createCtx(width, height, {
    maskValues: refreshedMask,
    currentLabel: 9,
  });
  painting.init(ctxPayload2.ctx);
  painting.floodFill(voidB);
  updated = ctxPayload2.getMaskValues();
  assert(updated[voidA.y * width + voidA.x] === 9, 'first void should remain filled after refresh');
  assert(updated[voidB.y * width + voidB.x] === 9, 'second void should fill after refresh');
}

function runBorderRefreshTest(painting) {
  const width = 16;
  const height = 8;
  const maskValues = new Uint32Array(width * height);
  for (let x = 0; x < width; x += 1) {
    maskValues[x] = 1;
  }
  let applyImmediate = 0;
  let redrawCalls = 0;
  const ctxPayload = createCtx(width, height, {
    maskValues,
    currentLabel: 7,
    overrides: {
      applyMaskRedrawImmediate: () => {
        applyImmediate += 1;
      },
      redrawMaskCanvas: () => {
        redrawCalls += 1;
      },
    },
  });
  painting.init(ctxPayload.ctx);
  painting.floodFill({ x: 0, y: 0 });
  const updated = ctxPayload.getMaskValues();
  for (let x = 0; x < width; x += 1) {
    assert(updated[x] === 7, 'border refresh: top row should adopt new label');
  }
  assert(ctxPayload.maskFullDirty.count > 0, 'border refresh: mask marked fully dirty');
  assert(ctxPayload.outlineFullDirty.count > 0, 'border refresh: outline marked fully dirty');
  assert(
    applyImmediate > 0 || redrawCalls > 0,
    'border refresh: should trigger immediate redraw',
  );
}

function runHugeComponentRefreshTest(painting) {
  const width = 24;
  const height = 24;
  const maskValues = new Uint32Array(width * height);
  for (let y = 4; y < height - 4; y += 1) {
    for (let x = 4; x < width - 4; x += 1) {
      maskValues[y * width + x] = 3;
    }
  }
  let applyImmediate = 0;
  let redrawCalls = 0;
  const ctxPayload = createCtx(width, height, {
    maskValues,
    currentLabel: 9,
    overrides: {
      applyMaskRedrawImmediate: () => {
        applyImmediate += 1;
      },
      redrawMaskCanvas: () => {
        redrawCalls += 1;
      },
    },
  });
  painting.init(ctxPayload.ctx);
  painting.floodFill({ x: 12, y: 12 });
  const updated = ctxPayload.getMaskValues();
  for (let y = 4; y < height - 4; y += 1) {
    for (let x = 4; x < width - 4; x += 1) {
      assert(updated[y * width + x] === 9, 'huge component: interior should adopt paint label');
    }
  }
  assert(ctxPayload.maskFullDirty.count > 0, 'huge component: mask dirty flagged');
  assert(ctxPayload.outlineFullDirty.count > 0, 'huge component: outline dirty flagged');
  assert(
    applyImmediate > 0 || redrawCalls > 0,
    'huge component: should trigger immediate redraw',
  );
}

function runWebglIncrementalFillRefreshTest(painting) {
  const width = 32;
  const height = 32;
  const maskValues = new Uint32Array(width * height);
  for (let y = 8; y < 24; y += 1) {
    for (let x = 4; x < 28; x += 1) {
      maskValues[y * width + x] = 5;
    }
  }
  let applyImmediate = 0;
  let redrawMaskCanvasCalls = 0;
  let markMaskFullDirty = 0;
  let markOutlineFullDirty = 0;
  const ctxPayload = createCtx(width, height, {
    maskValues,
    currentLabel: 11,
    overrides: {
      isWebglPipelineActive: () => true,
      markMaskTextureFullDirty: () => {
        markMaskFullDirty += 1;
      },
      markOutlineTextureFullDirty: () => {
        markOutlineFullDirty += 1;
      },
      applyMaskRedrawImmediate: () => {
        applyImmediate += 1;
      },
      redrawMaskCanvas: () => {
        redrawMaskCanvasCalls += 1;
      },
    },
  });
  painting.init(ctxPayload.ctx);
  painting.floodFill({ x: 16, y: 16 });
  assert(applyImmediate > 0 || redrawMaskCanvasCalls > 0, 'webgl incremental fill should force immediate redraw');
  assert(markMaskFullDirty > 0, 'webgl incremental fill should flag full mask dirty');
  assert(markOutlineFullDirty > 0, 'webgl incremental fill should flag full outline dirty');
  const updated = ctxPayload.getMaskValues();
  for (let y = 8; y < 24; y += 1) {
    for (let x = 4; x < 28; x += 1) {
      assert(updated[y * width + x] === 11, 'webgl incremental fill should update labels');
    }
  }
}

function drawRingOnMask(mask, width, height, centerX, centerY, innerRadius, thickness, label, background) {
  const innerSq = innerRadius * innerRadius;
  const outer = innerRadius + thickness;
  const outerSq = outer * outer;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const dx = x - centerX;
      const dy = y - centerY;
      const distSq = dx * dx + dy * dy;
      if (distSq <= outerSq) {
        mask[y * width + x] = label;
      }
    }
  }
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const dx = x - centerX;
      const dy = y - centerY;
      const distSq = dx * dx + dy * dy;
      if (distSq < innerSq) {
        mask[y * width + x] = background;
      }
    }
  }
}

function collectDiskIndices(target, x, y, radius, width, height) {
  const cx = Math.round(x);
  const cy = Math.round(y);
  const r = Math.max(1, Math.round(radius));
  const rSq = r * r;
  for (let dy = -r; dy <= r; dy += 1) {
    const yy = cy + dy;
    if (yy < 0 || yy >= height) continue;
    for (let dx = -r; dx <= r; dx += 1) {
      const xx = cx + dx;
      if (xx < 0 || xx >= width) continue;
      if ((dx * dx) + (dy * dy) <= rSq) {
        target.add(yy * width + xx);
      }
    }
  }
}

const NEIGHBORS_8 = [
  [-1, -1], [-1, 0], [-1, 1],
  [0, -1],           [0, 1],
  [1, -1],  [1, 0],  [1, 1],
];
const NEIGHBORS_4 = [
  [-1, 0],
  [0, -1], [0, 1],
  [1, 0],
];

function collectComponentIndicesForLabel(mask, width, height, startIdx, targetLabel) {
  if (targetLabel === undefined || targetLabel === null) {
    return [];
  }
  if (startIdx < 0 || startIdx >= mask.length) {
    return [];
  }
  const total = mask.length;
  const visited = new Uint8Array(total);
  const queue = new Uint32Array(total);
  let head = 0;
  let tail = 0;
  const indices = [];
  const steps = (targetLabel | 0) > 0 ? NEIGHBORS_8 : NEIGHBORS_4;
  queue[tail++] = startIdx;
  while (head < tail) {
    const idx = queue[head++] | 0;
    if (idx < 0 || idx >= total) {
      continue;
    }
    if (visited[idx]) {
      continue;
    }
    if ((mask[idx] | 0) !== (targetLabel | 0)) {
      continue;
    }
    visited[idx] = 1;
    indices.push(idx);
    const x = idx % width;
    const y = (idx / width) | 0;
    for (let i = 0; i < steps.length; i += 1) {
      const nx = x + steps[i][1];
      const ny = y + steps[i][0];
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
        continue;
      }
      const nidx = ny * width + nx;
      if (!visited[nidx]) {
        queue[tail++] = nidx;
      }
    }
  }
  return indices;
}

function drawCircleStroke(painting, centerX, centerY, radius, opts = {}) {
  const steps = opts.steps || 180;
  const points = [];
  for (let i = 0; i <= steps; i += 1) {
    const angle = (Math.PI * 2 * i) / steps;
    points.push({
      x: centerX + radius * Math.cos(angle),
      y: centerY + radius * Math.sin(angle),
    });
  }
  painting.beginStroke(points[0]);
  for (let i = 1; i < points.length; i += 1) {
    painting.queuePaintPoint(points[i]);
  }
  painting.processPaintQueue();
  painting.finalizeStroke();
}

function drawRingWithBrush(painting, centerX, centerY, innerRadius, thickness, options = {}) {
  const brushRadius = options.brushRadius || 4;
  const steps = options.steps || 180;
  const passes = Math.max(1, Math.ceil(thickness / (brushRadius * 1.5)));
  const startRadius = innerRadius;
  const endRadius = innerRadius + thickness;
  for (let pass = 0; pass < passes; pass += 1) {
    const t = passes === 1 ? 0.5 : pass / (passes - 1);
    const radius = startRadius + t * (endRadius - startRadius);
    drawCircleStroke(painting, centerX, centerY, radius, { steps });
  }
}

function drawLineWithBrush(painting, x0, y0, x1, y1, options = {}) {
  const steps = options.steps || Math.ceil(Math.hypot(x1 - x0, y1 - y0));
  if (steps <= 0) {
    return;
  }
  const points = [];
  for (let i = 0; i <= steps; i += 1) {
    const t = steps === 0 ? 0 : i / steps;
    points.push({
      x: x0 + (x1 - x0) * t,
      y: y0 + (y1 - y0) * t,
    });
  }
  painting.beginStroke(points[0]);
  for (let i = 1; i < points.length; i += 1) {
    painting.queuePaintPoint(points[i]);
  }
  painting.processPaintQueue();
  painting.finalizeStroke();
}

function seededRandom(seed = 1337) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0xffffffff;
  };
}

function drawRandomScribble(painting, width, height, rand, options = {}) {
  const ctx = painting.__debugGetState().ctx;
  const originalCollect = ctx.collectBrushIndices;
  const brushRadius = options.brushRadius || 4;
  const steps = options.steps || 40;
  ctx.collectBrushIndices = (target, x, y) => collectDiskIndices(target, x, y, brushRadius, width, height);
  const startX = Math.floor(rand() * width);
  const startY = Math.floor(rand() * height);
  const points = [{ x: startX, y: startY }];
  let angle = rand() * Math.PI * 2;
  let stepLength = Math.max(width, height) * 0.2;
  for (let i = 0; i < steps; i += 1) {
    angle += (rand() - 0.5) * Math.PI * 0.5;
    stepLength *= 0.8 + rand() * 0.4;
    const x = Math.max(0, Math.min(width - 1, points[points.length - 1].x + Math.cos(angle) * stepLength));
    const y = Math.max(0, Math.min(height - 1, points[points.length - 1].y + Math.sin(angle) * stepLength));
    points.push({ x, y });
    if (rand() < 0.1) {
      break;
    }
  }
  painting.beginStroke(points[0]);
  for (let i = 1; i < points.length; i += 1) {
    painting.queuePaintPoint(points[i]);
  }
  painting.processPaintQueue();
  painting.finalizeStroke();
  ctx.collectBrushIndices = originalCollect;
}

function drawRectangleStroke(painting, minX, minY, maxX, maxY) {
  const points = [];
  for (let x = minX; x <= maxX; x += 1) {
    points.push({ x, y: minY });
  }
  for (let y = minY + 1; y <= maxY; y += 1) {
    points.push({ x: maxX, y });
  }
  for (let x = maxX - 1; x >= minX; x -= 1) {
    points.push({ x, y: maxY });
  }
  for (let y = maxY - 1; y > minY; y -= 1) {
    points.push({ x: minX, y });
  }
  if (points.length === 0) {
    return;
  }
  painting.beginStroke(points[0]);
  for (let i = 1; i < points.length; i += 1) {
    painting.queuePaintPoint(points[i]);
  }
  painting.processPaintQueue();
  painting.finalizeStroke();
}

function carveGridWithBrush(painting, width, height, columns, rows, brushRadius = 4) {
  const cellWidth = Math.floor(width / columns);
  const cellHeight = Math.floor(height / rows);
  const ctx = painting.__debugGetState().ctx;
  const originalCollect = ctx.collectBrushIndices;
  ctx.collectBrushIndices = (target, x, y) => collectDiskIndices(target, x, y, brushRadius, width, height);
  for (let col = 0; col < columns; col += 1) {
    for (let row = 0; row < rows; row += 1) {
      const minX = col * cellWidth;
      const minY = row * cellHeight;
      const maxX = Math.min(width - 1, (col + 1) * cellWidth - 1);
      const maxY = Math.min(height - 1, (row + 1) * cellHeight - 1);
      drawRectangleStroke(painting, minX, minY, maxX, maxY);
    }
  }
  ctx.collectBrushIndices = originalCollect;
}

// Reproduces the "draw loop, fill inside, fill outside" workflow that
// previously left the outer region unchanged (or produced a same-label noop)
// after a few color changes.
function runRepeatedCircleFillTest(painting) {
  const width = 250;
  const height = 250;
  const maskValues = new Uint32Array(width * height);
  const ctxPayload = createCtx(width, height, {
    maskValues,
    currentLabel: 1,
    overrides: {
      collectBrushIndices: (target, x, y) => collectDiskIndices(target, x, y, 4, width, height),
    },
  });
  painting.init(ctxPayload.ctx);
  const centerX = width / 2;
  const centerY = height / 2;
  const colors = [1, 2, 3, 4];
  let backgroundLabel = 0;
  colors.forEach((color, index) => {
    maskValues.fill(backgroundLabel);
    painting.rebuildComponents();
    const innerRadius = 30 + index * 20;
    const thickness = 12;
    ctxPayload.ctx.setCurrentLabel(color);
    drawRingWithBrush(painting, centerX, centerY, innerRadius, thickness, {
      brushRadius: 4,
      steps: 360,
    });
    const maskBefore = ctxPayload.getMaskValues().slice();
    const centerIdx = (centerY * width + centerX) | 0;
    const beforeLabel = maskBefore[centerIdx] | 0;
    const componentBefore = collectComponentIndicesForLabel(maskBefore, width, height, centerIdx, beforeLabel);
    ctxPayload.ctx.setCurrentLabel(color);
    painting.floodFill({ x: centerX, y: centerY });
    const insideResult = painting.__debugGetState().lastFillResult;
    assert(insideResult && insideResult.status === 'ok', 'circle loop inside fill should succeed');
    const maskAfterInside = ctxPayload.getMaskValues();
    componentBefore.forEach((idx) => {
      if (maskAfterInside[idx] !== color) {
        throw new Error('circle loop inside fill missing index ' + idx);
      }
    });
    const cornerIdx = 5 * width + 5;
    const cornerBefore = maskAfterInside[cornerIdx];
    assert(cornerBefore === backgroundLabel, 'circle loop: corner pixel should remain background before outside fill');
    const maskBeforeOutside = maskAfterInside.slice();
    const outsideComponent = collectComponentIndicesForLabel(maskBeforeOutside, width, height, 0, maskBeforeOutside[0] | 0);
    ctxPayload.ctx.setCurrentLabel(color);
    painting.floodFill({ x: 5, y: 5 });
    const outsideResult = painting.__debugGetState().lastFillResult;
    assert(outsideResult && outsideResult.status === 'ok', 'circle loop outside fill should succeed');
    const maskAfterOutside = ctxPayload.getMaskValues();
    outsideComponent.forEach((idx) => {
      if (maskAfterOutside[idx] !== color) {
        throw new Error('circle loop outside fill missing index ' + idx);
      }
    });
    const cornerAfter = maskAfterOutside[cornerIdx];
    assert(cornerAfter === color, 'circle loop: outside corner should adopt new color');
    backgroundLabel = color;
  });
}

function runRepeatedCrossFillRegressionTest(painting) {
  const width = 500;
  const height = 500;
  const ctxPayload = createCtx(width, height, {
    currentLabel: 1,
    overrides: {
      collectBrushIndices: (target, x, y) => collectDiskIndices(target, x, y, 5, width, height),
    },
  });
  painting.init(ctxPayload.ctx);
  painting.rebuildComponents();
  const centers = [
    { x: Math.floor(width * 0.25), y: Math.floor(height * 0.25) },
    { x: Math.floor(width * 0.75), y: Math.floor(height * 0.25) },
    { x: Math.floor(width * 0.25), y: Math.floor(height * 0.75) },
    { x: Math.floor(width * 0.75), y: Math.floor(height * 0.75) },
  ];
  const colors = [1, 2, 3];
  colors.forEach((color) => {
    ctxPayload.ctx.setCurrentLabel(color);
    drawLineWithBrush(painting, width / 2, 0, width / 2, height - 1);
    drawLineWithBrush(painting, 0, height / 2, width - 1, height / 2);
    centers.forEach(({ x, y }) => {
      const sx = Math.max(0, Math.min(width - 1, x));
      const sy = Math.max(0, Math.min(height - 1, y));
      const startIdx = sy * width + sx;
      const maskBefore = ctxPayload.getMaskValues().slice();
      const beforeLabel = maskBefore[startIdx] | 0;
      const componentBefore = collectComponentIndicesForLabel(maskBefore, width, height, startIdx, beforeLabel);
      ctxPayload.ctx.setCurrentLabel(color);
      painting.floodFill({ x: sx, y: sy });
      const result = painting.__debugGetState().lastFillResult;
      if (!result || result.status !== 'ok') {
        throw new Error(`cross fill failed for color ${color} at (${sx},${sy}): ${JSON.stringify(result)}`);
      }
      const maskAfter = ctxPayload.getMaskValues();
      componentBefore.forEach((idx) => {
        if ((maskAfter[idx] | 0) !== color) {
          throw new Error(`cross fill component mismatch for color ${color} index ${idx} expected ${color} got ${maskAfter[idx] | 0}`);
        }
      });
    });
  });
}

function runGridDirtyAliasRegressionTest(painting) {
  const width = 200;
  const height = 180;
  const dirtyRefs = [];
  const dirtySnapshots = [];
  const ctxPayload = createCtx(width, height, {
    currentLabel: 1,
    overrides: {
      collectBrushIndices: (target, x, y) => collectDiskIndices(target, x, y, 4, width, height),
      markMaskIndicesDirty: (indices) => {
        dirtyRefs.push(indices);
        dirtySnapshots.push(new Uint32Array(indices));
      },
      markMaskTextureFullDirty: () => {},
      markOutlineTextureFullDirty: () => {},
    },
  });
  painting.init(ctxPayload.ctx);
  carveGridWithBrush(painting, width, height, 5, 4, 4);
  painting.rebuildComponents();
  let label = 3;
  for (let row = 0; row < 4; row += 1) {
    for (let col = 0; col < 5; col += 1) {
      const cx = Math.max(1, Math.min(width - 2, Math.floor((col + 0.5) * width / 5)));
      const cy = Math.max(1, Math.min(height - 2, Math.floor((row + 0.5) * height / 4)));
      const maskBefore = ctxPayload.getMaskValues().slice();
      const startIdx = cy * width + cx;
      const existing = maskBefore[startIdx] | 0;
      const componentBefore = collectComponentIndicesForLabel(maskBefore, width, height, startIdx, existing);
      label = ((label + 11) % 255) || 1;
      ctxPayload.ctx.setCurrentLabel(label);
      painting.floodFill({ x: cx, y: cy });
      const result = painting.__debugGetState().lastFillResult;
      if (!result || result.status !== 'ok') {
        throw new Error(`grid dirty alias fill failed at cell ${row},${col}`);
      }
      const maskAfter = ctxPayload.getMaskValues();
      componentBefore.forEach((idx) => {
        if ((maskAfter[idx] | 0) !== label) {
          throw new Error(`grid dirty alias component mismatch at cell ${row},${col} index ${idx}`);
        }
      });
    }
  }
  assert(dirtyRefs.length > 0, 'grid dirty alias regression should mark mask indices dirty');
  for (let i = 0; i < dirtyRefs.length; i += 1) {
    const ref = dirtyRefs[i];
    const snapshot = dirtySnapshots[i];
    assert(ref.length === snapshot.length, 'dirty alias regression length mismatch');
    for (let j = 0; j < ref.length; j += 1) {
      if ((ref[j] | 0) !== (snapshot[j] | 0)) {
        throw new Error('dirty alias regression detected mutated indices array');
      }
    }
  }
}

function runGridFillStressTest(painting) {
  const width = 300;
  const height = 300;
  const ctxPayload = createCtx(width, height, {
    currentLabel: 1,
    overrides: {
      collectBrushIndices: (target, x, y) => collectDiskIndices(target, x, y, 4, width, height),
    },
  });
  painting.init(ctxPayload.ctx);
  const maxSubdivisions = 12;
  const rand = seededRandom(42);
  for (let grid = 2; grid <= maxSubdivisions; grid += 1) {
    ctxPayload.ctx.setCurrentLabel(0);
    const mask = ctxPayload.ctx.maskValues;
    for (let i = 0; i < mask.length; i += 1) { mask[i] = 0; }
    painting.rebuildComponents();
    ctxPayload.ctx.setCurrentLabel(1);
    carveGridWithBrush(painting, width, height, grid, grid, 4);
    const scribbleCount = Math.max(16, grid * grid);
    const scribbleSteps = Math.max(40, Math.floor(200 / Math.max(1, grid - 1)));
    for (let s = 0; s < scribbleCount; s += 1) {
      const scribbleLabel = ((s + 3) % 255) || 1;
      ctxPayload.ctx.setCurrentLabel(scribbleLabel);
      drawRandomScribble(painting, width, height, rand, { brushRadius: 3, steps: scribbleSteps });
    }
    painting.rebuildComponents();
    let label = 5;
    const cells = [];
    for (let row = 0; row < grid; row += 1) {
      for (let col = 0; col < grid; col += 1) {
        cells.push({ row, col });
      }
    }
    for (let i = cells.length - 1; i > 0; i -= 1) {
      const j = Math.floor(rand() * (i + 1));
      const tmp = cells[i];
      cells[i] = cells[j];
      cells[j] = tmp;
    }
    cells.forEach(({ row, col }) => {
      const centerX = Math.max(1, Math.min(width - 2, Math.floor((col + 0.5) * width / grid)));
      const centerY = Math.max(1, Math.min(height - 2, Math.floor((row + 0.5) * height / grid)));
      const maskBefore = ctxPayload.getMaskValues().slice();
      const centerIdx = centerY * width + centerX;
      const existing = maskBefore[centerIdx] | 0;
      label = ((label + 7) % 255) || 1;
      if (label === existing) {
        label = ((label + 11) % 255) || 1;
      }
      const componentBefore = collectComponentIndicesForLabel(maskBefore, width, height, centerIdx, existing);
      ctxPayload.ctx.setCurrentLabel(label);
      painting.floodFill({ x: centerX, y: centerY });
      const result = painting.__debugGetState().lastFillResult;
      if (!result || result.status !== 'ok') {
        throw new Error(`grid fill failed at grid ${grid} cell ${row},${col}: ${JSON.stringify(result)}`);
      }
      const maskAfter = ctxPayload.getMaskValues();
      componentBefore.forEach((idx) => {
        if (maskAfter[idx] !== label) {
          throw new Error(`grid fill component mismatch at grid ${grid} cell ${row},${col} index ${idx} value ${maskAfter[idx]} expected ${label}`);
        }
      });
    });
  }
}

function runLargeGridRedrawTest(painting) {
  const width = 288;
  const height = 288;
  const redrawStats = { count: 0 };
  const requestStats = { count: 0 };
  const markNeedsStats = { count: 0 };
  const ctxPayload = createCtx(width, height, {
    currentLabel: 1,
    redrawStats,
    requestPaintStats: requestStats,
    markNeedsStats,
    overrides: {
      collectBrushIndices: (target, x, y) => collectDiskIndices(target, x, y, 3, width, height),
    },
  });
  painting.init(ctxPayload.ctx);
  const grid = 12;
  carveGridWithBrush(painting, width, height, grid, grid, 3);
  painting.rebuildComponents();
  const cells = [];
  for (let row = 0; row < grid; row += 1) {
    for (let col = 0; col < grid; col += 1) {
      cells.push({ row, col });
    }
  }
  const rand = seededRandom(99);
  for (let i = cells.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rand() * (i + 1));
    const tmp = cells[i];
    cells[i] = cells[j];
    cells[j] = tmp;
  }
  let paintLabel = 5;
  cells.forEach(({ row, col }) => {
    const centerX = Math.max(1, Math.min(width - 2, Math.floor((col + 0.5) * width / grid)));
    const centerY = Math.max(1, Math.min(height - 2, Math.floor((row + 0.5) * height / grid)));
    const maskBefore = ctxPayload.getMaskValues().slice();
    const centerIdx = centerY * width + centerX;
    const existing = maskBefore[centerIdx] | 0;
    const componentBefore = collectComponentIndicesForLabel(maskBefore, width, height, centerIdx, existing);
    paintLabel = ((paintLabel + 13) % 255) || 1;
    if (paintLabel === existing) {
      paintLabel = ((paintLabel + 17) % 255) || 1;
    }
    ctxPayload.maskDirty.length = 0;
    ctxPayload.outlineDirty.length = 0;
    const maskDirtyStart = ctxPayload.maskDirty.length;
    const maskFullStart = ctxPayload.maskFullDirty.count;
    const outlineFullStart = ctxPayload.outlineFullDirty.count;
    ctxPayload.redrawStats.count = 0;
    ctxPayload.requestPaintStats.count = 0;
    ctxPayload.markNeedsStats.count = 0;
    ctxPayload.ctx.setCurrentLabel(paintLabel);
    painting.floodFill({ x: centerX, y: centerY });
    const result = painting.__debugGetState().lastFillResult;
    if (!result || result.status !== 'ok') {
      throw new Error(`large grid fill failed at ${row},${col}: ${JSON.stringify(result)}`);
    }
    const maskAfter = ctxPayload.getMaskValues();
    componentBefore.forEach((idx) => {
      if ((maskAfter[idx] | 0) !== paintLabel) {
        throw new Error(`large grid fill mismatch at ${row},${col} index ${idx}`);
      }
    });
    const maskDirtyDelta = ctxPayload.maskDirty.length - maskDirtyStart;
    const maskFullDelta = ctxPayload.maskFullDirty.count - maskFullStart;
    const outlineFullDelta = ctxPayload.outlineFullDirty.count - outlineFullStart;
    if (maskFullDelta === 0) {
      throw new Error(`large grid fill at ${row},${col} did not mark mask texture fully dirty`);
    }
    if (outlineFullDelta === 0) {
      throw new Error(`large grid fill at ${row},${col} did not mark outline texture fully dirty`);
    }
    if (maskDirtyDelta !== 0) {
      throw new Error(`large grid fill at ${row},${col} produced unexpected incremental mask dirty regions`);
    }
    if (ctxPayload.redrawStats.count === 0 && ctxPayload.requestPaintStats.count === 0) {
      throw new Error(`large grid fill at ${row},${col} did not trigger redraw`);
    }
    if (outlineFullDelta === 0 && outlineDirtyDelta === 0) {
      throw new Error(`large grid fill at ${row},${col} did not update outline texture`);
    }
    if (ctxPayload.markNeedsStats.count === 0 && ctxPayload.redrawStats.count === 0) {
      throw new Error(`large grid fill at ${row},${col} missing markNeedsMaskRedraw`);
    }
  });
}

function runEdgeVoidFillTest(painting) {
  const width = 500;
  const height = 500;
  for (let subdivisions = 3; subdivisions <= 7; subdivisions += 1) {
    const ctxPayload = createCtx(width, height, {
      currentLabel: 1,
      overrides: {
        collectBrushIndices: (target, x, y) => collectDiskIndices(target, x, y, 4, width, height),
      },
    });
    painting.init(ctxPayload.ctx);
    carveGridWithBrush(painting, width, height, subdivisions, subdivisions, 4);
    const rand = seededRandom(2000 + subdivisions);
    const scribbleCount = Math.max(64, subdivisions * subdivisions * 6);
    for (let s = 0; s < scribbleCount; s += 1) {
      const label = ((s * 13) % 255) || 1;
      ctxPayload.ctx.setCurrentLabel(label);
      drawRandomScribble(painting, width, height, rand, { brushRadius: 3, steps: 160 });
    }
    painting.rebuildComponents();
    const mid = Math.floor(subdivisions / 2);
    const targets = [
      { row: 0, col: mid },
      { row: subdivisions - 1, col: mid },
      { row: mid, col: 0 },
      { row: mid, col: subdivisions - 1 },
      { row: mid, col: mid },
    ];
    let paintLabel = 7;
    targets.forEach(({ row, col }) => {
      const cx = Math.max(1, Math.min(width - 2, Math.floor((col + 0.5) * width / subdivisions)));
      const cy = Math.max(1, Math.min(height - 2, Math.floor((row + 0.5) * height / subdivisions)));
      const maskBefore = ctxPayload.getMaskValues().slice();
      const startIdx = cy * width + cx;
      const existing = maskBefore[startIdx] | 0;
      const componentBefore = collectComponentIndicesForLabel(maskBefore, width, height, startIdx, existing);
      if (!componentBefore.length) {
        throw new Error(`edge void test component missing at ${row},${col} (grid ${subdivisions})`);
      }
      paintLabel = ((paintLabel + 19) % 255) || 1;
      if (paintLabel === existing) {
        paintLabel = ((paintLabel + 37) % 255) || 1;
      }
      ctxPayload.maskDirty.length = 0;
      ctxPayload.outlineDirty.length = 0;
      const maskFullStart = ctxPayload.maskFullDirty.count;
      const outlineFullStart = ctxPayload.outlineFullDirty.count;
      ctxPayload.ctx.setCurrentLabel(paintLabel);
      painting.floodFill({ x: cx, y: cy });
      const result = painting.__debugGetState().lastFillResult;
      if (!result || result.status !== 'ok') {
        throw new Error(`edge void fill failed for ${row},${col} (grid ${subdivisions}): ${JSON.stringify(result)}`);
      }
      const maskAfter = ctxPayload.getMaskValues();
      componentBefore.forEach((idx) => {
        if ((maskAfter[idx] | 0) !== paintLabel) {
          throw new Error(`edge void fill mismatch at ${row},${col} (grid ${subdivisions}) index ${idx}`);
        }
      });
      if ((ctxPayload.maskFullDirty.count - maskFullStart) === 0) {
        throw new Error(`edge void fill ${row},${col} (grid ${subdivisions}) did not mark mask texture fully dirty`);
      }
      if ((ctxPayload.outlineFullDirty.count - outlineFullStart) === 0) {
        throw new Error(`edge void fill ${row},${col} (grid ${subdivisions}) did not mark outline texture fully dirty`);
      }
      if (ctxPayload.maskDirty.length > 0) {
        throw new Error(`edge void fill ${row},${col} (grid ${subdivisions}) produced incremental mask dirty regions`);
      }
    });
  }
}

function runMassiveScribbleFillTest(painting) {
  const width = 256;
  const height = 256;
  const redrawStats = { count: 0 };
  const requestStats = { count: 0 };
  const markNeedsStats = { count: 0 };
  const ctxPayload = createCtx(width, height, {
    currentLabel: 1,
    redrawStats,
    requestPaintStats: requestStats,
    markNeedsStats,
    overrides: {
      collectBrushIndices: (target, x, y) => collectDiskIndices(target, x, y, 4, width, height),
    },
  });
  painting.init(ctxPayload.ctx);
  const scribbleCount = 160;
  const rand = seededRandom(321);
  for (let i = 0; i < scribbleCount; i += 1) {
    const brushLabel = ((i * 17) % 255) || 1;
    ctxPayload.ctx.setCurrentLabel(brushLabel);
    drawRandomScribble(painting, width, height, rand, { brushRadius: 4, steps: 180 });
  }
  painting.rebuildComponents();
  const fillSamples = 40;
  for (let fillIdx = 0; fillIdx < fillSamples; fillIdx += 1) {
    const sx = Math.max(2, Math.min(width - 3, Math.floor(rand() * width)));
    const sy = Math.max(2, Math.min(height - 3, Math.floor(rand() * height)));
    const maskBefore = ctxPayload.getMaskValues().slice();
    const startIdx = sy * width + sx;
    const targetLabel = maskBefore[startIdx] | 0;
    const componentBefore = collectComponentIndicesForLabel(maskBefore, width, height, startIdx, targetLabel);
    if (!componentBefore.length) {
      continue;
    }
    let paintLabel = ((fillIdx * 23) % 255) || 1;
    if (paintLabel === targetLabel) {
      paintLabel = ((paintLabel + 31) % 255) || 1;
    }
    ctxPayload.maskDirty.length = 0;
    ctxPayload.outlineDirty.length = 0;
    const maskDirtyStart = ctxPayload.maskDirty.length;
    const maskFullStart = ctxPayload.maskFullDirty.count;
    const outlineFullStart = ctxPayload.outlineFullDirty.count;
    ctxPayload.redrawStats.count = 0;
    ctxPayload.requestPaintStats.count = 0;
    ctxPayload.markNeedsStats.count = 0;
    ctxPayload.ctx.setCurrentLabel(paintLabel);
    painting.floodFill({ x: sx, y: sy });
    const result = painting.__debugGetState().lastFillResult;
    if (!result || result.status !== 'ok') {
      throw new Error(`massive scribble fill failed at sample ${fillIdx}: ${JSON.stringify(result)}`);
    }
    const maskAfter = ctxPayload.getMaskValues();
    componentBefore.forEach((idx) => {
      if ((maskAfter[idx] | 0) !== paintLabel) {
        throw new Error(`massive scribble fill mismatch at sample ${fillIdx} index ${idx}`);
      }
    });
    const maskDirtyDelta = ctxPayload.maskDirty.length - maskDirtyStart;
    const maskFullDelta = ctxPayload.maskFullDirty.count - maskFullStart;
    const outlineFullDelta = ctxPayload.outlineFullDirty.count - outlineFullStart;
    if (maskFullDelta === 0) {
      throw new Error(`massive scribble fill sample ${fillIdx} did not mark mask texture fully dirty`);
    }
    if (outlineFullDelta === 0) {
      throw new Error(`massive scribble fill sample ${fillIdx} did not mark outline texture fully dirty`);
    }
    if (maskDirtyDelta !== 0) {
      throw new Error(`massive scribble fill sample ${fillIdx} produced unexpected incremental mask dirty regions`);
    }
    if (ctxPayload.redrawStats.count === 0 && ctxPayload.requestPaintStats.count === 0) {
      throw new Error(`massive scribble fill sample ${fillIdx} did not trigger redraw pathway`);
    }
    if (ctxPayload.markNeedsStats.count === 0 && ctxPayload.redrawStats.count === 0) {
      throw new Error(`massive scribble fill sample ${fillIdx} missing markNeedsMaskRedraw`);
    }
    ctxPayload.maskDirty.length = maskDirtyStart;
  }
}

function runCircleOutsideFillTest(painting) {
  const width = 64;
  const height = 64;
  const maskValues = new Uint32Array(width * height);
  const radius = 24;
  const centerX = width / 2;
  const centerY = height / 2;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const dx = x - centerX;
      const dy = y - centerY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist > radius) {
        maskValues[y * width + x] = 1;
      }
    }
  }
  const ctxPayload = createCtx(width, height, {
    maskValues,
    currentLabel: 2,
  });
  painting.init(ctxPayload.ctx);
  painting.floodFill({ x: centerX, y: centerY });
  ctxPayload.ctx.setCurrentLabel(3);
  painting.floodFill({ x: 1, y: 1 });
  const updated = ctxPayload.getMaskValues();
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = y * width + x;
      const dx = x - centerX;
      const dy = y - centerY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist > radius) {
        assert(updated[idx] === 3, 'circle outside fill: exterior should adopt paint label');
      } else {
        assert(updated[idx] === 2, 'circle outside fill: interior should retain label');
      }
    }
  }
}

function main() {
  try {
    const painting = loadPainting();
    runBackgroundFillTest(painting);
    runForegroundFillTest(painting);
    runBarrierRegressionTest(painting);
    runTallStripeFillTest(painting);
    runVoidFillNoopTest(painting);
    runAdjacentVoidFillTest(painting);
    runMultiRefreshVoidFillTest(painting);
    runBorderRefreshTest(painting);
    runHugeComponentRefreshTest(painting);
    runWebglIncrementalFillRefreshTest(painting);
    runCircleOutsideFillTest(painting);
    runRepeatedCrossFillRegressionTest(painting);
    runGridDirtyAliasRegressionTest(painting);
    runRepeatedCircleFillTest(painting);
    runLargeGridRedrawTest(painting);
    runEdgeVoidFillTest(painting);
    runMassiveScribbleFillTest(painting);
    runGridFillStressTest(painting);
    process.stdout.write(JSON.stringify({ success: true }));
  } catch (err) {
    const message = err && err.stack ? err.stack : String(err);
    console.error(message);
    process.exit(1);
  }
}

main();
