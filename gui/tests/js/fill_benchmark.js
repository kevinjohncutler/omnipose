#!/usr/bin/env node
const assert = require('assert');

function setupGlobals() {
  if (typeof global.window === 'undefined') {
    global.window = global;
  }
  if (typeof global.globalThis === 'undefined') {
    global.globalThis = global;
  }
  if (typeof global.performance === 'undefined') {
    global.performance = { now: () => 0 };
  }
  if (typeof global.atob === 'undefined') {
    global.atob = (str) => Buffer.from(str, 'base64').toString('binary');
  }
  if (typeof global.btoa === 'undefined') {
    global.btoa = (str) => Buffer.from(str, 'binary').toString('base64');
  }
  if (typeof global.requestAnimationFrame === 'undefined') {
    global.requestAnimationFrame = (cb) => setTimeout(() => cb(performance.now()), 0);
  }
  if (typeof global.cancelAnimationFrame === 'undefined') {
    global.cancelAnimationFrame = (handle) => clearTimeout(handle);
  }
  if (typeof global.__OMNI_FILL_DEBUG__ === 'undefined') {
    global.__OMNI_FILL_DEBUG__ = false;
  }
}

function loadPainting() {
  setupGlobals();
  delete require.cache[require.resolve('../../gui/web/js/painting.js')];
  require('../../gui/web/js/painting.js');
  const api = global.OmniPainting;
  assert(api && typeof api.init === 'function', 'OmniPainting not initialised');
  return api;
}

function makeMockContext(width, height, initialMask, callLog) {
  const state = {
    currentLabel: 0,
    maskHasNonZero: false,
  };
  const ctx = {
    maskValues: initialMask,
    getImageDimensions: () => ({ width, height }),
    getCurrentLabel: () => state.currentLabel,
    setCurrentLabel: (label) => { state.currentLabel = label | 0; },
    hasNColor: false,
    isNColorActive: () => false,
    getMaskHasNonZero: () => state.maskHasNonZero,
    setMaskHasNonZero: (value) => { state.maskHasNonZero = Boolean(value); },
    markMaskIndicesDirty: () => {},
    markOutlineIndicesDirty: () => {},
    markMaskTextureFullDirty: () => { callLog.maskFull += 1; },
    markOutlineTextureFullDirty: () => { callLog.outlineFull += 1; },
    markNeedsMaskRedraw: () => { callLog.needsMaskRedraw += 1; },
    requestPaintFrame: () => { callLog.requestPaintFrame += 1; },
    applyMaskRedrawImmediate: () => { callLog.applyMaskImmediate += 1; },
    scheduleDraw: () => { callLog.scheduleDraw += 1; },
    scheduleStateSave: () => { callLog.scheduleStateSave += 1; },
    updateAffinityGraphForIndices: (indices) => { callLog.affinityUpdates.push(Array.from(indices || [])); },
    markAffinityGeometryDirty: () => { callLog.affinityGeometryDirty += 1; },
    clearColorCaches: () => { callLog.clearColorCaches += 1; },
    isWebglPipelineActive: () => true,
    getAffinitySteps: () => [
      [-1, 0],
      [1, 0],
      [0, -1],
      [0, 1],
    ],
    debugFillPerformance: false,
    floodRebuildThreshold: 0.35,
    floodBulkPixelThreshold: 1000,
    collectBrushIndices: null,
    ensure: () => {},
    rebuildLocalAffinityGraph: () => { callLog.rebuildLocalAffinityGraph += 1; },
    markMaskIndicesDirtyDuringBrush: () => {},
    redrawMaskCanvas: () => { callLog.redrawMaskCanvas += 1; },
    isWebglPipelineReady: () => false,
    enqueueAffinityIndexBatch: null,
    getPendingSegmentationPayload: () => null,
    setPendingSegmentationPayload: () => {},
    getPendingMaskRebuild: () => false,
    setPendingMaskRebuild: () => {},
    getSegmentationTimer: () => null,
    setSegmentationTimer: () => {},
    canRebuildMask: () => false,
    triggerMaskRebuild: () => {},
    log: () => {},
  };
  return ctx;
}

function expectMask(mask, width, expected, msg) {
  assert.strictEqual(mask.length, expected.length, `${msg}: mask length mismatch`);
  for (let i = 0; i < expected.length; i += 1) {
    if (mask[i] !== expected[i]) {
      throw new Error(`${msg}: mask mismatch at index ${i} (expected ${expected[i]}, found ${mask[i]})`);
    }
  }
}

function resetCallLog(log) {
  log.maskDirty.length = 0;
  log.outlineDirty.length = 0;
  log.affinityUpdates.length = 0;
  log.maskFull = 0;
  log.outlineFull = 0;
  log.requestPaintFrame = 0;
  log.applyMaskImmediate = 0;
  log.scheduleDraw = 0;
  log.scheduleStateSave = 0;
  log.needsMaskRedraw = 0;
  log.affinityGeometryDirty = 0;
  log.clearColorCaches = 0;
  log.rebuildLocalAffinityGraph = 0;
  log.redrawMaskCanvas = 0;
}

function runBenchmark() {
  const OmniPainting = loadPainting();
  const width = 6;
  const height = 4;
  const initialMask = new Uint32Array(width * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = y * width + x;
      initialMask[idx] = x < width / 2 ? 0 : 2;
    }
  }
  const callLog = {
    maskDirty: [],
    outlineDirty: [],
    affinityUpdates: [],
    maskFull: 0,
    outlineFull: 0,
    requestPaintFrame: 0,
    applyMaskImmediate: 0,
    scheduleDraw: 0,
    scheduleStateSave: 0,
    needsMaskRedraw: 0,
    affinityGeometryDirty: 0,
    clearColorCaches: 0,
    rebuildLocalAffinityGraph: 0,
    redrawMaskCanvas: 0,
  };
  const ctx = makeMockContext(width, height, initialMask, callLog);
  OmniPainting.init(ctx);
  OmniPainting.rebuildComponents();
  const wasmMask = ctx.maskValues;

  function fillAndCheck(point, label, expectedMask) {
    ctx.setCurrentLabel(label);
    resetCallLog(callLog);
    OmniPainting.floodFill(point);
    expectMask(wasmMask, width, expectedMask, `mask after fill to ${label}`);
    assert.strictEqual(callLog.maskDirty.length, 0, 'incremental mask dirty regions should not be emitted');
    assert.strictEqual(callLog.outlineDirty.length, 0, 'incremental outline dirty regions should not be emitted');
    assert.ok(callLog.maskFull > 0, 'mask texture should be marked fully dirty');
    assert.ok(callLog.outlineFull > 0, 'outline texture should be marked fully dirty');
    assert.ok(callLog.requestPaintFrame > 0 || callLog.applyMaskImmediate > 0,
      'paint frame or immediate redraw should be scheduled');
    assert.ok(callLog.rebuildLocalAffinityGraph > 0, 'affinity graph should rebuild after fill');
    assert.ok(callLog.needsMaskRedraw > 0 || callLog.applyMaskImmediate > 0,
      'mask redraw should be queued or applied');
    const maxMaskValue = wasmMask.reduce((acc, value) => Math.max(acc, value), 0);
    assert.ok(maxMaskValue <= 0xffff, 'mask contains values beyond 16-bit range');
  }

  // Fill the left half (label 0) with label 1
  const expectedAfterFirst = new Uint32Array(width * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = y * width + x;
      expectedAfterFirst[idx] = x < width / 2 ? 1 : 2;
    }
  }
  fillAndCheck({ x: 0, y: 0 }, 1, expectedAfterFirst);

  // Fill the right half (label 2) with label 3
  const expectedAfterSecond = new Uint32Array(width * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = y * width + x;
      expectedAfterSecond[idx] = x < width / 2 ? 1 : 3;
    }
  }
  fillAndCheck({ x: width - 1, y: 0 }, 3, expectedAfterSecond);

  // Refill left half with the same label should result in no mask change
  resetCallLog(callLog);
  ctx.setCurrentLabel(1);
  OmniPainting.floodFill({ x: 1, y: 1 });
  expectMask(wasmMask, width, expectedAfterSecond, 'mask should remain unchanged when filling with same label');
  assert.ok(callLog.maskDirty.length === 0, 'mask should not be marked dirty when label unchanged');
  assert.ok(callLog.maskFull === 0, 'mask should not be marked fully dirty when label unchanged');
  assert.ok(callLog.outlineFull === 0, 'outline should remain untouched when label unchanged');

  console.log('Fill benchmark completed successfully.');
}

function runFullMaskBarrierScenario() {
  const OmniPainting = loadPainting();
  const width = 16;
  const height = 8;
  const total = width * height;
  const callLog = {
    maskDirty: [],
    outlineDirty: [],
    affinityUpdates: [],
    maskFull: 0,
    outlineFull: 0,
    requestPaintFrame: 0,
    applyMaskImmediate: 0,
    scheduleDraw: 0,
    scheduleStateSave: 0,
    needsMaskRedraw: 0,
    affinityGeometryDirty: 0,
    clearColorCaches: 0,
    rebuildLocalAffinityGraph: 0,
    redrawMaskCanvas: 0,
  };
  const initialMask = new Uint32Array(total);
  const ctx = makeMockContext(width, height, initialMask, callLog);
  OmniPainting.init(ctx);
  OmniPainting.rebuildComponents();
  const mask = ctx.maskValues;
  ctx.setCurrentLabel(1);
  OmniPainting.floodFill({ x: 0, y: 0 });
  for (let i = 0; i < mask.length; i += 1) {
    assert.strictEqual(mask[i], 1, 'mask not filled to 1');
  }
  const state = OmniPainting.__debugGetState();
  assert(state && state.componentTracker, 'component tracker unavailable');
  const barrierY = Math.floor(height / 2);
  ctx.setCurrentLabel(0);
  OmniPainting.beginStroke({ x: 0, y: barrierY });
  OmniPainting.queuePaintPoint({ x: width - 1, y: barrierY });
  OmniPainting.processPaintQueue();
  OmniPainting.finalizeStroke();
  ctx.setCurrentLabel(2);
  assert.doesNotThrow(() => OmniPainting.floodFill({ x: 1, y: 1 }), 'flood fill after barrier threw');
  assert.strictEqual(mask[0], 2, 'flood fill did not recolor top region');
  console.log('Full-mask barrier scenario completed successfully.');
}

runBenchmark();
runFullMaskBarrierScenario();
console.log('All fill tests completed successfully.');
