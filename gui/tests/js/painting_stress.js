const fs = require('fs');
const path = require('path');
const vm = require('vm');

const paintingPath = path.resolve(__dirname, '../../gui/web/js/painting.js');
const code = fs.readFileSync(paintingPath, 'utf8');
const sandbox = {
  console,
  performance: { now: () => Date.now() },
  requestAnimationFrame: (cb) => setTimeout(() => cb(), 0),
  setTimeout,
  clearTimeout,
  Buffer,
};
sandbox.global = sandbox;
sandbox.window = sandbox;
vm.createContext(sandbox);
new vm.Script(code).runInContext(sandbox);
const painting = sandbox.window.OmniPainting;

const width = 256;
const height = 256;
let maskValues = new Uint32Array(width * height);
const ctx = {
  maskValues,
  outlineState: new Uint8Array(width * height),
  viewState: {},
  getImageDimensions: () => ({ width, height }),
  getCurrentLabel: () => currentLabel,
  setCurrentLabel: (value) => {
    currentLabel = value;
  },
  getMaskHasNonZero: () => true,
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
  onMaskBufferReplaced: (next) => {
    maskValues = next;
    ctx.maskValues = next;
  },
};

painting.init(ctx);
let currentLabel = 1;

function paintBlock(x0, y0, size) {
  const label = currentLabel;
  for (let y = y0; y < y0 + size; y += 1) {
    for (let x = x0; x < x0 + size; x += 1) {
      const idx = y * width + x;
      if (idx >= 0 && idx < maskValues.length) {
        maskValues[idx] = label;
      }
    }
  }
}

for (let iter = 0; iter < 2000; iter += 1) {
  currentLabel = (iter % 15) + 1;
  const size = 5 + (iter % 12);
  const x = Math.max(0, Math.min(width - size - 1, Math.floor(Math.random() * width)));
  const y = Math.max(0, Math.min(height - size - 1, Math.floor(Math.random() * height)));
  paintBlock(x, y, size);
  if ((iter % 25) === 0) {
    const fx = Math.floor(Math.random() * width);
    const fy = Math.floor(Math.random() * height);
    painting.floodFill({ x: fx, y: fy });
  }
}

console.log('stress done');
