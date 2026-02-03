const fs = require('fs');
const path = require('path');
const vm = require('vm');

function noop() {}

const IMG_WIDTH = 392;
const IMG_HEIGHT = 384;

function collectDiskIndices(target, centerX, centerY, radius = 4, width = IMG_WIDTH, height = IMG_HEIGHT) {
  const cx = Math.round(centerX);
  const cy = Math.round(centerY);
  const r = Math.max(1, radius | 0);
  const r2 = r * r;
  for (let dy = -r; dy <= r; dy += 1) {
    const y = cy + dy;
    if (y < 0 || y >= height) {
      continue;
    }
    for (let dx = -r; dx <= r; dx += 1) {
      const x = cx + dx;
      if (x < 0 || x >= width) {
        continue;
      }
      if ((dx * dx) + (dy * dy) > r2) {
        continue;
      }
      target.add(y * width + x);
    }
  }
}

function makeCanvas2dStub() {
  return {
    getImageData: () => ({ data: new Uint8ClampedArray(IMG_WIDTH * IMG_HEIGHT * 4) }),
    createImageData: () => ({ data: new Uint8ClampedArray(IMG_WIDTH * IMG_HEIGHT * 4) }),
    putImageData: noop,
    clearRect: noop,
    drawImage: noop,
    fillRect: noop,
    beginPath: noop,
    moveTo: noop,
    lineTo: noop,
    stroke: noop,
    save: noop,
    restore: noop,
    translate: noop,
    scale: noop,
    setLineDash: noop,
    lineWidth: 1,
    strokeStyle: '',
  };
}

function makeWebglStub() {
  return {
    canvas: null,
    viewport: noop,
    clearColor: noop,
    clear: noop,
    pixelStorei: noop,
    createVertexArray: () => ({}),
    bindVertexArray: noop,
    createBuffer: () => ({}),
    bindBuffer: noop,
    bufferData: noop,
    enableVertexAttribArray: noop,
    vertexAttribPointer: noop,
    createTexture: () => ({}),
    bindTexture: noop,
    texParameteri: noop,
    texImage2D: noop,
    texSubImage2D: noop,
    deleteTexture: noop,
    createShader: () => ({}),
    shaderSource: noop,
    compileShader: noop,
    getShaderParameter: () => true,
    getShaderInfoLog: () => '',
    attachShader: noop,
    linkProgram: noop,
    getProgramParameter: () => true,
    getProgramInfoLog: () => '',
    createProgram: () => ({}),
    deleteShader: noop,
    deleteProgram: noop,
    getUniformLocation: () => ({}),
    uniform1i: noop,
    useProgram: noop,
    uniformMatrix3fv: noop,
    activeTexture: noop,
    uniform4f: noop,
    uniform1f: noop,
    enable: noop,
    disable: noop,
    blendFunc: noop,
    drawArrays: noop,
    RGBA: 0x1908,
    TEXTURE_2D: 0x0DE1,
    RG: 0x8227,
    UNSIGNED_BYTE: 0x1401,
    RED: 0x1903,
  };
}

function makeCanvasStub(webglStub, canvas2dStub) {
  const canvas = {
    width: IMG_WIDTH,
    height: IMG_HEIGHT,
    style: {},
    classList: { add: noop, remove: noop, contains: () => false },
    setAttribute: noop,
    getAttribute: () => null,
    addEventListener: noop,
    removeEventListener: noop,
    dispatchEvent: noop,
    getBoundingClientRect: () => ({ width: IMG_WIDTH, height: IMG_HEIGHT }),
    focus: noop,
  };
  canvas.getContext = (type) => {
    if (type === 'webgl2') return webglStub;
    if (type === '2d') return canvas2dStub;
    return null;
  };
  return canvas;
}

function makeElement(overrides = {}) {
  return Object.assign({
    style: { setProperty: noop, removeProperty: noop },
    classList: { add: noop, remove: noop, contains: () => false },
    addEventListener: noop,
    removeEventListener: noop,
    dispatchEvent: noop,
    setAttribute: noop,
    getAttribute: () => null,
    focus: noop,
    contains: () => false,
    getBoundingClientRect: () => ({ width: 100, height: 100 }),
    querySelector: () => null,
    querySelectorAll: () => [],
    appendChild: (child) => child,
    removeChild: noop,
  }, overrides);
}

function buildSandbox() {
  const canvas2dStub = makeCanvas2dStub();
  const webglStub = makeWebglStub();
  const mainCanvas = makeCanvasStub(webglStub, canvas2dStub);
  webglStub.canvas = mainCanvas;

  const sandbox = {
    console,
    performance: { now: () => 0 },
    requestAnimationFrame: (cb) => {
      if (typeof cb === 'function') {
        return setTimeout(cb, 0);
      }
      return 0;
    },
    cancelAnimationFrame: (handle) => {
      if (handle) clearTimeout(handle);
    },
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
    window: {},
    document: {},
    navigator: { userAgent: 'node' },
    WebGL2RenderingContext: function WebGL2RenderingContext() {},
  };

  const elements = {
    canvas: mainCanvas,
    viewer: makeElement(),
    dropOverlay: makeElement(),
    sidebar: makeElement(),
    leftPanel: makeElement(),
    brushPreview: makeCanvasStub(webglStub, canvas2dStub),
    maskCanvas: makeCanvasStub(webglStub, canvas2dStub),
    outlineCanvas: makeCanvasStub(webglStub, canvas2dStub),
  };

  sandbox.window = sandbox;
  sandbox.globalThis = sandbox;
  sandbox.window.Buffer = Buffer;
  sandbox.window.WebGL2RenderingContext = sandbox.WebGL2RenderingContext;
  sandbox.window.__OMNI_FORCE_GRID_MASK__ = true;
  sandbox.window.__OMNI_SKIP_GRID_REPAINT__ = false;
  sandbox.window.__OMNI_CONFIG__ = {
    width: IMG_WIDTH,
    height: IMG_HEIGHT,
    colorTable: [],
  };
  sandbox.window.devicePixelRatio = 1;
  sandbox.window.getComputedStyle = () => ({ getPropertyValue: () => '0' });

  const ImageCtor = function Image() {
    this.complete = true;
    this.width = IMG_WIDTH;
    this.height = IMG_HEIGHT;
  };
  ImageCtor.prototype.addEventListener = noop;
  sandbox.Image = ImageCtor;

  sandbox.document = {
    documentElement: makeElement(),
    getElementById: (id) => elements[id] || makeElement(),
    createElement: () => makeCanvasStub(webglStub, canvas2dStub),
    body: makeElement(),
    addEventListener: noop,
    removeEventListener: noop,
    readyState: 'complete',
    querySelectorAll: () => [],
  };

  sandbox.window.addEventListener = noop;
  sandbox.window.removeEventListener = noop;
  sandbox.window.matchMedia = () => ({ matches: false, addEventListener: noop, removeEventListener: noop });
  sandbox.window.__omniLogPush = noop;
  sandbox.window.localStorage = { getItem: () => null, setItem: noop, removeItem: noop };
  sandbox.window.history = { replaceState: noop };
  sandbox.window.CustomEvent = function CustomEvent() {};
  sandbox.window.MutationObserver = function MutationObserver() { this.observe = noop; this.disconnect = noop; };
  sandbox.window.ResizeObserver = function ResizeObserver() { this.observe = noop; this.disconnect = noop; };
  sandbox.window.IntersectionObserver = function IntersectionObserver() { this.observe = noop; this.disconnect = noop; };
  sandbox.window.Map = Map;
  sandbox.window.Set = Set;
  sandbox.window.WeakMap = WeakMap;
  sandbox.window.OmniPointer = {
    POINTER_OPTIONS: { stylus: {}, touch: {}, mouse: {} },
    createPointerState: () => ({}),
  };
  sandbox.window.OmniLog = { log: noop, warn: noop, error: noop };
  sandbox.window.OmniHistory = { init: noop, push: noop, clear: noop, serialize: () => ({ undo: [], redo: [] }), restore: noop };
  sandbox.window.OmniBrush = {
    init: noop,
    resizePreviewCanvas: noop,
    drawBrushPreview: noop,
    collectBrushIndices: (target, x, y) => collectDiskIndices(target, x, y, 4),
  };
  sandbox.window.OmniInteractions = { init: noop };
  sandbox.window.__webglStub = webglStub;
  sandbox.window.__canvas2dStub = canvas2dStub;

  vm.createContext(sandbox);

  const paintingCode = fs.readFileSync(path.resolve(__dirname, '../../gui/web/js/painting.js'), 'utf8');
  new vm.Script(paintingCode, { filename: 'painting.js' }).runInContext(sandbox);

  const appCode = fs.readFileSync(path.resolve(__dirname, '../../gui/web/app.js'), 'utf8');
  new vm.Script(appCode, { filename: 'app.js' }).runInContext(sandbox);

  return sandbox;
}

async function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function main() {
  const sandbox = buildSandbox();
  const painting = sandbox.window.OmniPainting;
  if (!painting || typeof painting.__debugGetState !== 'function') {
    throw new Error('failed to load painting from app context');
  }
  const state = painting.__debugGetState();
  const ctx = state.ctx;
  const dims = ctx.getImageDimensions();
  const width = dims.width | 0;
  const height = dims.height | 0;

  if (typeof painting.__debugApplyGridIfNeeded === 'function') {
    painting.__debugApplyGridIfNeeded(true);
  }
  // Ensure the grid has been applied and the async finalize ran.
  await delay(50);

  // Simulate a user stroke across the grid to mirror manual repro steps.
  if (typeof ctx.setCurrentLabel === 'function') {
    ctx.setCurrentLabel(1);
  }
  const strokePoints = [];
  const strokeY = Math.floor(height * 0.5);
  for (let x = 10; x < width - 10; x += 8) {
    strokePoints.push({ x, y: strokeY + Math.sin(x * 0.05) * 5 });
  }
  if (strokePoints.length >= 2 && typeof painting.beginStroke === 'function') {
    painting.beginStroke(strokePoints[0]);
    for (let i = 1; i < strokePoints.length; i += 1) {
      painting.queuePaintPoint(strokePoints[i]);
    }
    if (typeof painting.processPaintQueue === 'function') {
      painting.processPaintQueue();
    }
    if (typeof painting.finalizeStroke === 'function') {
      painting.finalizeStroke();
    }
  }
  await delay(20);

  const indicesToVisit = [];
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = y * width + x;
      if ((ctx.maskValues[idx] | 0) === 0) {
        indicesToVisit.push(idx);
      }
    }
  }

  const visited = new Set();
  let fills = 0;

  function floodComponent(startIdx) {
    const component = [];
    const mask = ctx.maskValues;
    const queue = [startIdx];
    const seen = new Set(queue);
    while (queue.length) {
      const idx = queue.pop();
      if ((mask[idx] | 0) !== 0) continue;
      component.push(idx);
      const x = idx % width;
      const y = (idx / width) | 0;
      const neighbors = [];
      if (y > 0) neighbors.push(idx - width);
      if (y < height - 1) neighbors.push(idx + width);
      if (x > 0) neighbors.push(idx - 1);
      if (x < width - 1) neighbors.push(idx + 1);
      for (const n of neighbors) {
        if (!seen.has(n)) {
          seen.add(n);
          queue.push(n);
        }
      }
    }
    return component;
  }

  for (const idx of indicesToVisit) {
      if ((ctx.maskValues[idx] | 0) !== 0 || visited.has(idx)) {
        continue;
      }
      const component = floodComponent(idx);
    component.forEach((i) => visited.add(i));
    const x = idx % width;
    const y = (idx / width) | 0;
    ctx.setCurrentLabel(1);
    painting.floodFill({ x, y });
    await delay(5);
    let failures = 0;
    for (const pixel of component) {
      if ((ctx.maskValues[pixel] | 0) !== 1) {
        failures += 1;
      }
    }
    if (failures > 0) {
      console.error('Fill failure detected', {
        fills,
        x,
        y,
        componentSize: component.length,
        failures,
        fillResult: painting.__debugGetState().lastFillResult,
      });
      process.exit(1);
    }
    fills += 1;
  }

  console.log(JSON.stringify({ fills }));
}

main().catch((err) => {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
