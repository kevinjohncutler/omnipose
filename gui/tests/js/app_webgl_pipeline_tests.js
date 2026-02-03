const fs = require('fs');
const path = require('path');
const vm = require('vm');

function noop() {}

const IMG_WIDTH = 392;
const IMG_HEIGHT = 384;

const glConstants = {
  ARRAY_BUFFER: 0x8892,
  STATIC_DRAW: 0x88e4,
  TEXTURE_2D: 0x0de1,
  RGBA: 0x1908,
  RG: 0x8227,
  RED: 0x1903,
  UNSIGNED_BYTE: 0x1401,
  NEAREST: 0x2600,
  LINEAR: 0x2601,
  CLAMP_TO_EDGE: 0x812f,
  RG8: 0x822b,
  R8: 0x8229,
  RGBA8: 0x8058,
  FLOAT: 0x1406,
  TRIANGLE_STRIP: 0x0005,
  LINES: 0x0001,
  COLOR_BUFFER_BIT: 0x4000,
  FRAMEBUFFER: 0x8d40,
  BLEND: 0x0be2,
  SRC_ALPHA: 0x0302,
  ONE_MINUS_SRC_ALPHA: 0x0303,
  UNPACK_ALIGNMENT: 0x0cf5,
  UNPACK_FLIP_Y_WEBGL: 0x9240,
};

function makeStyle() {
  return {
    setProperty: noop,
    removeProperty: noop,
  };
}

function makeClassList() {
  return {
    add: noop,
    remove: noop,
    contains: () => false,
  };
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
    strokeStyle: '',
    lineWidth: 1,
    createLinearGradient: () => ({ addColorStop: noop }),
  };
}

function makeWebglStub() {
  return Object.assign({
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
    detachShader: noop,
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
    createFramebuffer: () => ({}),
    bindFramebuffer: noop,
    framebufferTexture2D: noop,
    deleteFramebuffer: noop,
    readBuffer: noop,
    readPixels: noop,
    deleteVertexArray: noop,
    deleteBuffer: noop,
  }, glConstants);
}

function makeCanvasStub(webglStub, canvas2dStub) {
  const canvas = {
    width: IMG_WIDTH,
    height: IMG_HEIGHT,
    style: makeStyle(),
    classList: makeClassList(),
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
    style: makeStyle(),
    classList: makeClassList(),
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

function makeImageStub() {
  return function ImageStub() {
    this.width = IMG_WIDTH;
    this.height = IMG_HEIGHT;
    this.complete = true;
    this._listeners = {};
  };
}

function attachImagePrototype(ImageCtor) {
  ImageCtor.prototype.addEventListener = function addEventListener(type, handler) {
    this._listeners[type] = handler;
    if (type === 'load' && this.complete && typeof handler === 'function') {
      handler({ target: this });
    }
  };
  ImageCtor.prototype.removeEventListener = function removeEventListener(type) {
    delete this._listeners[type];
  };
  Object.defineProperty(ImageCtor.prototype, 'src', {
    set(value) {
      this._src = value;
      const handler = this._listeners.load;
      if (typeof handler === 'function') {
        handler({ target: this });
      }
    },
    get() {
      return this._src;
    },
  });
}

function buildSandbox() {
  const canvas2dStub = makeCanvas2dStub();
  const webglStub = makeWebglStub();
  const mainCanvas = makeCanvasStub(webglStub, canvas2dStub);
  webglStub.canvas = mainCanvas;

  const sandbox = {
    console,
    performance: { now: () => 0 },
    requestAnimationFrame: (cb) => { if (typeof cb === 'function') { return globalThis.setTimeout(cb, 0); } return 0; },
    cancelAnimationFrame: (handle) => { if (handle) { globalThis.clearTimeout(handle); } },
    setTimeout: (cb, delay) => { if (typeof cb === 'function') { return globalThis.setTimeout(cb, delay || 0); } return 0; },
    clearTimeout: (handle) => { if (handle) { globalThis.clearTimeout(handle); } },
    Uint8Array,
    Uint16Array,
    Uint32Array,
    Uint8ClampedArray,
    Float32Array,
    Array,
    Math,
    Date,
    Buffer,
    atob: (str) => Buffer.from(str, 'base64').toString('binary'),
    btoa: (str) => Buffer.from(str, 'binary').toString('base64'),
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
    labelValueInput: makeElement({ value: '' }),
  };

  sandbox.window = sandbox;
  sandbox.__OMNI_SKIP_GRID_REPAINT__ = true;
  sandbox.window.__OMNI_SKIP_GRID_REPAINT__ = true;
  sandbox.globalThis = sandbox;
  sandbox.window.Buffer = Buffer;
  sandbox.window.WebGL2RenderingContext = sandbox.WebGL2RenderingContext;
  sandbox.window.__OMNI_CONFIG__ = {
    width: IMG_WIDTH,
    height: IMG_HEIGHT,
    colorTable: [],
  };
  sandbox.window.devicePixelRatio = 1;
  sandbox.window.getComputedStyle = () => ({ getPropertyValue: () => '0' });

  const ImageCtor = makeImageStub();
  attachImagePrototype(ImageCtor);
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
  sandbox.window.OmniHistory = { init: noop, push: noop, clear: noop };
  sandbox.window.OmniBrush = {
    init: noop,
    resizePreviewCanvas: noop,
    drawBrushPreview: noop,
    collectBrushIndices: noop,
  };
  sandbox.window.OmniInteractions = { init: noop };
  sandbox.window.__webglStub = webglStub;
  sandbox.window.__canvas2dStub = canvas2dStub;

  vm.createContext(sandbox);

  const paintingPath = path.resolve(__dirname, '../../gui/web/js/painting.js');
  const paintingCode = fs.readFileSync(paintingPath, 'utf8');
  new vm.Script(paintingCode, { filename: 'painting.js' }).runInContext(sandbox);

  const appPath = path.resolve(__dirname, '../../gui/web/app.js');
  const appCode = fs.readFileSync(appPath, 'utf8');
  new vm.Script(appCode, { filename: 'app.js' }).runInContext(sandbox);

  return sandbox;
}

async function runWebglStartupFillTest() {
  const sandbox = buildSandbox();
  const painting = sandbox.window.OmniPainting;
  const debug = sandbox.window.__OMNI_DEBUG__;
  if (!painting || !debug) {
    throw new Error('failed to load painting or debug helpers');
  }

  const state = painting.__debugGetState();
  const ctx = state.ctx;
  if (!ctx || typeof ctx.setCurrentLabel !== 'function') {
    throw new Error('painting context unavailable');
  }

  ctx.setCurrentLabel(1);
  if (typeof ctx.getMaskHasNonZero === 'function' && ctx.getMaskHasNonZero()) {
    ctx.setMaskHasNonZero(false);
  }

  if (typeof debug.resetWebglPipeline === 'function') {
    debug.resetWebglPipeline();
  }
  debug.resetCounters();

  painting.floodFill({ x: 0, y: 0 });

  debug.resetCounters();
  debug.initializeWebglPipelineResources();
  const counters = debug.getCounters();
  if (!counters || counters.draw <= 0) {
    throw new Error('initializeWebglPipelineResources should trigger an immediate draw after WebGL becomes ready');
  }
  if (!debug.isWebglPipelineActive()) {
    throw new Error('WebGL pipeline not active after initialization');
  }
  if (typeof debug.getAffinityInfo === 'function') {
    await new Promise((resolve) => setTimeout(resolve, 0));
    const info = debug.getAffinityInfo();
    if (!info || !info.hasGraph || !info.nonZeroEdges) {
      console.error('affinity-info', info, 'finalizeRuns', debug.gridFinalizeRuns || 0);
      throw new Error('Affinity graph missing or empty after debug grid seeding');
    }
    if (!info.showAffinityGraph) {
      throw new Error('Affinity overlay disabled unexpectedly');
    }
  }
}


async function main() {
  try {
    await runWebglStartupFillTest();
    process.stdout.write(JSON.stringify({ success: true }));
  } catch (err) {
    const message = err && err.stack ? err.stack : String(err);
    console.error(message);
    process.exit(1);
  }
}

main();
