const CONFIG = window.__OMNI_CONFIG__ || {};
const imgWidth = CONFIG.width || 0;
const imgHeight = CONFIG.height || 0;
const imageDataUrl = CONFIG.imageDataUrl || '';
const colorTable = CONFIG.colorTable || [];
const initialBrushRadius = CONFIG.brushRadius ?? 6;
let defaultPalette = [];
let nColorPalette = [];

// Centralized input policy so stylus, touch, and mouse behavior can be tuned per platform.
const POINTER_OPTIONS = {
  stylus: {
    allowSimultaneousTouchGestures: false,
    barrelButtonPan: true,
  },
  touch: {
    enablePan: true,
    enablePinchZoom: true,
    enableRotate: true,
    rotationDeadzoneDegrees: 0.1,
  },
  mouse: {
    primaryDraw: true,
    secondaryPan: true,
  },
};

const RAD_TO_DEG = 180 / Math.PI;

function createPointerState(options = {}) {
  const merged = {
    stylus: {
      allowSimultaneousTouchGestures: false,
      barrelButtonPan: true,
      ...(options.stylus || {}),
    },
    touch: {
      enablePan: true,
      enablePinchZoom: true,
      enableRotate: true,
      rotationDeadzoneDegrees: 0.1,
      ...(options.touch || {}),
    },
    mouse: {
      primaryDraw: true,
      secondaryPan: true,
      ...(options.mouse || {}),
    },
  };
  let activePenId = null;
  let penButtons = 0;
  let penBarrelPan = false;

  function isStylusPointer(evt) {
    if (!evt) {
      return false;
    }
    if (evt.pointerType === 'pen') {
      return true;
    }
    if (evt.pointerType === 'touch' && typeof evt.touchType === 'string') {
      return evt.touchType.toLowerCase() === 'stylus';
    }
    return false;
  }

  function registerPenButtons(evt) {
    penButtons = typeof evt.buttons === 'number' ? evt.buttons : 0;
    penBarrelPan = merged.stylus.barrelButtonPan && (penButtons & ~1) !== 0;
  }

  return {
    options: merged,
    isStylusPointer,
    registerPointerDown(evt) {
      if (isStylusPointer(evt)) {
        activePenId = evt.pointerId;
        registerPenButtons(evt);
      }
    },
    registerPointerMove(evt) {
      if (isStylusPointer(evt) && evt.pointerId === activePenId) {
        registerPenButtons(evt);
      }
    },
    registerPointerUp(evt) {
      if (isStylusPointer(evt) && evt.pointerId === activePenId) {
        registerPenButtons(evt);
        activePenId = null;
        penBarrelPan = false;
        penButtons = 0;
      }
    },
    isActivePen(pointerId) {
      return activePenId !== null && pointerId === activePenId;
    },
    hasActivePen() {
      return activePenId !== null;
    },
    isBarrelPanActive() {
      return penBarrelPan;
    },
    shouldIgnoreTouch(evt) {
      if (!evt || evt.pointerType !== 'touch') {
        return false;
      }
      if (!merged.stylus.allowSimultaneousTouchGestures && activePenId !== null) {
        return true;
      }
      return false;
    },
    resetPen() {
      activePenId = null;
      penButtons = 0;
      penBarrelPan = false;
    },
  };
}

const supportsGestureEvents = typeof window !== 'undefined'
  && (typeof window.GestureEvent === 'function' || 'ongesturestart' in window);
const pointerOptionsOverride = CONFIG.pointerOptions || {};
const pointerState = createPointerState({
  stylus: { ...POINTER_OPTIONS.stylus, ...(pointerOptionsOverride.stylus || {}) },
  touch: { ...POINTER_OPTIONS.touch, ...(pointerOptionsOverride.touch || {}) },
  mouse: { ...POINTER_OPTIONS.mouse, ...(pointerOptionsOverride.mouse || {}) },
});
const debugTouchOverlay = CONFIG.debugTouchOverlay ?? false;

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const viewer = document.getElementById('viewer');
const dpr = window.devicePixelRatio || 1;
const rootStyle = window.getComputedStyle(document.documentElement);
const sidebarWidthRaw = rootStyle.getPropertyValue('--sidebar-width');
const sidebarWidthValue = Number.parseFloat(sidebarWidthRaw || '');
const sidebarWidth = Number.isFinite(sidebarWidthValue) ? Math.max(0, sidebarWidthValue) : 260;
const accentColor = (rootStyle.getPropertyValue('--accent-color') || '#d8a200').trim();
const histogramWindowColor = (rootStyle.getPropertyValue('--histogram-window-color') || 'rgba(140, 140, 140, 0.35)').trim();
const panelTextColor = (rootStyle.getPropertyValue('--panel-text-color') || '#f4f4f4').trim();

const offscreen = document.createElement('canvas');
offscreen.width = imgWidth;
offscreen.height = imgHeight;
const offCtx = offscreen.getContext('2d');

const maskCanvas = document.createElement('canvas');
maskCanvas.width = imgWidth;
maskCanvas.height = imgHeight;
const maskCtx = maskCanvas.getContext('2d');
const maskData = maskCtx.createImageData(imgWidth, imgHeight);
const maskValues = new Uint8Array(imgWidth * imgHeight);
const previewCanvas = document.getElementById('brushPreview');
const previewCtx = previewCanvas.getContext('2d');
previewCanvas.width = canvas.width;
previewCanvas.height = canvas.height;

const loadingOverlay = null;
const loadingOverlayMessage = null;
let overlayDismissed = true;

function setLoadingOverlay() {}

function hideLoadingOverlay() {}

const pendingLogs = [];
let pywebviewReady = false;
let loggedPixelSample = false;
let drawLogCount = 0;
let logFlushTimer = null;
const startTime = (typeof performance !== 'undefined' && performance.now)
  ? performance.now()
  : Date.now();
let shuttingDown = false;
const pywebviewPoll = setInterval(() => {
  if (pywebviewReady) {
    clearInterval(pywebviewPoll);
    return;
  }
  const api = window.pywebview ? window.pywebview.api : null;
  if (api && api.log) {
    pywebviewReady = true;
    clearInterval(pywebviewPoll);
    log('pywebview api detected via poll');
    flushLogs();
  }
}, 50);

window.addEventListener('error', (evt) => {
  try {
    const message = evt && evt.message ? evt.message : String(evt);
    log('uncaught error: ' + message);
  } catch (_) {
    /* ignore */
  }
});

function pushToQueue(msg) {
  pendingLogs.push(msg);
  if (pendingLogs.length > 200) {
    pendingLogs.shift();
  }
  scheduleLogFlush();
}

function flushLogs() {
  if (logFlushTimer !== null) {
    clearTimeout(logFlushTimer);
    logFlushTimer = null;
  }
  if (!pendingLogs.length) {
    return;
  }
  if (pywebviewReady) {
    const api = window.pywebview && window.pywebview.api && window.pywebview.api.log
      ? window.pywebview.api
      : null;
    if (!api || !api.log) {
      return;
    }
    while (pendingLogs.length) {
      api.log(pendingLogs.shift());
    }
    return;
  }
  if (typeof fetch !== 'function') {
    return;
  }
  const payload = { messages: pendingLogs.splice(0, pendingLogs.length) };
  const body = JSON.stringify(payload);
  try {
    if (navigator && typeof navigator.sendBeacon === 'function') {
      const ok = navigator.sendBeacon('/api/log', new Blob([body], { type: 'application/json' }));
      if (!ok) {
        pendingLogs.unshift(...payload.messages);
        scheduleLogFlush();
      }
    } else {
      fetch('/api/log', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
        keepalive: true,
      }).catch(() => {
        pendingLogs.unshift(...payload.messages);
        scheduleLogFlush();
      });
    }
  } catch (_) {
    pendingLogs.unshift(...payload.messages);
    scheduleLogFlush();
  }
}

function scheduleLogFlush(delay = 200) {
  if (logFlushTimer !== null) {
    return;
  }
  logFlushTimer = setTimeout(() => {
    logFlushTimer = null;
    flushLogs();
  }, delay);
}

function formatMessage(message) {
  const now = (typeof performance !== 'undefined' && performance.now)
    ? (performance.now() - startTime)
    : (Date.now() - startTime);
  const timestamp = now.toFixed(1).padStart(7, ' ');
  return '[' + timestamp + ' ms] ' + message;
}

function log(message) {
  const formatted = formatMessage(String(message));
  try {
    console.log('[pywebview]', formatted);
  } catch (_) {
    /* console unavailable */
  }
  const api = window.pywebview ? window.pywebview.api : null;
  if (api && api.log) {
    api.log(formatted);
  } else {
    pushToQueue(formatted);
  }
}

function clearHoverPreview() {
  hoverPoint = null;
  drawBrushPreview(null);
  updateHoverInfo(null);
}

const sliderRegistry = new Map();
const dropdownRegistry = new Map();
let dropdownOpenId = null;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function valueToPercent(input) {
  const min = Number(input.min || 0);
  const max = Number(input.max || 1);
  const span = max - min;
  if (!Number.isFinite(span) || span === 0) {
    return 0;
  }
  const value = Number(input.value || min);
  return clamp((value - min) / span, 0, 1);
}

function percentToValue(percent, input) {
  const min = Number(input.min || 0);
  const max = Number(input.max || 1);
  const span = max - min;
  const raw = min + clamp(percent, 0, 1) * span;
  const step = Number(input.step || '1');
  if (!Number.isFinite(step) || step <= 0) {
    return clamp(raw, min, max);
  }
  const snapped = Math.round((raw - min) / step) * step + min;
  const precision = (step.toString().split('.')[1] || '').length;
  const factor = 10 ** precision;
  return clamp(Math.round(snapped * factor) / factor, min, max);
}

function pointerPercent(evt, container) {
  const rect = container.getBoundingClientRect();
  if (rect.width <= 0) {
    return 0;
  }
  const ratio = (evt.clientX - rect.left) / rect.width;
  return clamp(ratio, 0, 1);
}

function registerSlider(root) {
  const id = root.dataset.sliderId || root.dataset.slider || root.id;
  if (!id) {
    return;
  }
  const type = (root.dataset.sliderType || 'single').toLowerCase();
  const inputs = Array.from(root.querySelectorAll('input[type="range"]'));
  if (!inputs.length) {
    return;
  }
  if (type === 'dual' && inputs.length < 2) {
    console.warn(`slider ${id} configured as dual but only one range input found`);
    return;
  }

  root.innerHTML = '';
  const track = document.createElement('div');
  track.className = 'slider-track';
  const fill = document.createElement('div');
  fill.className = 'slider-fill';
  root.appendChild(track);
  root.appendChild(fill);

  inputs.forEach((input) => {
    input.classList.add('slider-input');
    input.setAttribute('aria-hidden', 'true');
    root.appendChild(input);
  });

  const thumbs = inputs.map((_, idx) => {
    const thumb = document.createElement('div');
    thumb.className = 'slider-thumb';
    thumb.dataset.index = String(idx);
    root.appendChild(thumb);
    return thumb;
  });

  const entry = {
    id,
    type: type === 'dual' ? 'dual' : 'single',
    root,
    inputs,
    track,
    fill,
    thumbs,
    activePointer: null,
    activeThumb: null,
  };

  const apply = () => {
    if (entry.type === 'dual') {
      const minInput = entry.inputs[0];
      const maxInput = entry.inputs[1];
      let minValue = Number(minInput.value);
      let maxValue = Number(maxInput.value);
      if (minValue > maxValue) {
        const temp = minValue;
        minValue = maxValue;
        maxValue = temp;
        minInput.value = String(minValue);
        maxInput.value = String(maxValue);
      }
      const minPercent = valueToPercent(minInput);
      const maxPercent = valueToPercent(maxInput);
      const left = (minPercent * 100).toFixed(3) + '%';
      const rightPercent = (maxPercent * 100).toFixed(3) + '%';
      const width = Math.max(0, (maxPercent - minPercent) * 100).toFixed(3) + '%';
      entry.fill.style.left = left;
      entry.fill.style.width = width;
      entry.thumbs[0].style.left = left;
      entry.thumbs[1].style.left = rightPercent;
    } else {
      const input = entry.inputs[0];
      const percent = valueToPercent(input);
      const position = (percent * 100).toFixed(3) + '%';
      entry.fill.style.left = '0%';
      entry.fill.style.width = position;
      entry.thumbs[0].style.left = position;
    }
  };

  const setValueFromPercent = (index, percent) => {
    const input = entry.inputs[index];
    if (!input) {
      return;
    }
    let value = percentToValue(percent, input);
    if (entry.type === 'dual') {
      if (index === 0) {
        const other = Number(entry.inputs[1].value);
        if (value > other) {
          value = other;
        }
      } else {
        const other = Number(entry.inputs[0].value);
        if (value < other) {
          value = other;
        }
      }
    }
    input.value = String(value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
    apply();
  };

  const pickThumb = (percent) => {
    if (entry.type !== 'dual') {
      return 0;
    }
    const distances = entry.inputs.map((input) => Math.abs(percent - valueToPercent(input)));
    let bestIndex = 0;
    let bestDistance = distances[0];
    for (let i = 1; i < distances.length; i += 1) {
      if (distances[i] < bestDistance) {
        bestDistance = distances[i];
        bestIndex = i;
      }
    }
    return bestIndex;
  };

  const onPointerDown = (evt) => {
    evt.preventDefault();
    const percent = pointerPercent(evt, entry.root);
    const thumbIndex = entry.type === 'dual' ? pickThumb(percent) : 0;
    entry.activePointer = evt.pointerId;
    entry.activeThumb = thumbIndex;
    entry.root.setPointerCapture(entry.activePointer);
    entry.root.dataset.active = 'true';
    const targetInput = entry.inputs[thumbIndex];
    if (targetInput) {
      targetInput.focus();
    }
    setValueFromPercent(thumbIndex, percent);
  };

  const onPointerMove = (evt) => {
    if (entry.activePointer === null || evt.pointerId !== entry.activePointer) {
      return;
    }
    const percent = pointerPercent(evt, entry.root);
    setValueFromPercent(entry.activeThumb ?? 0, percent);
  };

  const onPointerRelease = (evt) => {
    if (entry.activePointer === null || evt.pointerId !== entry.activePointer) {
      return;
    }
    try {
      entry.root.releasePointerCapture(entry.activePointer);
    } catch (_) {
      /* ignore */
    }
    const percent = pointerPercent(evt, entry.root);
    setValueFromPercent(entry.activeThumb ?? 0, percent);
    entry.activePointer = null;
    entry.activeThumb = null;
    entry.root.dataset.active = 'false';
  };

  entry.root.addEventListener('pointerdown', onPointerDown);
  entry.root.addEventListener('pointermove', onPointerMove);
  entry.root.addEventListener('pointerup', onPointerRelease);
  entry.root.addEventListener('pointercancel', onPointerRelease);

  inputs.forEach((input) => {
    input.addEventListener('focus', () => {
      entry.root.dataset.focused = 'true';
    });
    input.addEventListener('blur', () => {
      entry.root.dataset.focused = 'false';
    });
    input.addEventListener('input', apply);
    input.addEventListener('change', apply);
  });

  entry.apply = apply;
  apply();
  sliderRegistry.set(id, entry);
}

function refreshSlider(id) {
  const entry = sliderRegistry.get(id);
  if (entry && typeof entry.apply === 'function') {
    entry.apply();
  }
}

function attachNumberInputStepper(input, onAdjust) {
  if (!input || typeof onAdjust !== 'function') {
    return;
  }
  input.addEventListener('keydown', (evt) => {
    if (evt.key !== 'ArrowUp' && evt.key !== 'ArrowDown') {
      return;
    }
    evt.preventDefault();
    const base = Number(input.step || '1');
    const step = Number.isFinite(base) && base > 0 ? base : 1;
    const factor = evt.shiftKey ? 5 : 1;
    const direction = evt.key === 'ArrowUp' ? 1 : -1;
    onAdjust(step * factor * direction);
  });
}

function closeDropdown(entry) {
  if (!entry) {
    return;
  }
  entry.root.dataset.open = 'false';
  entry.button.setAttribute('aria-expanded', 'false');
  if (entry.menu) {
    entry.menu.setAttribute('aria-hidden', 'true');
    entry.menu.scrollTop = 0;
  }
  dropdownOpenId = null;
}

function openDropdown(entry) {
  if (!entry) {
    return;
  }
  if (dropdownOpenId && dropdownOpenId !== entry.id) {
    closeDropdown(dropdownRegistry.get(dropdownOpenId));
  }
  entry.root.dataset.open = 'true';
  positionDropdown(entry);
  entry.button.setAttribute('aria-expanded', 'true');
  if (entry.menu) {
    entry.menu.setAttribute('aria-hidden', 'false');
  }
  dropdownOpenId = entry.id;
}

function toggleDropdown(entry) {
  if (!entry) {
    return;
  }
  const isOpen = entry.root.dataset.open === 'true';
  if (isOpen) {
    closeDropdown(entry);
  } else {
    openDropdown(entry);
  }
}

function positionDropdown(entry) {
  if (!entry || !entry.menu) {
    return;
  }
  entry.menu.style.minWidth = '100%';
}

function registerDropdown(root) {
  const select = root.querySelector('select');
  if (!select) {
    return;
  }
  const id = root.dataset.dropdownId || select.id || `dropdown-${dropdownRegistry.size}`;
  root.dataset.dropdownId = id;
  root.dataset.open = root.dataset.open || 'false';

  const originalOptions = Array.from(select.options).map((opt) => ({
    value: opt.value,
    label: opt.textContent || opt.value,
    disabled: opt.disabled,
  }));

  select.classList.add('dropdown-input');
  root.innerHTML = '';
  root.appendChild(select);

  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'dropdown-toggle';
  button.setAttribute('aria-haspopup', 'listbox');
  button.setAttribute('aria-expanded', 'false');
  const menu = document.createElement('div');
  menu.className = 'dropdown-menu';
  menu.setAttribute('role', 'listbox');
  menu.setAttribute('aria-hidden', 'true');
  menu.id = `${id}-menu`;
  button.setAttribute('aria-controls', menu.id);
  root.appendChild(button);
  const menuWrapper = document.createElement('div');
  menuWrapper.className = 'dropdown-menu-wrap';
  menuWrapper.appendChild(menu);
  root.appendChild(menuWrapper);

  const entry = {
    id,
    root,
    select,
    button,
    menu,
    menuWrapper,
    options: originalOptions,
  };

  const applySelection = () => {
    const selectedOption = select.options[select.selectedIndex];
    button.textContent = selectedOption ? selectedOption.textContent : 'Select';
    menu.querySelectorAll('.dropdown-option').forEach((child) => {
      child.dataset.selected = child.dataset.value === select.value ? 'true' : 'false';
    });
  };

  const buildMenu = () => {
    menu.innerHTML = '';
    entry.options.forEach((opt) => {
      const item = document.createElement('div');
      item.className = 'dropdown-option';
      item.dataset.value = opt.value;
      item.textContent = opt.label;
      item.setAttribute('role', 'option');
      if (opt.disabled) {
        item.setAttribute('aria-disabled', 'true');
        item.style.opacity = '0.45';
        item.style.pointerEvents = 'none';
      }
      item.addEventListener('pointerdown', (evt) => {
        evt.preventDefault();
        if (opt.disabled) {
          return;
        }
        select.value = opt.value;
        select.dispatchEvent(new Event('change', { bubbles: true }));
        applySelection();
        closeDropdown(entry);
      });
      menu.appendChild(item);
    });
    applySelection();
  };

  button.addEventListener('click', () => {
    toggleDropdown(entry);
  });

  select.addEventListener('change', () => {
    applySelection();
  });

  buildMenu();
  entry.applySelection = applySelection;
  entry.buildMenu = buildMenu;
  positionDropdown(entry);
  dropdownRegistry.set(id, entry);
}

document.addEventListener('pointerdown', (evt) => {
  if (!dropdownOpenId) {
    return;
  }
  const entry = dropdownRegistry.get(dropdownOpenId);
  if (!entry) {
    dropdownOpenId = null;
    return;
  }
  if (!entry.root.contains(evt.target)) {
    closeDropdown(entry);
  }
});

function refreshDropdown(id) {
  const entry = dropdownRegistry.get(id);
  if (entry && typeof entry.applySelection === 'function') {
    entry.applySelection();
  }
}

window.addEventListener('blur', () => {
  touchPointers.clear();
  pinchState = null;
  panPointerId = null;
  pointerState.resetPen();
  stylusToolOverride = null;
  gestureState = null;
  wheelRotationBuffer = 0;
  clearHoverPreview();
  if (dropdownOpenId) {
    closeDropdown(dropdownRegistry.get(dropdownOpenId));
  }
});

document.addEventListener('mouseleave', (evt) => {
  if (!evt.relatedTarget && !evt.toElement) {
    clearHoverPreview();
  }
});

function currentTouchPoints() {
  return Array.from(touchPointers.entries()).map(([id, data]) => ({ id, x: data.x, y: data.y }));
}

function beginPinchGesture() {
  if (touchPointers.size < 2) {
    return;
  }
  const rect = canvas.getBoundingClientRect();
  const points = currentTouchPoints().slice(0, 2);
  const dx = points[0].x - points[1].x;
  const dy = points[0].y - points[1].y;
  const distance = Math.hypot(dx, dy) || 1;
  const angle = Math.atan2(dy, dx);
  const midpoint = {
    x: (points[0].x + points[1].x) / 2,
    y: (points[0].y + points[1].y) / 2,
  };
  const canvasMid = {
    x: midpoint.x - rect.left,
    y: midpoint.y - rect.top,
  };
  const imageMid = screenToImage(canvasMid);
  pinchState = {
    pointers: [points[0].id, points[1].id],
    startDistance: distance,
    startScale: viewState.scale,
    startAngle: angle,
    startRotation: normalizeAngle(viewState.rotation),
    imageMid,
    lastLoggedRotation: 0,
  };
  log('pinch gesture start distance=' + distance.toFixed(2) + ' angle=' + (angle * RAD_TO_DEG).toFixed(2) + ' deg');
  wheelRotationBuffer = 0;
  isPanning = false;
  isPainting = false;
  hoverPoint = null;
  drawBrushPreview(null);
}

function handleTouchPointerDown(evt, pointer) {
  if (pointerState.shouldIgnoreTouch(evt)) {
    evt.preventDefault();
    return;
  }
  evt.preventDefault();
  touchPointers.set(evt.pointerId, { x: evt.clientX, y: evt.clientY });
  try {
    canvas.setPointerCapture(evt.pointerId);
  } catch (_) {
    /* ignore */
  }
  if (pinchState) {
    // already in pinch gesture, ignore additional touches
    return;
  }
  if (touchPointers.size >= 2) {
    if (!pointerState.options.touch.enablePinchZoom && !pointerState.options.touch.enableRotate) {
      return;
    }
    if (panPointerId !== null) {
      try {
        canvas.releasePointerCapture(panPointerId);
      } catch (_) {
        /* ignore */
      }
      panPointerId = null;
    }
    beginPinchGesture();
    return;
  }
  // single finger: treat as pan gesture
  if (!pointerState.options.touch.enablePan) {
    return;
  }
  isPainting = false;
  strokeChanges = null;
  isPanning = true;
  panPointerId = evt.pointerId;
  try {
    canvas.setPointerCapture(evt.pointerId);
  } catch (_) {
    /* ignore */
  }
  hoverPoint = null;
  drawBrushPreview(null);
  updateHoverInfo(null);
  lastPoint = pointer;
  updateCursor();
  drawBrushPreview(hoverPoint);
}

function handleGestureStart(evt) {
  if (!pointerState.options.touch.enableRotate && !pointerState.options.touch.enablePinchZoom) {
    return;
  }
  if (!canvas) {
    return;
  }
  evt.preventDefault();
  const origin = resolveGestureOrigin(evt);
  const imagePoint = screenToImage(origin);
  gestureState = {
    startScale: viewState.scale,
    startRotation: normalizeAngle(viewState.rotation),
    origin,
    imagePoint,
  };
  wheelRotationBuffer = 0;
  hoverPoint = null;
  drawBrushPreview(null);
  isPanning = false;
  spacePan = false;
}

function handleGestureChange(evt) {
  if (!gestureState) {
    return;
  }
  evt.preventDefault();
  const currentOrigin = resolveGestureOrigin(evt);
  gestureState.origin = currentOrigin;
  gestureState.imagePoint = screenToImage(currentOrigin);
  if (pointerState.options.touch.enablePinchZoom) {
    const nextScale = gestureState.startScale * evt.scale;
    if (Number.isFinite(nextScale) && nextScale > 0) {
      viewState.scale = Math.min(Math.max(nextScale, 0.1), 40);
    }
  }
  if (pointerState.options.touch.enableRotate) {
    const deadzone = Math.max(pointerState.options.touch.rotationDeadzoneDegrees || 0, 0);
    const rotationDegrees = Number.isFinite(evt.rotation) ? evt.rotation : 0;
    if (Math.abs(rotationDegrees) >= deadzone) {
      viewState.rotation = normalizeAngle(gestureState.startRotation + (rotationDegrees * Math.PI / 180));
    } else {
      viewState.rotation = gestureState.startRotation;
    }
  }
  setOffsetForImagePoint(gestureState.imagePoint, gestureState.origin);
  userAdjustedScale = true;
  autoFitPending = false;
  draw();
  drawBrushPreview(null);
}

function handleGestureEnd(evt) {
  if (!gestureState) {
    return;
  }
  if (evt && typeof evt.preventDefault === 'function') {
    evt.preventDefault();
  }
  gestureState = null;
  drawBrushPreview(hoverPoint);
}

const MIN_BRUSH_DIAMETER = 1;
const MAX_BRUSH_DIAMETER = 25;

const BRUSH_KERNEL_MODES = {
  SMOOTH: 'smooth',
  SNAPPED: 'snapped',
};
let brushKernelMode = BRUSH_KERNEL_MODES.SMOOTH;
const snappedKernelCache = new Map();

function clampBrushDiameter(value) {
  const numeric = Number(value);
  if (Number.isNaN(numeric)) {
    return MIN_BRUSH_DIAMETER;
  }
  return Math.max(MIN_BRUSH_DIAMETER, Math.min(MAX_BRUSH_DIAMETER, Math.round(numeric)));
}

const defaultDiameter = clampBrushDiameter(initialBrushRadius * 2 + 1);
let brushDiameter = defaultDiameter;

const brushSizeSlider = document.getElementById('brushSizeSlider');
const brushSizeInput = document.getElementById('brushSizeInput');
const brushKernelModeSelect = document.getElementById('brushKernelMode');
if (brushSizeSlider) {
  brushSizeSlider.min = String(MIN_BRUSH_DIAMETER);
  brushSizeSlider.max = String(MAX_BRUSH_DIAMETER);
  brushSizeSlider.step = '1';
}
if (brushSizeInput) {
  brushSizeInput.min = String(MIN_BRUSH_DIAMETER);
  brushSizeInput.max = String(MAX_BRUSH_DIAMETER);
  brushSizeInput.step = '1';
}
if (brushKernelModeSelect) {
  brushKernelModeSelect.value = brushKernelMode;
}
document.querySelectorAll('[data-slider-id]').forEach((root) => {
  registerSlider(root);
});
document.querySelectorAll('[data-dropdown-id]').forEach((root) => {
  registerDropdown(root);
});
updateBrushControls();

window.addEventListener('resize', () => {
  if (dropdownOpenId) {
    const entry = dropdownRegistry.get(dropdownOpenId);
    positionDropdown(entry);
  }
});

const gammaSlider = document.getElementById('gamma');
const gammaValue = document.getElementById('gammaValue');
const maskLabel = document.getElementById('maskLabel');
const maskVisibility = document.getElementById('maskVisibility');
const toolInfo = document.getElementById('toolInfo');
const segmentButton = document.getElementById('segmentButton');
const segmentStatus = document.getElementById('segmentStatus');
const colorMode = document.getElementById('colorMode');
const gammaInput = document.getElementById('gammaInput');
const histogramCanvas = document.getElementById('histogram');
const histRangeLabel = document.getElementById('histRange');
const hoverInfo = document.getElementById('hoverInfo');
const maskOpacitySlider = document.getElementById('maskOpacity');
const maskOpacityInput = document.getElementById('maskOpacityInput');
const maskOpacityValue = document.getElementById('maskOpacityValue');

attachNumberInputStepper(brushSizeInput, (delta) => {
  setBrushDiameter(brushDiameter + delta, true);
});

attachNumberInputStepper(gammaInput, (delta) => {
  setGamma(currentGamma + delta);
});

const HISTORY_LIMIT = 200;
const undoStack = [];
const redoStack = [];
const viewState = { scale: 1.0, offsetX: 0.0, offsetY: 0.0, rotation: 0.0 };
let maskVisible = true;
let currentLabel = 1;
let originalImageData = null;
let isPanning = false;
let isPainting = false;
let lastPoint = { x: 0, y: 0 };
let lastPaintPoint = null;
let strokeChanges = null;
let tool = 'brush';
let spacePan = false;
let stylusToolOverride = null;

let hoverPoint = null;
let eraseActive = false;
let erasePreviousLabel = null;
let isSegmenting = false;
let useNColor = false;
let histogramData = null;
let windowLow = 0;
let windowHigh = 255;
let currentGamma = 1.0;
let histDragTarget = null;
let histDragOffset = 0;
let cursorInsideCanvas = false;
let cursorInsideImage = false;
let maskOpacity = CONFIG.maskOpacity ?? 0.8;
let userAdjustedScale = false;
const touchPointers = new Map();
let pinchState = null;
let panPointerId = null;
let activePointerId = null;
let gestureState = null;
let wheelRotationBuffer = 0;

const MIN_GAMMA = 0.1;
const MAX_GAMMA = 6.0;
const DEFAULT_GAMMA = 1.0;
const HIST_HANDLE_THRESHOLD = 8;
let autoFitPending = true;

function resizePreviewCanvas() {
  previewCanvas.width = canvas.width;
  previewCanvas.height = canvas.height;
}

function clearPreview() {
  previewCtx.setTransform(1, 0, 0, 1, 0, 0);
  previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
}

function drawTouchOverlay() {
  if (!debugTouchOverlay) {
    return;
  }
  const rect = canvas.getBoundingClientRect();
  previewCtx.save();
  previewCtx.setTransform(1, 0, 0, 1, 0, 0);
  previewCtx.lineWidth = 2;
  previewCtx.strokeStyle = 'rgba(0, 200, 255, 0.9)';
  previewCtx.fillStyle = 'rgba(0, 200, 255, 0.2)';
  let rendered = false;
  touchPointers.forEach((data) => {
    const x = data.x - rect.left;
    const y = data.y - rect.top;
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      return;
    }
    rendered = true;
    previewCtx.beginPath();
    previewCtx.arc(x, y, 16, 0, Math.PI * 2);
    previewCtx.fill();
    previewCtx.stroke();
  });
  if (!rendered && gestureState && gestureState.origin) {
    // Optional debug marker for gesture origin when no raw touch pointers exist.
    previewCtx.beginPath();
    previewCtx.arc(gestureState.origin.x, gestureState.origin.y, 18, 0, Math.PI * 2);
    previewCtx.strokeStyle = 'rgba(255, 180, 0, 0.6)';
    previewCtx.setLineDash([6, 6]);
    previewCtx.stroke();
  }
  previewCtx.restore();
}

function drawBrushPreview(point) {
  clearPreview();
  if (point) {
    const pixels = enumerateBrushPixels(point.x, point.y);
    if (pixels.length) {
      const center = getBrushKernelCenter(point.x, point.y);
      previewCtx.save();
      applyViewTransform(previewCtx, { includeDpr: true });
      previewCtx.imageSmoothingEnabled = false;
      previewCtx.fillStyle = 'rgba(255, 255, 255, 0.24)';
      const size = 1;
      for (const pixel of pixels) {
        previewCtx.fillRect(pixel.x, pixel.y, size, size);
      }
      const radius = (brushDiameter - 1) / 2;
      if (radius >= 0) {
        previewCtx.lineWidth = 1 / Math.max(viewState.scale * dpr, 1);
        previewCtx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
        previewCtx.beginPath();
        const centerX = center.x + 0.5;
        const centerY = center.y + 0.5;
        previewCtx.arc(centerX, centerY, radius + 0.5, 0, Math.PI * 2);
        previewCtx.stroke();
      }
      previewCtx.restore();
    }
  }
  drawTouchOverlay();
}

function updateCursor() {
  if (spacePan || isPanning) {
    canvas.style.cursor = 'move';
  } else if (tool === 'brush' && cursorInsideImage) {
    canvas.style.cursor = 'none';
  } else {
    canvas.style.cursor = 'default';
  }
}

function updateMaskLabel() {
  maskLabel.textContent = 'Mask Label: ' + currentLabel;
}

function updateMaskVisibilityLabel() {
  maskVisibility.textContent = 'Mask Layer: ' + (maskVisible ? 'On' : 'Off') + " (toggle with 'M')";
}

function updateMaskOpacityLabel() {
  if (maskOpacityValue) {
    maskOpacityValue.textContent = 'Mask Opacity: ' + maskOpacity.toFixed(2);
  }
}

function updateToolInfo() {
  let description = 'Brush (B=Brush, G=Fill, I=Picker)';
  if (tool === 'fill') {
    description = 'Flood Fill (B=Brush, G=Fill, I=Picker)';
  } else if (tool === 'picker') {
    description = 'Picker (B=Brush, G=Fill, I=Picker)';
  }
  toolInfo.textContent = 'Tool: ' + description;
}

function startEraseOverride() {
  if (eraseActive) {
    return;
  }
  eraseActive = true;
  erasePreviousLabel = currentLabel;
  setTool('brush');
  currentLabel = 0;
  updateMaskLabel();
}

function stopEraseOverride() {
  if (!eraseActive) {
    return;
  }
  if (erasePreviousLabel !== null) {
    currentLabel = erasePreviousLabel;
  }
  eraseActive = false;
  erasePreviousLabel = null;
  updateMaskLabel();
}

function setTool(nextTool) {
  if (tool === nextTool) {
    return;
  }
  tool = nextTool;
  updateToolInfo();
  updateCursor();
}

function updateBrushControls() {
  if (brushSizeSlider) {
    brushSizeSlider.value = String(brushDiameter);
    refreshSlider('brushSizeSlider');
  }
  if (brushSizeInput) {
    brushSizeInput.value = String(brushDiameter);
  }
}

function setBrushDiameter(nextDiameter, fromUser = false) {
  const clamped = clampBrushDiameter(nextDiameter);
  if (brushDiameter === clamped) {
    if (fromUser) {
      setTool('brush');
    }
    return;
  }
  brushDiameter = clamped;
  updateBrushControls();
  log('brush size set to ' + brushDiameter);
  if (fromUser) {
    setTool('brush');
  }
  refreshSlider('brushSizeSlider');
  drawBrushPreview(hoverPoint);
}

function syncMaskOpacityControls() {
  if (maskOpacitySlider) {
    maskOpacitySlider.value = maskOpacity.toFixed(2);
    refreshSlider('maskOpacity');
  }
  if (maskOpacityInput && document.activeElement !== maskOpacityInput) {
    maskOpacityInput.value = maskOpacity.toFixed(2);
  }
  updateMaskOpacityLabel();
}

function setMaskOpacity(value, { silent = false } = {}) {
  const numeric = typeof value === 'number' ? value : parseFloat(value);
  if (Number.isNaN(numeric)) {
    syncMaskOpacityControls();
    return;
  }
  const clamped = clamp(numeric, 0, 1);
  if (maskOpacity === clamped && !silent) {
    syncMaskOpacityControls();
    return;
  }
  maskOpacity = clamped;
  syncMaskOpacityControls();
  if (!silent) {
    redrawMaskCanvas();
    draw();
  }
}

function pushHistory(indices, before, after) {
  undoStack.push({ indices, before, after });
  if (undoStack.length > HISTORY_LIMIT) {
    undoStack.shift();
  }
  redoStack.length = 0;
}

function applyHistoryEntry(entry, useAfter) {
  const values = useAfter ? entry.after : entry.before;
  const idxs = entry.indices;
  for (let i = 0; i < idxs.length; i += 1) {
    maskValues[idxs[i]] = values[i];
  }
}

function undo() {
  if (!undoStack.length) {
    return;
  }
  const entry = undoStack.pop();
  applyHistoryEntry(entry, false);
  redoStack.push(entry);
  redrawMaskCanvas();
  draw();
}

function redo() {
  if (!redoStack.length) {
    return;
  }
  const entry = redoStack.pop();
  applyHistoryEntry(entry, true);
  undoStack.push(entry);
  redrawMaskCanvas();
  draw();
}

function collectBrushIndices(target, centerX, centerY) {
  const pixels = enumerateBrushPixels(centerX, centerY);
  for (const pixel of pixels) {
    target.add(pixel.y * imgWidth + pixel.x);
  }
}

function setBrushKernelMode(nextMode) {
  const normalized = nextMode === BRUSH_KERNEL_MODES.SNAPPED
    ? BRUSH_KERNEL_MODES.SNAPPED
    : BRUSH_KERNEL_MODES.SMOOTH;
  if (brushKernelMode === normalized) {
    return;
  }
  brushKernelMode = normalized;
  if (brushKernelModeSelect) {
    brushKernelModeSelect.value = brushKernelMode;
    refreshDropdown('brushKernelMode');
  }
  log('brush kernel mode set to ' + brushKernelMode);
  drawBrushPreview(hoverPoint);
}

function getBrushKernelCenter(x, y) {
  if (brushKernelMode === BRUSH_KERNEL_MODES.SNAPPED) {
    return { x: Math.round(x), y: Math.round(y) };
  }
  return { x, y };
}

function enumerateBrushPixels(centerX, centerY) {
  if (brushKernelMode === BRUSH_KERNEL_MODES.SNAPPED) {
    return enumerateSnappedBrushPixels(centerX, centerY);
  }
  return enumerateSmoothBrushPixels(centerX, centerY);
}

function enumerateSmoothBrushPixels(centerX, centerY) {
  const pixels = [];
  const radius = (brushDiameter - 1) / 2;
  if (radius <= 0) {
    const x = Math.round(centerX);
    const y = Math.round(centerY);
    if (x >= 0 && x < imgWidth && y >= 0 && y < imgHeight) {
      pixels.push({ x, y });
    }
    return pixels;
  }
  const threshold = (radius + 0.35) * (radius + 0.35);
  const minX = Math.max(0, Math.floor(centerX - radius - 1));
  const maxX = Math.min(imgWidth - 1, Math.ceil(centerX + radius + 1));
  for (let x = minX; x <= maxX; x += 1) {
    const dx = x - centerX;
    if (dx * dx > threshold) {
      continue;
    }
    const dy = Math.sqrt(Math.max(0, threshold - dx * dx));
    let yMin = Math.ceil(centerY - dy);
    let yMax = Math.floor(centerY + dy);
    if (yMax < yMin) {
      continue;
    }
    if (yMin < 0) {
      yMin = 0;
    }
    if (yMax >= imgHeight) {
      yMax = imgHeight - 1;
    }
    for (let y = yMin; y <= yMax; y += 1) {
      pixels.push({ x, y });
    }
  }
  return pixels;
}

function enumerateSnappedBrushPixels(centerX, centerY) {
  const center = getBrushKernelCenter(centerX, centerY);
  const offsets = getSnappedKernelOffsets(brushDiameter);
  const pixels = [];
  for (const offset of offsets) {
    const x = center.x + offset.x;
    const y = center.y + offset.y;
    if (x >= 0 && x < imgWidth && y >= 0 && y < imgHeight) {
      pixels.push({ x, y });
    }
  }
  return pixels;
}

function getSnappedKernelOffsets(diameter) {
  if (snappedKernelCache.has(diameter)) {
    return snappedKernelCache.get(diameter);
  }
  const radius = (diameter - 1) / 2;
  const threshold = (radius + 0.35) * (radius + 0.35);
  const span = Math.ceil(radius + 1);
  const offsets = [];
  for (let y = -span; y <= span; y += 1) {
    for (let x = -span; x <= span; x += 1) {
      if (x * x + y * y <= threshold) {
        offsets.push({ x, y });
      }
    }
  }
  snappedKernelCache.set(diameter, offsets);
  return offsets;
}

function traverseLine(x0, y0, x1, y1, callback) {
  let ix0 = Math.round(x0);
  let iy0 = Math.round(y0);
  const ix1 = Math.round(x1);
  const iy1 = Math.round(y1);
  const dx = Math.abs(ix1 - ix0);
  const sx = ix0 < ix1 ? 1 : -1;
  const dy = -Math.abs(iy1 - iy0);
  const sy = iy0 < iy1 ? 1 : -1;
  let err = dx + dy;
  while (true) {
    callback(ix0, iy0);
    if (ix0 === ix1 && iy0 === iy1) break;
    const e2 = 2 * err;
    if (e2 >= dy) {
      err += dy;
      ix0 += sx;
    }
    if (e2 <= dx) {
      err += dx;
      iy0 += sy;
    }
  }
}

function paintStroke(point) {
  if (!strokeChanges) {
    strokeChanges = new Map();
  }
  const start = lastPaintPoint ? { x: lastPaintPoint.x, y: lastPaintPoint.y } : { x: point.x, y: point.y };
  const local = new Set();
  const dx = point.x - start.x;
  const dy = point.y - start.y;
  const dist = Math.hypot(dx, dy);
  const spacing = Math.max(0.15, 0.5 / Math.max(viewState.scale, 0.0001));
  const steps = Math.max(1, Math.ceil(dist / spacing));
  for (let i = 0; i <= steps; i += 1) {
    const t = steps === 0 ? 1 : i / steps;
    const px = start.x + dx * t;
    const py = start.y + dy * t;
    collectBrushIndices(local, px, py);
  }
  let changed = false;
  local.forEach((idx) => {
    if (!strokeChanges.has(idx)) {
      const original = maskValues[idx];
      if (original === currentLabel) {
        return;
      }
      strokeChanges.set(idx, original);
    }
    if (maskValues[idx] !== currentLabel) {
      maskValues[idx] = currentLabel;
      changed = true;
    }
  });
  if (changed) {
    redrawMaskCanvas();
    draw();
  }
  lastPaintPoint = { x: point.x, y: point.y };
}

function finalizeStroke() {
  if (!strokeChanges || strokeChanges.size === 0) {
    strokeChanges = null;
    return;
  }
  const keys = Array.from(strokeChanges.keys()).sort((a, b) => a - b);
  const count = keys.length;
  const indices = new Uint32Array(count);
  const before = new Uint8Array(count);
  const after = new Uint8Array(count);
  for (let i = 0; i < count; i += 1) {
    const idx = keys[i];
    indices[i] = idx;
    before[i] = strokeChanges.get(idx);
    after[i] = currentLabel;
  }
  pushHistory(indices, before, after);
  strokeChanges = null;
}

function floodFill(point) {
  const sx = Math.round(point.x);
  const sy = Math.round(point.y);
  if (sx < 0 || sy < 0 || sx >= imgWidth || sy >= imgHeight) {
    return;
  }
  const startIdx = sy * imgWidth + sx;
  const targetLabel = maskValues[startIdx];
  if (targetLabel === currentLabel) {
    return;
  }
  const visited = new Uint8Array(maskValues.length);
  const stack = [startIdx];
  const indices = [];
  while (stack.length) {
    const idx = stack.pop();
    if (visited[idx]) {
      continue;
    }
    visited[idx] = 1;
    if (maskValues[idx] !== targetLabel) {
      continue;
    }
    indices.push(idx);
    const x = idx % imgWidth;
    const y = (idx / imgWidth) | 0;
    if (x > 0) { stack.push(idx - 1); }
    if (x + 1 < imgWidth) { stack.push(idx + 1); }
    if (y > 0) { stack.push(idx - imgWidth); }
    if (y + 1 < imgHeight) { stack.push(idx + imgWidth); }
  }
  if (!indices.length) {
    return;
  }
  const sorted = Array.from(new Set(indices)).sort((a, b) => a - b);
  const count = sorted.length;
  const idxArr = new Uint32Array(count);
  const before = new Uint8Array(count);
  const after = new Uint8Array(count);
  for (let i = 0; i < count; i += 1) {
    const idx = sorted[i];
    idxArr[i] = idx;
    before[i] = maskValues[idx];
    after[i] = currentLabel;
    maskValues[idx] = currentLabel;
  }
  pushHistory(idxArr, before, after);
  redrawMaskCanvas();
  draw();
}

function pickColor(point) {
  const sx = Math.round(point.x);
  const sy = Math.round(point.y);
  if (sx < 0 || sy < 0 || sx >= imgWidth || sy >= imgHeight) {
    return;
  }
  const idx = sy * imgWidth + sx;
  currentLabel = maskValues[idx];
  updateMaskLabel();
  log('picker set label ' + currentLabel);
}

function redrawMaskCanvas() {
  const data = maskData.data;
  const palette = useNColor ? nColorPalette : defaultPalette;
  for (let i = 0; i < maskValues.length; i += 1) {
    const label = maskValues[i];
    const color = getColorForLabel(palette, label);
    const p = i * 4;
    data[p] = color[0];
    data[p + 1] = color[1];
    data[p + 2] = color[2];
    data[p + 3] = color[3];
  }
  maskCtx.putImageData(maskData, 0, 0);
}

function shouldLogDraw() {
  if (drawLogCount < 20) {
    drawLogCount += 1;
    return true;
  }
  drawLogCount += 1;
  return drawLogCount % 50 === 0;
}

function applyViewTransform(context, { includeDpr = false } = {}) {
  const cos = Math.cos(viewState.rotation);
  const sin = Math.sin(viewState.rotation);
  const scaleFactor = viewState.scale * (includeDpr ? dpr : 1);
  const tx = viewState.offsetX * (includeDpr ? dpr : 1);
  const ty = viewState.offsetY * (includeDpr ? dpr : 1);
  context.setTransform(
    scaleFactor * cos,
    scaleFactor * sin,
    -scaleFactor * sin,
    scaleFactor * cos,
    tx,
    ty,
  );
}

function setOffsetForImagePoint(imagePoint, canvasPoint) {
  const cos = Math.cos(viewState.rotation);
  const sin = Math.sin(viewState.rotation);
  const scaledX = imagePoint.x * viewState.scale;
  const scaledY = imagePoint.y * viewState.scale;
  const rotatedX = scaledX * cos - scaledY * sin;
  const rotatedY = scaledX * sin + scaledY * cos;
  viewState.offsetX = canvasPoint.x - rotatedX;
  viewState.offsetY = canvasPoint.y - rotatedY;
}

function normalizeAngleDelta(delta) {
  if (!Number.isFinite(delta)) {
    return 0;
  }
  return Math.atan2(Math.sin(delta), Math.cos(delta));
}

function normalizeAngle(angle) {
  if (!Number.isFinite(angle)) {
    return 0;
  }
  return Math.atan2(Math.sin(angle), Math.cos(angle));
}

function rotateView(deltaRadians) {
  if (!Number.isFinite(deltaRadians) || deltaRadians === 0) {
    return;
  }
  if (!canvas) {
    return;
  }
  if (!Number.isFinite(imgWidth) || !Number.isFinite(imgHeight) || imgWidth <= 0 || imgHeight <= 0) {
    return;
  }
  const rect = canvas.getBoundingClientRect();
  if (!rect || rect.width <= 0 || rect.height <= 0) {
    return;
  }
  const imageCenter = {
    x: imgWidth / 2,
    y: imgHeight / 2,
  };
  const anchorScreen = imageToScreen(imageCenter);
  viewState.rotation = normalizeAngle(viewState.rotation + deltaRadians);
  log('rotate view delta=' + (deltaRadians * RAD_TO_DEG).toFixed(1) + ' deg, now ' + (viewState.rotation * RAD_TO_DEG).toFixed(1) + ' deg');
  setOffsetForImagePoint(imageCenter, anchorScreen);
  userAdjustedScale = true;
  autoFitPending = false;
  draw();
  drawBrushPreview(hoverPoint);
}

function draw() {
  if (shuttingDown) {
    return;
  }
  if (shouldLogDraw()) {
    log('draw start scale=' + viewState.scale.toFixed(3) + ' offset=' + viewState.offsetX.toFixed(1) + ',' + viewState.offsetY.toFixed(1));
  }
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.restore();
  ctx.save();
  applyViewTransform(ctx, { includeDpr: true });
  const smooth = viewState.scale < 1;
  ctx.imageSmoothingEnabled = smooth;
  ctx.imageSmoothingQuality = smooth ? 'high' : 'low';
  ctx.drawImage(offscreen, 0, 0);
  if (maskVisible) {
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(maskCanvas, 0, 0);
  }
  ctx.restore();
  drawBrushPreview(hoverPoint);
  if (!loggedPixelSample && canvas.width > 0 && canvas.height > 0) {
    try {
      const cx = Math.floor(canvas.width / 2);
      const cy = Math.floor(canvas.height / 2);
      const sample = ctx.getImageData(cx, cy, 1, 1).data;
      log('center pixel rgba=' + Array.from(sample).join(','));
    } catch (err) {
      log('center pixel read failed: ' + (err && err.message ? err.message : err));
    }
    loggedPixelSample = true;
    hideLoadingOverlay();
  }
}

function getPointerPosition(evt) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: evt.clientX - rect.left,
    y: evt.clientY - rect.top,
  };
}

function screenToImage(point) {
  const cos = Math.cos(viewState.rotation);
  const sin = Math.sin(viewState.rotation);
  const dx = point.x - viewState.offsetX;
  const dy = point.y - viewState.offsetY;
  const localX = (dx * cos + dy * sin) / viewState.scale;
  const localY = (-dx * sin + dy * cos) / viewState.scale;
  return {
    x: localX,
    y: localY,
  };
}

function imageToScreen(point) {
  const cos = Math.cos(viewState.rotation);
  const sin = Math.sin(viewState.rotation);
  const scaledX = point.x * viewState.scale;
  const scaledY = point.y * viewState.scale;
  const rotatedX = scaledX * cos - scaledY * sin;
  const rotatedY = scaledX * sin + scaledY * cos;
  return {
    x: rotatedX + viewState.offsetX,
    y: rotatedY + viewState.offsetY,
  };
}

function resolveGestureOrigin(evt) {
  if (!canvas) {
    return { x: 0, y: 0 };
  }
  const rect = canvas.getBoundingClientRect();
  let x = typeof evt.clientX === 'number'
    ? evt.clientX - rect.left
    : Number.NaN;
  let y = typeof evt.clientY === 'number'
    ? evt.clientY - rect.top
    : Number.NaN;
  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    x = rect.width / 2;
    y = rect.height / 2;
  }
  return { x, y };
}

function resizeCanvas() {
  if (shuttingDown) {
    return;
  }
  const rect = viewer.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) {
    log('resize skipped: viewer size ' + rect.width.toFixed(1) + 'x' + rect.height.toFixed(1));
    if (!shuttingDown) {
      requestAnimationFrame(resizeCanvas);
    }
    return;
  }
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  canvas.style.width = rect.width + 'px';
  canvas.style.height = rect.height + 'px';
  resizePreviewCanvas();
  if (!fitViewToWindow(rect)) {
    recenterView(rect);
  }
  draw();
  drawBrushPreview(hoverPoint);
}

function recenterView(bounds) {
  const rect = bounds || viewer.getBoundingClientRect();
  const width = rect ? rect.width : 0;
  const height = rect ? rect.height : 0;
  const usableWidth = Math.max(0, width - sidebarWidth);
  const imageCenter = { x: imgWidth / 2, y: imgHeight / 2 };
  const target = {
    x: usableWidth / 2,
    y: height / 2,
  };
  setOffsetForImagePoint(imageCenter, target);
  log('recenter to ' + viewState.offsetX.toFixed(1) + ',' + viewState.offsetY.toFixed(1));
}

function fitViewToWindow(bounds) {
  if (!viewer) {
    return false;
  }
  if (userAdjustedScale && !autoFitPending) {
    return false;
  }
  const rect = bounds || viewer.getBoundingClientRect();
  if (!rect || rect.width <= 0 || rect.height <= 0 || imgWidth === 0 || imgHeight === 0) {
    return false;
  }
  const usableWidth = Math.max(1, rect.width - sidebarWidth);
  const scaleX = usableWidth / imgWidth;
  const scaleY = rect.height / imgHeight;
  const nextScale = Math.min(scaleX, scaleY);
  if (!Number.isFinite(nextScale) || nextScale <= 0) {
    return false;
  }
  viewState.scale = nextScale;
  viewState.rotation = 0;
  autoFitPending = false;
  recenterView(rect);
  return true;
}

function resetView() {
  const rect = viewer ? viewer.getBoundingClientRect() : null;
  autoFitPending = true;
  userAdjustedScale = false;
  viewState.rotation = 0;
  if (!fitViewToWindow(rect)) {
    viewState.rotation = 0;
    recenterView(rect);
  }
  draw();
  if (hoverPoint) {
    drawBrushPreview(hoverPoint);
  } else {
    drawBrushPreview(null);
  }
}

function clampGammaValue(value) {
  if (Number.isNaN(value)) {
    return currentGamma;
  }
  return Math.min(MAX_GAMMA, Math.max(MIN_GAMMA, value));
}

function updateGammaLabel() {
  if (gammaValue) {
    gammaValue.textContent = 'Gamma: ' + currentGamma.toFixed(2);
  }
}

function syncGammaControls() {
  if (gammaSlider) {
    const sliderValue = Math.round(currentGamma * 100);
    gammaSlider.value = String(Math.min(600, Math.max(10, sliderValue)));
    refreshSlider('gamma');
  }
  if (gammaInput) {
    gammaInput.value = currentGamma.toFixed(2);
  }
  updateGammaLabel();
}

function setGamma(gamma, { emit = true } = {}) {
  currentGamma = clampGammaValue(gamma);
  syncGammaControls();
  if (emit) {
    applyImageAdjustments();
  } else {
    renderHistogram();
  }
}

function applyGamma(gamma) {
  setGamma(gamma, { emit: true });
}

function sinebowColor(t) {
  const angle = 2 * Math.PI * (t - Math.floor(t));
  const r = Math.sin(angle) * 0.5 + 0.5;
  const g = Math.sin(angle + (2 * Math.PI) / 3) * 0.5 + 0.5;
  const b = Math.sin(angle + (4 * Math.PI) / 3) * 0.5 + 0.5;
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255), 200];
}

function generateSinebowPalette(size, offset = 0) {
  const count = Math.max(size, 2);
  const palette = new Array(count);
  palette[0] = [0, 0, 0, 0];
  const golden = 0.61803398875;
  for (let i = 1; i < count; i += 1) {
    const t = (offset + i * golden) % 1;
    palette[i] = sinebowColor(t);
  }
  return palette;
}

defaultPalette = generateSinebowPalette(Math.max(colorTable.length || 0, 256), 0.0);
nColorPalette = generateSinebowPalette(Math.max(colorTable.length || 0, 256), 0.35);

function applyImageAdjustments() {
  if (!originalImageData) {
    return;
  }
  const source = originalImageData.data;
  const img = offCtx.createImageData(imgWidth, imgHeight);
  const target = img.data;
  const low = windowLow / 255;
  const high = windowHigh / 255;
  const range = Math.max(high - low, 1 / 255);
  for (let i = 0; i < source.length; i += 4) {
    let value = source[i] / 255;
    value = Math.min(Math.max((value - low) / range, 0), 1);
    value = Math.pow(value, currentGamma);
    const v = Math.round(value * 255);
    target[i] = v;
    target[i + 1] = v;
    target[i + 2] = v;
    target[i + 3] = source[i + 3];
  }
  offCtx.putImageData(img, 0, 0);
  draw();
  renderHistogram();
}

function computeHistogram() {
  if (!originalImageData) {
    histogramData = null;
    return;
  }
  histogramData = new Uint32Array(256);
  const data = originalImageData.data;
  for (let i = 0; i < data.length; i += 4) {
    histogramData[data[i]] += 1;
  }
}

function renderHistogram() {
  if (!histogramCanvas || !histogramData) {
    return;
  }
  const ctx = histogramCanvas.getContext('2d');
  if (!ctx) {
    return;
  }
  const width = histogramCanvas.width;
  const height = histogramCanvas.height;
  ctx.clearRect(0, 0, width, height);
  const maxCount = Math.max(...histogramData);
  if (maxCount > 0) {
    ctx.fillStyle = accentColor;
    const binWidth = Math.max(width / 256, 1);
    for (let i = 0; i < 256; i += 1) {
      const value = histogramData[i] / maxCount;
      const barHeight = Math.max(1, Math.round(value * (height - 4)));
      const x = Math.floor(i * binWidth);
      ctx.fillRect(x, height - barHeight, Math.ceil(binWidth), barHeight);
    }
  }
  const lowX = (windowLow / 255) * width;
  const highX = (windowHigh / 255) * width;
  ctx.fillStyle = histogramWindowColor;
  ctx.fillRect(lowX, 0, Math.max(highX - lowX, 1), height);
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(lowX, 0);
  ctx.lineTo(lowX, height);
  ctx.moveTo(highX, 0);
  ctx.lineTo(highX, height);
  ctx.stroke();
  const gammaCurveColor = panelTextColor || accentColor;
  if (windowHigh > windowLow) {
    ctx.strokeStyle = gammaCurveColor;
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    const startX = Math.max(0, Math.floor(lowX));
    const endX = Math.min(width, Math.ceil(highX));
    for (let x = startX; x <= endX; x += 1) {
      const intensity = (x / width) * 255;
      const y = gammaCurveY(intensity, width, height);
      if (x === startX) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
  }
  updateHistogramCursor();
}

function updateHistogramUI() {
  if (histRangeLabel) {
    histRangeLabel.textContent = 'Window: [' + windowLow + ', ' + windowHigh + ']';
  }
  renderHistogram();
}

function histogramValueFromEvent(evt) {
  const rect = histogramCanvas.getBoundingClientRect();
  const x = Math.min(Math.max(evt.clientX - rect.left, 0), rect.width);
  return Math.round((x / rect.width) * 255);
}

function gammaCurveY(intensity, width, height) {
  if (windowHigh <= windowLow) {
    return height - 2;
  }
  const clampedIntensity = Math.min(Math.max(intensity, windowLow), windowHigh);
  let t = (clampedIntensity - windowLow) / (windowHigh - windowLow);
  t = Math.min(Math.max(t, 0.0001), 0.9999);
  const mapped = Math.pow(t, 1 / currentGamma);
  const y = height - (mapped * (height - 4)) - 2;
  return Math.min(height - 2, Math.max(2, y));
}

function updateHistogramCursor(evt) {
  if (!histogramCanvas) {
    return;
  }
  if (histDragTarget) {
    histogramCanvas.style.cursor = (histDragTarget === 'range' || histDragTarget === 'gamma') ? 'grabbing' : 'ew-resize';
    return;
  }
  const rect = histogramCanvas.getBoundingClientRect();
  const x = evt ? evt.clientX - rect.left : NaN;
  const width = rect.width;
  const height = rect.height;
  const lowX = (windowLow / 255) * width;
  const highX = (windowHigh / 255) * width;
  const threshold = HIST_HANDLE_THRESHOLD;
  let cursor = 'crosshair';
  if (!Number.isNaN(x)) {
    if (Math.abs(x - lowX) < threshold || Math.abs(x - highX) < threshold) {
      cursor = 'ew-resize';
    } else if (x > lowX && x < highX) {
      cursor = 'grab';
      if (evt && windowHigh > windowLow) {
        const intensity = histogramValueFromEvent(evt);
        const y = evt.clientY - rect.top;
        const curveY = gammaCurveY(intensity, width, height);
        if (Math.abs(y - curveY) < threshold) {
          cursor = 'grab';
        }
      }
    }
  }
  histogramCanvas.style.cursor = cursor;
}

function setWindowBounds(low, high, { emit = true } = {}) {
  let clampedLow = Math.round(low);
  let clampedHigh = Math.round(high);
  if (Number.isNaN(clampedLow)) clampedLow = windowLow;
  if (Number.isNaN(clampedHigh)) clampedHigh = windowHigh;
  clampedLow = Math.max(0, Math.min(255, clampedLow));
  clampedHigh = Math.max(0, Math.min(255, clampedHigh));
  if (clampedHigh <= clampedLow) {
    if (histDragTarget === 'low') {
      clampedLow = Math.max(0, Math.min(254, clampedHigh - 1));
    } else if (histDragTarget === 'high') {
      clampedHigh = Math.min(255, Math.max(1, clampedLow + 1));
    } else {
      clampedHigh = Math.min(255, Math.max(1, clampedLow + 1));
    }
  }
  windowLow = clampedLow;
  windowHigh = clampedHigh;
  updateHistogramUI();
  if (emit) {
    applyImageAdjustments();
  }
}

function handleHistogramPointerDown(evt) {
  if (!histogramCanvas) {
    return;
  }
  evt.preventDefault();
  histogramCanvas.setPointerCapture(evt.pointerId);
  const rect = histogramCanvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  const x = Math.min(Math.max(evt.clientX - rect.left, 0), width);
  const y = Math.min(Math.max(evt.clientY - rect.top, 0), height);
  const intensity = (x / width) * 255;
  const lowX = (windowLow / 255) * width;
  const highX = (windowHigh / 255) * width;
  const threshold = HIST_HANDLE_THRESHOLD;
  histDragTarget = null;
  if (windowHigh > windowLow) {
    const curveY = gammaCurveY(intensity, width, height);
    if (intensity >= windowLow && intensity <= windowHigh && Math.abs(y - curveY) <= threshold) {
      histDragTarget = 'gamma';
    }
  }
  if (!histDragTarget && Math.abs(x - lowX) <= threshold) {
    histDragTarget = 'low';
  } else if (!histDragTarget && Math.abs(x - highX) <= threshold) {
    histDragTarget = 'high';
  } else if (!histDragTarget && x > lowX && x < highX) {
    histDragTarget = 'range';
    histDragOffset = histogramValueFromEvent(evt) - windowLow;
  } else if (!histDragTarget) {
    histDragTarget = Math.abs(x - lowX) < Math.abs(x - highX) ? 'low' : 'high';
  }
  if (histDragTarget !== 'range') {
    histDragOffset = 0;
  }
  updateHistogramCursor(evt);
  handleHistogramPointerMove(evt);
}

function handleHistogramPointerMove(evt) {
  if (!histogramCanvas) {
    return;
  }
  if (!histDragTarget) {
    updateHistogramCursor(evt);
    return;
  }
  evt.preventDefault();
  const value = histogramValueFromEvent(evt);
  if (histDragTarget === 'low') {
    setWindowBounds(Math.min(value, windowHigh - 1), windowHigh);
  } else if (histDragTarget === 'high') {
    setWindowBounds(windowLow, Math.max(value, windowLow + 1));
  } else if (histDragTarget === 'range') {
    const span = windowHigh - windowLow;
    let newLow = value - histDragOffset;
    newLow = Math.max(0, Math.min(255 - span, newLow));
    setWindowBounds(newLow, newLow + span);
  } else if (histDragTarget === 'gamma') {
    if (windowHigh > windowLow) {
      const rect = histogramCanvas.getBoundingClientRect();
      const height = rect.height;
      const width = rect.width;
      const clampedValue = Math.min(Math.max(value, windowLow + 0.5), windowHigh - 0.5);
      let t = (clampedValue - windowLow) / (windowHigh - windowLow);
      t = Math.min(Math.max(t, 0.0001), 0.9999);
      const yRatio = 1 - Math.min(Math.max((evt.clientY - rect.top) / height, 0.0001), 0.9999);
      let newGamma = Math.log(t) / Math.log(yRatio);
      if (!Number.isFinite(newGamma) || newGamma <= 0) {
        newGamma = currentGamma;
      }
      setGamma(newGamma);
    }
  }
  updateHistogramCursor(evt);
}

function handleHistogramPointerUp(evt) {
  if (!histogramCanvas) {
    return;
  }
  evt.preventDefault();
  histogramCanvas.releasePointerCapture(evt.pointerId);
  histDragTarget = null;
  histDragOffset = 0;
  updateHistogramCursor(evt);
}

function updateHoverInfo(point) {
  if (!hoverInfo) {
    cursorInsideImage = false;
    updateCursor();
    return;
  }
  if (!point || !originalImageData) {
    cursorInsideImage = false;
    hoverInfo.textContent = 'Y: --, X: --, Val: --';
    updateCursor();
    return;
  }
  const x = Math.round(point.x);
  const y = Math.round(point.y);
  if (x < 0 || y < 0 || x >= imgWidth || y >= imgHeight) {
    cursorInsideImage = false;
    hoverInfo.textContent = 'Y: --, X: --, Val: --';
    updateCursor();
    return;
  }
  const idx = (y * imgWidth + x) * 4;
  const value = originalImageData.data[idx];
  cursorInsideImage = true;
  hoverInfo.textContent = 'Y: ' + y + ', X: ' + x + ', Val: ' + value;
  updateCursor();
}

function getColorForLabel(palette, label) {
  if (label <= 0) {
    return [0, 0, 0, 0];
  }
  if (palette[label]) {
    const base = palette[label];
    return [base[0], base[1], base[2], Math.round(base[3] * maskOpacity)];
  }
  if (palette.length > 1) {
    const idx = ((label - 1) % (palette.length - 1)) + 1;
    const base = palette[idx];
    return [base[0], base[1], base[2], Math.round(base[3] * maskOpacity)];
  }
  return palette[0] || [0, 0, 0, 0];
}

function updateColorModeLabel() {
  if (!colorMode) {
    return;
  }
  const mode = useNColor ? 'N-Color' : 'Palette';
  colorMode.textContent = 'Mask Colors: ' + mode + " (toggle with 'N')";
}

function toggleColorMode() {
  useNColor = !useNColor;
  updateColorModeLabel();
  redrawMaskCanvas();
  draw();
}

function setSegmentStatus(message, isError = false) {
  if (!segmentStatus) {
    return;
  }
  segmentStatus.textContent = message || '';
  segmentStatus.style.color = isError ? '#ff8a8a' : '#9aa';
}

function applySegmentationMask(payload) {
  if (!payload || !payload.mask) {
    throw new Error('segment payload missing mask');
  }
  const binary = atob(payload.mask);
  if (binary.length !== maskValues.length) {
    throw new Error('mask size mismatch (' + binary.length + ' vs ' + maskValues.length + ')');
  }
  for (let i = 0; i < binary.length; i += 1) {
    maskValues[i] = binary.charCodeAt(i);
  }
  redrawMaskCanvas();
  draw();
  updateMaskLabel();
}

async function requestSegmentation() {
  if (window.pywebview && window.pywebview.api && window.pywebview.api.segment) {
    return window.pywebview.api.segment();
  }
  const response = await fetch('/api/segment', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });
  if (!response.ok) {
    throw new Error('HTTP ' + response.status);
  }
  return response.json();
}

async function runSegmentation() {
  if (!segmentButton || isSegmenting) {
    return;
  }
  isSegmenting = true;
  segmentButton.disabled = true;
  setSegmentStatus('Running segmentation…');
  try {
    const raw = await requestSegmentation();
    const payload = typeof raw === 'string' ? JSON.parse(raw) : raw;
    if (payload.error) {
      throw new Error(payload.error);
    }
    applySegmentationMask(payload);
    setSegmentStatus('Segmentation complete.');
  } catch (err) {
    console.error(err);
    const msg = err && err.message ? err.message : err;
    setSegmentStatus('Segmentation failed: ' + msg, true);
  } finally {
    isSegmenting = false;
    segmentButton.disabled = false;
  }
}

canvas.addEventListener('wheel', (evt) => {
  evt.preventDefault();
  const pointer = getPointerPosition(evt);
  const imagePoint = screenToImage(pointer);
  const deltaY = Number.isFinite(evt.deltaY) ? evt.deltaY : 0;
  const deltaZ = Number.isFinite(evt.deltaZ) ? evt.deltaZ : 0;

  let scaleChanged = false;
  if (deltaY !== 0) {
    const factor = deltaY < 0 ? 1.1 : 0.9;
    const nextScale = Math.min(Math.max(viewState.scale * factor, 0.1), 40);
    if (nextScale !== viewState.scale) {
      viewState.scale = nextScale;
      scaleChanged = true;
    }
  }

  let rotationApplied = false;
  if (pointerState.options.touch.enableRotate && deltaZ !== 0) {
    wheelRotationBuffer += deltaZ;
    log('wheel gesture deltaY=' + deltaY.toFixed(2) + ' deltaZ=' + deltaZ.toFixed(2) + ' buffer=' + wheelRotationBuffer.toFixed(2));
    const threshold = Math.max(pointerState.options.touch.rotationDeadzoneDegrees || 0, 0);
    if (Math.abs(wheelRotationBuffer) >= threshold) {
      const appliedDegrees = wheelRotationBuffer;
      viewState.rotation = normalizeAngle(viewState.rotation + (appliedDegrees * Math.PI / 180));
      wheelRotationBuffer = 0;
      log('wheel rotation applied ' + appliedDegrees.toFixed(2) + ' deg');
      rotationApplied = true;
    }
  } else if (deltaZ === 0) {
    wheelRotationBuffer = 0;
  }

  if (scaleChanged || rotationApplied) {
    setOffsetForImagePoint(imagePoint, pointer);
    userAdjustedScale = true;
    autoFitPending = false;
    draw();
  }
}, { passive: false });

function startPointerPan(evt) {
  isPainting = false;
  strokeChanges = null;
  isPanning = true;
  updateCursor();
  lastPoint = getPointerPosition(evt);
  wheelRotationBuffer = 0;
  try {
    canvas.setPointerCapture(evt.pointerId);
    activePointerId = evt.pointerId;
  } catch (_) {
    /* ignore */
  }
  hoverPoint = null;
  drawBrushPreview(null);
  updateHoverInfo(null);
}

function beginBrushStroke(evt, worldPoint) {
  strokeChanges = new Map();
  isPainting = true;
  canvas.classList.add('painting');
  updateCursor();
  try {
    canvas.setPointerCapture(evt.pointerId);
    activePointerId = evt.pointerId;
  } catch (_) {
    /* ignore */
  }
  lastPaintPoint = null;
  paintStroke(worldPoint);
  hoverPoint = worldPoint ? { x: worldPoint.x, y: worldPoint.y } : null;
  drawBrushPreview(hoverPoint);
  updateHoverInfo(hoverPoint);
}

canvas.addEventListener('pointerdown', (evt) => {
  gestureState = null;
  pointerState.registerPointerDown(evt);
  const pointer = getPointerPosition(evt);
  lastPoint = pointer;
  const world = screenToImage(pointer);
  updateHoverInfo(world);
  const isStylus = pointerState.isStylusPointer(evt);
  if (isStylus) {
    if (!pointerState.options.stylus.allowSimultaneousTouchGestures) {
      if (panPointerId !== null) {
        try {
          canvas.releasePointerCapture(panPointerId);
        } catch (_) {
          /* ignore */
        }
      }
      panPointerId = null;
      touchPointers.clear();
      pinchState = null;
      isPanning = false;
      spacePan = false;
    }
    if (pointerState.isBarrelPanActive()) {
      startPointerPan(evt);
      return;
    }
    if (stylusToolOverride === null && tool !== 'brush') {
      stylusToolOverride = tool;
      setTool('brush');
    }
    beginBrushStroke(evt, world);
    return;
  }
  if (evt.pointerType === 'touch') {
    handleTouchPointerDown(evt, pointer);
    return;
  }
  const isSecondaryButton = evt.button === 2
    || (evt.buttons & 2) !== 0
    || (evt.ctrlKey && evt.button === 0);
  if (isSecondaryButton && pointerState.options.mouse.secondaryPan) {
    if (typeof evt.preventDefault === 'function') {
      evt.preventDefault();
    }
    startPointerPan(evt);
    return;
  }
  if (evt.button === 0) {
    if (spacePan) {
      startPointerPan(evt);
      return;
    }
    if (!pointerState.options.mouse.primaryDraw) {
      startPointerPan(evt);
      return;
    }
    if (tool === 'fill') {
      floodFill(world);
      return;
    }
    if (tool === 'picker') {
      pickColor(world);
      return;
    }
    beginBrushStroke(evt, world);
    return;
  }
  startPointerPan(evt);
});

canvas.addEventListener('pointermove', (evt) => {
  pointerState.registerPointerMove(evt);
  const pointer = getPointerPosition(evt);
  const world = screenToImage(pointer);
  if (!isPanning && !isPainting && pointerState.isStylusPointer(evt) && pointerState.isBarrelPanActive()) {
    startPointerPan(evt);
    return;
  }
  if (!isPanning && evt.pointerType !== 'touch' && (evt.buttons & 2) !== 0) {
    if (typeof evt.preventDefault === 'function') {
      evt.preventDefault();
    }
    startPointerPan(evt);
    return;
  }
  if (evt.pointerType === 'touch') {
    if (pointerState.shouldIgnoreTouch(evt)) {
      evt.preventDefault();
      return;
    }
    if (touchPointers.has(evt.pointerId)) {
      touchPointers.set(evt.pointerId, { x: evt.clientX, y: evt.clientY });
    }
    if (pinchState && pinchState.pointers.every((id) => touchPointers.has(id))) {
      const rect = canvas.getBoundingClientRect();
      const [idA, idB] = pinchState.pointers;
      const ptA = touchPointers.get(idA);
      const ptB = touchPointers.get(idB);
      if (ptA && ptB) {
        const dx = ptA.x - ptB.x;
        const dy = ptA.y - ptB.y;
        const distance = Math.hypot(dx, dy);
        if (distance > 0.0 && pinchState.startDistance > 0.0) {
          let nextScale = viewState.scale;
          if (pointerState.options.touch.enablePinchZoom) {
            const ratio = distance / pinchState.startDistance;
            nextScale = Math.min(Math.max(pinchState.startScale * ratio, 0.1), 40);
          } else {
            nextScale = pinchState.startScale;
          }
          viewState.scale = nextScale;
          if (pointerState.options.touch.enableRotate) {
            const angle = Math.atan2(dy, dx);
            const delta = normalizeAngleDelta(angle - pinchState.startAngle);
            const rotationDegrees = delta * RAD_TO_DEG;
            const deadzone = Math.max(pointerState.options.touch.rotationDeadzoneDegrees || 0, 0);
            if (pinchState.lastLoggedRotation === undefined || Math.abs(rotationDegrees - pinchState.lastLoggedRotation) >= Math.max(1, deadzone * 2)) {
              log('pinch rotation raw=' + rotationDegrees.toFixed(2) + ' deg');
              pinchState.lastLoggedRotation = rotationDegrees;
            }
            const applyRotation = Math.abs(rotationDegrees) >= deadzone;
            viewState.rotation = applyRotation
              ? normalizeAngle(pinchState.startRotation + delta)
              : pinchState.startRotation;
            if (applyRotation) {
              log('pinch rotation applied delta=' + rotationDegrees.toFixed(2) + ' deg total=' + (viewState.rotation * RAD_TO_DEG).toFixed(2));
            }
          }
          const midpoint = {
            x: (ptA.x + ptB.x) / 2,
            y: (ptA.y + ptB.y) / 2,
          };
          const midCanvas = {
            x: midpoint.x - rect.left,
            y: midpoint.y - rect.top,
          };
          setOffsetForImagePoint(pinchState.imageMid, midCanvas);
          userAdjustedScale = true;
          autoFitPending = false;
          draw();
          drawBrushPreview(null);
        }
      }
      evt.preventDefault();
      return;
    }
    if (isPanning && evt.pointerId === panPointerId) {
      const dx = pointer.x - lastPoint.x;
      const dy = pointer.y - lastPoint.y;
      viewState.offsetX += dx;
      viewState.offsetY += dy;
      lastPoint = pointer;
      draw();
      updateHoverInfo(null);
      evt.preventDefault();
      return;
    }
  }
  if (isPainting) {
    paintStroke(world);
    hoverPoint = world;
    drawBrushPreview(hoverPoint);
    updateHoverInfo(world);
    return;
  }
  if (!isPanning && !spacePan) {
    if (tool === 'brush') {
      hoverPoint = world;
      drawBrushPreview(hoverPoint);
    } else {
      hoverPoint = null;
      drawBrushPreview(null);
    }
    updateHoverInfo(world);
  }
  if (!isPanning) {
    return;
  }
  const dx = pointer.x - lastPoint.x;
  const dy = pointer.y - lastPoint.y;
  viewState.offsetX += dx;
  viewState.offsetY += dy;
  lastPoint = pointer;
  draw();
  updateHoverInfo(screenToImage(pointer));
});

canvas.addEventListener('dblclick', (evt) => {
  if (spacePan) {
    evt.preventDefault();
    resetView();
  }
});

if (supportsGestureEvents && canvas) {
  canvas.addEventListener('gesturestart', handleGestureStart, { passive: false });
  canvas.addEventListener('gesturechange', handleGestureChange, { passive: false });
  canvas.addEventListener('gestureend', handleGestureEnd, { passive: false });
}
if (supportsGestureEvents && viewer && viewer !== canvas) {
  viewer.addEventListener('gesturestart', handleGestureStart, { passive: false });
  viewer.addEventListener('gesturechange', handleGestureChange, { passive: false });
  viewer.addEventListener('gestureend', handleGestureEnd, { passive: false });
}

function stopInteraction(evt) {
  const stylusEvent = evt ? pointerState.isStylusPointer(evt) : false;
  const penShouldFinalize = stylusEvent && evt && (evt.type === 'pointerup' || evt.type === 'pointercancel');
  if (evt && (evt.type === 'pointerup' || evt.type === 'pointercancel')) {
    pointerState.registerPointerUp(evt);
  }
  if (evt && evt.button === 2 && typeof evt.preventDefault === 'function') {
    evt.preventDefault();
  }
  if (evt && evt.pointerType === 'touch') {
    touchPointers.delete(evt.pointerId);
    if (pinchState && pinchState.pointers.includes(evt.pointerId)) {
      pinchState = null;
    }
    if (evt.pointerId === panPointerId) {
      panPointerId = null;
    }
  }
  if (isPainting) {
    canvas.classList.remove('painting');
    finalizeStroke();
  }
  isPainting = false;
  isPanning = false;
  spacePan = false;
  updateCursor();
  lastPaintPoint = null;
  if (evt && evt.type === 'pointerleave') {
    clearHoverPreview();
  } else if (hoverPoint) {
    drawBrushPreview(hoverPoint);
    updateHoverInfo(hoverPoint);
  } else {
    clearHoverPreview();
  }
  if (evt && evt.pointerId !== undefined) {
    try {
      canvas.releasePointerCapture(evt.pointerId);
    } catch (_) {
      /* ignore */
    }
  } else if (activePointerId !== null) {
    try {
      canvas.releasePointerCapture(activePointerId);
    } catch (_) {
      /* ignore */
    }
  }
  activePointerId = null;
  if (penShouldFinalize && stylusToolOverride !== null) {
    const previousTool = stylusToolOverride;
    stylusToolOverride = null;
    if (previousTool !== tool) {
      setTool(previousTool);
    }
  }
  wheelRotationBuffer = 0;
}

function handleContextMenuEvent(evt) {
  if (evt && typeof evt.preventDefault === 'function') {
    evt.preventDefault();
  }
  if (activePointerId !== null) {
    try {
      canvas.releasePointerCapture(activePointerId);
    } catch (_) {
      /* ignore */
    }
    activePointerId = null;
  }
  if (isPainting) {
    finalizeStroke();
  }
  isPainting = false;
  isPanning = false;
  spacePan = false;
  updateCursor();
}

canvas.addEventListener('pointerup', stopInteraction);
canvas.addEventListener('pointerleave', stopInteraction);
canvas.addEventListener('pointercancel', stopInteraction);
canvas.addEventListener('mouseenter', () => {
  cursorInsideCanvas = true;
  updateCursor();
});
canvas.addEventListener('mouseleave', () => {
  cursorInsideCanvas = false;
  cursorInsideImage = false;
  updateCursor();
});
canvas.addEventListener('contextmenu', handleContextMenuEvent);
if (viewer) {
  viewer.addEventListener('contextmenu', handleContextMenuEvent);
}

window.addEventListener('keydown', (evt) => {
  const tag = evt.target && evt.target.tagName ? evt.target.tagName.toLowerCase() : '';
  if (tag === 'input') {
    return;
  }
  const key = evt.key.toLowerCase();
  const modifier = evt.ctrlKey || evt.metaKey;
  if (key === 'escape' && dropdownOpenId) {
    closeDropdown(dropdownRegistry.get(dropdownOpenId));
    evt.preventDefault();
    return;
  }
  if (!modifier && !evt.altKey && key === 'e') {
    startEraseOverride();
    evt.preventDefault();
  }
  if (!modifier && !evt.altKey && key === 'h') {
    resetView();
    evt.preventDefault();
    return;
  }
  if (modifier && key === 'z') {
    if (evt.shiftKey) {
      redo();
    } else {
      undo();
    }
    evt.preventDefault();
    return;
  }
  if (modifier && key === 'y') {
    redo();
    evt.preventDefault();
    return;
  }
  if (!modifier && !evt.altKey) {
    if (key === 'b') {
      setTool('brush');
      evt.preventDefault();
      return;
    }
    if (key === 'g') {
      setTool('fill');
      evt.preventDefault();
      return;
    }
    if (key === 'i') {
      setTool('picker');
      evt.preventDefault();
      return;
    }
    if (key === 'r') {
      const direction = evt.shiftKey ? -1 : 1;
      rotateView(direction * (Math.PI / 4));
      evt.preventDefault();
      return;
    }
    if (key === '[') {
      setBrushDiameter(brushDiameter - 1, true);
      evt.preventDefault();
      return;
    }
    if (key === ']') {
      setBrushDiameter(brushDiameter + 1, true);
      evt.preventDefault();
      return;
    }
    if (key === ' ') {
        spacePan = true;
      updateCursor();
      evt.preventDefault();
      return;
    }
  }
  if (key === 'm') {
    maskVisible = !maskVisible;
    updateMaskVisibilityLabel();
    draw();
    evt.preventDefault();
    return;
  }
  if (!modifier && !evt.altKey && key === 'n') {
    toggleColorMode();
    evt.preventDefault();
    return;
  }
  if (!modifier && key >= '0' && key <= '9') {
    const nextLabel = parseInt(key, 10);
    if (eraseActive) {
      erasePreviousLabel = nextLabel;
    } else {
      currentLabel = nextLabel;
      updateMaskLabel();
    }
    evt.preventDefault();
  }
});

window.addEventListener('keyup', (evt) => {
  if (evt.key === ' ') {
    spacePan = false;
    updateCursor();
  }
  if (evt.key && evt.key.toLowerCase() === 'e') {
    stopEraseOverride();
  }
});

function initialize() {
  log('initialize');
  autoFitPending = true;
  userAdjustedScale = false;
  const img = new Image();
  img.onload = () => {
    log('image loaded: ' + imgWidth + 'x' + imgHeight);
    offCtx.drawImage(img, 0, 0);
    originalImageData = offCtx.getImageData(0, 0, imgWidth, imgHeight);
    windowLow = 0;
    windowHigh = 255;
    currentGamma = DEFAULT_GAMMA;
    computeHistogram();
    setGamma(currentGamma, { emit: false });
    updateHistogramUI();
    applyImageAdjustments();
    redrawMaskCanvas();
    resizeCanvas();
    updateBrushControls();
  };
  img.onerror = (evt) => {
    const detail = evt?.message || 'unknown error';
    log('image load failed: ' + detail);
    setLoadingOverlay('Failed to load image', true);
  };
  img.src = imageDataUrl;
  updateCursor();
}

window.addEventListener('resize', resizeCanvas);
if (gammaSlider) {
  gammaSlider.addEventListener('input', (evt) => {
    const value = parseInt(evt.target.value, 10);
    const gamma = Number.isNaN(value) ? currentGamma : value / 100.0;
    setGamma(gamma);
    refreshSlider('gamma');
  });
}

if (gammaInput) {
  gammaInput.addEventListener('change', () => {
    const value = parseFloat(gammaInput.value);
    if (Number.isNaN(value)) {
      gammaInput.value = currentGamma.toFixed(2);
      return;
    }
    setGamma(value);
  });
}

brushSizeSlider.addEventListener('input', (evt) => {
  const value = parseInt(evt.target.value, 10);
  if (!Number.isNaN(value)) {
    setBrushDiameter(value, true);
    refreshSlider('brushSizeSlider');
  }
});

brushSizeInput.addEventListener('change', (evt) => {
  let value = parseInt(evt.target.value, 10);
  if (Number.isNaN(value)) {
    brushSizeInput.value = String(brushDiameter);
    return;
  }
  setBrushDiameter(value, true);
});

if (brushKernelModeSelect) {
  brushKernelModeSelect.addEventListener('change', (evt) => {
    setBrushKernelMode(evt.target.value);
  });
}

updateMaskLabel();
updateMaskVisibilityLabel();
updateToolInfo();
updateBrushControls();
updateColorModeLabel();
updateHoverInfo(null);
if (segmentButton) {
  segmentButton.addEventListener('click', () => {
    runSegmentation();
  });
}
if (histogramCanvas) {
  histogramCanvas.addEventListener('pointerdown', handleHistogramPointerDown);
  histogramCanvas.addEventListener('pointermove', handleHistogramPointerMove);
  histogramCanvas.addEventListener('pointerup', handleHistogramPointerUp);
  histogramCanvas.addEventListener('pointercancel', handleHistogramPointerUp);
  histogramCanvas.addEventListener('pointerleave', (evt) => {
    if (!histDragTarget) {
      updateHistogramCursor(evt);
    }
  });
  updateHistogramCursor();
}
if (maskOpacitySlider) {
  maskOpacitySlider.addEventListener('input', (evt) => {
    setMaskOpacity(evt.target.value);
  });
  maskOpacitySlider.addEventListener('change', (evt) => {
    setMaskOpacity(evt.target.value);
  });
}
if (maskOpacityInput) {
  maskOpacityInput.addEventListener('input', (evt) => {
    if (evt.target.value === '') {
      return;
    }
    setMaskOpacity(evt.target.value);
  });
  maskOpacityInput.addEventListener('change', (evt) => {
    setMaskOpacity(evt.target.value);
  });
  attachNumberInputStepper(maskOpacityInput, (delta) => {
    setMaskOpacity(maskOpacity + delta);
  });
}
syncMaskOpacityControls();

let bootstrapped = false;
function boot() {
  if (bootstrapped) {
    return;
  }
  bootstrapped = true;
  log('boot (readyState=' + document.readyState + ')');
  initialize();
  setTimeout(resizeCanvas, 100);
  flushLogs();
}

if (document.readyState === 'complete' || document.readyState === 'interactive') {
  boot();
} else {
  window.addEventListener('load', boot);
}
window.addEventListener('pywebviewready', () => {
  pywebviewReady = true;
  log('pywebview ready event');
  flushLogs();
  boot();
});
window.addEventListener('beforeunload', () => {
  shuttingDown = true;
  pendingLogs.length = 0;
});
