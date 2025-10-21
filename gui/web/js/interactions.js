(function initOmniInteractions(global) {
  'use strict';

  const state = {
    ctx: null,
    hoverPoint: null,
    hoverScreenPoint: null,
    pointerFrameScheduled: false,
    pendingPanPointer: null,
    hoverUpdatePending: false,
    pendingHoverScreenPoint: null,
    pendingHoverHasPreview: false,
    gestureFrameScheduled: false,
    pendingGestureUpdate: null,
  };

  function clonePoint(point) {
    if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y)) {
      return null;
    }
    return { x: Number(point.x), y: Number(point.y) };
  }

  function init(options) {
    state.ctx = Object.assign({
      pointerState: null,
      getTool: () => 'brush',
      getPreviewToolTypes: () => new Set(),
      getCrosshairToolTypes: () => new Set(),
      isCursorInsideImage: () => false,
      drawBrushPreview: () => {},
      updateHoverInfo: () => {},
      draw: () => {},
      screenToImage: (pt) => pt,
      setCursorHold: () => {},
      cursorStyles: { dot: 'crosshair' },
      normalizeAngle: (angle) => angle,
      setOffsetForImagePoint: () => {},
      getViewState: () => ({ scale: 1, rotation: 0, offsetX: 0, offsetY: 0 }),
      setViewStateScale: () => {},
      setViewStateRotation: () => {},
      applyPanDelta: () => {},
      markViewStateDirty: () => {},
      setUserAdjustedScale: () => {},
      setAutoFitPending: () => {},
      processPaintQueue: () => null,
      isPainting: () => false,
      isPanning: () => false,
      getLastPointer: () => ({ x: 0, y: 0 }),
      setLastPointer: () => {},
      getGestureState: () => null,
      setGestureState: () => {},
      requestAnimationFrame: typeof global.requestAnimationFrame === 'function'
        ? global.requestAnimationFrame.bind(global)
        : (fn) => global.setTimeout(fn, 16),
    }, options || {});
    state.hoverPoint = null;
    state.hoverScreenPoint = null;
    state.pointerFrameScheduled = false;
    state.pendingPanPointer = null;
    state.hoverUpdatePending = false;
    state.pendingHoverScreenPoint = null;
    state.pendingHoverHasPreview = false;
    state.gestureFrameScheduled = false;
    state.pendingGestureUpdate = null;
  }

  function ctx() {
    if (!state.ctx) {
      throw new Error('OmniInteractions.init must be called before use');
    }
    return state.ctx;
  }

  function getHoverPoint() {
    return state.hoverPoint ? { x: state.hoverPoint.x, y: state.hoverPoint.y } : null;
  }

  function getHoverScreenPoint() {
    return state.hoverScreenPoint ? { x: state.hoverScreenPoint.x, y: state.hoverScreenPoint.y } : null;
  }

  function setHoverState(worldPoint, screenPoint) {
    state.hoverPoint = clonePoint(worldPoint);
    state.hoverScreenPoint = clonePoint(screenPoint);
  }

  function clearHoverState() {
    state.hoverPoint = null;
    state.hoverScreenPoint = null;
  }

  function renderHoverPreview() {
    const context = ctx();
    const drawBrushPreview = context.drawBrushPreview;
    if (typeof drawBrushPreview !== 'function') {
      return;
    }
    const point = state.hoverPoint;
    if (point) {
      const previewTools = context.getPreviewToolTypes ? context.getPreviewToolTypes() : null;
      const crosshairTools = context.getCrosshairToolTypes ? context.getCrosshairToolTypes() : null;
      const tool = context.getTool ? context.getTool() : 'brush';
      if (previewTools && typeof previewTools.has === 'function' && previewTools.has(tool)) {
        drawBrushPreview(point);
      } else if (
        crosshairTools
        && typeof crosshairTools.has === 'function'
        && crosshairTools.has(tool)
        && context.isCursorInsideImage
        && context.isCursorInsideImage()
      ) {
        drawBrushPreview(point, { crosshairOnly: true });
      } else {
        drawBrushPreview(null);
      }
    } else {
      drawBrushPreview(null);
    }
  }

  function clearHoverPreview() {
    clearHoverState();
    const context = ctx();
    if (typeof context.drawBrushPreview === 'function') {
      context.drawBrushPreview(null);
    }
    if (typeof context.updateHoverInfo === 'function') {
      context.updateHoverInfo(null);
    }
  }

  function queuePanPointer(screenPoint) {
    state.pendingPanPointer = clonePoint(screenPoint);
    state.hoverUpdatePending = true;
    state.pendingHoverScreenPoint = null;
    state.pendingHoverHasPreview = false;
  }

  function queueHoverUpdate(screenPoint, { hasPreview = false } = {}) {
    state.pendingHoverScreenPoint = clonePoint(screenPoint);
    state.pendingHoverHasPreview = Boolean(hasPreview);
    state.hoverUpdatePending = true;
  }

  function clearPending() {
    state.pendingPanPointer = null;
    state.hoverUpdatePending = false;
    state.pendingHoverScreenPoint = null;
    state.pendingHoverHasPreview = false;
  }

  function setPendingGestureUpdate(update) {
    state.pendingGestureUpdate = update ? {
      origin: clonePoint(update.origin),
      scale: Number.isFinite(update.scale) ? update.scale : 1,
      rotation: Number.isFinite(update.rotation) ? update.rotation : 0,
    } : null;
  }

  function resetGestureScheduling() {
    state.pendingGestureUpdate = null;
    state.gestureFrameScheduled = false;
  }

  function scheduleGestureUpdate() {
    if (state.gestureFrameScheduled) {
      return;
    }
    state.gestureFrameScheduled = true;
    const raf = ctx().requestAnimationFrame;
    raf(() => {
      state.gestureFrameScheduled = false;
      applyPendingGestureUpdate();
    });
  }

  function applyPendingGestureUpdate() {
    const context = ctx();
    const gestureState = context.getGestureState ? context.getGestureState() : null;
    if (!gestureState || !state.pendingGestureUpdate) {
      return;
    }
    const pointerState = context.pointerState || {};
    const touchOptions = pointerState.options ? pointerState.options.touch || {} : {};
    const pending = state.pendingGestureUpdate;
    state.pendingGestureUpdate = null;
    const origin = clonePoint(pending.origin) || { x: 0, y: 0 };
    const scale = Number.isFinite(pending.scale) ? pending.scale : 1;
    const rotation = Number.isFinite(pending.rotation) ? pending.rotation : 0;
    gestureState.origin = origin;
    const imagePoint = context.screenToImage(origin);
    gestureState.imagePoint = imagePoint;
    setHoverState(imagePoint, origin);
    if (typeof context.setCursorHold === 'function' && context.cursorStyles && context.cursorStyles.dot) {
      context.setCursorHold(context.cursorStyles.dot);
    }
    if (touchOptions.enablePinchZoom) {
      const startScale = Number.isFinite(gestureState.startScale) ? gestureState.startScale : 1;
      const nextScale = startScale * scale;
      if (Number.isFinite(nextScale) && nextScale > 0) {
        const clampedScale = Math.min(Math.max(nextScale, 0.1), 40);
        if (typeof context.setViewStateScale === 'function') {
          context.setViewStateScale(clampedScale);
        }
      }
    }
    if (touchOptions.enableRotate) {
      const deadzone = Math.max(touchOptions.rotationDeadzoneDegrees || 0, 0);
      const startRotation = Number.isFinite(gestureState.startRotation) ? gestureState.startRotation : 0;
      if (Math.abs(rotation) >= deadzone) {
        const radians = (rotation * Math.PI) / 180;
        const normalized = context.normalizeAngle
          ? context.normalizeAngle(startRotation + radians)
          : startRotation + radians;
        if (typeof context.setViewStateRotation === 'function') {
          context.setViewStateRotation(normalized);
        }
        if (typeof context.setCursorHold === 'function' && context.cursorStyles && context.cursorStyles.dot) {
          context.setCursorHold(context.cursorStyles.dot);
        }
      } else if (typeof context.setViewStateRotation === 'function') {
        context.setViewStateRotation(startRotation);
      }
    }
    if (typeof context.setOffsetForImagePoint === 'function') {
      context.setOffsetForImagePoint(gestureState.imagePoint, gestureState.origin);
    }
    if (typeof context.setUserAdjustedScale === 'function') {
      context.setUserAdjustedScale(true);
    }
    if (typeof context.setAutoFitPending === 'function') {
      context.setAutoFitPending(false);
    }
    if (typeof context.draw === 'function') {
      context.draw();
    }
    if (typeof context.updateHoverInfo === 'function') {
      context.updateHoverInfo(state.hoverPoint || null);
    }
    renderHoverPreview();
  }

  function schedulePointerFrame() {
    if (state.pointerFrameScheduled) {
      return;
    }
    state.pointerFrameScheduled = true;
    const raf = ctx().requestAnimationFrame;
    raf(() => {
      state.pointerFrameScheduled = false;
      processPointerFrame();
    });
  }

  function processPointerFrame() {
    const context = ctx();
    let panUpdated = false;
    if (context.isPanning && context.isPanning() && state.pendingPanPointer) {
      const last = context.getLastPointer ? context.getLastPointer() : { x: 0, y: 0 };
      const dx = state.pendingPanPointer.x - last.x;
      const dy = state.pendingPanPointer.y - last.y;
      if (dx !== 0 || dy !== 0) {
        if (typeof context.applyPanDelta === 'function') {
          context.applyPanDelta(dx, dy);
        }
        if (typeof context.markViewStateDirty === 'function') {
          context.markViewStateDirty();
        }
        panUpdated = true;
      }
      if (typeof context.setLastPointer === 'function') {
        context.setLastPointer(state.pendingPanPointer);
      }
      const worldPoint = context.screenToImage(state.pendingPanPointer);
      setHoverState(worldPoint, state.pendingPanPointer);
      if (typeof context.updateHoverInfo === 'function') {
        context.updateHoverInfo(worldPoint);
      }
      state.hoverUpdatePending = false;
      state.pendingHoverScreenPoint = null;
      state.pendingHoverHasPreview = false;
      if (typeof context.setCursorHold === 'function' && context.cursorStyles && context.cursorStyles.dot) {
        context.setCursorHold(context.cursorStyles.dot);
      }
      state.pendingPanPointer = null;
    }

    if (context.isPainting && context.isPainting()) {
      const lastWorld = typeof context.processPaintQueue === 'function'
        ? context.processPaintQueue()
        : null;
      if (lastWorld) {
        setHoverState(lastWorld, null);
        if (typeof context.drawBrushPreview === 'function') {
          context.drawBrushPreview(lastWorld);
        }
        if (typeof context.updateHoverInfo === 'function') {
          context.updateHoverInfo(lastWorld);
        }
        state.hoverUpdatePending = false;
        state.pendingHoverScreenPoint = null;
        state.pendingHoverHasPreview = false;
      }
    } else if (state.hoverUpdatePending) {
      if (state.pendingHoverScreenPoint) {
        const world = context.screenToImage(state.pendingHoverScreenPoint);
        setHoverState(world, state.pendingHoverScreenPoint);
        const hasPreview = Boolean(state.pendingHoverHasPreview);
        const previewTools = context.getPreviewToolTypes ? context.getPreviewToolTypes() : null;
        const crosshairTools = context.getCrosshairToolTypes ? context.getCrosshairToolTypes() : null;
        const tool = context.getTool ? context.getTool() : 'brush';
        if (hasPreview && previewTools && previewTools.has(tool)) {
          if (typeof context.drawBrushPreview === 'function') {
            context.drawBrushPreview(world);
          }
        } else if (
          crosshairTools
          && crosshairTools.has
          && crosshairTools.has(tool)
          && context.isCursorInsideImage
          && context.isCursorInsideImage()
        ) {
          if (typeof context.drawBrushPreview === 'function') {
            context.drawBrushPreview(world, { crosshairOnly: true });
          }
        } else if (typeof context.drawBrushPreview === 'function') {
          context.drawBrushPreview(null);
        }
        if (typeof context.updateHoverInfo === 'function') {
          context.updateHoverInfo(world);
        }
      } else {
        clearHoverState();
        if (typeof context.drawBrushPreview === 'function') {
          context.drawBrushPreview(null);
        }
        if (typeof context.updateHoverInfo === 'function') {
          context.updateHoverInfo(null);
        }
      }
      state.hoverUpdatePending = false;
      state.pendingHoverScreenPoint = null;
      state.pendingHoverHasPreview = false;
    }

    if (panUpdated && typeof context.draw === 'function') {
      context.draw();
    }
  }

  const api = global.OmniInteractions || {};
  Object.assign(api, {
    init,
    getHoverPoint,
    getHoverScreenPoint,
    setHoverState,
    clearHoverState,
    renderHoverPreview,
    clearHoverPreview,
    queuePanPointer,
    queueHoverUpdate,
    clearPending,
    schedulePointerFrame,
    scheduleGestureUpdate,
    setPendingGestureUpdate,
    resetGestureScheduling,
    applyPendingGestureUpdate,
  });
  global.OmniInteractions = api;
})(typeof window !== 'undefined' ? window : globalThis);
