(function initOmniPointer(global) {
  'use strict';

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

  const existing = global.OmniPointer || {};
  existing.POINTER_OPTIONS = POINTER_OPTIONS;
  existing.createPointerState = createPointerState;
  global.OmniPointer = existing;
})(typeof window !== 'undefined' ? window : globalThis);
