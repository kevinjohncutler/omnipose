// Extracted from app.js lines 7693-8070

function initialize() {
  log('initialize');
  if (typeof refreshOppositeStepMapping === "function") {
    refreshOppositeStepMapping();
  }
  if (typeof clearAffinityGraphData === "function") {
    clearAffinityGraphData();
  }
  showAffinityGraph = true;
  if (affinityGraphToggle) {
    affinityGraphToggle.checked = true;
  }
  autoFitPending = true;
  userAdjustedScale = false;
  const img = new Image();
  img.onload = () => {
    log('image loaded: ' + imgWidth + 'x' + imgHeight);
    offCtx.drawImage(img, 0, 0);
    try {
      const sample = offCtx.getImageData(0, 0, 1, 1).data;
      log('offscreen sample rgba=' + Array.from(sample).join(','));
    } catch (sampleErr) {
      log('offscreen sample read failed: ' + (sampleErr && sampleErr.message ? sampleErr.message : sampleErr));
    }
    if (gl && webglPipelineRequested && !webglPipelineReady) {
      initializeWebglPipelineResources(img);
    }
    originalImageData = offCtx.getImageData(0, 0, imgWidth, imgHeight);
    windowLow = 0;
    windowHigh = 255;
    currentGamma = DEFAULT_GAMMA;
    computeHistogram();
    setGamma(currentGamma, { emit: false });
    updateHistogramUI();
    applyImageAdjustments();
    let restored = false;
    if (savedViewerState) {
      try {
        restoreViewerState(savedViewerState);
        restored = true;
      } catch (err) {
        console.warn('Failed to restore viewer state', err);
      }
    }
    if (!restored) {
      maskValues.fill(0);
      outlineState.fill(0);
      maskHasNonZero = false;
      undoStack.length = 0;
      redoStack.length = 0;
      needsMaskRedraw = true;
      updateHistoryButtons();
      applyMaskRedrawImmediate();
      // Force default camera + visibility so the image is guaranteed to be on-screen.
      viewState.scale = 1;
      viewState.offsetX = 0;
      viewState.offsetY = 0;
      viewState.rotation = 0;
      autoFitPending = true;
      userAdjustedScale = false;
      setImageVisible(true, { silent: true });
      maskVisible = true;
      if (maskVisibilityToggle) {
        maskVisibilityToggle.checked = true;
      }
      draw();
      resetView();
    }
    resizeCanvas();
    updateBrushControls();
    updateImageInfo();
    updateHistoryButtons();
  };
  img.onerror = (evt) => {
    const detail = evt?.message || 'unknown error';
    log('image load failed: ' + detail);
    setLoadingOverlay('Failed to load image', true);
  };
  img.src = imageDataUrl;
  updateCursor();
  ensureWebglOverlayReady();
  setupDragAndDrop();
  updateImageInfo();
}

window.addEventListener('resize', resizeCanvas);
let orientationResizePending = false;
window.addEventListener('orientationchange', () => {
  if (orientationResizePending) {
    return;
  }
  orientationResizePending = true;
  setTimeout(() => {
    orientationResizePending = false;
    resizeCanvas();
  }, 120);
});

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

if (brushSizeSlider) {
  brushSizeSlider.addEventListener('input', (evt) => {
  const value = parseInt(evt.target.value, 10);
  if (!Number.isNaN(value)) {
    setBrushDiameter(value, true);
    refreshSlider('brushSizeSlider');
  }
  });
}

if (brushSizeInput) {
  brushSizeInput.addEventListener('change', (evt) => {
  let value = parseInt(evt.target.value, 10);
  if (Number.isNaN(value)) {
    brushSizeInput.value = String(brushDiameter);
    return;
  }
  setBrushDiameter(value, true);
  });
}

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
if (clearMasksButton) {
  clearMasksButton.addEventListener('click', () => {
    promptClearMasks();
  });
}
if (clearCacheButton) {
  clearCacheButton.addEventListener('click', async () => {
    if (!confirm('Clear cached viewer state and reload?')) {
      return;
    }
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        const storageKeys = Object.keys(window.localStorage);
        storageKeys
          .filter((key) => key.startsWith('OMNI') || key.includes('omnipose'))
          .forEach((key) => {
            try {
              window.localStorage.removeItem(key);
            } catch (_) {
              /* ignore */
            }
          });
      }
    } catch (_) {
      // ignore storage access errors
    }
    try {
      const response = await fetch('/api/clear_cache', { method: 'POST', keepalive: true });
      if (!response.ok) {
        console.warn('clear_cache request failed', response.status);
      }
    } catch (_) {
      // ignore network errors
    }
    window.location.reload();
  });
}
if (toolStopButtons.length) {
  toolStopButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const mode = btn.getAttribute('data-mode') || 'draw';
      selectToolMode(mode);
      scheduleStateSave();
      if (viewer && typeof viewer.focus === 'function') {
        try {
          viewer.focus({ preventScroll: true });
        } catch (_) {
          viewer.focus();
        }
      }
    });
  });
}
updateToolButtons();
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
if (maskThresholdSlider) {
  maskThresholdSlider.addEventListener('input', (evt) => {
    setMaskThreshold(evt.target.value);
  });
  maskThresholdSlider.addEventListener('change', (evt) => {
    setMaskThreshold(evt.target.value);
  });
}
if (maskThresholdInput) {
  maskThresholdInput.addEventListener('input', (evt) => {
    if (evt.target.value === '') {
      return;
    }
    setMaskThreshold(evt.target.value);
  });
  maskThresholdInput.addEventListener('change', (evt) => {
    setMaskThreshold(evt.target.value);
  });
  attachNumberInputStepper(maskThresholdInput, (delta) => {
    setMaskThreshold(maskThreshold + delta);
  });
}
if (flowThresholdSlider) {
  flowThresholdSlider.addEventListener('input', (evt) => {
    setFlowThreshold(evt.target.value);
  });
  flowThresholdSlider.addEventListener('change', (evt) => {
    setFlowThreshold(evt.target.value);
  });
}
if (flowThresholdInput) {
  flowThresholdInput.addEventListener('input', (evt) => {
    if (evt.target.value === '') {
      return;
    }
    setFlowThreshold(evt.target.value);
  });
  flowThresholdInput.addEventListener('change', (evt) => {
    setFlowThreshold(evt.target.value);
  });
  attachNumberInputStepper(flowThresholdInput, (delta) => {
    setFlowThreshold(flowThreshold + delta);
  });
}
if (clusterToggle) {
  clusterToggle.addEventListener('change', (evt) => {
    setClusterEnabled(evt.target.checked);
    scheduleStateSave();
  });
}
if (affinityToggle) {
  affinityToggle.addEventListener('change', (evt) => {
    setAffinitySegEnabled(evt.target.checked);
    scheduleStateSave();
  });
}
if (affinityGraphToggle) {
  affinityGraphToggle.addEventListener('change', (evt) => {
    showAffinityGraph = Boolean(evt.target.checked);
    if (showAffinityGraph) {
      // Do not rebuild or mutate the affinity graph here; only (re)build segments if we already have values
      if (affinityGraphInfo && !affinityGraphInfo.segments) {
        buildAffinityGraphSegments();
      }
      if (webglOverlay && webglOverlay.enabled) {
        webglOverlay.needsGeometryRebuild = true;
      }
    } else {
      clearWebglOverlaySurface();
    }
    markAffinityGeometryDirty();
    draw();
    scheduleStateSave();
  });
}
if (flowOverlayToggle) {
  flowOverlayToggle.addEventListener('change', (evt) => {
    if (!flowOverlayImage || !flowOverlayImage.complete) {
      flowOverlayToggle.checked = false;
      showFlowOverlay = false;
      return;
    }
    showFlowOverlay = evt.target.checked;
    draw();
    scheduleStateSave();
  });
}
if (distanceOverlayToggle) {
  distanceOverlayToggle.addEventListener('change', (evt) => {
    if (!distanceOverlayImage || !distanceOverlayImage.complete) {
      distanceOverlayToggle.checked = false;
      showDistanceOverlay = false;
      return;
    }
    showDistanceOverlay = evt.target.checked;
    draw();
    scheduleStateSave();
  });
}
if (imageVisibilityToggle) {
  imageVisibilityToggle.addEventListener('change', (evt) => {
    const visible = Boolean(evt.target.checked);
    if (visible === imageVisible) {
      return;
    }
    setImageVisible(visible);
  });
}
if (maskVisibilityToggle) {
  maskVisibilityToggle.addEventListener('change', (evt) => {
    const visible = Boolean(evt.target.checked);
    if (visible === maskVisible) {
      return;
    }
    maskVisible = visible;
    updateMaskVisibilityLabel();
    draw();
    scheduleStateSave();
  });
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
syncMaskThresholdControls();
syncFlowThresholdControls();
setClusterEnabled(clusterEnabled, { silent: true });
setAffinitySegEnabled(affinitySegEnabled, { silent: true });
if (maskVisibilityToggle) {
  maskVisibilityToggle.checked = maskVisible;
}
setImageVisible(imageVisible, { silent: true });
if (imageVisibilityToggle) {
  imageVisibilityToggle.checked = imageVisible;
}
syncMaskOpacityControls();

let bootstrapped = false;
if (typeof window !== 'undefined') {
  window.__omniDebug = window.__omniDebug || {};
  window.__omniDebug.waitForFirstDraw = function waitForFirstDraw(timeoutMs = 4000) {
    const deadline = Date.now() + Math.max(0, timeoutMs);
    return new Promise((resolve) => {
      function check() {
        if (window.__OMNI_LAST_DRAW_TS) {
          resolve(true);
          return;
        }
        if (Date.now() >= deadline) {
          resolve(false);
          return;
        }
        setTimeout(check, 50);
      }
      check();
    });
  };
  window.__omniDebug.captureCanvas = function captureCanvas() {
    const canvasEl = document.getElementById('canvas');
    if (!canvasEl) {
      return { ok: false, reason: 'no-canvas' };
    }
    try {
      const dataUrl = canvasEl.toDataURL('image/png');
      return { ok: true, dataUrl };
    } catch (err) {
      return { ok: false, reason: String(err && err.message ? err.message : err) };
    }
  };
}
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
// no incremental rebuild method; geometry rebuilds in ensureWebglGeometry
