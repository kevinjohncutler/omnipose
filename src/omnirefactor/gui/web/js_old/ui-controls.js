// Extracted from app.js ui controls section

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

function updateColorModeLabel() {
  if (!colorMode) {
    return;
  }
  const mode = nColorActive ? 'N-Color' : 'Palette';
  colorMode.textContent = 'Mask Colors: ' + mode + " (toggle with 'N')";
}

function toggleColorMode() {
  if (!nColorActive) {
    // ON: compute groups from current mask and write into maskValues.
    lastLabelBeforeNColor = currentLabel;
    const prevLabel = currentLabel | 0;
    recomputeNColorFromCurrentMask(true).then((ok) => {
      if (!ok) console.warn('N-color mapping failed');
      // Repaint outlines with new palette without changing the graph
      rebuildOutlineFromAffinity();
      // Preserve currentLabel if valid in group space; otherwise default to 1
      try {
        let maxGroup = 0;
        for (let i = 0, n = maskValues.length; i < n; i += Math.max(1, Math.floor(n / 2048))) {
          const g = maskValues[i] | 0; if (g > maxGroup) maxGroup = g;
        }
        // Fallback full scan if sample returned 0
        if (maxGroup === 0) {
          for (let i = 0; i < maskValues.length; i += 1) { const g = maskValues[i] | 0; if (g > maxGroup) maxGroup = g; }
        }
        if (!(prevLabel >= 1 && prevLabel <= maxGroup)) {
          currentLabel = maxGroup >= 1 ? 1 : 0;
        }
      } catch (_) { currentLabel = 1; }
      updateMaskLabel();
      updateColorModeLabel();
      draw();
      scheduleStateSave();
    });
    return;
  }
  // OFF: relabel by affinity using current groups; result is instance labels
  try { window.__pendingRelabelSelection = null; } catch (_) {}
  const prevLabel = currentLabel | 0;
  relabelFromAffinity()
    .then((ok) => {
      if (!ok) console.warn('relabel_from_affinity failed during N-color OFF');
      nColorActive = false;
      clearColorCaches();
      if (isWebglPipelineActive()) {
        markMaskTextureFullDirty();
        markOutlineTextureFullDirty();
      } else {
        redrawMaskCanvas();
      }
      // Do not modify the affinity graph when toggling OFF
      // Preserve currentLabel if still present; else default to 1
      try {
        let found = false;
        if (prevLabel > 0) {
          for (let i = 0; i < maskValues.length; i += Math.max(1, Math.floor(maskValues.length / 2048))) {
            if ((maskValues[i] | 0) === prevLabel) { found = true; break; }
          }
          if (!found) {
            for (let i = 0; i < maskValues.length; i += 1) { if ((maskValues[i] | 0) === prevLabel) { found = true; break; } }
          }
        }
        if (!found) {
          currentLabel = 1;
        }
      } catch (_) { currentLabel = 1; }
      updateMaskLabel();
      updateColorModeLabel();
      draw();
      scheduleStateSave();
    })
    .catch((err) => {
      console.warn('N-color OFF relabel failed', err);
    });
}

// buildLinksFromCurrentGraph removed: we keep a single connectivity source based on raw labels only

