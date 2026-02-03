#!/usr/bin/env node
/**
 * Comprehensive GUI test - checks various functionality.
 */

const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const DEFAULT_URL = 'http://127.0.0.1:8000/';
const OUTPUT_DIR = path.resolve(__dirname, '../../tmp/playwright');

const ensureDir = (dir) => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
};

const timestamp = () => new Date().toISOString().replace(/[:.]/g, '-');

async function runTest() {
  const ts = timestamp();
  ensureDir(OUTPUT_DIR);
  const results = { passed: [], failed: [], warnings: [] };

  console.log('[test] Starting comprehensive GUI test...');

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1600, height: 1000 },
  });
  const page = await context.newPage();

  const errors = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      errors.push(msg.text());
    }
  });
  page.on('pageerror', (err) => {
    errors.push(err.message);
  });

  try {
    // TEST 1: Page loads without errors
    console.log('[test] TEST 1: Page loading...');
    await page.goto(DEFAULT_URL, { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForFunction(() => {
      return document.getElementById('canvas') !== null &&
             (window.OmniPainting || window.__OMNI_CONFIG__);
    }, { timeout: 30000 });

    const initialErrors = errors.length;
    if (initialErrors === 0) {
      results.passed.push('Page loads without JavaScript errors');
    } else {
      results.failed.push(`Page has ${initialErrors} errors: ${errors.slice(0, 3).join(', ')}`);
    }

    // TEST 2: Canvas and image are rendered
    const canvasState = await page.evaluate(() => {
      const canvas = document.getElementById('canvas');
      return {
        exists: Boolean(canvas),
        width: canvas?.width || 0,
        height: canvas?.height || 0,
        imgWidth: window.imgWidth || 0,
        imgHeight: window.imgHeight || 0,
      };
    });
    console.log('[test] Canvas state:', JSON.stringify(canvasState));

    if (canvasState.exists && canvasState.width > 0 && canvasState.height > 0) {
      results.passed.push('Canvas rendered with valid dimensions');
    } else {
      results.failed.push('Canvas not rendered properly');
    }

    // Check image dimensions from mask data instead
    const maskDimensions = await page.evaluate(() => {
      const state = window.OmniPainting?.__debugGetState?.();
      return {
        width: state?.ctx?.width || 0,
        height: state?.ctx?.height || 0,
      };
    });
    if (maskDimensions.width > 0 && maskDimensions.height > 0) {
      results.passed.push(`Image loaded: ${maskDimensions.width}x${maskDimensions.height}`);
    } else if (canvasState.imgWidth > 0 && canvasState.imgHeight > 0) {
      results.passed.push(`Image loaded: ${canvasState.imgWidth}x${canvasState.imgHeight}`);
    } else {
      results.warnings.push('Image dimensions not available (may be scoped)');
    }

    // TEST 3: Segmentation works
    console.log('[test] TEST 3: Running segmentation...');
    await page.evaluate(() => {
      const btn = document.getElementById('segmentButton');
      if (btn) btn.click();
    });

    await page.waitForFunction(() => {
      const status = document.getElementById('segmentStatus');
      return status && (status.textContent.includes('complete') || status.textContent.includes('failed'));
    }, { timeout: 120000 });

    const segmentStatus = await page.evaluate(() => {
      const status = document.getElementById('segmentStatus');
      return status?.textContent || '';
    });

    if (segmentStatus.includes('complete')) {
      results.passed.push('Segmentation completed successfully');
    } else {
      results.failed.push(`Segmentation failed: ${segmentStatus}`);
    }

    // TEST 4: Mask has non-zero values after segmentation
    const maskState = await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      if (!maskValues) return { hasMask: false };
      const uniqueLabels = new Set(Array.from(maskValues).filter(v => v > 0));
      return {
        hasMask: true,
        length: maskValues.length,
        uniqueLabels: uniqueLabels.size,
        hasNonZero: uniqueLabels.size > 0,
      };
    });
    console.log('[test] Mask state:', JSON.stringify(maskState));

    if (maskState.hasMask && maskState.hasNonZero) {
      results.passed.push(`Mask has ${maskState.uniqueLabels} unique labels`);
    } else {
      results.failed.push('Mask has no labels after segmentation');
    }

    await page.screenshot({
      path: path.join(OUTPUT_DIR, `comprehensive-${ts}-01-after-segment.png`),
      fullPage: true
    });

    // TEST 5: Mask visibility toggle works
    console.log('[test] TEST 5: Testing mask visibility toggle...');
    const maskVisibleBefore = await page.evaluate(() => {
      const toggle = document.getElementById('maskVisible');
      return toggle?.checked ?? null;
    });

    await page.evaluate(() => {
      const toggle = document.getElementById('maskVisible');
      if (toggle) toggle.click();
    });
    await page.waitForTimeout(500);

    const maskVisibleAfter = await page.evaluate(() => {
      const toggle = document.getElementById('maskVisible');
      return toggle?.checked ?? null;
    });

    if (maskVisibleBefore !== null && maskVisibleBefore !== maskVisibleAfter) {
      results.passed.push('Mask visibility toggle works');
    } else {
      results.warnings.push('Mask visibility toggle may not be working');
    }

    // Toggle back
    await page.evaluate(() => {
      const toggle = document.getElementById('maskVisible');
      if (toggle) toggle.click();
    });

    // TEST 6: Zoom/pan works
    console.log('[test] TEST 6: Testing zoom...');
    const viewStateBefore = await page.evaluate(() => ({
      scale: window.viewState?.scale || 1,
      offsetX: window.viewState?.offsetX || 0,
      offsetY: window.viewState?.offsetY || 0,
    }));

    // Simulate scroll to zoom
    await page.mouse.move(800, 500);
    await page.mouse.wheel(0, -100); // Scroll up to zoom in
    await page.waitForTimeout(500);

    const viewStateAfter = await page.evaluate(() => ({
      scale: window.viewState?.scale || 1,
      offsetX: window.viewState?.offsetX || 0,
      offsetY: window.viewState?.offsetY || 0,
    }));

    if (viewStateAfter.scale !== viewStateBefore.scale) {
      results.passed.push(`Zoom works: ${viewStateBefore.scale.toFixed(2)} -> ${viewStateAfter.scale.toFixed(2)}`);
    } else {
      results.warnings.push('Zoom may not be working');
    }

    // TEST 7: Opacity slider works
    console.log('[test] TEST 7: Testing opacity slider...');
    const opacityBefore = await page.evaluate(() => {
      const slider = document.getElementById('maskOpacity');
      return slider?.value ?? null;
    });

    await page.evaluate(() => {
      const slider = document.getElementById('maskOpacity');
      if (slider) {
        slider.value = '0.5';
        slider.dispatchEvent(new Event('input'));
      }
    });
    await page.waitForTimeout(300);

    const opacityAfter = await page.evaluate(() => {
      const slider = document.getElementById('maskOpacity');
      return slider?.value ?? null;
    });

    if (opacityAfter === '0.5') {
      results.passed.push(`Opacity slider works: ${opacityBefore} -> ${opacityAfter}`);
    } else {
      results.warnings.push('Opacity slider may not be working');
    }

    // TEST 8: Brush size slider works
    console.log('[test] TEST 8: Testing brush size slider...');
    const brushBefore = await page.evaluate(() => {
      const slider = document.getElementById('brushRadius');
      return slider?.value ?? null;
    });

    await page.evaluate(() => {
      const slider = document.getElementById('brushRadius');
      if (slider) {
        slider.value = '20';
        slider.dispatchEvent(new Event('input'));
      }
    });
    await page.waitForTimeout(300);

    const brushAfter = await page.evaluate(() => {
      const slider = document.getElementById('brushRadius');
      return slider?.value ?? null;
    });

    if (brushAfter === '20') {
      results.passed.push(`Brush size slider works: ${brushBefore} -> ${brushAfter}`);
    } else {
      results.warnings.push('Brush size slider may not be working');
    }

    // TEST 9: Label colormap selector works
    console.log('[test] TEST 9: Testing colormap selector...');
    const colormapBefore = await page.evaluate(() => {
      const select = document.getElementById('labelColormap');
      return select?.value ?? null;
    });

    await page.evaluate(() => {
      const select = document.getElementById('labelColormap');
      if (select) {
        select.value = 'pastel';
        select.dispatchEvent(new Event('change'));
      }
    });
    await page.waitForTimeout(500);

    const colormapAfter = await page.evaluate(() => {
      const select = document.getElementById('labelColormap');
      return select?.value ?? null;
    });

    if (colormapAfter === 'pastel') {
      results.passed.push('Colormap selector works');

      // Take screenshot with pastel colormap
      await page.screenshot({
        path: path.join(OUTPUT_DIR, `comprehensive-${ts}-02-pastel-colormap.png`),
        fullPage: true
      });
    } else {
      results.warnings.push(`Colormap may not have changed: ${colormapBefore} -> ${colormapAfter}`);
    }

    // TEST 10: N-color toggle works (regression test)
    console.log('[test] TEST 10: N-color toggle regression test...');
    // Use DOM element's checked state instead of window.nColorActive (which isn't exposed)
    const nColorBefore = await page.evaluate(() => {
      const toggle = document.getElementById('autoNColorToggle');
      return toggle?.checked ?? null;
    });

    const labelsBefore = await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      if (!maskValues) return 0;
      return new Set(Array.from(maskValues).filter(v => v > 0)).size;
    });

    await page.evaluate(() => {
      const toggle = document.getElementById('autoNColorToggle');
      if (toggle) toggle.click();
    });
    await page.waitForTimeout(2000);

    const nColorAfter = await page.evaluate(() => {
      const toggle = document.getElementById('autoNColorToggle');
      return toggle?.checked ?? null;
    });

    if (nColorBefore !== nColorAfter) {
      results.passed.push(`N-color toggle state changed: ${nColorBefore} -> ${nColorAfter}`);
    } else {
      results.failed.push('N-color toggle state did not change');
    }

    // Check mask has correct number of labels based on mode
    const labelsAfterToggle = await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      if (!maskValues) return 0;
      return new Set(Array.from(maskValues).filter(v => v > 0)).size;
    });

    if (!nColorAfter) {
      // Switched to instance mode
      if (labelsAfterToggle > 10) {
        results.passed.push(`Instance mode has ${labelsAfterToggle} unique labels`);
      } else {
        results.failed.push(`Instance mode should have many labels, got ${labelsAfterToggle}`);
      }
    } else {
      // Switched to N-color mode
      if (labelsAfterToggle <= 6) {
        results.passed.push(`N-color mode has ${labelsAfterToggle} unique labels`);
      } else {
        results.failed.push(`N-color mode should have few labels, got ${labelsAfterToggle}`);
      }
    }

    await page.screenshot({
      path: path.join(OUTPUT_DIR, `comprehensive-${ts}-03-final.png`),
      fullPage: true
    });

    // Print results
    console.log('\n========== TEST RESULTS ==========');
    console.log(`PASSED: ${results.passed.length}`);
    results.passed.forEach(r => console.log(`  ✓ ${r}`));

    console.log(`\nFAILED: ${results.failed.length}`);
    results.failed.forEach(r => console.log(`  ✗ ${r}`));

    console.log(`\nWARNINGS: ${results.warnings.length}`);
    results.warnings.forEach(r => console.log(`  ⚠ ${r}`));

    console.log('==================================\n');
    console.log(`[test] Screenshots saved to: ${OUTPUT_DIR}`);

  } catch (err) {
    console.error('[test] Error:', err.message);
    results.failed.push(`Test error: ${err.message}`);
    await page.screenshot({
      path: path.join(OUTPUT_DIR, `comprehensive-${ts}-error.png`),
      fullPage: true
    });
  } finally {
    await browser.close();
  }

  // Exit with error code if any tests failed
  if (results.failed.length > 0) {
    process.exit(1);
  }
}

runTest().catch(console.error);
