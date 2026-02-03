#!/usr/bin/env node
/**
 * Test drawing/filling functionality and affinity graph updates.
 * Verifies that painting updates the affinity graph so toggle works correctly.
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

  console.log('[test] Starting drawing/affinity test...');

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1600, height: 1000 },
  });
  const page = await context.newPage();

  page.on('console', (msg) => {
    if (msg.type() === 'log' || msg.type() === 'warn' || msg.type() === 'error') {
      console.log(`[page][${msg.type()}] ${msg.text()}`);
    }
  });

  try {
    await page.goto(DEFAULT_URL, { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForFunction(() => {
      return document.getElementById('canvas') !== null &&
             (window.OmniPainting || window.__OMNI_CONFIG__);
    }, { timeout: 30000 });

    console.log('[test] Page loaded, triggering segmentation first...');

    // Trigger segmentation
    await page.evaluate(() => {
      const btn = document.getElementById('segmentButton');
      if (btn) btn.click();
    });

    // Wait for segmentation to complete
    console.log('[test] Waiting for segmentation to complete...');
    await page.waitForFunction(() => {
      const status = document.getElementById('segmentStatus');
      return status && (status.textContent.includes('complete') || status.textContent.includes('failed'));
    }, { timeout: 120000 });
    await page.waitForTimeout(2000);

    // Check initial state
    const initialState = await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      const uniqueLabels = maskValues ? new Set(Array.from(maskValues).filter(v => v > 0)) : new Set();
      return {
        nColorActive: window.nColorActive,
        uniqueLabelCount: uniqueLabels.size,
        hasAffinityGraph: Boolean(window.affinityGraphInfo && window.affinityGraphInfo.graph),
        affinityNodeCount: window.affinityGraphInfo?.graph?.size || 0,
      };
    });
    console.log('[test] After segmentation:', JSON.stringify(initialState, null, 2));

    // Take screenshot
    await page.screenshot({
      path: path.join(OUTPUT_DIR, `drawing-test-${ts}-01-after-segment.png`),
      fullPage: true
    });

    // STEP 1: Toggle N-color OFF to get instance labels
    console.log('[test] STEP 1: Toggling N-color OFF...');
    await page.evaluate(() => {
      const toggle = document.getElementById('autoNColorToggle');
      if (toggle && toggle.checked) toggle.click();
    });
    await page.waitForTimeout(2000);

    const afterToggleOff = await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      const uniqueLabels = maskValues ? new Set(Array.from(maskValues).filter(v => v > 0)) : new Set();
      return {
        nColorActive: window.nColorActive,
        uniqueLabelCount: uniqueLabels.size,
        maxLabel: Math.max(0, ...Array.from(uniqueLabels)),
        hasAffinityGraph: Boolean(window.affinityGraphInfo && window.affinityGraphInfo.graph),
        affinityNodeCount: window.affinityGraphInfo?.graph?.size || 0,
      };
    });
    console.log('[test] After toggle OFF:', JSON.stringify(afterToggleOff, null, 2));

    // STEP 2: Draw/paint on two cells to merge them
    console.log('[test] STEP 2: Simulating paint to merge cells...');

    // Find two adjacent cells and get their coordinates
    const paintInfo = await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      if (!maskValues) return { error: 'no maskValues' };

      const width = window.imgWidth || 392;
      const height = window.imgHeight || 384;

      // Find first non-zero label and its neighbor with different label
      let firstLabel = 0;
      let firstX = 0, firstY = 0;
      let secondLabel = 0;
      let secondX = 0, secondY = 0;

      for (let y = 10; y < height - 10; y++) {
        for (let x = 10; x < width - 10; x++) {
          const idx = y * width + x;
          const label = maskValues[idx];
          if (label > 0 && firstLabel === 0) {
            firstLabel = label;
            firstX = x;
            firstY = y;
          }
          if (label > 0 && label !== firstLabel && secondLabel === 0) {
            secondLabel = label;
            secondX = x;
            secondY = y;
          }
          if (firstLabel > 0 && secondLabel > 0) break;
        }
        if (firstLabel > 0 && secondLabel > 0) break;
      }

      return {
        firstLabel, firstX, firstY,
        secondLabel, secondX, secondY,
        uniqueLabels: new Set(Array.from(maskValues).filter(v => v > 0)).size,
      };
    });
    console.log('[test] Paint info:', JSON.stringify(paintInfo, null, 2));

    // Use the fill tool to change second cell's label to first cell's label
    if (paintInfo.firstLabel && paintInfo.secondLabel) {
      console.log(`[test] Filling cell ${paintInfo.secondLabel} with label ${paintInfo.firstLabel}...`);

      // Set current label to first cell's label, then fill second cell
      const fillResult = await page.evaluate(async (info) => {
        // Get the painting context to set current label
        const paintingState = window.OmniPainting?.__debugGetState?.();
        if (paintingState && paintingState.ctx && typeof paintingState.ctx.setCurrentLabel === 'function') {
          paintingState.ctx.setCurrentLabel(info.firstLabel);
        } else {
          // Fallback: try to set currentLabel directly
          window.currentLabel = info.firstLabel;
        }

        // Call floodFill with point object {x, y}
        const point = { x: info.secondX, y: info.secondY };

        if (window.OmniPainting && typeof window.OmniPainting.floodFill === 'function') {
          window.OmniPainting.floodFill(point);
          return { filled: true, method: 'OmniPainting.floodFill' };
        }

        // Try the global floodFill if exposed
        if (typeof window.floodFill === 'function') {
          window.floodFill(point);
          return { filled: true, method: 'window.floodFill' };
        }

        return { filled: false, reason: 'no fill function found' };
      }, paintInfo);

      console.log('[test] Fill result:', JSON.stringify(fillResult, null, 2));
      await page.waitForTimeout(1000);

      // Check state after fill
      const afterFill = await page.evaluate(() => {
        const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
        const uniqueLabels = maskValues ? new Set(Array.from(maskValues).filter(v => v > 0)) : new Set();
        return {
          uniqueLabelCount: uniqueLabels.size,
          hasAffinityGraph: Boolean(window.affinityGraphInfo && window.affinityGraphInfo.graph),
          affinityNodeCount: window.affinityGraphInfo?.graph?.size || 0,
        };
      });
      console.log('[test] After fill:', JSON.stringify(afterFill, null, 2));

      // Take screenshot
      await page.screenshot({
        path: path.join(OUTPUT_DIR, `drawing-test-${ts}-02-after-fill.png`),
        fullPage: true
      });

      // STEP 3: Toggle N-color back ON and verify it uses updated affinity
      console.log('[test] STEP 3: Toggling N-color back ON...');
      await page.evaluate(() => {
        const toggle = document.getElementById('autoNColorToggle');
        if (toggle && !toggle.checked) toggle.click();
      });
      await page.waitForTimeout(2000);

      const afterToggleOn = await page.evaluate(() => {
        const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
        const uniqueLabels = maskValues ? new Set(Array.from(maskValues).filter(v => v > 0)) : new Set();
        return {
          nColorActive: window.nColorActive,
          uniqueLabelCount: uniqueLabels.size,
          hasAffinityGraph: Boolean(window.affinityGraphInfo && window.affinityGraphInfo.graph),
          affinityNodeCount: window.affinityGraphInfo?.graph?.size || 0,
        };
      });
      console.log('[test] After toggle ON:', JSON.stringify(afterToggleOn, null, 2));

      // Take final screenshot
      await page.screenshot({
        path: path.join(OUTPUT_DIR, `drawing-test-${ts}-03-after-toggle-on.png`),
        fullPage: true
      });

      // Verify results
      if (afterFill.uniqueLabelCount < afterToggleOff.uniqueLabelCount) {
        console.log('[test] SUCCESS: Fill reduced unique label count (cells were merged)');
      } else {
        console.log('[test] WARNING: Fill did not reduce unique label count');
      }

      if (afterToggleOn.uniqueLabelCount <= 6) {
        console.log('[test] SUCCESS: N-color mode has expected label count after fill');
      } else {
        console.log('[test] WARNING: N-color mode has more labels than expected');
      }
    }

    console.log('[test] Test completed!');
    console.log(`[test] Screenshots saved to: ${OUTPUT_DIR}`);

  } catch (err) {
    console.error('[test] Error:', err.message);
    await page.screenshot({
      path: path.join(OUTPUT_DIR, `drawing-test-${ts}-error.png`),
      fullPage: true
    });
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
