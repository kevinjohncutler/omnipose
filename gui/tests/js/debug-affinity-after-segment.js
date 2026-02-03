#!/usr/bin/env node
/**
 * Debug test for affinity graph updates after segmentation.
 * Tests if incremental affinity updates work when drawing after segmentation.
 */

const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const DEFAULT_URL = process.env.TEST_URL || 'http://127.0.0.1:8001/';
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

  console.log('[test] Starting affinity-after-segment debug test...');
  console.log('[test] URL:', DEFAULT_URL);

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1600, height: 1000 },
  });
  const page = await context.newPage();

  // Capture all console output
  page.on('console', (msg) => {
    console.log(`[page][${msg.type()}] ${msg.text()}`);
  });

  try {
    await page.goto(DEFAULT_URL, { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForFunction(() => {
      return document.getElementById('canvas') !== null &&
             (window.OmniPainting || window.__OMNI_DEBUG__);
    }, { timeout: 30000 });

    console.log('[test] Page loaded');

    // STEP 1: Check initial state
    const initialState = await page.evaluate(() => {
      return window.__OMNI_DEBUG__ && window.__OMNI_DEBUG__.getAffinityState
        ? window.__OMNI_DEBUG__.getAffinityState()
        : { error: 'getAffinityState not available' };
    });
    console.log('[test] Initial state:', JSON.stringify(initialState, null, 2));

    // STEP 2: Draw something before segmentation
    console.log('[test] STEP 2: Drawing before segmentation...');
    const drawResult1 = await page.evaluate(() => {
      if (!window.__OMNI_DEBUG__ || !window.__OMNI_DEBUG__.paintAndUpdateAffinity) {
        return { error: 'paintAndUpdateAffinity not available' };
      }
      // Paint a 10x10 square at position (50, 50) with label 1
      return window.__OMNI_DEBUG__.paintAndUpdateAffinity(50, 50, 10, 10, 1);
    });
    console.log('[test] Draw result before segmentation:', JSON.stringify(drawResult1, null, 2));

    // Force a redraw
    await page.evaluate(() => {
      if (window.__OMNI_DEBUG__ && window.__OMNI_DEBUG__.draw) {
        window.__OMNI_DEBUG__.draw();
      }
    });
    await page.waitForTimeout(500);

    await page.screenshot({
      path: path.join(OUTPUT_DIR, `affinity-debug-${ts}-01-before-segment.png`),
      fullPage: true
    });

    // STEP 3: Trigger segmentation
    console.log('[test] STEP 3: Running segmentation...');
    await page.evaluate(() => {
      const btn = document.getElementById('segmentButton');
      if (btn) btn.click();
    });

    // Wait for segmentation to complete
    await page.waitForFunction(() => {
      const status = document.getElementById('segmentStatus');
      return status && (status.textContent.includes('complete') || status.textContent.includes('failed'));
    }, { timeout: 120000 });
    await page.waitForTimeout(2000);

    const afterSegment = await page.evaluate(() => {
      return window.__OMNI_DEBUG__ && window.__OMNI_DEBUG__.getAffinityState
        ? window.__OMNI_DEBUG__.getAffinityState()
        : { error: 'getAffinityState not available' };
    });
    console.log('[test] After segmentation:', JSON.stringify(afterSegment, null, 2));

    await page.screenshot({
      path: path.join(OUTPUT_DIR, `affinity-debug-${ts}-02-after-segment.png`),
      fullPage: true
    });

    // STEP 4: Draw something after segmentation
    console.log('[test] STEP 4: Drawing after segmentation...');

    // Find a label from the segmentation to use for painting
    const labelInfo = await page.evaluate(() => {
      const maskValues = window.__OMNI_DEBUG__ && window.__OMNI_DEBUG__.getMaskValues
        ? window.__OMNI_DEBUG__.getMaskValues()
        : null;
      if (!maskValues) return { error: 'no maskValues' };

      // Find first non-zero label
      for (let i = 0; i < maskValues.length; i++) {
        if (maskValues[i] > 0) {
          return { label: maskValues[i], index: i };
        }
      }
      return { label: 1, index: -1 };
    });
    console.log('[test] Label info:', JSON.stringify(labelInfo, null, 2));

    const drawResult2 = await page.evaluate((useLabel) => {
      if (!window.__OMNI_DEBUG__ || !window.__OMNI_DEBUG__.paintAndUpdateAffinity) {
        return { error: 'paintAndUpdateAffinity not available' };
      }
      // Paint a 10x10 square at position (100, 100) with the label from segmentation
      return window.__OMNI_DEBUG__.paintAndUpdateAffinity(100, 100, 10, 10, useLabel);
    }, labelInfo.label || 1);
    console.log('[test] Draw result after segmentation:', JSON.stringify(drawResult2, null, 2));

    // Force a redraw
    await page.evaluate(() => {
      if (window.__OMNI_DEBUG__ && window.__OMNI_DEBUG__.draw) {
        window.__OMNI_DEBUG__.draw();
      }
    });
    await page.waitForTimeout(500);

    await page.screenshot({
      path: path.join(OUTPUT_DIR, `affinity-debug-${ts}-03-after-draw.png`),
      fullPage: true
    });

    // STEP 5: Check final state
    const finalState = await page.evaluate(() => {
      const state = window.__OMNI_DEBUG__ && window.__OMNI_DEBUG__.getAffinityState
        ? window.__OMNI_DEBUG__.getAffinityState()
        : { error: 'getAffinityState not available' };

      // Also check the painted region specifically
      const regionCheck = window.__OMNI_DEBUG__ && window.__OMNI_DEBUG__.checkAffinityInRegion
        ? window.__OMNI_DEBUG__.checkAffinityInRegion(100, 100, 10, 10)
        : { error: 'checkAffinityInRegion not available' };

      return {
        state,
        regionCheck,
      };
    });
    console.log('[test] Final state:', JSON.stringify(finalState, null, 2));

    // Analysis
    console.log('\n[test] === ANALYSIS ===');

    if (drawResult2.error) {
      console.log('[test] FAILURE: Could not execute paint operation:', drawResult2.error);
    } else if (finalState.regionCheck && finalState.regionCheck.edgeCount > 0) {
      console.log('[test] SUCCESS: Painted area has', finalState.regionCheck.edgeCount, 'affinity edges');
    } else {
      console.log('[test] FAILURE: Painted area has NO affinity edges');
      console.log('[test] This indicates updateAffinityGraphForIndices is not working after segmentation');

      // More detailed analysis
      if (drawResult2.stateBefore && drawResult2.stateAfter) {
        console.log('[test] Before update:');
        console.log('  - showAffinityGraph:', drawResult2.stateBefore.showAffinityGraph);
        console.log('  - affinitySegEnabled:', drawResult2.stateBefore.affinitySegEnabled);
        console.log('  - affinityGraphSource:', drawResult2.stateBefore.affinityGraphSource);
        console.log('  - hasSegments:', drawResult2.stateBefore.hasSegments);
        console.log('[test] After update:');
        console.log('  - affinityGraphSource:', drawResult2.stateAfter.affinityGraphSource);
        console.log('  - hasSegments:', drawResult2.stateAfter.hasSegments);
      }
    }

  } catch (err) {
    console.error('[test] Error:', err.message);
    console.error(err.stack);
    await page.screenshot({
      path: path.join(OUTPUT_DIR, `affinity-debug-${ts}-error.png`),
      fullPage: true
    });
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
