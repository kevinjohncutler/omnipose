#!/usr/bin/env node
/**
 * Test to verify the nColor source mask issue.
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

async function runTest() {
  ensureDir(OUTPUT_DIR);

  console.log('[test] Starting nColor source mask test...');

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1600, height: 1000 },
  });
  const page = await context.newPage();

  page.on('console', (msg) => {
    console.log(`[page][${msg.type()}] ${msg.text()}`);
  });

  try {
    await page.goto(DEFAULT_URL, { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForFunction(() => {
      return window.OmniPainting && window.__OMNI_DEBUG__;
    }, { timeout: 30000 });

    console.log('[test] Page loaded');

    // Run segmentation
    console.log('[test] Running segmentation...');
    await page.evaluate(() => {
      const btn = document.getElementById('segmentButton');
      if (btn) btn.click();
    });

    await page.waitForFunction(() => {
      const status = document.getElementById('segmentStatus');
      return status && (status.textContent.includes('complete') || status.textContent.includes('failed'));
    }, { timeout: 120000 });
    await page.waitForTimeout(2000);

    // Check nColor state
    const nColorState = await page.evaluate(() => {
      const painting = window.OmniPainting;
      const state = painting && painting.__debugGetState ? painting.__debugGetState() : null;
      const ctx = state ? state.ctx : null;

      // Check if nColor is active
      const isNColorActive = ctx && typeof ctx.isNColorActive === 'function' ? ctx.isNColorActive() : 'unknown';

      // Get mask values from different sources
      const maskValues = window.__OMNI_DEBUG__.getMaskValues();
      const sampleIdx = 50 * 392 + 50;

      return {
        nColorActive: window.nColorActive,
        isNColorActiveFn: isNColorActive,
        maskValuesSample: maskValues ? maskValues[sampleIdx] : 'no maskValues',
        hasNColorInstanceMask: Boolean(window.nColorInstanceMask),
        nColorInstanceMaskSample: window.nColorInstanceMask ? window.nColorInstanceMask[sampleIdx] : 'no nColorInstanceMask',
      };
    });
    console.log('[test] nColor state:', JSON.stringify(nColorState, null, 2));

    // Find an empty region and paint it
    const testResult = await page.evaluate(() => {
      const maskValues = window.__OMNI_DEBUG__.getMaskValues();
      const width = 392;
      const height = 384;

      // Find empty region
      let emptyX = -1, emptyY = -1;
      for (let y = 10; y < height - 20; y += 10) {
        for (let x = 10; x < width - 20; x += 10) {
          let isEmpty = true;
          for (let dy = 0; dy < 10 && isEmpty; dy++) {
            for (let dx = 0; dx < 10 && isEmpty; dx++) {
              if (maskValues[(y + dy) * width + (x + dx)] !== 0) {
                isEmpty = false;
              }
            }
          }
          if (isEmpty) {
            emptyX = x;
            emptyY = y;
            break;
          }
        }
        if (emptyX >= 0) break;
      }

      if (emptyX < 0) {
        return { error: 'No empty region found' };
      }

      // Paint with label 99
      const label = 99;
      const indices = [];
      for (let dy = 0; dy < 10; dy++) {
        for (let dx = 0; dx < 10; dx++) {
          const idx = (emptyY + dy) * width + (emptyX + dx);
          maskValues[idx] = label;
          indices.push(idx);
        }
      }

      // Check what the source mask returns for one of these indices
      const testIdx = (emptyY + 5) * width + (emptyX + 5);
      const maskValue = maskValues[testIdx];
      const instanceMaskValue = window.nColorInstanceMask ? window.nColorInstanceMask[testIdx] : 'N/A';

      // If nColor is active and we're using nColorInstanceMask, the source mask won't have our painted label
      const painting = window.OmniPainting;
      const state = painting && painting.__debugGetState ? painting.__debugGetState() : null;
      const ctx = state ? state.ctx : null;

      // Try updating both masks
      if (window.nColorInstanceMask && window.nColorActive) {
        console.log('[test] nColor is active - also updating nColorInstanceMask');
        for (const idx of indices) {
          window.nColorInstanceMask[idx] = label;
        }
      }

      // Now call updateAffinityGraphForIndices
      if (ctx && typeof ctx.updateAffinityGraphForIndices === 'function') {
        ctx.updateAffinityGraphForIndices(indices);
      }

      // Check the region now
      const region = window.__OMNI_DEBUG__.checkAffinityInRegion(emptyX, emptyY, 10, 10);

      return {
        emptyRegion: { x: emptyX, y: emptyY },
        nColorActive: window.nColorActive,
        maskValue,
        instanceMaskValue,
        regionEdges: region.edgeCount,
        afterInstanceMaskUpdate: window.nColorInstanceMask ? window.nColorInstanceMask[testIdx] : 'N/A',
      };
    });

    console.log('[test] Test result:', JSON.stringify(testResult, null, 2));

    // Analysis
    console.log('\n[test] === ANALYSIS ===');
    if (testResult.nColorActive) {
      console.log('[test] nColor mode IS active');
      console.log('[test] When nColor is active, updateAffinityGraphForIndices uses nColorInstanceMask');
      console.log('[test] If we only modify maskValues, the affinity update sees the wrong values!');
    }
    if (testResult.regionEdges > 0) {
      console.log('[test] SUCCESS: Got', testResult.regionEdges, 'edges after updating both masks');
    } else {
      console.log('[test] Still no edges - something else is wrong');
    }

  } catch (err) {
    console.error('[test] Error:', err.message);
    console.error(err.stack);
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
