#!/usr/bin/env node
/**
 * Debug test to understand the source mask issue.
 */

const { chromium } = require('playwright');

const DEFAULT_URL = process.env.TEST_URL || 'http://127.0.0.1:8001/';

async function runTest() {
  console.log('[test] Starting source mask debug test...');

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
      document.getElementById('segmentButton')?.click();
    });

    await page.waitForFunction(() => {
      const status = document.getElementById('segmentStatus');
      return status && (status.textContent.includes('complete') || status.textContent.includes('failed'));
    }, { timeout: 120000 });
    await page.waitForTimeout(2000);

    // Check nColor state with new debug functions
    const nColorState = await page.evaluate(() => {
      return window.__OMNI_DEBUG__.getNColorState();
    });
    console.log('[test] nColor state:', JSON.stringify(nColorState, null, 2));

    // Find an empty region
    const testInfo = await page.evaluate(() => {
      const maskValues = window.__OMNI_DEBUG__.getMaskValues();
      const sourceMask = window.__OMNI_DEBUG__.getAffinitySourceMask();
      const nColorInstanceMask = window.__OMNI_DEBUG__.getNColorInstanceMask();

      const width = 392;
      const height = 384;

      // Find empty region in source mask
      let emptyX = -1, emptyY = -1;
      for (let y = 10; y < height - 20; y += 10) {
        for (let x = 10; x < width - 20; x += 10) {
          let isEmpty = true;
          for (let dy = 0; dy < 10 && isEmpty; dy++) {
            for (let dx = 0; dx < 10 && isEmpty; dx++) {
              const idx = (y + dy) * width + (x + dx);
              if (sourceMask && sourceMask[idx] !== 0) {
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

      // Sample values at test location
      const sampleIdx = 50 * width + 50;
      return {
        emptyRegion: emptyX >= 0 ? { x: emptyX, y: emptyY } : null,
        maskValuesSample: maskValues ? maskValues[sampleIdx] : 'N/A',
        sourceMaskSample: sourceMask ? sourceMask[sampleIdx] : 'N/A',
        instanceMaskSample: nColorInstanceMask ? nColorInstanceMask[sampleIdx] : 'N/A',
        isSameAsSource: sourceMask === maskValues ? 'maskValues' : (sourceMask === nColorInstanceMask ? 'nColorInstanceMask' : 'unknown'),
      };
    });
    console.log('[test] Test info:', JSON.stringify(testInfo, null, 2));

    if (!testInfo.emptyRegion) {
      console.log('[test] No empty region found');
      return;
    }

    // Now paint in the source mask and test
    const paintResult = await page.evaluate((region) => {
      const sourceMask = window.__OMNI_DEBUG__.getAffinitySourceMask();
      const width = 392;
      const label = 99;
      const indices = [];

      // Paint in the SOURCE mask (not maskValues)
      for (let dy = 0; dy < 10; dy++) {
        for (let dx = 0; dx < 10; dx++) {
          const idx = (region.y + dy) * width + (region.x + dx);
          sourceMask[idx] = label;
          indices.push(idx);
        }
      }

      // Get affinity state before update
      const before = window.__OMNI_DEBUG__.getAffinityState();
      const regionBefore = window.__OMNI_DEBUG__.checkAffinityInRegion(region.x, region.y, 10, 10);

      // Call update
      const painting = window.OmniPainting;
      const state = painting.__debugGetState ? painting.__debugGetState() : null;
      const ctx = state ? state.ctx : null;
      if (ctx && typeof ctx.updateAffinityGraphForIndices === 'function') {
        console.log('[test] Calling updateAffinityGraphForIndices');
        ctx.updateAffinityGraphForIndices(indices);
      }

      // Get affinity state after update
      const after = window.__OMNI_DEBUG__.getAffinityState();
      const regionAfter = window.__OMNI_DEBUG__.checkAffinityInRegion(region.x, region.y, 10, 10);

      // Force draw
      window.__OMNI_DEBUG__.draw();

      return {
        paintedCount: indices.length,
        before: {
          source: before.affinityGraphSource,
          nextSlot: before.nextSlot,
          regionEdges: regionBefore.edgeCount,
        },
        after: {
          source: after.affinityGraphSource,
          nextSlot: after.nextSlot,
          regionEdges: regionAfter.edgeCount,
        },
      };
    }, testInfo.emptyRegion);

    console.log('[test] Paint result:', JSON.stringify(paintResult, null, 2));

    // Analysis
    console.log('\n[test] === ANALYSIS ===');
    console.log('[test] Source mask is:', testInfo.isSameAsSource);
    console.log('[test] Slot change:', paintResult.before.nextSlot, '->', paintResult.after.nextSlot);
    console.log('[test] Edge change:', paintResult.before.regionEdges, '->', paintResult.after.regionEdges);

    if (paintResult.after.regionEdges > paintResult.before.regionEdges) {
      console.log('[test] SUCCESS: Edges increased when painting in source mask');
    } else {
      console.log('[test] FAILURE: Still no edges');
    }

  } catch (err) {
    console.error('[test] Error:', err.message);
    console.error(err.stack);
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
