#!/usr/bin/env node
/**
 * Test affinity graph updates in an empty region.
 * This isolates the issue by painting in a region with no existing edges.
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

  console.log('[test] Starting empty region affinity test...');

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

    // Find an empty region (background)
    const emptyRegion = await page.evaluate(() => {
      const maskValues = window.__OMNI_DEBUG__.getMaskValues();
      const width = 392;
      const height = 384;

      // Find a 10x10 region that's all zeros
      for (let startY = 10; startY < height - 20; startY += 10) {
        for (let startX = 10; startX < width - 20; startX += 10) {
          let isEmpty = true;
          for (let dy = 0; dy < 10 && isEmpty; dy++) {
            for (let dx = 0; dx < 10 && isEmpty; dx++) {
              const idx = (startY + dy) * width + (startX + dx);
              if (maskValues[idx] !== 0) {
                isEmpty = false;
              }
            }
          }
          if (isEmpty) {
            return { x: startX, y: startY, w: 10, h: 10 };
          }
        }
      }
      return null;
    });

    console.log('[test] Empty region found:', emptyRegion);

    if (!emptyRegion) {
      console.log('[test] ERROR: No empty region found');
      return;
    }

    // Check affinity in empty region before painting
    const before = await page.evaluate((region) => {
      return {
        affinity: window.__OMNI_DEBUG__.getAffinityState(),
        region: window.__OMNI_DEBUG__.checkAffinityInRegion(region.x, region.y, region.w, region.h),
      };
    }, emptyRegion);

    console.log('[test] Before painting:');
    console.log('  - affinityGraphSource:', before.affinity.affinityGraphSource);
    console.log('  - nextSlot:', before.affinity.nextSlot);
    console.log('  - region edges:', before.region.edgeCount);

    // Paint the empty region with label 99
    const paintResult = await page.evaluate((region) => {
      const maskValues = window.__OMNI_DEBUG__.getMaskValues();
      const width = 392;
      const label = 99;  // Use a unique label

      const indices = [];
      for (let dy = 0; dy < region.h; dy++) {
        for (let dx = 0; dx < region.w; dx++) {
          const idx = (region.y + dy) * width + (region.x + dx);
          maskValues[idx] = label;
          indices.push(idx);
        }
      }

      // Get the painting context and call updateAffinityGraphForIndices
      const painting = window.OmniPainting;
      const state = painting && painting.__debugGetState ? painting.__debugGetState() : null;
      const ctx = state ? state.ctx : null;

      if (!ctx || typeof ctx.updateAffinityGraphForIndices !== 'function') {
        return { error: 'No updateAffinityGraphForIndices' };
      }

      console.log('[test] Calling updateAffinityGraphForIndices with', indices.length, 'indices');
      ctx.updateAffinityGraphForIndices(indices);

      // Force draw
      if (window.__OMNI_DEBUG__.draw) {
        window.__OMNI_DEBUG__.draw();
      }

      return {
        success: true,
        paintedCount: indices.length,
        label,
      };
    }, emptyRegion);

    console.log('[test] Paint result:', paintResult);

    await page.waitForTimeout(500);

    // Check affinity in region after painting
    const after = await page.evaluate((region) => {
      return {
        affinity: window.__OMNI_DEBUG__.getAffinityState(),
        region: window.__OMNI_DEBUG__.checkAffinityInRegion(region.x, region.y, region.w, region.h),
      };
    }, emptyRegion);

    console.log('[test] After painting:');
    console.log('  - affinityGraphSource:', after.affinity.affinityGraphSource);
    console.log('  - nextSlot:', after.affinity.nextSlot);
    console.log('  - region edges:', after.region.edgeCount);

    // Analysis
    console.log('\n[test] === ANALYSIS ===');
    console.log('[test] Slot change:', before.affinity.nextSlot, '->', after.affinity.nextSlot);
    console.log('[test] Edge count change:', before.region.edgeCount, '->', after.region.edgeCount);

    if (after.region.edgeCount > before.region.edgeCount) {
      console.log('[test] SUCCESS: New affinity edges created');
    } else if (after.region.edgeCount === 0 && before.region.edgeCount === 0) {
      console.log('[test] FAILURE: No edges created in empty region');
      console.log('[test] Expected ~720 edges for 10x10 8-connected region');

      // Debug: check if values were updated
      const valuesCheck = await page.evaluate((region) => {
        const info = window.affinityGraphInfo;
        if (!info || !info.values) return { error: 'no values' };

        const width = info.width;
        const planeStride = width * info.height;
        const stepCount = info.stepCount;

        // Check a few specific edges in the region
        const samples = [];
        const x = region.x + 5;
        const y = region.y + 5;
        const idx = y * width + x;

        for (let s = 0; s < stepCount; s++) {
          const offset = s * planeStride + idx;
          samples.push({ step: s, value: info.values[offset] });
        }

        return {
          index: idx,
          samples,
          maskValue: window.__OMNI_DEBUG__.getMaskValues()[idx],
        };
      }, emptyRegion);
      console.log('[test] Values check:', JSON.stringify(valuesCheck, null, 2));
    } else {
      console.log('[test] UNEXPECTED: Edges decreased or unchanged');
    }

    await page.screenshot({
      path: path.join(OUTPUT_DIR, `empty-region-${ts}.png`),
      fullPage: true
    });

  } catch (err) {
    console.error('[test] Error:', err.message);
    console.error(err.stack);
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
