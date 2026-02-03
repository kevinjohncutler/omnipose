#!/usr/bin/env node
/**
 * Test painting module affinity updates.
 * Simulates the actual painting flow through OmniPainting API.
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

  console.log('[test] Starting painting module affinity test...');
  console.log('[test] URL:', DEFAULT_URL);

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
      return document.getElementById('canvas') !== null &&
             window.OmniPainting && window.__OMNI_DEBUG__;
    }, { timeout: 30000 });

    console.log('[test] Page loaded');

    // STEP 1: Run segmentation first
    console.log('[test] STEP 1: Running segmentation...');
    await page.evaluate(() => {
      const btn = document.getElementById('segmentButton');
      if (btn) btn.click();
    });

    await page.waitForFunction(() => {
      const status = document.getElementById('segmentStatus');
      return status && (status.textContent.includes('complete') || status.textContent.includes('failed'));
    }, { timeout: 120000 });
    await page.waitForTimeout(2000);

    const afterSegment = await page.evaluate(() => window.__OMNI_DEBUG__.getAffinityState());
    console.log('[test] After segmentation:', JSON.stringify(afterSegment, null, 2));

    await page.screenshot({
      path: path.join(OUTPUT_DIR, `painting-module-${ts}-01-after-segment.png`),
      fullPage: true
    });

    // STEP 2: Simulate painting through OmniPainting module
    console.log('[test] STEP 2: Simulating painting through OmniPainting...');

    const paintResult = await page.evaluate(() => {
      const painting = window.OmniPainting;
      if (!painting) return { error: 'OmniPainting not found' };

      const state = painting.__debugGetState ? painting.__debugGetState() : null;
      if (!state || !state.ctx) return { error: 'Painting state not available' };

      const ctx = state.ctx;
      const maskValues = ctx.maskValues;
      if (!maskValues) return { error: 'No maskValues in context' };

      const width = 392; // from earlier logs
      const height = 384;

      // Verify ctx has the updateAffinityGraphForIndices function
      const hasUpdateAffinity = typeof ctx.updateAffinityGraphForIndices === 'function';

      // Get current label or use 1
      const currentLabel = typeof ctx.getCurrentLabel === 'function' ? ctx.getCurrentLabel() : 1;

      // Find an existing label near our paint area
      const existingLabel = maskValues[100 * width + 100] || currentLabel || 1;

      // Get affinity state before painting
      const affinityBefore = window.__OMNI_DEBUG__.getAffinityState();
      const regionBefore = window.__OMNI_DEBUG__.checkAffinityInRegion(150, 150, 10, 10);

      // Log what we're about to do
      console.log('[paint-test] About to paint. hasUpdateAffinity:', hasUpdateAffinity, 'label:', existingLabel);

      // Simulate painting by:
      // 1. Setting the current label
      if (typeof ctx.setCurrentLabel === 'function') {
        ctx.setCurrentLabel(existingLabel);
      }

      // 2. Directly modifying mask values like the painting module does
      const paintedIndices = [];
      for (let dy = 0; dy < 10; dy++) {
        for (let dx = 0; dx < 10; dx++) {
          const x = 150 + dx;
          const y = 150 + dy;
          const idx = y * width + x;
          if (idx >= 0 && idx < maskValues.length) {
            maskValues[idx] = existingLabel;
            paintedIndices.push(idx);
          }
        }
      }

      // 3. Call updateAffinityGraphForIndices as the painting module would
      if (hasUpdateAffinity) {
        console.log('[paint-test] Calling updateAffinityGraphForIndices with', paintedIndices.length, 'indices');
        ctx.updateAffinityGraphForIndices(paintedIndices);
      }

      // 4. Trigger mask dirty notification
      if (typeof ctx.markMaskIndicesDirty === 'function') {
        ctx.markMaskIndicesDirty(paintedIndices);
      }

      // 5. Request paint frame (like the painting module does)
      if (typeof ctx.requestPaintFrame === 'function') {
        ctx.requestPaintFrame();
      }

      // Get affinity state after painting
      const affinityAfter = window.__OMNI_DEBUG__.getAffinityState();
      const regionAfter = window.__OMNI_DEBUG__.checkAffinityInRegion(150, 150, 10, 10);

      return {
        hasUpdateAffinity,
        label: existingLabel,
        paintedCount: paintedIndices.length,
        affinityBefore,
        affinityAfter,
        regionBefore,
        regionAfter,
      };
    });

    console.log('[test] Paint result:', JSON.stringify(paintResult, null, 2));

    await page.waitForTimeout(500);

    await page.screenshot({
      path: path.join(OUTPUT_DIR, `painting-module-${ts}-02-after-paint.png`),
      fullPage: true
    });

    // Analysis
    console.log('\n[test] === ANALYSIS ===');

    if (paintResult.error) {
      console.log('[test] FAILURE:', paintResult.error);
    } else if (!paintResult.hasUpdateAffinity) {
      console.log('[test] FAILURE: ctx.updateAffinityGraphForIndices not available');
    } else if (paintResult.regionAfter.edgeCount > paintResult.regionBefore.edgeCount) {
      console.log('[test] SUCCESS: Affinity edges increased from', paintResult.regionBefore.edgeCount, 'to', paintResult.regionAfter.edgeCount);
    } else if (paintResult.regionAfter.edgeCount > 0) {
      console.log('[test] PARTIAL SUCCESS: Region has', paintResult.regionAfter.edgeCount, 'edges (before:', paintResult.regionBefore.edgeCount, ')');
    } else {
      console.log('[test] FAILURE: No affinity edges in painted region');
      console.log('[test] Before:', paintResult.regionBefore);
      console.log('[test] After:', paintResult.regionAfter);
    }

    // STEP 3: Test actual mouse-based painting
    console.log('\n[test] STEP 3: Testing actual mouse-based painting...');

    // First get canvas position
    const canvasBox = await page.locator('#canvas').boundingBox();
    console.log('[test] Canvas bounds:', canvasBox);

    if (canvasBox) {
      // Get affinity state before mouse painting
      const beforeMouse = await page.evaluate(() => ({
        affinity: window.__OMNI_DEBUG__.getAffinityState(),
        region: window.__OMNI_DEBUG__.checkAffinityInRegion(200, 200, 20, 20),
      }));

      // Simulate mouse painting
      const startX = canvasBox.x + 200;
      const startY = canvasBox.y + 200;

      // Mouse down, move, up
      await page.mouse.move(startX, startY);
      await page.mouse.down();
      await page.mouse.move(startX + 20, startY + 20, { steps: 10 });
      await page.mouse.up();

      await page.waitForTimeout(500);

      // Get affinity state after mouse painting
      const afterMouse = await page.evaluate(() => ({
        affinity: window.__OMNI_DEBUG__.getAffinityState(),
        region: window.__OMNI_DEBUG__.checkAffinityInRegion(200, 200, 20, 20),
      }));

      console.log('[test] Before mouse paint:', JSON.stringify(beforeMouse.region, null, 2));
      console.log('[test] After mouse paint:', JSON.stringify(afterMouse.region, null, 2));

      await page.screenshot({
        path: path.join(OUTPUT_DIR, `painting-module-${ts}-03-after-mouse.png`),
        fullPage: true
      });

      if (afterMouse.region.edgeCount > beforeMouse.region.edgeCount) {
        console.log('[test] MOUSE SUCCESS: Edges increased from', beforeMouse.region.edgeCount, 'to', afterMouse.region.edgeCount);
      } else {
        console.log('[test] MOUSE FAILURE: No new edges from mouse painting');
      }
    }

  } catch (err) {
    console.error('[test] Error:', err.message);
    console.error(err.stack);
    await page.screenshot({
      path: path.join(OUTPUT_DIR, `painting-module-${ts}-error.png`),
      fullPage: true
    });
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
