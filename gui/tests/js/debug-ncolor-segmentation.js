#!/usr/bin/env node
/**
 * Debug N-color segmentation - test segmenting with N-color OFF from the start
 */

const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const DEFAULT_URL = 'http://127.0.0.1:8000/';  // User's server on port 8000
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

  console.log('[debug] Starting N-color segmentation debug test...');
  console.log('[debug] Connecting to', DEFAULT_URL);

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
             (window.OmniPainting || window.__OMNI_CONFIG__);
    }, { timeout: 30000 });

    console.log('[debug] Page loaded');

    // Check initial nColorActive state
    const initialState = await page.evaluate(() => {
      return {
        nColorActive: window.nColorActive,
        labelColormap: window.labelColormap,
        labelShuffle: window.labelShuffle,
        paletteTextureDirty: window.paletteTextureDirty,
        colorTableLength: window.colorTable ? window.colorTable.length : 0,
        nColorPaletteColorsLength: window.nColorPaletteColors ? window.nColorPaletteColors.length : 0,
      };
    });
    console.log('[debug] Initial state:', JSON.stringify(initialState, null, 2));

    // STEP 1: Toggle N-color OFF first
    console.log('[debug] STEP 1: Toggling N-color OFF...');
    await page.evaluate(() => {
      const toggle = document.getElementById('autoNColorToggle');
      if (toggle && toggle.checked) {
        toggle.click();
      }
    });
    await page.waitForTimeout(2000);

    const afterToggleOff = await page.evaluate(() => {
      return {
        nColorActive: window.nColorActive,
        toggleChecked: document.getElementById('autoNColorToggle')?.checked,
        paletteTextureDirty: window.paletteTextureDirty,
      };
    });
    console.log('[debug] After toggle OFF:', JSON.stringify(afterToggleOff, null, 2));

    // Take screenshot with N-color OFF (before segmentation)
    await page.screenshot({
      path: path.join(OUTPUT_DIR, `debug-ncolor-${ts}-01-ncolor-off.png`),
      fullPage: true
    });

    // STEP 2: Segment with N-color OFF
    console.log('[debug] STEP 2: Clicking Segment button with N-color OFF...');
    await page.evaluate(() => {
      const btn = document.getElementById('segmentButton');
      if (btn) btn.click();
    });

    // Wait for segmentation to complete
    console.log('[debug] Waiting for segmentation...');
    await page.waitForFunction(() => {
      const status = document.getElementById('segmentStatus');
      return status && (status.textContent.includes('complete') || status.textContent.includes('failed'));
    }, { timeout: 120000 });
    await page.waitForTimeout(1000);

    // Check state after segmentation
    const afterSegment = await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      const uniqueLabels = maskValues ? new Set(Array.from(maskValues).filter(v => v > 0)) : new Set();

      // Check what buildPaletteTextureData would return
      let palettePreview = [];
      if (typeof window.buildPaletteTextureData === 'function') {
        // Can't call directly, but can check the state
      }

      return {
        nColorActive: window.nColorActive,
        paletteTextureDirty: window.paletteTextureDirty,
        uniqueLabelCount: uniqueLabels.size,
        maxLabel: Math.max(0, ...Array.from(uniqueLabels)),
        sampleLabels: Array.from(uniqueLabels).slice(0, 20),
        labelColormap: window.labelColormap,
        labelShuffle: window.labelShuffle,
        colorTableLength: window.colorTable ? window.colorTable.length : 0,
        nColorPaletteColorsLength: window.nColorPaletteColors ? window.nColorPaletteColors.length : 0,
        nColorPaletteColors: window.nColorPaletteColors ? window.nColorPaletteColors.slice(0, 6) : [],
      };
    });
    console.log('[debug] After segmentation:', JSON.stringify(afterSegment, null, 2));

    // Take screenshot after segmentation
    await page.screenshot({
      path: path.join(OUTPUT_DIR, `debug-ncolor-${ts}-02-after-segment.png`),
      fullPage: true
    });

    // STEP 3: Check if the palette is using N-color or instance colors
    console.log('[debug] STEP 3: Checking color sampling...');
    const colorCheck = await page.evaluate(() => {
      // Sample some pixel colors from the mask
      const results = {};

      // Check what getColormapColor returns for various labels
      if (typeof window.getColormapColor === 'function') {
        results.colormapColors = {};
        for (let i = 1; i <= 10; i++) {
          results.colormapColors[i] = window.getColormapColor(i);
        }
      }

      // Check the N-color palette
      results.nColorPalette = window.nColorPaletteColors ? window.nColorPaletteColors.slice(0, 6) : [];

      // Check the color table
      results.colorTable = window.colorTable ? window.colorTable.slice(0, 6) : [];

      return results;
    });
    console.log('[debug] Color check:', JSON.stringify(colorCheck, null, 2));

    // STEP 4: Force palette rebuild and redraw
    console.log('[debug] STEP 4: Forcing palette rebuild...');
    await page.evaluate(() => {
      window.paletteTextureDirty = true;
      window.clearColorCaches();
      if (typeof window.draw === 'function') {
        window.draw();
      }
    });
    await page.waitForTimeout(500);

    // Take screenshot after forced rebuild
    await page.screenshot({
      path: path.join(OUTPUT_DIR, `debug-ncolor-${ts}-03-after-rebuild.png`),
      fullPage: true
    });

    const afterRebuild = await page.evaluate(() => {
      return {
        nColorActive: window.nColorActive,
        paletteTextureDirty: window.paletteTextureDirty,
      };
    });
    console.log('[debug] After rebuild:', JSON.stringify(afterRebuild, null, 2));

    console.log('[debug] Test completed!');
    console.log(`[debug] Screenshots saved to: ${OUTPUT_DIR}`);

  } catch (err) {
    console.error('[debug] Error:', err.message);
    await page.screenshot({
      path: path.join(OUTPUT_DIR, `debug-ncolor-${ts}-error.png`),
      fullPage: true
    });
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
