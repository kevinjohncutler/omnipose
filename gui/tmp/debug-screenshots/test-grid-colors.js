#!/usr/bin/env node
/**
 * Test with a known grid of masks - not using segmentation
 * Creates a simple 10x10 grid of labeled cells
 */

const { chromium } = require('playwright');
const path = require('path');

const DEFAULT_URL = 'http://127.0.0.1:8000/';
const OUTPUT_DIR = path.resolve(__dirname);

async function runTest() {
  console.log('='.repeat(60));
  console.log('GRID COLOR TEST');
  console.log('='.repeat(60));

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1600, height: 1000 },
  });
  const page = await context.newPage();

  try {
    await page.goto(DEFAULT_URL, { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForFunction(() => document.getElementById('canvas') !== null, { timeout: 30000 });
    await page.waitForTimeout(2000);

    // Create a simple grid pattern in the mask
    console.log('\n[1] Creating a 10x10 grid of labeled cells...');
    await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      if (!maskValues) {
        console.error('No mask values!');
        return;
      }

      const width = window.imgWidth || 392;
      const height = window.imgHeight || 384;

      // Create 10x10 grid (100 cells)
      const gridCols = 10;
      const gridRows = 10;
      const cellWidth = Math.floor(width / gridCols);
      const cellHeight = Math.floor(height / gridRows);

      // Fill mask with grid pattern
      for (let row = 0; row < gridRows; row++) {
        for (let col = 0; col < gridCols; col++) {
          const label = row * gridCols + col + 1; // Labels 1-100

          const startX = col * cellWidth;
          const startY = row * cellHeight;
          const endX = Math.min(startX + cellWidth, width);
          const endY = Math.min(startY + cellHeight, height);

          for (let y = startY; y < endY; y++) {
            for (let x = startX; x < endX; x++) {
              const idx = y * width + x;
              maskValues[idx] = label;
            }
          }
        }
      }

      // Rebuild painting state
      if (window.paintingApi && typeof window.paintingApi.rebuildComponents === 'function') {
        window.paintingApi.rebuildComponents();
      }

      // Mark textures dirty
      window.paletteTextureDirty = true;
      if (typeof window.markMaskTextureFullDirty === 'function') {
        window.markMaskTextureFullDirty();
      }
      if (typeof window.markOutlineTextureFullDirty === 'function') {
        window.markOutlineTextureFullDirty();
      }

      // Force draw
      if (typeof window.draw === 'function') {
        window.draw();
      }
    });

    await page.waitForTimeout(1000);

    // Verify grid was created
    const gridInfo = await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      const uniqueLabels = new Set();
      for (let i = 0; i < maskValues.length; i++) {
        if (maskValues[i] > 0) uniqueLabels.add(maskValues[i]);
      }
      return {
        uniqueLabels: uniqueLabels.size,
        sampleLabels: Array.from(uniqueLabels).sort((a,b) => a-b).slice(0, 20),
      };
    });
    console.log(`  Created ${gridInfo.uniqueLabels} unique labels`);

    // Take screenshot WITH N-color ON (default)
    console.log('\n[2] Screenshot with N-color ON...');
    await page.screenshot({ path: path.join(OUTPUT_DIR, 'grid-1-ncolor-on.png'), fullPage: true });

    // Toggle N-color OFF
    console.log('\n[3] Toggling N-color OFF...');
    await page.evaluate(() => {
      // First make sure N-color is on
      if (!window.nColorActive) {
        const toggle = document.getElementById('autoNColorToggle');
        if (toggle && !toggle.checked) toggle.click();
      }
    });
    await page.waitForTimeout(500);

    // Now toggle it OFF
    await page.evaluate(() => {
      const toggle = document.getElementById('autoNColorToggle');
      if (toggle && toggle.checked) toggle.click();
    });
    await page.waitForTimeout(3000);

    // Take screenshot WITH N-color OFF
    console.log('\n[4] Screenshot with N-color OFF...');
    await page.screenshot({ path: path.join(OUTPUT_DIR, 'grid-2-ncolor-off.png'), fullPage: true });

    // Get colors for some labels
    const colors = await page.evaluate(() => {
      const result = {};
      for (let i = 1; i <= 20; i++) {
        if (typeof window.getColormapColor === 'function') {
          result[i] = window.getColormapColor(i);
        }
      }
      return result;
    });

    console.log('\n[5] Colors for labels 1-20 (N-color OFF):');
    for (let i = 1; i <= 20; i++) {
      const c = colors[i];
      console.log(`  Label ${String(i).padStart(2)}: rgb(${c ? c.join(',') : 'null'})`);
    }

    // Check for duplicate colors
    const colorStrs = Object.values(colors).map(c => c ? c.join(',') : 'null');
    const uniqueColors = new Set(colorStrs);
    console.log(`\n  Unique colors: ${uniqueColors.size} / 20`);

    if (uniqueColors.size < 20) {
      const dupes = {};
      for (const [label, c] of Object.entries(colors)) {
        const key = c ? c.join(',') : 'null';
        if (!dupes[key]) dupes[key] = [];
        dupes[key].push(label);
      }
      console.log('  Duplicate color groups:');
      for (const [color, labels] of Object.entries(dupes)) {
        if (labels.length > 1) {
          console.log(`    rgb(${color}): labels ${labels.join(', ')}`);
        }
      }
    }

    // Toggle N-color back ON
    console.log('\n[6] Toggling N-color back ON...');
    await page.evaluate(() => {
      const toggle = document.getElementById('autoNColorToggle');
      if (toggle && !toggle.checked) toggle.click();
    });
    await page.waitForTimeout(2000);

    // Take screenshot
    console.log('\n[7] Screenshot with N-color ON again...');
    await page.screenshot({ path: path.join(OUTPUT_DIR, 'grid-3-ncolor-on-again.png'), fullPage: true });

    console.log('\n' + '='.repeat(60));
    console.log('Screenshots saved. Compare:');
    console.log('  grid-1-ncolor-on.png     - Initial with N-color ON');
    console.log('  grid-2-ncolor-off.png    - After toggle to OFF (instance colors)');
    console.log('  grid-3-ncolor-on-again.png - After toggle back to ON');
    console.log('='.repeat(60));

  } catch (err) {
    console.error('\n[ERROR]', err.message);
    console.error(err.stack);
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
