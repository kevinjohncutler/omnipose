#!/usr/bin/env node
/**
 * Test grid with N-color OFF and shuffle OFF - should see rainbow snake pattern
 */

const { chromium } = require('playwright');
const path = require('path');

const DEFAULT_URL = 'http://127.0.0.1:8000/';
const OUTPUT_DIR = path.resolve(__dirname);

async function runTest() {
  console.log('='.repeat(60));
  console.log('GRID SHUFFLE OFF TEST - Should see rainbow snake');
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

    // Create grid
    console.log('\n[1] Creating 10x10 grid...');
    await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      if (!maskValues) return;

      const width = window.imgWidth || 392;
      const height = window.imgHeight || 384;
      const gridCols = 10;
      const gridRows = 10;
      const cellWidth = Math.floor(width / gridCols);
      const cellHeight = Math.floor(height / gridRows);

      for (let row = 0; row < gridRows; row++) {
        for (let col = 0; col < gridCols; col++) {
          const label = row * gridCols + col + 1;
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

      if (window.paintingApi?.rebuildComponents) window.paintingApi.rebuildComponents();
      window.paletteTextureDirty = true;
      if (window.markMaskTextureFullDirty) window.markMaskTextureFullDirty();
      if (window.draw) window.draw();
    });

    // Toggle N-color OFF
    console.log('\n[2] Setting N-color OFF...');
    await page.evaluate(() => {
      const toggle = document.getElementById('autoNColorToggle');
      if (toggle && toggle.checked) toggle.click();
    });
    await page.waitForTimeout(3000);

    // Toggle shuffle OFF
    console.log('\n[3] Setting shuffle OFF...');
    await page.evaluate(() => {
      const toggle = document.getElementById('labelShuffleToggle');
      if (toggle && toggle.checked) toggle.click();
    });
    await page.waitForTimeout(1000);

    // Take screenshot
    const screenshotPath = path.join(OUTPUT_DIR, 'grid-shuffle-off.png');
    await page.screenshot({ path: screenshotPath, fullPage: true });
    console.log(`  Saved: grid-shuffle-off.png`);

    // Get colors for first 20 labels to verify rainbow order
    const colors = await page.evaluate(() => {
      const result = [];
      for (let i = 1; i <= 20; i++) {
        const c = window.getColormapColor ? window.getColormapColor(i) : null;
        result.push({ label: i, color: c ? c.join(',') : 'null' });
      }
      return result;
    });

    console.log('\n[4] Colors for labels 1-20 (shuffle OFF):');
    for (const { label, color } of colors) {
      console.log(`  Label ${String(label).padStart(2)}: rgb(${color})`);
    }

    // Copy to grid folder
    const fs = require('fs');
    fs.copyFileSync(screenshotPath, path.join(OUTPUT_DIR, '../grid/grid-shuffle-off.png'));
    console.log('\n  Copied to: ../grid/grid-shuffle-off.png');

  } catch (err) {
    console.error('\n[ERROR]', err.message);
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
