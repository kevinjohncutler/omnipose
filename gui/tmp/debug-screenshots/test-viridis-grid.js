#!/usr/bin/env node
/**
 * Test viridis with grid pattern - no shuffle - should see gradient
 */

const { chromium } = require('playwright');
const path = require('path');

const DEFAULT_URL = 'http://127.0.0.1:8000/';
const OUTPUT_DIR = path.resolve(__dirname);

async function runTest() {
  console.log('='.repeat(60));
  console.log('VIRIDIS GRID TEST - NO SHUFFLE');
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

    // Create 10x10 grid
    console.log('\n[1] Creating 10x10 grid (100 labels)...');
    await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      if (!maskValues) {
        console.error('No mask values!');
        return;
      }

      const width = window.imgWidth || 392;
      const height = window.imgHeight || 384;
      const gridCols = 10;
      const gridRows = 10;
      const cellWidth = Math.floor(width / gridCols);
      const cellHeight = Math.floor(height / gridRows);

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

      // Update currentMaxLabel
      window.currentMaxLabel = 100;

      // Rebuild
      if (window.paintingApi?.rebuildComponents) window.paintingApi.rebuildComponents();
      window.paletteTextureDirty = true;
      if (window.markMaskTextureFullDirty) window.markMaskTextureFullDirty();
      if (window.draw) window.draw();
    });
    await page.waitForTimeout(1000);

    // Select viridis
    console.log('\n[2] Selecting Viridis...');
    await page.evaluate(() => {
      const select = document.getElementById('cmapSelect');
      if (select) {
        select.value = 'viridis';
        select.dispatchEvent(new Event('change'));
      }
    });
    await page.waitForTimeout(500);

    // Turn OFF ncolor (instance mode)
    console.log('\n[3] Turning OFF Ncolor (instance mode)...');
    await page.evaluate(() => {
      const toggle = document.getElementById('autoNColorToggle');
      if (toggle && toggle.checked) toggle.click();
    });
    await page.waitForTimeout(500);

    // Turn OFF shuffle
    console.log('\n[4] Turning OFF Shuffle...');
    await page.evaluate(() => {
      window.labelShuffle = false;
      const toggle = document.getElementById('labelShuffleToggle');
      if (toggle && toggle.checked) toggle.click();
      window.paletteTextureDirty = true;
      if (window.markMaskTextureFullDirty) window.markMaskTextureFullDirty();
      if (window.draw) window.draw();
    });
    await page.waitForTimeout(1000);

    // Check colors for labels 1, 25, 50, 75, 100
    const colorInfo = await page.evaluate(() => {
      const result = {
        currentMaxLabel: window.currentMaxLabel,
        labelShuffle: window.labelShuffle,
        labelColormap: window.labelColormap,
        colors: []
      };
      for (const label of [1, 10, 25, 50, 75, 100]) {
        const c = window.getColormapColor ? window.getColormapColor(label) : null;
        result.colors.push({ label, color: c ? `rgb(${c.join(',')})` : 'null' });
      }
      return result;
    });

    console.log(`\n  currentMaxLabel: ${colorInfo.currentMaxLabel}`);
    console.log(`  labelShuffle: ${colorInfo.labelShuffle}`);
    console.log(`  labelColormap: ${colorInfo.labelColormap}`);
    console.log('\n  Sample label colors (should go purple → blue → teal → green → yellow):');
    colorInfo.colors.forEach(c => console.log(`    Label ${String(c.label).padStart(3)}: ${c.color}`));

    await page.screenshot({ path: path.join(OUTPUT_DIR, 'viridis-grid-no-shuffle.png'), fullPage: true });
    console.log('\n  Screenshot saved: viridis-grid-no-shuffle.png');

  } catch (err) {
    console.error('\n[ERROR]', err.message);
    console.error(err.stack);
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
