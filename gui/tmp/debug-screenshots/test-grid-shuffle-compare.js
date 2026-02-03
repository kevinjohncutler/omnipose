#!/usr/bin/env node
/**
 * Test grid with shuffle ON vs OFF - compare colors used
 */

const { chromium } = require('playwright');
const path = require('path');

const DEFAULT_URL = 'http://127.0.0.1:8000/';
const OUTPUT_DIR = '/Volumes/DataDrive/omnipose/gui/tmp/grid';

async function runTest() {
  console.log('='.repeat(60));
  console.log('GRID SHUFFLE COMPARISON TEST');
  console.log('='.repeat(60));

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1600, height: 1000 },
  });
  const page = await context.newPage();

  // Capture console logs
  page.on('console', msg => {
    if (msg.type() === 'log' || msg.type() === 'warn') {
      console.log(`  [browser] ${msg.text()}`);
    }
  });

  try {
    await page.goto(DEFAULT_URL, { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForFunction(() => document.getElementById('canvas') !== null, { timeout: 30000 });
    await page.waitForTimeout(2000);

    // Create a 10x10 grid of labeled cells
    console.log('\n[1] Creating 10x10 grid (100 labels)...');
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

    // Ensure N-color is OFF (instance mode)
    console.log('\n[2] Setting N-color OFF (instance mode)...');
    await page.evaluate(() => {
      const toggle = document.getElementById('autoNColorToggle');
      if (toggle && toggle.checked) toggle.click();
    });
    await page.waitForTimeout(1000);

    // Test SHUFFLE OFF
    console.log('\n[3] Testing SHUFFLE OFF...');
    await page.evaluate(() => {
      const shuffleToggle = document.getElementById('labelShuffleToggle');
      if (shuffleToggle && shuffleToggle.checked) shuffleToggle.click();
      window.paletteTextureDirty = true;
      if (typeof window.markMaskTextureFullDirty === 'function') {
        window.markMaskTextureFullDirty();
      }
      if (typeof window.draw === 'function') {
        window.draw();
      }
    });
    await page.waitForTimeout(2000);

    // Get colors for labels 1-20 with shuffle OFF
    const colorsShuffleOff = await page.evaluate(() => {
      const colors = [];
      for (let i = 1; i <= 100; i++) {
        if (typeof window.getColormapColor === 'function') {
          const c = window.getColormapColor(i);
          colors.push(c ? c.join(',') : 'null');
        }
      }
      return colors;
    });

    await page.screenshot({ path: path.join(OUTPUT_DIR, 'grid-shuffle-off-new.png'), fullPage: true });
    console.log('  Screenshot saved: grid-shuffle-off-new.png');

    // Test SHUFFLE ON
    console.log('\n[4] Testing SHUFFLE ON...');
    await page.evaluate(() => {
      const shuffleToggle = document.getElementById('labelShuffleToggle');
      if (shuffleToggle && !shuffleToggle.checked) shuffleToggle.click();
      window.paletteTextureDirty = true;
      if (typeof window.markMaskTextureFullDirty === 'function') {
        window.markMaskTextureFullDirty();
      }
      if (typeof window.draw === 'function') {
        window.draw();
      }
    });
    await page.waitForTimeout(2000);

    // Get colors for labels 1-20 with shuffle ON
    const colorsShuffleOn = await page.evaluate(() => {
      const colors = [];
      for (let i = 1; i <= 100; i++) {
        if (typeof window.getColormapColor === 'function') {
          const c = window.getColormapColor(i);
          colors.push(c ? c.join(',') : 'null');
        }
      }
      return colors;
    });

    await page.screenshot({ path: path.join(OUTPUT_DIR, 'grid-shuffle-on-new.png'), fullPage: true });
    console.log('  Screenshot saved: grid-shuffle-on-new.png');

    // Analyze color usage
    console.log('\n[5] Color Analysis:');

    const uniqueColorsOff = new Set(colorsShuffleOff);
    const uniqueColorsOn = new Set(colorsShuffleOn);

    console.log(`  Shuffle OFF: ${uniqueColorsOff.size} unique colors for 100 labels`);
    console.log(`  Shuffle ON:  ${uniqueColorsOn.size} unique colors for 100 labels`);

    // Check if same set of colors
    const sameColors = [...uniqueColorsOff].every(c => uniqueColorsOn.has(c)) &&
                       [...uniqueColorsOn].every(c => uniqueColorsOff.has(c));
    console.log(`  Same color set: ${sameColors ? 'YES' : 'NO'}`);

    // Show first 10 colors for each mode
    console.log('\n  First 10 labels:');
    console.log('  Label | Shuffle OFF          | Shuffle ON');
    console.log('  ' + '-'.repeat(55));
    for (let i = 0; i < 10; i++) {
      console.log(`  ${String(i+1).padStart(5)} | ${colorsShuffleOff[i].padEnd(20)} | ${colorsShuffleOn[i]}`);
    }

    // Check color range (hue diversity)
    const parseColor = (str) => str.split(',').map(Number);

    const analyzeHueRange = (colors) => {
      const rgbColors = colors.map(parseColor);
      // Simple check: count how many distinct "color categories" we have
      const categories = new Set();
      for (const [r, g, b] of rgbColors) {
        // Rough hue categorization
        if (r > g && r > b) categories.add('red');
        if (g > r && g > b) categories.add('green');
        if (b > r && b > g) categories.add('blue');
        if (r > 200 && g > 200 && b < 100) categories.add('yellow');
        if (r > 200 && b > 200 && g < 150) categories.add('magenta');
        if (g > 200 && b > 200 && r < 150) categories.add('cyan');
        if (r > 200 && g > 100 && g < 200 && b < 100) categories.add('orange');
        if (r > 100 && r < 200 && b > 200) categories.add('purple');
      }
      return categories;
    };

    const huesOff = analyzeHueRange(colorsShuffleOff);
    const huesOn = analyzeHueRange(colorsShuffleOn);

    console.log(`\n  Hue categories present:`);
    console.log(`    Shuffle OFF: ${[...huesOff].join(', ')}`);
    console.log(`    Shuffle ON:  ${[...huesOn].join(', ')}`);

    console.log('\n' + '='.repeat(60));

  } catch (err) {
    console.error('\n[ERROR]', err.message);
    console.error(err.stack);
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
