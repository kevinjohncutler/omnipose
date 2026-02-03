#!/usr/bin/env node
/**
 * Test grid with N-color OFF and shuffle OFF - verify state and force rebuild
 */

const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

const DEFAULT_URL = 'http://127.0.0.1:8000/';
const OUTPUT_DIR = path.resolve(__dirname);

async function runTest() {
  console.log('='.repeat(60));
  console.log('GRID SHUFFLE OFF TEST v2');
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
    });

    // Toggle N-color OFF first
    console.log('\n[2] Setting N-color OFF...');
    await page.evaluate(() => {
      const toggle = document.getElementById('autoNColorToggle');
      if (toggle && toggle.checked) toggle.click();
    });
    await page.waitForTimeout(3000);

    // Check initial shuffle state
    const initialState = await page.evaluate(() => {
      return {
        labelShuffle: window.labelShuffle,
        toggleChecked: document.getElementById('labelShuffleToggle')?.checked,
      };
    });
    console.log(`  Initial shuffle state: labelShuffle=${initialState.labelShuffle}, toggle=${initialState.toggleChecked}`);

    // Force shuffle OFF by setting the variable directly AND clicking toggle
    console.log('\n[3] Forcing shuffle OFF...');
    await page.evaluate(() => {
      window.labelShuffle = false;
      const toggle = document.getElementById('labelShuffleToggle');
      if (toggle) {
        toggle.checked = false;
        toggle.dispatchEvent(new Event('change'));
      }
      // Clear caches and rebuild
      if (window.clearColorCaches) window.clearColorCaches();
      window.paletteTextureDirty = true;
      if (window.markMaskTextureFullDirty) window.markMaskTextureFullDirty();
      if (window.draw) window.draw();
    });
    await page.waitForTimeout(1000);

    // Verify shuffle state
    const afterState = await page.evaluate(() => {
      return {
        labelShuffle: window.labelShuffle,
        toggleChecked: document.getElementById('labelShuffleToggle')?.checked,
      };
    });
    console.log(`  After setting: labelShuffle=${afterState.labelShuffle}, toggle=${afterState.toggleChecked}`);

    // Take screenshot with shuffle OFF
    await page.screenshot({ path: path.join(OUTPUT_DIR, 'grid-shuffle-off.png'), fullPage: true });
    console.log('  Saved: grid-shuffle-off.png');

    // Get colors
    const colorsOff = await page.evaluate(() => {
      const result = [];
      for (let i = 1; i <= 10; i++) {
        const c = window.getColormapColor ? window.getColormapColor(i) : null;
        result.push({ label: i, color: c ? c.join(',') : 'null' });
      }
      return { colors: result, shuffle: window.labelShuffle };
    });
    console.log(`\n[4] Colors with shuffle=${colorsOff.shuffle}:`);
    for (const { label, color } of colorsOff.colors) {
      console.log(`  Label ${String(label).padStart(2)}: rgb(${color})`);
    }

    // Now turn shuffle ON
    console.log('\n[5] Setting shuffle ON...');
    await page.evaluate(() => {
      window.labelShuffle = true;
      const toggle = document.getElementById('labelShuffleToggle');
      if (toggle) {
        toggle.checked = true;
        toggle.dispatchEvent(new Event('change'));
      }
      if (window.clearColorCaches) window.clearColorCaches();
      window.paletteTextureDirty = true;
      if (window.markMaskTextureFullDirty) window.markMaskTextureFullDirty();
      if (window.draw) window.draw();
    });
    await page.waitForTimeout(1000);

    await page.screenshot({ path: path.join(OUTPUT_DIR, 'grid-shuffle-on.png'), fullPage: true });
    console.log('  Saved: grid-shuffle-on.png');

    const colorsOn = await page.evaluate(() => {
      const result = [];
      for (let i = 1; i <= 10; i++) {
        const c = window.getColormapColor ? window.getColormapColor(i) : null;
        result.push({ label: i, color: c ? c.join(',') : 'null' });
      }
      return { colors: result, shuffle: window.labelShuffle };
    });
    console.log(`\n[6] Colors with shuffle=${colorsOn.shuffle}:`);
    for (const { label, color } of colorsOn.colors) {
      console.log(`  Label ${String(label).padStart(2)}: rgb(${color})`);
    }

    // Compare
    console.log('\n[7] Comparing shuffle OFF vs ON:');
    let same = true;
    for (let i = 0; i < 10; i++) {
      if (colorsOff.colors[i].color !== colorsOn.colors[i].color) {
        same = false;
        console.log(`  Label ${i+1}: OFF=${colorsOff.colors[i].color} vs ON=${colorsOn.colors[i].color}`);
      }
    }
    if (same) {
      console.log('  WARNING: Colors are the same! Shuffle toggle may not be working.');
    } else {
      console.log('  Colors are different - shuffle toggle is working!');
    }

    // Copy to grid folder
    fs.copyFileSync(path.join(OUTPUT_DIR, 'grid-shuffle-off.png'), path.join(OUTPUT_DIR, '../grid/grid-shuffle-off.png'));
    fs.copyFileSync(path.join(OUTPUT_DIR, 'grid-shuffle-on.png'), path.join(OUTPUT_DIR, '../grid/grid-shuffle-on.png'));
    console.log('\n  Copied to ../grid/');

  } catch (err) {
    console.error('\n[ERROR]', err.message);
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
