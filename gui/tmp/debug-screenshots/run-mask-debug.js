#!/usr/bin/env node
/**
 * Debug mask values and palette - check what's actually in the data structures
 */

const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const DEFAULT_URL = 'http://127.0.0.1:8000/';
const OUTPUT_DIR = path.resolve(__dirname);

async function runTest() {
  console.log('='.repeat(60));
  console.log('MASK AND PALETTE DEBUG');
  console.log('='.repeat(60));

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1600, height: 1000 },
  });
  const page = await context.newPage();

  try {
    await page.goto(DEFAULT_URL, { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForFunction(() => document.getElementById('canvas') !== null, { timeout: 30000 });

    // Run segmentation
    console.log('\n[1] Running segmentation...');
    await page.evaluate(() => {
      const btn = document.getElementById('segmentButton');
      if (btn) btn.click();
    });
    await page.waitForFunction(() => {
      const status = document.getElementById('segmentStatus');
      return status && (status.textContent.includes('complete') || status.textContent.includes('failed'));
    }, { timeout: 120000 });
    await page.waitForTimeout(2000);

    console.log('\n[2] State with N-color ON:');
    const ncolorOnState = await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      if (!maskValues) return { error: 'No mask values' };

      const labelCounts = {};
      for (let i = 0; i < maskValues.length; i++) {
        const label = maskValues[i];
        if (label > 0) {
          labelCounts[label] = (labelCounts[label] || 0) + 1;
        }
      }

      return {
        uniqueLabels: Object.keys(labelCounts).length,
        labelCounts,
        nColorActive: document.getElementById('autoNColorToggle')?.checked,
      };
    });
    console.log(`  Unique labels: ${ncolorOnState.uniqueLabels}`);
    console.log(`  N-color active: ${ncolorOnState.nColorActive}`);
    console.log(`  Label counts: ${JSON.stringify(ncolorOnState.labelCounts)}`);

    await page.screenshot({ path: path.join(OUTPUT_DIR, 'mask-debug-01-ncolor-on.png'), fullPage: true });

    // Toggle N-color OFF
    console.log('\n[3] Toggling N-color OFF...');
    await page.evaluate(() => {
      const toggle = document.getElementById('autoNColorToggle');
      if (toggle && toggle.checked) toggle.click();
    });
    await page.waitForTimeout(3000);

    console.log('\n[4] State with N-color OFF:');
    const ncolorOffState = await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      if (!maskValues) return { error: 'No mask values' };

      const labelCounts = {};
      for (let i = 0; i < maskValues.length; i++) {
        const label = maskValues[i];
        if (label > 0) {
          labelCounts[label] = (labelCounts[label] || 0) + 1;
        }
      }

      // Get colors for each label
      const labelColors = {};
      if (typeof window.getColormapColor === 'function') {
        for (const label of Object.keys(labelCounts)) {
          labelColors[label] = window.getColormapColor(Number(label));
        }
      }

      return {
        uniqueLabels: Object.keys(labelCounts).length,
        labelCounts,
        labelColors,
        nColorActive: document.getElementById('autoNColorToggle')?.checked,
      };
    });

    console.log(`  Unique labels: ${ncolorOffState.uniqueLabels}`);
    console.log(`  N-color active: ${ncolorOffState.nColorActive}`);
    console.log(`  Label counts (first 10):`);

    const labels = Object.keys(ncolorOffState.labelCounts).slice(0, 10);
    for (const label of labels) {
      const count = ncolorOffState.labelCounts[label];
      const color = ncolorOffState.labelColors[label];
      const colorStr = color ? `rgb(${color.join(',')})` : 'N/A';
      console.log(`    Label ${label}: ${count} pixels, color=${colorStr}`);
    }

    // Check for duplicate colors
    const colorStrings = Object.values(ncolorOffState.labelColors).map(c =>
      c ? c.join(',') : 'null'
    );
    const uniqueColors = new Set(colorStrings);
    console.log(`\n  Total labels: ${colorStrings.length}`);
    console.log(`  Unique colors from getColormapColor: ${uniqueColors.size}`);

    if (uniqueColors.size < colorStrings.length) {
      console.log('\n  DUPLICATE COLORS DETECTED!');

      // Find which labels share colors
      const colorToLabels = {};
      for (const [label, color] of Object.entries(ncolorOffState.labelColors)) {
        const key = color ? color.join(',') : 'null';
        if (!colorToLabels[key]) colorToLabels[key] = [];
        colorToLabels[key].push(label);
      }

      for (const [color, labelsWithColor] of Object.entries(colorToLabels)) {
        if (labelsWithColor.length > 1) {
          console.log(`    Color ${color}: labels ${labelsWithColor.join(', ')}`);
        }
      }
    } else {
      console.log('  All labels have unique colors from getColormapColor!');
    }

    await page.screenshot({ path: path.join(OUTPUT_DIR, 'mask-debug-02-ncolor-off.png'), fullPage: true });

    // Check the palette texture data directly
    console.log('\n[5] Checking palette texture...');
    const paletteInfo = await page.evaluate(() => {
      // Try to get palette data
      if (typeof window.buildPaletteTextureData === 'function') {
        const data = window.buildPaletteTextureData();
        const colors = [];
        for (let i = 0; i < Math.min(20, data.length / 4); i++) {
          colors.push([data[i*4], data[i*4+1], data[i*4+2]]);
        }
        return { paletteColors: colors };
      }
      return { error: 'buildPaletteTextureData not found' };
    });

    if (paletteInfo.paletteColors) {
      console.log('  Palette colors (indices 0-19):');
      for (let i = 0; i < paletteInfo.paletteColors.length; i++) {
        const c = paletteInfo.paletteColors[i];
        console.log(`    [${i}]: rgb(${c.join(',')})`);
      }

      // Check uniqueness
      const paletteStrings = paletteInfo.paletteColors.map(c => c.join(','));
      const uniquePalette = new Set(paletteStrings);
      console.log(`\n  Unique palette colors: ${uniquePalette.size} out of ${paletteStrings.length}`);
    }

    // Save full debug data
    fs.writeFileSync(
      path.join(OUTPUT_DIR, 'mask-debug.json'),
      JSON.stringify({ ncolorOnState, ncolorOffState, paletteInfo }, null, 2)
    );
    console.log('\n  Full debug data saved to: mask-debug.json');

    console.log('\n' + '='.repeat(60));

  } catch (err) {
    console.error('\n[ERROR]', err.message);
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
