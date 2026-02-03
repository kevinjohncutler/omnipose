#!/usr/bin/env node
/**
 * Debug test - verify mask values and palette data match
 */

const { chromium } = require('playwright');
const path = require('path');

const DEFAULT_URL = 'http://127.0.0.1:8000/';

async function runTest() {
  console.log('='.repeat(60));
  console.log('MASK & PALETTE DEBUG TEST');
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

    // Get state BEFORE toggling N-color off
    const beforeToggle = await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      if (!maskValues) return { error: 'no mask' };

      // Sample first 20 unique labels and their values
      const labelSamples = {};
      const seenLabels = new Set();
      for (let i = 0; i < maskValues.length && seenLabels.size < 20; i++) {
        const v = maskValues[i];
        if (v > 0 && !seenLabels.has(v)) {
          seenLabels.add(v);
          labelSamples[v] = { firstIndex: i };
        }
      }

      return {
        nColorActive: window.nColorActive,
        uniqueLabels: Array.from(seenLabels).sort((a,b) => a-b),
        labelSamples,
      };
    });
    console.log('\n[2] BEFORE N-color OFF:');
    console.log(`  nColorActive: ${beforeToggle.nColorActive}`);
    console.log(`  uniqueLabels: ${beforeToggle.uniqueLabels?.join(', ')}`);

    // Toggle N-color OFF
    console.log('\n[3] Toggling N-color OFF...');
    await page.evaluate(() => {
      const toggle = document.getElementById('autoNColorToggle');
      if (toggle && toggle.checked) toggle.click();
    });
    await page.waitForTimeout(5000); // Wait for relabel to complete

    // Get state AFTER toggling N-color off
    const afterToggle = await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      if (!maskValues) return { error: 'no mask' };

      // Sample first 20 unique labels and their values
      const labelSamples = {};
      const seenLabels = new Set();
      for (let i = 0; i < maskValues.length && seenLabels.size < 20; i++) {
        const v = maskValues[i];
        if (v > 0 && !seenLabels.has(v)) {
          seenLabels.add(v);
          labelSamples[v] = {
            firstIndex: i,
            color: typeof window.getColormapColor === 'function' ? window.getColormapColor(v) : null,
          };
        }
      }

      // Check palette texture data
      let paletteColors = null;
      if (window.webglPipeline && window.webglPipeline.gl) {
        // We can't directly read the palette texture, but we can check buildPaletteTextureData
        if (typeof window.buildPaletteTextureData === 'function') {
          const data = window.buildPaletteTextureData();
          paletteColors = {};
          for (const label of Array.from(seenLabels)) {
            const base = label * 4;
            paletteColors[label] = [data[base], data[base+1], data[base+2]];
          }
        }
      }

      return {
        nColorActive: window.nColorActive,
        uniqueLabels: Array.from(seenLabels).sort((a,b) => a-b),
        labelSamples,
        paletteColors,
        paletteTextureDirty: window.paletteTextureDirty,
      };
    });

    console.log('\n[4] AFTER N-color OFF:');
    console.log(`  nColorActive: ${afterToggle.nColorActive}`);
    console.log(`  paletteTextureDirty: ${afterToggle.paletteTextureDirty}`);
    console.log(`  uniqueLabels (${afterToggle.uniqueLabels?.length}): ${afterToggle.uniqueLabels?.slice(0, 15).join(', ')}...`);

    // Show label -> color mapping
    console.log('\n[5] Label -> Color mapping (from getColormapColor):');
    const labels = afterToggle.uniqueLabels?.slice(0, 15) || [];
    for (const label of labels) {
      const sample = afterToggle.labelSamples?.[label];
      const color = sample?.color;
      const colorStr = color ? `rgb(${color.join(',')})` : 'null';
      console.log(`  Label ${label}: ${colorStr}`);
    }

    // Show palette texture colors if available
    if (afterToggle.paletteColors) {
      console.log('\n[6] Palette texture colors (at index = label):');
      for (const label of labels) {
        const color = afterToggle.paletteColors[label];
        const colorStr = color ? `rgb(${color.join(',')})` : 'null';
        console.log(`  Palette[${label}]: ${colorStr}`);
      }
    }

    // Check if getColormapColor matches palette texture
    if (afterToggle.paletteColors) {
      console.log('\n[7] Comparing getColormapColor vs palette texture:');
      let mismatches = 0;
      for (const label of labels) {
        const expected = afterToggle.labelSamples?.[label]?.color;
        const actual = afterToggle.paletteColors[label];
        if (expected && actual) {
          const match = expected[0] === actual[0] && expected[1] === actual[1] && expected[2] === actual[2];
          if (!match) {
            mismatches++;
            console.log(`  MISMATCH Label ${label}: expected rgb(${expected.join(',')}) vs palette rgb(${actual.join(',')})`);
          }
        }
      }
      console.log(`  Total mismatches: ${mismatches}`);
    }

    // Take screenshot
    await page.screenshot({ path: path.join(__dirname, 'mask-palette-test.png'), fullPage: true });
    console.log('\n  Screenshot saved: mask-palette-test.png');

  } catch (err) {
    console.error('\n[ERROR]', err.message);
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
