#!/usr/bin/env node
/**
 * Debug test - saves screenshots to gui/tmp/debug-screenshots/
 * Run this from the omnipose root: node gui/tmp/debug-screenshots/run-debug-test.js
 */

const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const DEFAULT_URL = 'http://127.0.0.1:8000/';
const OUTPUT_DIR = path.resolve(__dirname);

async function runTest() {
  console.log('='.repeat(60));
  console.log('DEBUG TEST - Screenshots will be saved to:');
  console.log(OUTPUT_DIR);
  console.log('='.repeat(60));

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1600, height: 1000 },
  });
  const page = await context.newPage();

  const logs = [];
  page.on('console', (msg) => {
    logs.push(`[${msg.type()}] ${msg.text()}`);
  });

  try {
    // 1. Initial page load
    console.log('\n[STEP 1] Loading page...');
    await page.goto(DEFAULT_URL, { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForFunction(() => {
      return document.getElementById('canvas') !== null;
    }, { timeout: 30000 });

    await page.screenshot({ path: path.join(OUTPUT_DIR, '01-initial-load.png'), fullPage: true });
    console.log('  Saved: 01-initial-load.png');

    // Check initial state
    const initialState = await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      const uniqueLabels = maskValues ? new Set(Array.from(maskValues).filter(v => v > 0)) : new Set();
      const toggle = document.getElementById('autoNColorToggle');
      return {
        hasMask: Boolean(maskValues),
        uniqueLabelCount: uniqueLabels.size,
        nColorToggleChecked: toggle?.checked,
        localStorage: Object.keys(localStorage).filter(k => k.startsWith('OMNI_')),
      };
    });
    console.log('  Initial state:', JSON.stringify(initialState, null, 2));

    // 2. Run segmentation
    console.log('\n[STEP 2] Running segmentation...');
    await page.evaluate(() => {
      const btn = document.getElementById('segmentButton');
      if (btn) btn.click();
    });

    await page.waitForFunction(() => {
      const status = document.getElementById('segmentStatus');
      return status && (status.textContent.includes('complete') || status.textContent.includes('failed'));
    }, { timeout: 120000 });
    await page.waitForTimeout(2000);

    await page.screenshot({ path: path.join(OUTPUT_DIR, '02-after-segmentation.png'), fullPage: true });
    console.log('  Saved: 02-after-segmentation.png');

    const afterSegState = await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      const uniqueLabels = maskValues ? new Set(Array.from(maskValues).filter(v => v > 0)) : new Set();
      const toggle = document.getElementById('autoNColorToggle');
      return {
        uniqueLabelCount: uniqueLabels.size,
        nColorToggleChecked: toggle?.checked,
        sampleLabels: Array.from(uniqueLabels).slice(0, 10),
      };
    });
    console.log('  After segmentation:', JSON.stringify(afterSegState, null, 2));

    // 3. Toggle N-color OFF
    console.log('\n[STEP 3] Toggling N-color OFF...');
    await page.evaluate(() => {
      const toggle = document.getElementById('autoNColorToggle');
      if (toggle && toggle.checked) toggle.click();
    });
    await page.waitForTimeout(3000);

    await page.screenshot({ path: path.join(OUTPUT_DIR, '03-ncolor-off.png'), fullPage: true });
    console.log('  Saved: 03-ncolor-off.png');

    const ncolorOffState = await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      const uniqueLabels = maskValues ? new Set(Array.from(maskValues).filter(v => v > 0)) : new Set();
      const toggle = document.getElementById('autoNColorToggle');

      // Get colors for first 10 labels
      const colors = {};
      if (typeof window.getColormapColor === 'function') {
        const labels = Array.from(uniqueLabels).slice(0, 10);
        for (const label of labels) {
          colors[label] = window.getColormapColor(label);
        }
      }

      return {
        uniqueLabelCount: uniqueLabels.size,
        nColorToggleChecked: toggle?.checked,
        sampleLabels: Array.from(uniqueLabels).slice(0, 10),
        colors,
      };
    });
    console.log('  N-color OFF state:', JSON.stringify(ncolorOffState, null, 2));

    // Check if colors are unique
    const colorValues = Object.values(ncolorOffState.colors);
    const colorStrings = colorValues.map(c => c ? c.join(',') : 'null');
    const uniqueColors = new Set(colorStrings);
    console.log(`  Unique colors: ${uniqueColors.size} out of ${colorStrings.length}`);
    if (uniqueColors.size < colorStrings.length) {
      console.log('  WARNING: Duplicate colors detected!');
      console.log('  Color strings:', colorStrings);
    } else {
      console.log('  SUCCESS: All colors are unique!');
    }

    // 4. Toggle N-color back ON
    console.log('\n[STEP 4] Toggling N-color back ON...');
    await page.evaluate(() => {
      const toggle = document.getElementById('autoNColorToggle');
      if (toggle && !toggle.checked) toggle.click();
    });
    await page.waitForTimeout(2000);

    await page.screenshot({ path: path.join(OUTPUT_DIR, '04-ncolor-on.png'), fullPage: true });
    console.log('  Saved: 04-ncolor-on.png');

    const ncolorOnState = await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      const uniqueLabels = maskValues ? new Set(Array.from(maskValues).filter(v => v > 0)) : new Set();
      const toggle = document.getElementById('autoNColorToggle');
      return {
        uniqueLabelCount: uniqueLabels.size,
        nColorToggleChecked: toggle?.checked,
        sampleLabels: Array.from(uniqueLabels).slice(0, 10),
      };
    });
    console.log('  N-color ON state:', JSON.stringify(ncolorOnState, null, 2));

    // Save console logs
    fs.writeFileSync(path.join(OUTPUT_DIR, 'console-logs.txt'), logs.join('\n'));
    console.log('\n  Saved: console-logs.txt');

    console.log('\n' + '='.repeat(60));
    console.log('TEST COMPLETE - Check screenshots in:');
    console.log(OUTPUT_DIR);
    console.log('='.repeat(60));

  } catch (err) {
    console.error('\n[ERROR]', err.message);
    await page.screenshot({ path: path.join(OUTPUT_DIR, 'error.png'), fullPage: true });
    console.log('  Saved: error.png');
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
