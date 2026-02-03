#!/usr/bin/env node
/**
 * Test shuffle toggle and seed functionality
 */

const { chromium } = require('playwright');
const path = require('path');

const DEFAULT_URL = 'http://127.0.0.1:8000/';

async function runTest() {
  console.log('='.repeat(60));
  console.log('SHUFFLE TOGGLE AND SEED TEST');
  console.log('='.repeat(60));

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1600, height: 1000 },
  });
  const page = await context.newPage();

  try {
    // Load page
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

    // Toggle N-color OFF to work with instance colors
    console.log('\n[2] Toggling N-color OFF...');
    await page.evaluate(() => {
      const toggle = document.getElementById('autoNColorToggle');
      if (toggle && toggle.checked) toggle.click();
    });
    await page.waitForTimeout(1000);

    // Get colors with shuffle ON (default)
    const colorsShuffleOn = await page.evaluate(() => {
      const colors = {};
      for (let i = 1; i <= 10; i++) {
        const c = window.getColormapColor(i);
        colors[i] = c ? c.join(',') : 'null';
      }
      return colors;
    });
    console.log('\n[3] Colors with shuffle ON (default):');
    for (const [label, color] of Object.entries(colorsShuffleOn)) {
      console.log(`  Label ${label}: ${color}`);
    }

    // Toggle shuffle OFF
    console.log('\n[4] Toggling shuffle OFF...');
    await page.evaluate(() => {
      const toggle = document.getElementById('labelShuffleToggle');
      if (toggle && toggle.checked) toggle.click();
    });
    await page.waitForTimeout(500);

    const colorsShuffleOff = await page.evaluate(() => {
      const colors = {};
      for (let i = 1; i <= 10; i++) {
        const c = window.getColormapColor(i);
        colors[i] = c ? c.join(',') : 'null';
      }
      return colors;
    });
    console.log('\n[5] Colors with shuffle OFF:');
    for (const [label, color] of Object.entries(colorsShuffleOff)) {
      console.log(`  Label ${label}: ${color}`);
    }

    // Check if colors changed
    let shuffleChanged = false;
    for (let i = 1; i <= 10; i++) {
      if (colorsShuffleOn[i] !== colorsShuffleOff[i]) {
        shuffleChanged = true;
        break;
      }
    }
    console.log(`\n  Shuffle toggle effect: ${shuffleChanged ? 'WORKING - colors changed' : 'NOT WORKING - colors same'}`);

    // Toggle shuffle back ON and change seed
    console.log('\n[6] Toggling shuffle ON and testing seed...');
    await page.evaluate(() => {
      const toggle = document.getElementById('labelShuffleToggle');
      if (toggle && !toggle.checked) toggle.click();
    });
    await page.waitForTimeout(500);

    const colorsSeed0 = await page.evaluate(() => {
      const colors = {};
      for (let i = 1; i <= 10; i++) {
        const c = window.getColormapColor(i);
        colors[i] = c ? c.join(',') : 'null';
      }
      return colors;
    });
    console.log('\n[7] Colors with shuffle ON, seed=0:');
    for (const [label, color] of Object.entries(colorsSeed0)) {
      console.log(`  Label ${label}: ${color}`);
    }

    // Change seed to 42
    console.log('\n[8] Changing seed to 42...');
    await page.evaluate(() => {
      const input = document.getElementById('labelShuffleSeed');
      if (input) {
        input.value = '42';
        input.dispatchEvent(new Event('change'));
      }
    });
    await page.waitForTimeout(500);

    const colorsSeed42 = await page.evaluate(() => {
      const colors = {};
      for (let i = 1; i <= 10; i++) {
        const c = window.getColormapColor(i);
        colors[i] = c ? c.join(',') : 'null';
      }
      return colors;
    });
    console.log('\n[9] Colors with shuffle ON, seed=42:');
    for (const [label, color] of Object.entries(colorsSeed42)) {
      console.log(`  Label ${label}: ${color}`);
    }

    // Check if seed changed colors
    let seedChanged = false;
    for (let i = 1; i <= 10; i++) {
      if (colorsSeed0[i] !== colorsSeed42[i]) {
        seedChanged = true;
        break;
      }
    }
    console.log(`\n  Seed change effect: ${seedChanged ? 'WORKING - colors changed' : 'NOT WORKING - colors same'}`);

    console.log('\n' + '='.repeat(60));
    console.log('TEST SUMMARY');
    console.log('='.repeat(60));
    console.log(`Shuffle toggle: ${shuffleChanged ? 'PASS' : 'FAIL'}`);
    console.log(`Seed change: ${seedChanged ? 'PASS' : 'FAIL'}`);

  } catch (err) {
    console.error('\n[ERROR]', err.message);
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
