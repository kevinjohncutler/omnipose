#!/usr/bin/env node
/**
 * Test state persistence across page refresh
 */

const { chromium } = require('playwright');
const path = require('path');

const DEFAULT_URL = 'http://127.0.0.1:8000/';
const OUTPUT_DIR = path.resolve(__dirname);

async function runTest() {
  console.log('='.repeat(60));
  console.log('STATE PERSISTENCE TEST');
  console.log('='.repeat(60));

  const browser = await chromium.launch({ headless: true });
  // Use persistent context to maintain localStorage across navigations
  const context = await browser.newContext({
    viewport: { width: 1600, height: 1000 },
  });
  const page = await context.newPage();

  try {
    // Load page
    console.log('\n[1] Loading page...');
    await page.goto(DEFAULT_URL, { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForFunction(() => document.getElementById('canvas') !== null, { timeout: 30000 });

    // Check initial state
    const initialMaskState = await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      const uniqueLabels = maskValues ? new Set(Array.from(maskValues).filter(v => v > 0)) : new Set();
      return {
        hasMask: Boolean(maskValues),
        uniqueLabelCount: uniqueLabels.size,
        localStorageKeys: Object.keys(localStorage).filter(k => k.startsWith('OMNI_')),
      };
    });
    console.log('  Initial state:', JSON.stringify(initialMaskState, null, 2));

    // Run segmentation
    console.log('\n[2] Running segmentation...');
    await page.evaluate(() => {
      const btn = document.getElementById('segmentButton');
      if (btn) btn.click();
    });
    await page.waitForFunction(() => {
      const status = document.getElementById('segmentStatus');
      return status && (status.textContent.includes('complete') || status.textContent.includes('failed'));
    }, { timeout: 120000 });
    await page.waitForTimeout(2000);

    // Check state after segmentation
    const afterSegState = await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      const uniqueLabels = maskValues ? new Set(Array.from(maskValues).filter(v => v > 0)) : new Set();
      return {
        hasMask: Boolean(maskValues),
        uniqueLabelCount: uniqueLabels.size,
        sampleLabels: Array.from(uniqueLabels).slice(0, 5),
      };
    });
    console.log('  After segmentation:', JSON.stringify(afterSegState, null, 2));

    // Wait for state to be saved
    console.log('\n[3] Waiting for state save...');
    await page.waitForTimeout(3000);

    // Check localStorage
    const storageInfo = await page.evaluate(() => {
      const result = {};
      for (const key of Object.keys(localStorage)) {
        if (key.startsWith('OMNI_')) {
          try {
            const value = localStorage.getItem(key);
            const parsed = JSON.parse(value);
            result[key] = {
              size: value.length,
              hasMask: Boolean(parsed.mask),
              maskSize: parsed.mask ? parsed.mask.length : 0,
              maskHasNonZero: parsed.maskHasNonZero,
            };
          } catch (e) {
            result[key] = { error: e.message };
          }
        }
      }
      return result;
    });
    console.log('  LocalStorage info:', JSON.stringify(storageInfo, null, 2));

    // Take screenshot before refresh
    await page.screenshot({ path: path.join(OUTPUT_DIR, 'persistence-1-before-refresh.png'), fullPage: true });
    console.log('  Saved: persistence-1-before-refresh.png');

    // Refresh the page
    console.log('\n[4] Refreshing page...');
    await page.reload({ waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForFunction(() => document.getElementById('canvas') !== null, { timeout: 30000 });
    await page.waitForTimeout(2000);

    // Check state after refresh
    const afterRefreshState = await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      const uniqueLabels = maskValues ? new Set(Array.from(maskValues).filter(v => v > 0)) : new Set();
      return {
        hasMask: Boolean(maskValues),
        uniqueLabelCount: uniqueLabels.size,
        sampleLabels: Array.from(uniqueLabels).slice(0, 5),
        localStorageKeys: Object.keys(localStorage).filter(k => k.startsWith('OMNI_')),
      };
    });
    console.log('  After refresh:', JSON.stringify(afterRefreshState, null, 2));

    // Take screenshot after refresh
    await page.screenshot({ path: path.join(OUTPUT_DIR, 'persistence-2-after-refresh.png'), fullPage: true });
    console.log('  Saved: persistence-2-after-refresh.png');

    // Check if mask was preserved
    const maskPreserved = afterRefreshState.uniqueLabelCount > 0 &&
                          afterRefreshState.uniqueLabelCount === afterSegState.uniqueLabelCount;

    console.log('\n' + '='.repeat(60));
    console.log('TEST SUMMARY');
    console.log('='.repeat(60));
    console.log(`Labels before refresh: ${afterSegState.uniqueLabelCount}`);
    console.log(`Labels after refresh: ${afterRefreshState.uniqueLabelCount}`);
    console.log(`State persistence: ${maskPreserved ? 'PASS' : 'FAIL'}`);

  } catch (err) {
    console.error('\n[ERROR]', err.message);
    await page.screenshot({ path: path.join(OUTPUT_DIR, 'persistence-error.png'), fullPage: true });
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
