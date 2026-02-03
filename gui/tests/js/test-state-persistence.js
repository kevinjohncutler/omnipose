#!/usr/bin/env node
/**
 * Test state persistence across page refreshes.
 */

const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const DEFAULT_URL = 'http://127.0.0.1:8000/';
const OUTPUT_DIR = path.resolve(__dirname, '../../tmp/playwright');

const ensureDir = (dir) => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
};

const timestamp = () => new Date().toISOString().replace(/[:.]/g, '-');

async function runTest() {
  const ts = timestamp();
  ensureDir(OUTPUT_DIR);

  console.log('[test] Starting state persistence test...');

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1600, height: 1000 },
  });
  const page = await context.newPage();

  page.on('console', (msg) => {
    if (msg.type() === 'error' || msg.type() === 'warn') {
      console.log(`[page][${msg.type()}] ${msg.text()}`);
    }
  });

  try {
    // Initial page load
    await page.goto(DEFAULT_URL, { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForFunction(() => {
      return document.getElementById('canvas') !== null &&
             (window.OmniPainting || window.__OMNI_CONFIG__);
    }, { timeout: 30000 });

    console.log('[test] Page loaded, triggering segmentation...');

    // Trigger segmentation
    await page.evaluate(() => {
      const btn = document.getElementById('segmentButton');
      if (btn) btn.click();
    });

    // Wait for segmentation to complete
    await page.waitForFunction(() => {
      const status = document.getElementById('segmentStatus');
      return status && (status.textContent.includes('complete') || status.textContent.includes('failed'));
    }, { timeout: 120000 });
    await page.waitForTimeout(2000);

    // Get mask state before refresh
    const stateBefore = await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      const uniqueLabels = maskValues ? new Set(Array.from(maskValues).filter(v => v > 0)) : new Set();

      // Check localStorage
      const localStorageKeys = Object.keys(localStorage);
      const omniKeys = localStorageKeys.filter(k => k.startsWith('OMNI_'));

      return {
        uniqueLabelCount: uniqueLabels.size,
        maskLength: maskValues ? maskValues.length : 0,
        hasNonZero: uniqueLabels.size > 0,
        localStorageKeys: omniKeys,
        nColorActive: document.getElementById('autoNColorToggle')?.checked,
      };
    });
    console.log('[test] State before refresh:', JSON.stringify(stateBefore, null, 2));

    // Take screenshot before refresh
    await page.screenshot({
      path: path.join(OUTPUT_DIR, `persistence-${ts}-01-before-refresh.png`),
      fullPage: true
    });

    // Wait a bit for state to be saved
    console.log('[test] Waiting for state to be saved...');
    await page.waitForTimeout(2000);

    // Check localStorage content
    const localStorageContent = await page.evaluate(() => {
      const keys = Object.keys(localStorage).filter(k => k.startsWith('OMNI_'));
      const result = {};
      for (const key of keys) {
        try {
          const value = localStorage.getItem(key);
          if (value) {
            const parsed = JSON.parse(value);
            result[key] = {
              hasMask: Boolean(parsed.mask),
              maskLength: parsed.mask ? parsed.mask.length : 0,
              hasNColorActive: typeof parsed.nColorActive === 'boolean',
              nColorActive: parsed.nColorActive,
              maskHasNonZero: parsed.maskHasNonZero,
              keys: Object.keys(parsed),
            };
          }
        } catch (e) {
          result[key] = { error: e.message };
        }
      }
      return result;
    });
    console.log('[test] LocalStorage content:', JSON.stringify(localStorageContent, null, 2));

    // Refresh the page
    console.log('[test] Refreshing page...');
    await page.reload({ waitUntil: 'networkidle', timeout: 60000 });

    // Wait for app to initialize
    await page.waitForFunction(() => {
      return document.getElementById('canvas') !== null &&
             (window.OmniPainting || window.__OMNI_CONFIG__);
    }, { timeout: 30000 });
    await page.waitForTimeout(2000);

    // Get mask state after refresh
    const stateAfter = await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      const uniqueLabels = maskValues ? new Set(Array.from(maskValues).filter(v => v > 0)) : new Set();

      return {
        uniqueLabelCount: uniqueLabels.size,
        maskLength: maskValues ? maskValues.length : 0,
        hasNonZero: uniqueLabels.size > 0,
        nColorActive: document.getElementById('autoNColorToggle')?.checked,
      };
    });
    console.log('[test] State after refresh:', JSON.stringify(stateAfter, null, 2));

    // Take screenshot after refresh
    await page.screenshot({
      path: path.join(OUTPUT_DIR, `persistence-${ts}-02-after-refresh.png`),
      fullPage: true
    });

    // Verify results
    if (stateAfter.hasNonZero && stateAfter.uniqueLabelCount > 0) {
      console.log('[test] SUCCESS: Mask was restored after refresh');
      if (stateAfter.uniqueLabelCount === stateBefore.uniqueLabelCount) {
        console.log(`[test] SUCCESS: Label count matches (${stateAfter.uniqueLabelCount})`);
      } else {
        console.log(`[test] WARNING: Label count changed: ${stateBefore.uniqueLabelCount} -> ${stateAfter.uniqueLabelCount}`);
      }
    } else {
      console.log('[test] FAILURE: Mask was NOT restored after refresh');
      console.log('[test] This indicates a state persistence bug');
    }

    console.log('[test] Test completed!');
    console.log(`[test] Screenshots saved to: ${OUTPUT_DIR}`);

  } catch (err) {
    console.error('[test] Error:', err.message);
    await page.screenshot({
      path: path.join(OUTPUT_DIR, `persistence-${ts}-error.png`),
      fullPage: true
    });
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
