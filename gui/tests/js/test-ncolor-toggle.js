#!/usr/bin/env node
/**
 * Test N-color toggle functionality.
 * Verifies that toggling from N-color to instance mode produces correct labels.
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

  console.log('[test] Starting N-color toggle test...');

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1600, height: 1000 },
  });
  const page = await context.newPage();

  page.on('console', (msg) => {
    if (msg.type() === 'log' || msg.type() === 'warn' || msg.type() === 'error') {
      console.log(`[page][${msg.type()}] ${msg.text()}`);
    }
  });

  try {
    await page.goto(DEFAULT_URL, { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForFunction(() => {
      return document.getElementById('canvas') !== null &&
             (window.OmniPainting || window.__OMNI_CONFIG__);
    }, { timeout: 30000 });

    console.log('[test] Page loaded, triggering segmentation first...');

    // Trigger segmentation
    const segmentButton = await page.$('#segmentButton');
    if (segmentButton) {
      console.log('[test] Clicking Segment button...');
      await page.evaluate(() => {
        const btn = document.getElementById('segmentButton');
        if (btn) btn.click();
      });

      // Wait for segmentation to complete (watch for status change or timeout)
      console.log('[test] Waiting for segmentation to complete...');
      await page.waitForFunction(() => {
        const status = document.getElementById('segmentStatus');
        return status && (status.textContent.includes('complete') || status.textContent.includes('failed'));
      }, { timeout: 120000 });

      console.log('[test] Segmentation completed, waiting for UI update...');
      await page.waitForTimeout(2000);
    }

    // Check state after segmentation
    const initialState = await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      const uniqueLabels = maskValues ? new Set(Array.from(maskValues).filter(v => v > 0)) : new Set();
      return {
        nColorActive: window.nColorActive,
        hasNColorInstanceMask: Boolean(window.nColorInstanceMask && window.nColorInstanceMask.length > 0),
        maskValuesLength: maskValues ? maskValues.length : 0,
        uniqueLabelCount: uniqueLabels.size,
        sampleLabels: Array.from(uniqueLabels).slice(0, 10),
      };
    });

    console.log('[test] State after segmentation:', JSON.stringify(initialState, null, 2));

    // Take initial screenshot
    await page.screenshot({
      path: path.join(OUTPUT_DIR, `ncolor-test-${ts}-01-initial.png`),
      fullPage: true
    });

    // Get N-color toggle element
    const toggleExists = await page.evaluate(() => {
      return Boolean(document.getElementById('autoNColorToggle'));
    });

    if (!toggleExists) {
      console.log('[test] N-color toggle not found in DOM');
      await browser.close();
      return;
    }

    // Check current toggle state
    const toggleState = await page.evaluate(() => {
      const toggle = document.getElementById('autoNColorToggle');
      return toggle ? toggle.checked : null;
    });

    console.log(`[test] N-color toggle checked: ${toggleState}`);

    // If N-color is ON, toggle it OFF
    if (toggleState === true) {
      console.log('[test] Toggling N-color OFF...');

      // Use evaluate to click directly (bypasses pointer interception by styled toggle)
      await page.evaluate(() => {
        const toggle = document.getElementById('autoNColorToggle');
        if (toggle) toggle.click();
      });
      await page.waitForTimeout(2000); // Wait for relabelFromAffinity to complete

      // Check state after toggle
      const afterToggleState = await page.evaluate(() => {
        const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
        const uniqueLabels = maskValues ? new Set(Array.from(maskValues).filter(v => v > 0)) : new Set();
        return {
          nColorActive: window.nColorActive,
          hasNColorInstanceMask: Boolean(window.nColorInstanceMask && window.nColorInstanceMask.length > 0),
          uniqueLabelCount: uniqueLabels.size,
          sampleLabels: Array.from(uniqueLabels).slice(0, 20),
          maxLabel: Math.max(0, ...Array.from(uniqueLabels)),
        };
      });

      console.log('[test] After toggle OFF:', JSON.stringify(afterToggleState, null, 2));

      // Take screenshot after toggle
      await page.screenshot({
        path: path.join(OUTPUT_DIR, `ncolor-test-${ts}-02-after-toggle-off.png`),
        fullPage: true
      });

      // Check if we have more than 4 unique labels (indicating instance mode)
      if (afterToggleState.uniqueLabelCount <= 4) {
        console.log('[test] WARNING: Only 4 or fewer unique labels after toggle OFF!');
        console.log('[test] This indicates the N-color to instance conversion may have failed.');
      } else {
        console.log(`[test] SUCCESS: Found ${afterToggleState.uniqueLabelCount} unique labels after toggle OFF`);
      }

      // Now toggle back ON
      console.log('[test] Toggling N-color back ON...');
      await page.evaluate(() => {
        const toggle = document.getElementById('autoNColorToggle');
        if (toggle) toggle.click();
      });
      await page.waitForTimeout(2000);

      const afterToggleOnState = await page.evaluate(() => {
        const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
        const uniqueLabels = maskValues ? new Set(Array.from(maskValues).filter(v => v > 0)) : new Set();
        return {
          nColorActive: window.nColorActive,
          uniqueLabelCount: uniqueLabels.size,
          sampleLabels: Array.from(uniqueLabels).slice(0, 10),
          maxLabel: Math.max(0, ...Array.from(uniqueLabels)),
        };
      });

      console.log('[test] After toggle ON:', JSON.stringify(afterToggleOnState, null, 2));

      // Take final screenshot
      await page.screenshot({
        path: path.join(OUTPUT_DIR, `ncolor-test-${ts}-03-after-toggle-on.png`),
        fullPage: true
      });

      // In N-color mode, we should have <= ~6 unique labels (the N-color groups)
      if (afterToggleOnState.uniqueLabelCount > 10) {
        console.log('[test] WARNING: More than 10 unique labels in N-color mode!');
        console.log('[test] This indicates the instance to N-color conversion may have failed.');
      } else {
        console.log(`[test] SUCCESS: Found ${afterToggleOnState.uniqueLabelCount} unique labels in N-color mode (expected ~4-6)`);
      }
    }

    console.log('[test] Test completed!');
    console.log(`[test] Screenshots saved to: ${OUTPUT_DIR}`);

  } catch (err) {
    console.error('[test] Error:', err.message);
    await page.screenshot({
      path: path.join(OUTPUT_DIR, `ncolor-test-${ts}-error.png`),
      fullPage: true
    });
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
