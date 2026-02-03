#!/usr/bin/env node
/**
 * Debug state persistence - check localStorage keys and content.
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

async function runTest() {
  ensureDir(OUTPUT_DIR);

  console.log('[test] Starting state debug test...');

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1600, height: 1000 },
  });
  const page = await context.newPage();

  page.on('console', (msg) => {
    console.log(`[page][${msg.type()}] ${msg.text()}`);
  });

  try {
    // Initial page load
    await page.goto(DEFAULT_URL, { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForFunction(() => {
      return document.getElementById('canvas') !== null;
    }, { timeout: 30000 });

    // Check what localStorage key is being used
    const configInfo = await page.evaluate(() => {
      return {
        imagePath: window.__OMNI_CONFIG__?.imagePath || 'not found',
        imageName: window.__OMNI_CONFIG__?.imageName || 'not found',
        localStorageKeys: Object.keys(localStorage).filter(k => k.startsWith('OMNI_')),
      };
    });
    console.log('[test] Config info:', JSON.stringify(configInfo, null, 2));

    // Check current mask state
    const maskState = await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      const uniqueLabels = maskValues ? new Set(Array.from(maskValues).filter(v => v > 0)) : new Set();
      return {
        hasMask: Boolean(maskValues),
        uniqueLabelCount: uniqueLabels.size,
        hasNonZero: uniqueLabels.size > 0,
      };
    });
    console.log('[test] Initial mask state:', JSON.stringify(maskState, null, 2));

    // If no mask, run segmentation
    if (!maskState.hasNonZero) {
      console.log('[test] No mask found, running segmentation...');
      await page.evaluate(() => {
        const btn = document.getElementById('segmentButton');
        if (btn) btn.click();
      });

      await page.waitForFunction(() => {
        const status = document.getElementById('segmentStatus');
        return status && (status.textContent.includes('complete') || status.textContent.includes('failed'));
      }, { timeout: 120000 });
      await page.waitForTimeout(2000);

      const maskStateAfter = await page.evaluate(() => {
        const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
        const uniqueLabels = maskValues ? new Set(Array.from(maskValues).filter(v => v > 0)) : new Set();
        return {
          hasMask: Boolean(maskValues),
          uniqueLabelCount: uniqueLabels.size,
          hasNonZero: uniqueLabels.size > 0,
        };
      });
      console.log('[test] Mask state after segmentation:', JSON.stringify(maskStateAfter, null, 2));
    }

    // Wait for state to be saved
    console.log('[test] Waiting for state to be saved...');
    await page.waitForTimeout(3000);

    // Check localStorage content in detail
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
              timestamp: parsed.timestamp,
            };
          } catch (e) {
            result[key] = { error: e.message };
          }
        }
      }
      return result;
    });
    console.log('[test] LocalStorage info:', JSON.stringify(storageInfo, null, 2));

    // Test color picker consistency
    console.log('[test] Testing color consistency...');

    // Toggle N-color OFF first
    await page.evaluate(() => {
      const toggle = document.getElementById('autoNColorToggle');
      if (toggle && toggle.checked) toggle.click();
    });
    await page.waitForTimeout(2000);

    const colorConsistency = await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      if (!maskValues) return { error: 'no mask' };

      // Find first few unique labels
      const labels = [];
      for (let i = 0; i < maskValues.length && labels.length < 10; i++) {
        const label = maskValues[i];
        if (label > 0 && !labels.includes(label)) {
          labels.push(label);
        }
      }

      // Get colors from getColormapColor for these labels
      const colors = {};
      if (typeof window.getColormapColor === 'function') {
        for (const label of labels) {
          colors[label] = window.getColormapColor(label);
        }
      } else {
        return { error: 'getColormapColor not found' };
      }

      // Check if colors are unique
      const colorStrings = Object.values(colors).map(c => c ? c.join(',') : 'null');
      const uniqueColors = new Set(colorStrings);
      const allUnique = uniqueColors.size === colorStrings.length;

      // Try to access through __OMNI_CONFIG__
      const config = window.__OMNI_CONFIG__ || {};

      return {
        labels,
        colors,
        configColorTableLength: config.colorTable ? config.colorTable.length : 'not in config',
        allColorsUnique: allUnique,
        duplicateColors: allUnique ? [] : colorStrings.filter((c, i) => colorStrings.indexOf(c) !== i),
        // Check if adjacent cells have same color
        sampleColorPairs: labels.slice(0, 5).map((l, i) => ({
          label: l,
          color: colors[l] ? colors[l].join(',') : 'null'
        })),
      };
    });
    console.log('[test] Color consistency check:', JSON.stringify(colorConsistency, null, 2));

    console.log('[test] Test completed!');

  } catch (err) {
    console.error('[test] Error:', err.message);
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
