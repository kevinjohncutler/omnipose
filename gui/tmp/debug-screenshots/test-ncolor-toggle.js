#!/usr/bin/env node
/**
 * Debug N-color toggle - verify nColorInstanceMask is saved correctly
 */

const { chromium } = require('playwright');
const path = require('path');

const DEFAULT_URL = 'http://127.0.0.1:8000/';

async function runTest() {
  console.log('='.repeat(60));
  console.log('N-COLOR TOGGLE DEBUG TEST');
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

    // Check initial state
    const initial = await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      return {
        nColorActive: window.nColorActive,
        nColorInstanceMask: window.nColorInstanceMask ? window.nColorInstanceMask.length : null,
        maskUniqueLabels: maskValues ? new Set(Array.from(maskValues).filter(v => v > 0)).size : 0,
      };
    });
    console.log('\n[1] Initial state:');
    console.log(`  nColorActive: ${initial.nColorActive}`);
    console.log(`  nColorInstanceMask length: ${initial.nColorInstanceMask}`);
    console.log(`  mask unique labels: ${initial.maskUniqueLabels}`);

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
    await page.waitForTimeout(3000);

    // Check state after segmentation
    const afterSeg = await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      const uniqueMask = maskValues ? new Set(Array.from(maskValues).filter(v => v > 0)) : new Set();
      const uniqueInstance = window.nColorInstanceMask
        ? new Set(Array.from(window.nColorInstanceMask).filter(v => v > 0))
        : new Set();
      return {
        nColorActive: window.nColorActive,
        nColorInstanceMask: window.nColorInstanceMask ? window.nColorInstanceMask.length : null,
        maskUniqueLabels: uniqueMask.size,
        maskSample: Array.from(uniqueMask).slice(0, 10),
        instanceUniqueLabels: uniqueInstance.size,
        instanceSample: Array.from(uniqueInstance).slice(0, 10),
      };
    });
    console.log('\n[3] After segmentation (N-color should be ON):');
    console.log(`  nColorActive: ${afterSeg.nColorActive}`);
    console.log(`  maskValues unique labels: ${afterSeg.maskUniqueLabels} (sample: ${afterSeg.maskSample.join(', ')})`);
    console.log(`  nColorInstanceMask unique labels: ${afterSeg.instanceUniqueLabels} (sample: ${afterSeg.instanceSample.join(', ')})`);

    // Toggle N-color OFF
    console.log('\n[4] Toggling N-color OFF...');
    await page.evaluate(() => {
      const toggle = document.getElementById('autoNColorToggle');
      if (toggle && toggle.checked) toggle.click();
    });
    await page.waitForTimeout(3000);

    // Check state after toggle OFF
    const afterOff = await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      const uniqueMask = maskValues ? new Set(Array.from(maskValues).filter(v => v > 0)) : new Set();
      return {
        nColorActive: window.nColorActive,
        nColorInstanceMask: window.nColorInstanceMask,
        maskUniqueLabels: uniqueMask.size,
        maskSample: Array.from(uniqueMask).slice(0, 10),
      };
    });
    console.log('\n[5] After N-color OFF:');
    console.log(`  nColorActive: ${afterOff.nColorActive}`);
    console.log(`  nColorInstanceMask: ${afterOff.nColorInstanceMask}`);
    console.log(`  maskValues unique labels: ${afterOff.maskUniqueLabels} (sample: ${afterOff.maskSample.join(', ')})`);

    // Toggle N-color back ON
    console.log('\n[6] Toggling N-color back ON...');
    await page.evaluate(() => {
      const toggle = document.getElementById('autoNColorToggle');
      if (toggle && !toggle.checked) toggle.click();
    });
    await page.waitForTimeout(3000);

    // Check state after toggle ON
    const afterOn = await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      const uniqueMask = maskValues ? new Set(Array.from(maskValues).filter(v => v > 0)) : new Set();
      const uniqueInstance = window.nColorInstanceMask
        ? new Set(Array.from(window.nColorInstanceMask).filter(v => v > 0))
        : new Set();
      return {
        nColorActive: window.nColorActive,
        nColorInstanceMask: window.nColorInstanceMask ? window.nColorInstanceMask.length : null,
        maskUniqueLabels: uniqueMask.size,
        maskSample: Array.from(uniqueMask).slice(0, 10),
        instanceUniqueLabels: uniqueInstance.size,
        instanceSample: Array.from(uniqueInstance).slice(0, 10),
      };
    });
    console.log('\n[7] After N-color back ON:');
    console.log(`  nColorActive: ${afterOn.nColorActive}`);
    console.log(`  maskValues unique labels: ${afterOn.maskUniqueLabels} (sample: ${afterOn.maskSample.join(', ')})`);
    console.log(`  nColorInstanceMask unique labels: ${afterOn.instanceUniqueLabels} (sample: ${afterOn.instanceSample.join(', ')})`);

    console.log('\n' + '='.repeat(60));
    console.log('SUMMARY:');
    console.log(`  After seg instance labels: ${afterSeg.instanceUniqueLabels}`);
    console.log(`  After OFF mask labels: ${afterOff.maskUniqueLabels}`);
    console.log(`  Labels preserved: ${afterSeg.instanceUniqueLabels === afterOff.maskUniqueLabels ? 'YES' : 'NO'}`);

  } catch (err) {
    console.error('\n[ERROR]', err.message);
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
