#!/usr/bin/env node
/**
 * Test viridis in instance mode - verify colors spread across full range
 */

const { chromium } = require('playwright');
const path = require('path');

const DEFAULT_URL = 'http://127.0.0.1:8000/';
const OUTPUT_DIR = path.resolve(__dirname);

async function runTest() {
  console.log('='.repeat(60));
  console.log('VIRIDIS INSTANCE MODE TEST');
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

    // Run segmentation first to get real labels
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

    // Check currentMaxLabel
    const maxLabel = await page.evaluate(() => window.currentMaxLabel);
    console.log(`  currentMaxLabel: ${maxLabel}`);

    // Select viridis
    console.log('\n[2] Selecting Viridis...');
    await page.evaluate(() => {
      const select = document.getElementById('cmapSelect');
      if (select) {
        select.value = 'viridis';
        select.dispatchEvent(new Event('change'));
      }
    });
    await page.waitForTimeout(1000);

    // Toggle ncolor OFF (instance mode)
    console.log('\n[3] Toggle Ncolor OFF (instance mode)...');
    await page.evaluate(() => {
      const toggle = document.getElementById('autoNColorToggle');
      if (toggle && toggle.checked) toggle.click();
    });
    await page.waitForTimeout(1000);

    // Check color distribution
    const colorInfo = await page.evaluate(() => {
      const colors = [];
      for (let label = 1; label <= Math.min(10, window.currentMaxLabel); label++) {
        const c = window.getColormapColor ? window.getColormapColor(label) : null;
        colors.push({ label, color: c ? `rgb(${c.join(',')})` : 'null' });
      }
      return {
        maxLabel: window.currentMaxLabel,
        labelColormap: window.labelColormap,
        colors
      };
    });

    console.log(`  Colormap: ${colorInfo.labelColormap}`);
    console.log(`  Max label: ${colorInfo.maxLabel}`);
    console.log('  First 10 label colors:');
    colorInfo.colors.forEach(c => console.log(`    Label ${c.label}: ${c.color}`));

    await page.screenshot({ path: path.join(OUTPUT_DIR, 'viridis-instance-mode.png'), fullPage: true });
    console.log('\n  Screenshot saved: viridis-instance-mode.png');

  } catch (err) {
    console.error('\n[ERROR]', err.message);
    console.error(err.stack);
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
