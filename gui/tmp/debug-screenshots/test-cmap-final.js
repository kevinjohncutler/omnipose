#!/usr/bin/env node
const { chromium } = require('playwright');
const path = require('path');

const DEFAULT_URL = 'http://127.0.0.1:8000/';
const OUTPUT_DIR = '/Volumes/DataDrive/omnipose/gui/tmp/grid';

async function runTest() {
  console.log('CMAP PANEL FINAL TEST');
  console.log('='.repeat(40));

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });

  try {
    await page.goto(DEFAULT_URL, { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForTimeout(2000);

    // Get current colormap
    const cmap = await page.$eval('#cmapSelect', el => el.value);
    console.log(`Current colormap: ${cmap}`);

    // Screenshot 1: Sinebow with N-color ON
    await page.screenshot({ path: path.join(OUTPUT_DIR, 'cmap-1-sinebow-ncolor-on.png'), fullPage: true });
    console.log('Screenshot 1: cmap-1-sinebow-ncolor-on.png');

    // Screenshot 2: Change to viridis
    await page.selectOption('#cmapSelect', 'viridis');
    await page.waitForTimeout(500);
    await page.screenshot({ path: path.join(OUTPUT_DIR, 'cmap-2-viridis-ncolor-on.png'), fullPage: true });
    console.log('Screenshot 2: cmap-2-viridis-ncolor-on.png');

    // Screenshot 3: Toggle N-color OFF via JavaScript
    await page.evaluate(() => {
      const toggle = document.getElementById('autoNColorToggle');
      if (toggle && toggle.checked) {
        toggle.checked = false;
        toggle.dispatchEvent(new Event('change', { bubbles: true }));
      }
    });
    await page.waitForTimeout(1000);
    await page.screenshot({ path: path.join(OUTPUT_DIR, 'cmap-3-viridis-ncolor-off.png'), fullPage: true });
    console.log('Screenshot 3: cmap-3-viridis-ncolor-off.png');

    // Screenshot 4: Back to sinebow with N-color OFF
    await page.selectOption('#cmapSelect', 'sinebow');
    await page.waitForTimeout(500);
    await page.screenshot({ path: path.join(OUTPUT_DIR, 'cmap-4-sinebow-ncolor-off.png'), fullPage: true });
    console.log('Screenshot 4: cmap-4-sinebow-ncolor-off.png');

    // Check slider gradient style
    const sliderBg = await page.$eval('#cmapHueOffset', el => el.style.background);
    console.log(`\nSlider gradient: ${sliderBg ? sliderBg.substring(0, 60) + '...' : 'not set'}`);

    console.log('\nAll screenshots saved to tmp/grid/');

  } catch (err) {
    console.error('Error:', err.message);
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
