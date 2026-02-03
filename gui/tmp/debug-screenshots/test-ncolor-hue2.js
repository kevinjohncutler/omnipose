#!/usr/bin/env node
const { chromium } = require('playwright');
const path = require('path');

const DEFAULT_URL = 'http://127.0.0.1:8000/';
const OUTPUT_DIR = '/Volumes/DataDrive/omnipose/gui/tmp/grid';

async function runTest() {
  console.log('N-COLOR HUE OFFSET TEST');
  console.log('='.repeat(40));

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });

  try {
    await page.goto(DEFAULT_URL, { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForTimeout(2000);

    // Check the rendered swatches
    const swatches = await page.$$eval('#ncolorSwatches .ncolor-swatch', els => {
      return els.map(el => el.style.getPropertyValue('--swatch-color'));
    });
    console.log(`N-color swatches (${swatches.length}):`);
    swatches.forEach((c, i) => console.log(`  ${i + 1}: ${c}`));

    // Get the hue offset value
    const hueOffset = await page.$eval('#ncolorHueOffset', el => el.value);
    console.log(`\nHue offset slider value: ${hueOffset}`);

    // Get the colormap preview background
    const previewBg = await page.$eval('#ncolorColormapPreview', el => el.style.background);
    console.log(`N-color preview background: ${previewBg.substring(0, 80)}...`);

    // Take screenshot with hue=0
    await page.screenshot({ path: path.join(OUTPUT_DIR, 'ncolor-hue-0.png'), fullPage: true });
    console.log('\nScreenshot saved: ncolor-hue-0.png');

    // Change hue offset to 50%
    console.log('\nChanging hue offset to 50%...');
    await page.fill('#ncolorHueOffset', '50');
    await page.dispatchEvent('#ncolorHueOffset', 'input');
    await page.waitForTimeout(500);

    const newSwatches = await page.$$eval('#ncolorSwatches .ncolor-swatch', els => {
      return els.map(el => el.style.getPropertyValue('--swatch-color'));
    });
    console.log(`N-color swatches after change (${newSwatches.length}):`);
    newSwatches.forEach((c, i) => console.log(`  ${i + 1}: ${c}`));

    // Take screenshot with hue=50
    await page.screenshot({ path: path.join(OUTPUT_DIR, 'ncolor-hue-50.png'), fullPage: true });
    console.log('\nScreenshot saved: ncolor-hue-50.png');

  } catch (err) {
    console.error('Error:', err.message);
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
