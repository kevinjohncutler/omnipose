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

  page.on('console', msg => {
    if (msg.type() === 'error') console.log(`  [error] ${msg.text()}`);
  });

  try {
    await page.goto(DEFAULT_URL, { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForTimeout(2000);

    // Check if hue offset slider exists
    const hueSlider = await page.$('#ncolorHueOffset');
    console.log(`Hue offset slider exists: ${hueSlider ? 'YES' : 'NO'}`);

    // Check if colormap previews exist
    const ncolorPreview = await page.$('#ncolorColormapPreview');
    const instancePreview = await page.$('#instanceColormapPreview');
    console.log(`N-color preview exists: ${ncolorPreview ? 'YES' : 'NO'}`);
    console.log(`Instance preview exists: ${instancePreview ? 'YES' : 'NO'}`);

    // Get current N-color palette colors
    const colors = await page.evaluate(() => {
      if (window.nColorPaletteColors && window.nColorPaletteColors.length) {
        return window.nColorPaletteColors.map(c => `rgb(${c.join(',')})`);
      }
      return [];
    });
    console.log(`\nN-color palette (${colors.length} colors):`);
    colors.forEach((c, i) => console.log(`  ${i + 1}: ${c}`));

    // Test changing hue offset
    if (hueSlider) {
      console.log('\nChanging hue offset to 50%...');
      await page.evaluate(() => {
        const slider = document.getElementById('ncolorHueOffset');
        if (slider) {
          slider.value = 50;
          slider.dispatchEvent(new Event('input', { bubbles: true }));
        }
      });
      await page.waitForTimeout(500);

      const newColors = await page.evaluate(() => {
        if (window.nColorPaletteColors && window.nColorPaletteColors.length) {
          return window.nColorPaletteColors.map(c => `rgb(${c.join(',')})`);
        }
        return [];
      });
      console.log(`N-color palette after hue change:`);
      newColors.forEach((c, i) => console.log(`  ${i + 1}: ${c}`));
    }

    // Take screenshot
    await page.screenshot({ path: path.join(OUTPUT_DIR, 'ncolor-hue-test.png'), fullPage: true });
    console.log('\nScreenshot saved: ncolor-hue-test.png');

  } catch (err) {
    console.error('Error:', err.message);
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
