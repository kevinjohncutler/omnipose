#!/usr/bin/env node
/**
 * Test cmap with ncolor - verify viridis works with ncolor swatches
 */

const { chromium } = require('playwright');
const path = require('path');

const DEFAULT_URL = 'http://127.0.0.1:8000/';
const OUTPUT_DIR = path.resolve(__dirname);

async function runTest() {
  console.log('='.repeat(60));
  console.log('CMAP + NCOLOR TEST');
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

    // Take screenshot with default sinebow
    console.log('\n[1] Default state (Sinebow + Ncolor ON):');
    await page.screenshot({ path: path.join(OUTPUT_DIR, 'cmap-sinebow-ncolor.png'), fullPage: true });

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
    await page.screenshot({ path: path.join(OUTPUT_DIR, 'cmap-viridis-ncolor.png'), fullPage: true });

    // Check swatch colors
    const swatchColors = await page.evaluate(() => {
      const swatches = document.querySelectorAll('#ncolorSwatches .ncolor-swatch');
      return Array.from(swatches).map(s => {
        const style = s.style.getPropertyValue('--swatch-color') || getComputedStyle(s).getPropertyValue('--swatch-color');
        return style;
      });
    });
    console.log('  Swatch colors:', swatchColors);

    // Toggle ncolor OFF
    console.log('\n[3] Toggle Ncolor OFF (show Shuffle/Seed):');
    await page.evaluate(() => {
      const toggle = document.getElementById('autoNColorToggle');
      if (toggle && toggle.checked) toggle.click();
    });
    await page.waitForTimeout(500);
    await page.screenshot({ path: path.join(OUTPUT_DIR, 'cmap-viridis-ncolor-off.png'), fullPage: true });

    console.log('\n  Screenshots saved!');

  } catch (err) {
    console.error('\n[ERROR]', err.message);
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
