#!/usr/bin/env node
/**
 * Test ncolor swatches - verify 5 per row and color resampling
 */

const { chromium } = require('playwright');
const path = require('path');

const DEFAULT_URL = 'http://127.0.0.1:8000/';
const OUTPUT_DIR = path.resolve(__dirname);

async function runTest() {
  console.log('='.repeat(60));
  console.log('NCOLOR SWATCHES TEST');
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

    // Check initial swatches (should be 4)
    console.log('\n[1] Initial state (4 swatches):');
    let swatchCount = await page.evaluate(() => {
      return document.querySelectorAll('#ncolorSwatches .ncolor-swatch').length;
    });
    console.log(`  Swatch count: ${swatchCount}`);
    await page.screenshot({ path: path.join(OUTPUT_DIR, 'ncolor-4-swatches.png'), fullPage: true });

    // Add one color (5 swatches)
    console.log('\n[2] Adding 1 color (5 swatches):');
    await page.click('#ncolorAddColor');
    await page.waitForTimeout(500);
    swatchCount = await page.evaluate(() => {
      return document.querySelectorAll('#ncolorSwatches .ncolor-swatch').length;
    });
    console.log(`  Swatch count: ${swatchCount}`);
    await page.screenshot({ path: path.join(OUTPUT_DIR, 'ncolor-5-swatches.png'), fullPage: true });

    // Add another color (6 swatches - should wrap)
    console.log('\n[3] Adding 1 more color (6 swatches):');
    await page.click('#ncolorAddColor');
    await page.waitForTimeout(500);
    swatchCount = await page.evaluate(() => {
      return document.querySelectorAll('#ncolorSwatches .ncolor-swatch').length;
    });
    console.log(`  Swatch count: ${swatchCount}`);
    await page.screenshot({ path: path.join(OUTPUT_DIR, 'ncolor-6-swatches.png'), fullPage: true });

    // Add more to test button position
    console.log('\n[4] Adding 2 more colors (8 swatches):');
    await page.click('#ncolorAddColor');
    await page.click('#ncolorAddColor');
    await page.waitForTimeout(500);
    swatchCount = await page.evaluate(() => {
      return document.querySelectorAll('#ncolorSwatches .ncolor-swatch').length;
    });
    console.log(`  Swatch count: ${swatchCount}`);
    await page.screenshot({ path: path.join(OUTPUT_DIR, 'ncolor-8-swatches.png'), fullPage: true });

    console.log('\n  Screenshots saved!');

  } catch (err) {
    console.error('\n[ERROR]', err.message);
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
