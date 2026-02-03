#!/usr/bin/env node
const { chromium } = require('playwright');
const path = require('path');

const DEFAULT_URL = 'http://127.0.0.1:8000/';
const OUTPUT_DIR = '/Volumes/DataDrive/omnipose/gui/tmp/grid';

async function runTest() {
  console.log('CMAP PANEL TEST');
  console.log('='.repeat(40));

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });

  page.on('console', msg => {
    if (msg.type() === 'error') console.log(`  [error] ${msg.text()}`);
  });

  try {
    await page.goto(DEFAULT_URL, { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForTimeout(2000);

    // Check if new elements exist
    const cmapPanel = await page.$('#cmapPanel');
    const cmapSelect = await page.$('#cmapSelect');
    const cmapHueOffset = await page.$('#cmapHueOffset');
    const cmapPreview = await page.$('#cmapPreview');
    const ncolorSubsection = await page.$('#ncolorSubsection');

    console.log(`cmapPanel exists: ${cmapPanel ? 'YES' : 'NO'}`);
    console.log(`cmapSelect exists: ${cmapSelect ? 'YES' : 'NO'}`);
    console.log(`cmapHueOffset exists: ${cmapHueOffset ? 'YES' : 'NO'}`);
    console.log(`cmapPreview exists: ${cmapPreview ? 'YES' : 'NO'}`);
    console.log(`ncolorSubsection exists: ${ncolorSubsection ? 'YES' : 'NO'}`);

    // Get current colormap
    if (cmapSelect) {
      const cmap = await cmapSelect.evaluate(el => el.value);
      console.log(`\nCurrent colormap: ${cmap}`);
    }

    // Take screenshot with sinebow
    await page.screenshot({ path: path.join(OUTPUT_DIR, 'cmap-panel-sinebow.png'), fullPage: true });
    console.log('\nScreenshot saved: cmap-panel-sinebow.png');

    // Change to viridis
    console.log('\nChanging to viridis...');
    await page.selectOption('#cmapSelect', 'viridis');
    await page.waitForTimeout(500);

    // Check if offset slider is hidden
    const offsetVisible = await page.$eval('#cmapHueOffset', el => {
      const style = window.getComputedStyle(el);
      return style.pointerEvents !== 'none';
    }).catch(() => false);
    console.log(`Offset slider interactive: ${offsetVisible}`);

    await page.screenshot({ path: path.join(OUTPUT_DIR, 'cmap-panel-viridis.png'), fullPage: true });
    console.log('Screenshot saved: cmap-panel-viridis.png');

    // Toggle N-color off
    console.log('\nToggling N-color off...');
    const ncolorToggle = await page.$('#autoNColorToggle');
    if (ncolorToggle) {
      await ncolorToggle.click();
      await page.waitForTimeout(500);
    }

    await page.screenshot({ path: path.join(OUTPUT_DIR, 'cmap-panel-ncolor-off.png'), fullPage: true });
    console.log('Screenshot saved: cmap-panel-ncolor-off.png');

  } catch (err) {
    console.error('Error:', err.message);
    console.error(err.stack);
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
