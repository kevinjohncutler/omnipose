#!/usr/bin/env node
const { chromium } = require('playwright');
const path = require('path');

const DEFAULT_URL = 'http://127.0.0.1:8000/';
const OUTPUT_DIR = path.resolve(__dirname);

async function runTest() {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1600, height: 1000 } });
  const page = await context.newPage();

  try {
    await page.goto(DEFAULT_URL, { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForFunction(() => document.getElementById('canvas') !== null, { timeout: 30000 });
    await page.waitForTimeout(2000);

    // Select a label to show the value input styling
    await page.evaluate(() => {
      if (window.setCurrentLabel) window.setCurrentLabel(5);
    });
    await page.waitForTimeout(500);

    await page.screenshot({ path: path.join(OUTPUT_DIR, 'ui-final.png'), fullPage: true });
    console.log('Screenshot saved: ui-final.png');

  } catch (err) {
    console.error('ERROR:', err.message);
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
