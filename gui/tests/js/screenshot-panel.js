#!/usr/bin/env node
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

const OUTPUT_DIR = path.resolve(__dirname, '../../tmp/playwright');

async function run() {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1600, height: 1000 } });
  const page = await context.newPage();

  await page.goto('http://127.0.0.1:8001/', { waitUntil: 'networkidle', timeout: 60000 });
  await page.waitForTimeout(2000);

  // Take full screenshot
  await page.screenshot({
    path: path.join(OUTPUT_DIR, 'panel-check.png'),
    fullPage: true
  });

  console.log('Screenshot saved to', path.join(OUTPUT_DIR, 'panel-check.png'));

  await browser.close();
}

run().catch(console.error);
