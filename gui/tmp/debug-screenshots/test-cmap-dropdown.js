#!/usr/bin/env node
/**
 * Test cmap dropdown - verify viridis appears
 */

const { chromium } = require('playwright');
const path = require('path');

const DEFAULT_URL = 'http://127.0.0.1:8000/';
const OUTPUT_DIR = path.resolve(__dirname);

async function runTest() {
  console.log('='.repeat(60));
  console.log('CMAP DROPDOWN TEST');
  console.log('='.repeat(60));

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1600, height: 1000 },
  });
  const page = await context.newPage();

  page.on('console', msg => {
    if (msg.type() === 'log' || msg.type() === 'warn' || msg.type() === 'error') {
      console.log(`  [browser ${msg.type()}] ${msg.text()}`);
    }
  });

  try {
    await page.goto(DEFAULT_URL, { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForFunction(() => document.getElementById('canvas') !== null, { timeout: 30000 });
    await page.waitForTimeout(2000);

    // Check the cmap dropdown
    const dropdownInfo = await page.evaluate(() => {
      const select = document.getElementById('cmapSelect');
      if (!select) return { error: 'cmapSelect not found' };

      const options = Array.from(select.options).map(opt => ({
        value: opt.value,
        label: opt.textContent,
      }));

      return {
        optionCount: options.length,
        options: options,
        currentValue: select.value,
        hasViridis: options.some(o => o.value === 'viridis'),
      };
    });

    console.log('\n[1] Dropdown info:');
    console.log(`  Option count: ${dropdownInfo.optionCount}`);
    console.log(`  Current value: ${dropdownInfo.currentValue}`);
    console.log(`  Has viridis: ${dropdownInfo.hasViridis}`);
    console.log('\n  All options:');
    if (dropdownInfo.options) {
      dropdownInfo.options.forEach(opt => {
        console.log(`    - ${opt.value}: "${opt.label}"`);
      });
    }

    // Take screenshot of the left panel
    await page.screenshot({ path: path.join(OUTPUT_DIR, 'cmap-dropdown.png'), fullPage: true });
    console.log('\n  Screenshot saved: cmap-dropdown.png');

    // Try selecting viridis
    if (dropdownInfo.hasViridis) {
      console.log('\n[2] Selecting viridis...');
      await page.evaluate(() => {
        const select = document.getElementById('cmapSelect');
        if (select) {
          select.value = 'viridis';
          select.dispatchEvent(new Event('change'));
        }
      });
      await page.waitForTimeout(1000);
      await page.screenshot({ path: path.join(OUTPUT_DIR, 'cmap-viridis.png'), fullPage: true });
      console.log('  Screenshot saved: cmap-viridis.png');
    }

  } catch (err) {
    console.error('\n[ERROR]', err.message);
    console.error(err.stack);
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
