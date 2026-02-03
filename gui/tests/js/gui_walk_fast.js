#!/usr/bin/env node
const puppeteer = require('puppeteer');

async function main() {
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
    defaultViewport: { width: 1600, height: 1200, deviceScaleFactor: 1 },
  });
  try {
    const page = await browser.newPage();
    page.on('console', (msg) => {
      try {
        const location = msg.location()
          ? `${msg.location().url || 'unknown'}:${msg.location().lineNumber || 0}`
          : 'unknown';
        console.log(`[page-console][${msg.type()}][${location}]`, msg.text());
      } catch (err) {
        console.log('[page-console]', msg.type(), msg.text());
      }
    });
    page.on('pageerror', (err) => {
      console.error('[page-error]', err && err.stack ? err.stack : err);
    });
    await page.evaluateOnNewDocument(() => {
      window.__OMNI_CONFIG__ = {
        width: 392,
        height: 384,
        imageName: 'headless-grid.png',
        imagePath: '/gui/logo.png',
        imageDataUrl: '/gui/logo.png',
        colorTable: [],
        brushRadius: 6,
        sessionId: 'headless-grid-walk-fast',
        pointerOptions: {},
        enableMaskPipelineV2: true,
      };
      window.__OMNI_FORCE_GRID_MASK__ = true;
      window.__OMNI_MASK_PIPELINE_V2__ = true;
    });
    await page.goto('http://127.0.0.1:8765/tests/headless_index.html', {
      waitUntil: 'networkidle2',
      timeout: 120000,
    });
    await page.waitForFunction(
      () => window.__OMNI_DEBUG__ && typeof window.__OMNI_DEBUG__.walkGridFill === 'function',
      { timeout: 60000 },
    );
    const result = await page.evaluate(async () => {
      const walker = window.__OMNI_DEBUG__.walkGridFill;
      const entries = await walker({ fast: true, visual: false, forceGrid: true });
      return {
        entryCount: entries.length,
        first: entries[0] || null,
        last: entries[entries.length - 1] || null,
        counters: window.__OMNI_DEBUG__.getCounters ? window.__OMNI_DEBUG__.getCounters() : null,
      };
    });
    console.log(JSON.stringify(result, null, 2));
  } finally {
    await browser.close();
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
