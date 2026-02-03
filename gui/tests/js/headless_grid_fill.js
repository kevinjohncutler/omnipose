#!/usr/bin/env node
/**
 * Launch the Omnipose web GUI inside a real Chromium instance, seed the debug grid,
 * and walk every grid component sequentially so we can detect the "nth fill" failure
 * under a production-like environment (WebGL, history stack, etc.).
 *
 * Usage:
 *   node tests/js/headless_grid_fill.js [url] [delay_ms]
 *
 * Defaults:
 *   url      -> http://127.0.0.1:8765/gui/web/index.html
 *   delay_ms -> 10
 */
const puppeteer = require('puppeteer');

const DEFAULT_URL = 'http://127.0.0.1:8765/tests/headless_index.html';
const DEFAULT_DELAY = 10;

function buildViewerConfig() {
  return {
    width: 392,
    height: 384,
    imageName: 'headless-grid.png',
    imagePath: '/gui/logo.png',
    imageDataUrl: '/gui/logo.png',
    colorTable: [],
    brushRadius: 6,
    sessionId: 'headless-grid',
    pointerOptions: {},
    enableMaskPipelineV2: true,
  };
}

async function main() {
  const targetUrl = process.argv[2] || DEFAULT_URL;
  const delay = Number.isFinite(Number(process.argv[3])) ? Number(process.argv[3]) : DEFAULT_DELAY;

  const browser = await puppeteer.launch({
    headless: 'new',
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
    ],
    defaultViewport: {
      width: 1400,
      height: 900,
      deviceScaleFactor: 1,
    },
  });

  try {
    const page = await browser.newPage();
    page.on('console', (msg) => {
      try {
        const text = msg.text();
        if (text && text.includes('debug-walk-grid')) {
          console.log(text);
        }
      } catch (_) {
        /* ignore */
      }
    });
    await page.evaluateOnNewDocument((config) => {
      window.__OMNI_CONFIG__ = config;
      window.__OMNI_FORCE_GRID_MASK__ = true;
      window.__OMNI_SKIP_GRID_REPAINT__ = false;
      window.__OMNI_MASK_PIPELINE_V2__ = Boolean(config.enableMaskPipelineV2);
    }, buildViewerConfig());

    await page.goto(targetUrl, { waitUntil: 'networkidle2', timeout: 60000 });
    await page.waitForFunction(
      () => window.__OMNI_DEBUG__ && typeof window.__OMNI_DEBUG__.walkGridFill === 'function',
      { timeout: 60000 },
    );

    // Give the async grid seeding a moment to finish.
    await new Promise((resolve) => setTimeout(resolve, 100));

    const result = await page.evaluate(async (options) => {
      const walker = window.__OMNI_DEBUG__.walkGridFill;
      if (typeof walker !== 'function') {
        throw new Error('walkGridFill missing from __OMNI_DEBUG__');
      }
      const entries = await walker(options);
      const failures = entries.filter((entry) => entry && entry.failures && entry.failures > 0);
      const lastFill =
        window.OmniPainting
        && typeof window.OmniPainting.__debugGetState === 'function'
          ? window.OmniPainting.__debugGetState().lastFillResult
          : null;
      return {
        entries,
        failures,
        lastFill,
      };
    }, { forceGrid: true, delay });

    const total = result.entries ? result.entries.length : 0;
    console.log(`grid-walk completed ${total} tiles`);
    if (result.failures && result.failures.length) {
      console.error('Detected fill failures:', JSON.stringify(result.failures, null, 2));
      process.exitCode = 1;
    } else {
      console.log('No failures detected.');
    }
    if (result.lastFill) {
      console.log('Last fill result:', JSON.stringify(result.lastFill, null, 2));
    }
  } finally {
    await browser.close();
  }
}

main().catch((err) => {
  console.error(err && err.stack ? err.stack : err);
  process.exit(1);
});
