#!/usr/bin/env node
const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');

const DEFAULT_URL = 'http://127.0.0.1:8765/tests/headless_index.html';
const SCREENSHOT_PATH = path.resolve(__dirname, '../../tmp/gui_fill_capture.png');
const REPORT_PATH = path.resolve(__dirname, '../../tmp/gui_fill_capture.json');
const ENTRIES_PATH = path.resolve(__dirname, '../../tmp/gui_fill_entries.json');

const ensureDirs = () => {
  const outDir = path.resolve(__dirname, '../../tmp');
  if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
  }
  return outDir;
};

const buildConfig = () => ({
  width: 392,
  height: 384,
  imageName: 'headless-grid.png',
  imagePath: '/gui/logo.png',
  imageDataUrl: '/gui/logo.png',
  colorTable: [],
  brushRadius: 6,
  sessionId: 'headless-grid-capture',
  pointerOptions: {},
  enableMaskPipelineV2: true,
});

(async () => {
  ensureDirs();
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
    defaultViewport: {
      width: 1600,
      height: 1200,
      deviceScaleFactor: 1,
    },
  });
  const page = await browser.newPage();
  if (typeof page.setCacheEnabled === 'function') {
    await page.setCacheEnabled(false);
  }
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
  await page.evaluateOnNewDocument((config) => {
    window.__OMNI_CONFIG__ = config;
    window.__OMNI_FORCE_GRID_MASK__ = true;
    window.__OMNI_SKIP_GRID_REPAINT__ = false;
    window.__OMNI_MASK_PIPELINE_V2__ = true;
  }, buildConfig());

  await page.goto(DEFAULT_URL, { waitUntil: 'networkidle2', timeout: 120000 });
  await page.waitForFunction(
    () => window.__OMNI_DEBUG__
      && typeof window.__OMNI_DEBUG__.walkGridFill === 'function'
      && typeof window.OmniPainting === 'object',
    { timeout: 60000 },
  );

  await page.screenshot({ path: SCREENSHOT_PATH.replace('.png', '_before.png'), fullPage: true });
  const result = await page.evaluate(async () => {
    const walker = window.__OMNI_DEBUG__.walkGridFill;
    const entries = await walker({ forceGrid: true, delay: 35 });
    const report = window.__OMNI_DEBUG__.collectFillDiagnostics();
    const coords = entries.map((entry) => ({
      tile: entry.tile,
      sample: entry.sample,
      maskValueAfter: window.OmniPainting.__debugGetState().ctx.maskValues[entry.sample.idx] | 0,
    }));
    const state = window.OmniPainting.__debugGetState();
    const ctx = state && state.ctx;
    let maskPreview = null;
    if (ctx && ctx.maskValues && ctx.getImageDimensions) {
      const dims = ctx.getImageDimensions();
      const maskCanvas = document.createElement('canvas');
      maskCanvas.width = dims.width;
      maskCanvas.height = dims.height;
      const maskCtx = maskCanvas.getContext('2d');
      const imageData = maskCtx.createImageData(dims.width, dims.height);
      for (let i = 0; i < ctx.maskValues.length; i += 1) {
        const label = ctx.maskValues[i] | 0;
        const base = i * 4;
        if (label === 0) {
          imageData.data[base] = 0;
          imageData.data[base + 1] = 0;
          imageData.data[base + 2] = 0;
          imageData.data[base + 3] = 255;
        } else {
          const color = window.__OMNI_DEBUG__ && typeof window.__OMNI_DEBUG__.hashColor === 'function'
            ? window.__OMNI_DEBUG__.hashColor(label)
            : [41, 89, 252];
          imageData.data[base] = color[0];
          imageData.data[base + 1] = color[1];
          imageData.data[base + 2] = color[2];
          imageData.data[base + 3] = 255;
        }
      }
      maskCtx.putImageData(imageData, 0, 0);
      maskPreview = maskCanvas.toDataURL('image/png');
    }
    return { entries, report, coords, maskPreview };
  });

  await page.screenshot({ path: SCREENSHOT_PATH, fullPage: true });
  fs.writeFileSync(REPORT_PATH, JSON.stringify(result.report, null, 2));
  fs.writeFileSync(ENTRIES_PATH, JSON.stringify(result.entries, null, 2));
  fs.writeFileSync(REPORT_PATH.replace('.json', '_coords.json'), JSON.stringify(result.coords, null, 2));
  if (result.maskPreview) {
    const base64 = result.maskPreview.split(',')[1];
    fs.writeFileSync(REPORT_PATH.replace('.json', '_mask.png'), Buffer.from(base64, 'base64'));
  }
  console.log('Saved headless capture to:', SCREENSHOT_PATH);
  console.log('Diagnostic report:', REPORT_PATH);
  console.log('Entries:', ENTRIES_PATH);

  await browser.close();
})();
