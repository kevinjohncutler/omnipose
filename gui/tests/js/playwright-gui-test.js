#!/usr/bin/env node
/**
 * Playwright-based GUI test for Omnipose viewer.
 * Supports Chrome and WebKit (Safari) with trace recording for debugging.
 *
 * Usage (from gui/ directory):
 *   node tests/js/playwright-gui-test.js [url] [browser] [--headed]
 *
 * Examples:
 *   node tests/js/playwright-gui-test.js http://127.0.0.1:8765 chromium
 *   node tests/js/playwright-gui-test.js http://127.0.0.1:8765 webkit --headed
 *
 * Output: gui/tmp/playwright/
 */

const { chromium, webkit } = require('playwright');
const fs = require('fs');
const path = require('path');

const DEFAULT_URL = 'http://127.0.0.1:8765/';
const OUTPUT_DIR = path.resolve(__dirname, '../../tmp/playwright');

const parseArgs = () => {
  const args = process.argv.slice(2);
  let url = DEFAULT_URL;
  let browserType = 'chromium';
  let headed = false;

  for (const arg of args) {
    if (arg.startsWith('http')) url = arg;
    else if (arg === 'chromium' || arg === 'webkit') browserType = arg;
    else if (arg === '--headed') headed = true;
  }

  return { url, browserType, headed };
};

const ensureDir = (dir) => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
};

const timestamp = () => new Date().toISOString().replace(/[:.]/g, '-');

async function runTest() {
  const { url, browserType, headed } = parseArgs();
  const ts = timestamp();

  ensureDir(OUTPUT_DIR);

  console.log(`[playwright] Testing ${url} with ${browserType} (headed=${headed})`);

  // Launch browser
  const browserLauncher = browserType === 'webkit' ? webkit : chromium;
  const browser = await browserLauncher.launch({
    headless: !headed,
  });

  // Create context with tracing enabled
  const context = await browser.newContext({
    viewport: { width: 1600, height: 1000 },
    deviceScaleFactor: 1,
  });

  // Start tracing for debugging
  await context.tracing.start({
    screenshots: true,
    snapshots: true,
    sources: true,
  });

  const page = await context.newPage();

  // Capture console messages
  const consoleLogs = [];
  page.on('console', (msg) => {
    const entry = {
      type: msg.type(),
      text: msg.text(),
      location: msg.location(),
    };
    consoleLogs.push(entry);
    console.log(`[page][${msg.type()}] ${msg.text()}`);
  });

  // Capture errors
  const errors = [];
  page.on('pageerror', (err) => {
    errors.push({ message: err.message, stack: err.stack });
    console.error(`[page-error] ${err.message}`);
  });

  try {
    // Navigate to the GUI
    console.log('[playwright] Loading page...');
    await page.goto(url, { waitUntil: 'networkidle', timeout: 60000 });

    // Wait for app initialization (flexible - don't require OmniPainting.fill which needs WebGL)
    console.log('[playwright] Waiting for app initialization...');
    await page.waitForFunction(() => {
      // Check if basic app structure exists - more resilient to WebGL failures
      return document.getElementById('canvas') !== null &&
             document.getElementById('leftPanel') !== null &&
             (window.OmniPainting || window.__OMNI_CONFIG__);
    }, { timeout: 30000 });

    // Take initial screenshot
    const screenshotBefore = path.join(OUTPUT_DIR, `${browserType}-${ts}-01-initial.png`);
    await page.screenshot({ path: screenshotBefore, fullPage: true });
    console.log(`[playwright] Screenshot: ${screenshotBefore}`);

    // Get page state info
    const pageInfo = await page.evaluate(() => {
      const canvas = document.getElementById('canvas');
      const viewer = document.getElementById('viewer');
      const leftPanel = document.getElementById('leftPanel');
      const sidebar = document.getElementById('sidebar');

      return {
        viewport: {
          width: window.innerWidth,
          height: window.innerHeight
        },
        canvas: canvas ? {
          width: canvas.width,
          height: canvas.height,
          offsetWidth: canvas.offsetWidth,
          offsetHeight: canvas.offsetHeight,
          boundingRect: canvas.getBoundingClientRect(),
        } : null,
        viewer: viewer ? viewer.getBoundingClientRect() : null,
        leftPanel: leftPanel ? leftPanel.getBoundingClientRect() : null,
        sidebar: sidebar ? sidebar.getBoundingClientRect() : null,
        omniPaintingLoaded: typeof window.OmniPainting === 'object',
        omniDebugAvailable: typeof window.__OMNI_DEBUG__ === 'object',
      };
    });

    console.log('[playwright] Page info:', JSON.stringify(pageInfo, null, 2));

    // Test interactions - click on tools
    console.log('[playwright] Testing tool buttons...');

    // Click draw tool
    const drawButton = page.locator('button[data-mode="draw"]');
    if (await drawButton.count() > 0) {
      await drawButton.click();
      console.log('[playwright] Clicked draw tool');
    }

    // Click fill tool
    const fillButton = page.locator('button[data-mode="fill"]');
    if (await fillButton.count() > 0) {
      await fillButton.click();
      console.log('[playwright] Clicked fill tool');
    }

    // Take screenshot after tool interactions
    const screenshotTools = path.join(OUTPUT_DIR, `${browserType}-${ts}-02-tools.png`);
    await page.screenshot({ path: screenshotTools, fullPage: true });
    console.log(`[playwright] Screenshot: ${screenshotTools}`);

    // Test slider interactions
    console.log('[playwright] Testing sliders...');
    const brushSlider = page.locator('#brushSizeSlider');
    if (await brushSlider.count() > 0) {
      const box = await brushSlider.boundingBox();
      if (box) {
        // Drag slider to middle
        await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
        console.log('[playwright] Adjusted brush size slider');
      }
    }

    // Screenshot after slider interaction
    const screenshotSliders = path.join(OUTPUT_DIR, `${browserType}-${ts}-03-sliders.png`);
    await page.screenshot({ path: screenshotSliders, fullPage: true });
    console.log(`[playwright] Screenshot: ${screenshotSliders}`);

    // Test canvas interaction if canvas exists
    const canvasInfo = await page.evaluate(() => {
      const canvas = document.getElementById('canvas');
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      return { x: rect.x, y: rect.y, width: rect.width, height: rect.height };
    });

    if (canvasInfo && canvasInfo.width > 0 && canvasInfo.height > 0) {
      console.log('[playwright] Testing canvas drawing...');
      const centerX = canvasInfo.x + canvasInfo.width / 2;
      const centerY = canvasInfo.y + canvasInfo.height / 2;

      // Draw a stroke
      await page.mouse.move(centerX - 50, centerY);
      await page.mouse.down();
      await page.mouse.move(centerX + 50, centerY, { steps: 10 });
      await page.mouse.up();

      // Take screenshot after drawing
      const screenshotDraw = path.join(OUTPUT_DIR, `${browserType}-${ts}-04-after-draw.png`);
      await page.screenshot({ path: screenshotDraw, fullPage: true });
      console.log(`[playwright] Screenshot: ${screenshotDraw}`);
    }

    // Collect layout measurements for debugging
    const layoutReport = await page.evaluate(() => {
      const elements = ['leftPanel', 'viewer', 'sidebar', 'canvas', 'brushPreview'];
      const report = {};

      for (const id of elements) {
        const el = document.getElementById(id);
        if (el) {
          const rect = el.getBoundingClientRect();
          const styles = window.getComputedStyle(el);
          report[id] = {
            rect: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
            display: styles.display,
            position: styles.position,
            visibility: styles.visibility,
            zIndex: styles.zIndex,
          };
        }
      }

      return report;
    });

    // Save report
    const reportPath = path.join(OUTPUT_DIR, `${browserType}-${ts}-report.json`);
    fs.writeFileSync(reportPath, JSON.stringify({
      timestamp: ts,
      browser: browserType,
      url,
      pageInfo,
      layoutReport,
      consoleLogs,
      errors,
    }, null, 2));
    console.log(`[playwright] Report: ${reportPath}`);

    // Final screenshot
    const screenshotFinal = path.join(OUTPUT_DIR, `${browserType}-${ts}-05-final.png`);
    await page.screenshot({ path: screenshotFinal, fullPage: true });
    console.log(`[playwright] Final screenshot: ${screenshotFinal}`);

    console.log('\n[playwright] Test completed successfully!');
    console.log(`[playwright] Output directory: ${OUTPUT_DIR}`);

  } catch (err) {
    console.error('[playwright] Test failed:', err.message);

    // Take error screenshot
    const screenshotError = path.join(OUTPUT_DIR, `${browserType}-${ts}-error.png`);
    await page.screenshot({ path: screenshotError, fullPage: true });
    console.log(`[playwright] Error screenshot: ${screenshotError}`);

    throw err;
  } finally {
    // Stop tracing and save
    const tracePath = path.join(OUTPUT_DIR, `${browserType}-${ts}-trace.zip`);
    await context.tracing.stop({ path: tracePath });
    console.log(`[playwright] Trace saved: ${tracePath}`);
    console.log(`[playwright] View trace with: npx playwright show-trace ${tracePath}`);

    await browser.close();
  }
}

runTest().catch((err) => {
  console.error(err);
  process.exit(1);
});
