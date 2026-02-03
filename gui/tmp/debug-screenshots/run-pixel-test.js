#!/usr/bin/env node
/**
 * Pixel-level color test - samples actual rendered colors from the canvas
 */

const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const DEFAULT_URL = 'http://127.0.0.1:8000/';
const OUTPUT_DIR = path.resolve(__dirname);

async function runTest() {
  console.log('='.repeat(60));
  console.log('PIXEL COLOR TEST');
  console.log('='.repeat(60));

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1600, height: 1000 },
  });
  const page = await context.newPage();

  try {
    await page.goto(DEFAULT_URL, { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForFunction(() => document.getElementById('canvas') !== null, { timeout: 30000 });

    // Run segmentation
    console.log('\n[1] Running segmentation...');
    await page.evaluate(() => {
      const btn = document.getElementById('segmentButton');
      if (btn) btn.click();
    });
    await page.waitForFunction(() => {
      const status = document.getElementById('segmentStatus');
      return status && (status.textContent.includes('complete') || status.textContent.includes('failed'));
    }, { timeout: 120000 });
    await page.waitForTimeout(2000);

    // Toggle N-color OFF
    console.log('\n[2] Toggling N-color OFF...');
    await page.evaluate(() => {
      const toggle = document.getElementById('autoNColorToggle');
      if (toggle && toggle.checked) toggle.click();
    });
    await page.waitForTimeout(3000);

    // Take screenshot of just the canvas area
    const canvasHandle = await page.$('#canvas');
    const screenshotPath = path.join(OUTPUT_DIR, 'pixel-test-ncolor-off.png');
    await canvasHandle.screenshot({ path: screenshotPath });

    // Sample pixel colors from different cells using the screenshot
    console.log('\n[3] Sampling pixel colors from cells...');

    const pixelAnalysis = await page.evaluate(() => {
      const canvas = document.getElementById('canvas');
      const ctx = canvas.getContext('2d', { willReadFrequently: true });

      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      if (!maskValues) return { error: 'No mask values' };

      const width = window.imgWidth || 392;
      const height = window.imgHeight || 384;

      // Find center pixel of each unique label
      const labelPixels = {};
      const labelCounts = {};

      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const idx = y * width + x;
          const label = maskValues[idx];
          if (label > 0) {
            if (!labelPixels[label]) {
              labelPixels[label] = { sumX: 0, sumY: 0, count: 0 };
            }
            labelPixels[label].sumX += x;
            labelPixels[label].sumY += y;
            labelPixels[label].count += 1;
            labelCounts[label] = (labelCounts[label] || 0) + 1;
          }
        }
      }

      // Get center of each label
      const labelCenters = {};
      for (const [label, data] of Object.entries(labelPixels)) {
        labelCenters[label] = {
          x: Math.round(data.sumX / data.count),
          y: Math.round(data.sumY / data.count),
          pixelCount: data.count
        };
      }

      // Get view transform to convert image coords to canvas coords
      const viewState = window.viewState || { scale: 1, offsetX: 0, offsetY: 0 };
      const canvasCenterX = canvas.width / 2;
      const canvasCenterY = canvas.height / 2;

      // Sample rendered color from canvas for each label
      const results = [];
      const sortedLabels = Object.keys(labelCenters).map(Number).sort((a, b) => a - b).slice(0, 20);

      for (const label of sortedLabels) {
        const center = labelCenters[label];

        // Convert image coords to canvas coords
        const canvasX = Math.round((center.x - viewState.offsetX) * viewState.scale + canvasCenterX);
        const canvasY = Math.round((center.y - viewState.offsetY) * viewState.scale + canvasCenterY);

        // Sample pixel from canvas
        let renderedColor = null;
        try {
          const imageData = ctx.getImageData(canvasX, canvasY, 1, 1);
          renderedColor = [imageData.data[0], imageData.data[1], imageData.data[2]];
        } catch (e) {
          renderedColor = 'error: ' + e.message;
        }

        // Get expected color from getColormapColor
        let expectedColor = null;
        if (typeof window.getColormapColor === 'function') {
          expectedColor = window.getColormapColor(label);
        }

        results.push({
          label,
          imageCoords: { x: center.x, y: center.y },
          canvasCoords: { x: canvasX, y: canvasY },
          pixelCount: center.pixelCount,
          renderedColor,
          expectedColor,
          match: renderedColor && expectedColor ?
            (Math.abs(renderedColor[0] - expectedColor[0]) < 10 &&
             Math.abs(renderedColor[1] - expectedColor[1]) < 10 &&
             Math.abs(renderedColor[2] - expectedColor[2]) < 10) : false
        });
      }

      // Check for duplicate rendered colors
      const colorStrings = results.map(r =>
        Array.isArray(r.renderedColor) ? r.renderedColor.join(',') : 'invalid'
      );
      const uniqueRendered = new Set(colorStrings);

      return {
        totalLabels: Object.keys(labelCenters).length,
        sampledLabels: results.length,
        uniqueRenderedColors: uniqueRendered.size,
        viewState,
        samples: results,
        duplicateColors: colorStrings.length - uniqueRendered.size > 0 ?
          colorStrings.filter((c, i) => colorStrings.indexOf(c) !== i) : []
      };
    });

    console.log('\nAnalysis Results:');
    console.log(`  Total labels in mask: ${pixelAnalysis.totalLabels}`);
    console.log(`  Sampled labels: ${pixelAnalysis.sampledLabels}`);
    console.log(`  Unique RENDERED colors: ${pixelAnalysis.uniqueRenderedColors}`);

    if (pixelAnalysis.duplicateColors && pixelAnalysis.duplicateColors.length > 0) {
      console.log(`\n  WARNING: ${pixelAnalysis.duplicateColors.length} duplicate rendered colors!`);
      console.log(`  Duplicates: ${[...new Set(pixelAnalysis.duplicateColors)].join('; ')}`);
    }

    console.log('\n  Sample details (first 10):');
    for (const sample of pixelAnalysis.samples.slice(0, 10)) {
      const rendered = Array.isArray(sample.renderedColor) ?
        `rgb(${sample.renderedColor.join(',')})` : sample.renderedColor;
      const expected = Array.isArray(sample.expectedColor) ?
        `rgb(${sample.expectedColor.join(',')})` : 'N/A';
      console.log(`    Label ${sample.label}: rendered=${rendered}, expected=${expected}, match=${sample.match}`);
    }

    // Save full results
    fs.writeFileSync(
      path.join(OUTPUT_DIR, 'pixel-analysis.json'),
      JSON.stringify(pixelAnalysis, null, 2)
    );
    console.log('\n  Full analysis saved to: pixel-analysis.json');

    console.log('\n' + '='.repeat(60));

  } catch (err) {
    console.error('\n[ERROR]', err.message);
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
