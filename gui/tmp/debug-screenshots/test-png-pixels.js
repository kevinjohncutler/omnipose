#!/usr/bin/env node
/**
 * Sample actual pixels from PNG screenshot and compare to expected colors
 */

const { chromium } = require('playwright');
const { PNG } = require('pngjs');
const fs = require('fs');
const path = require('path');

const DEFAULT_URL = 'http://127.0.0.1:8000/';
const OUTPUT_DIR = path.resolve(__dirname);

async function runTest() {
  console.log('='.repeat(60));
  console.log('PNG PIXEL SAMPLING TEST');
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
    await page.waitForTimeout(5000);

    // Get canvas bounding box and view state
    const canvasInfo = await page.evaluate(() => {
      const canvas = document.getElementById('canvas');
      const rect = canvas.getBoundingClientRect();
      return {
        x: rect.x,
        y: rect.y,
        width: canvas.width,
        height: canvas.height,
        viewState: window.viewState || { scale: 1, offsetX: 0, offsetY: 0 },
        imgWidth: window.imgWidth || 392,
        imgHeight: window.imgHeight || 384,
      };
    });
    console.log(`  Canvas: ${canvasInfo.width}x${canvasInfo.height} at (${canvasInfo.x}, ${canvasInfo.y})`);
    console.log(`  View: scale=${canvasInfo.viewState.scale.toFixed(3)}, offset=(${canvasInfo.viewState.offsetX.toFixed(1)}, ${canvasInfo.viewState.offsetY.toFixed(1)})`);

    // Get mask data and label centers
    const labelInfo = await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      if (!maskValues) return { error: 'no mask' };

      const width = window.imgWidth || 392;
      const height = window.imgHeight || 384;

      // Find center of each unique label (first 20)
      const labelPixels = {};
      const uniqueLabels = [];

      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const idx = y * width + x;
          const label = maskValues[idx];
          if (label > 0) {
            if (!labelPixels[label]) {
              labelPixels[label] = { sumX: 0, sumY: 0, count: 0 };
              uniqueLabels.push(label);
            }
            labelPixels[label].sumX += x;
            labelPixels[label].sumY += y;
            labelPixels[label].count += 1;
          }
        }
      }

      // Get center and expected color for first 20 labels
      const result = [];
      const sortedLabels = uniqueLabels.sort((a, b) => a - b).slice(0, 20);

      for (const label of sortedLabels) {
        const data = labelPixels[label];
        const cx = Math.round(data.sumX / data.count);
        const cy = Math.round(data.sumY / data.count);

        // Get expected color from getColormapColor
        let expectedColor = null;
        if (typeof window.getColormapColor === 'function') {
          expectedColor = window.getColormapColor(label);
        }

        result.push({
          label,
          imageX: cx,
          imageY: cy,
          pixelCount: data.count,
          expectedColor,
        });
      }

      return {
        totalLabels: uniqueLabels.length,
        labels: result,
      };
    });

    console.log(`\n[3] Found ${labelInfo.totalLabels} unique labels`);

    // Take screenshot of just the canvas
    const canvasHandle = await page.$('#canvas');
    const screenshotPath = path.join(OUTPUT_DIR, 'png-test-canvas.png');
    await canvasHandle.screenshot({ path: screenshotPath });
    console.log(`  Screenshot saved: png-test-canvas.png`);

    // Read the PNG and sample pixels
    console.log('\n[4] Reading PNG and sampling pixels...');
    const pngData = fs.readFileSync(screenshotPath);
    const png = PNG.sync.read(pngData);

    const { scale, offsetX, offsetY } = canvasInfo.viewState;
    const canvasCenterX = canvasInfo.width / 2;
    const canvasCenterY = canvasInfo.height / 2;

    console.log('\n[5] Comparing expected vs actual colors:');
    console.log('    Label | Expected Color     | Actual Color       | Match?');
    console.log('    ' + '-'.repeat(60));

    let matchCount = 0;
    let mismatchCount = 0;
    const colorToLabels = {}; // Track which labels have which colors

    for (const info of labelInfo.labels) {
      // Convert image coords to canvas coords
      const canvasX = Math.round((info.imageX - offsetX) * scale + canvasCenterX);
      const canvasY = Math.round((info.imageY - offsetY) * scale + canvasCenterY);

      // Sample pixel from PNG
      let actualColor = null;
      if (canvasX >= 0 && canvasX < png.width && canvasY >= 0 && canvasY < png.height) {
        const pngIdx = (canvasY * png.width + canvasX) * 4;
        actualColor = [png.data[pngIdx], png.data[pngIdx + 1], png.data[pngIdx + 2]];
      }

      const expected = info.expectedColor;
      const actual = actualColor;

      // Check if match (with tolerance for blending)
      let match = false;
      if (expected && actual) {
        // Allow some tolerance for alpha blending with background
        const tol = 30;
        match = Math.abs(expected[0] - actual[0]) < tol &&
                Math.abs(expected[1] - actual[1]) < tol &&
                Math.abs(expected[2] - actual[2]) < tol;
      }

      const expectedStr = expected ? `rgb(${expected.join(',')})` : 'N/A';
      const actualStr = actual ? `rgb(${actual.join(',')})` : 'N/A';
      const matchStr = match ? 'YES' : 'NO';

      console.log(`    ${String(info.label).padStart(5)} | ${expectedStr.padEnd(18)} | ${actualStr.padEnd(18)} | ${matchStr}`);

      if (match) matchCount++;
      else mismatchCount++;

      // Track color to labels mapping
      if (actual) {
        const colorKey = actual.join(',');
        if (!colorToLabels[colorKey]) colorToLabels[colorKey] = [];
        colorToLabels[colorKey].push(info.label);
      }
    }

    console.log('\n[6] Summary:');
    console.log(`  Matches: ${matchCount}`);
    console.log(`  Mismatches: ${mismatchCount}`);

    // Check for duplicate actual colors
    console.log('\n[7] Checking for duplicate rendered colors:');
    let dupCount = 0;
    for (const [color, labels] of Object.entries(colorToLabels)) {
      if (labels.length > 1) {
        dupCount++;
        console.log(`  Color rgb(${color}) used by labels: ${labels.join(', ')}`);
      }
    }
    if (dupCount === 0) {
      console.log('  No duplicate colors found!');
    } else {
      console.log(`  WARNING: ${dupCount} colors used by multiple labels!`);
    }

  } catch (err) {
    console.error('\n[ERROR]', err.message);
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
