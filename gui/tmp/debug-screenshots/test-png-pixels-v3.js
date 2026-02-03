#!/usr/bin/env node
/**
 * Sample actual pixels from PNG screenshot - with proper view state
 */

const { chromium } = require('playwright');
const { PNG } = require('pngjs');
const fs = require('fs');
const path = require('path');

const DEFAULT_URL = 'http://127.0.0.1:8000/';
const OUTPUT_DIR = path.resolve(__dirname);

async function runTest() {
  console.log('='.repeat(60));
  console.log('PNG PIXEL SAMPLING TEST v3');
  console.log('='.repeat(60));

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1600, height: 1000 },
  });
  const page = await context.newPage();

  try {
    await page.goto(DEFAULT_URL, { waitUntil: 'networkidle', timeout: 60000 });
    await page.waitForFunction(() => document.getElementById('canvas') !== null, { timeout: 30000 });
    await page.waitForTimeout(2000);

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
    await page.waitForTimeout(3000);

    // Toggle N-color OFF
    console.log('\n[2] Toggling N-color OFF...');
    await page.evaluate(() => {
      const toggle = document.getElementById('autoNColorToggle');
      if (toggle && toggle.checked) toggle.click();
    });
    await page.waitForTimeout(5000);

    // Trigger recenter and wait
    console.log('\n[3] Triggering recenter...');
    await page.evaluate(() => {
      if (typeof window.recenterImage === 'function') {
        window.recenterImage();
      }
    });
    await page.waitForTimeout(1000);

    // Force a draw and get view state
    const info = await page.evaluate(() => {
      // Force draw to ensure view is updated
      if (typeof window.draw === 'function') {
        window.draw();
      }

      const canvas = document.getElementById('canvas');
      const vs = window.viewState || { scale: 1, offsetX: 0, offsetY: 0 };

      // Get mask data
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;

      const width = window.imgWidth || 392;
      const height = window.imgHeight || 384;

      // Find centers of first 15 unique labels
      const labelData = {};
      const seenLabels = new Set();

      if (maskValues) {
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const idx = y * width + x;
            const label = maskValues[idx];
            if (label > 0) {
              if (!labelData[label]) {
                labelData[label] = { sumX: 0, sumY: 0, count: 0 };
                seenLabels.add(label);
              }
              labelData[label].sumX += x;
              labelData[label].sumY += y;
              labelData[label].count += 1;
            }
          }
        }
      }

      const labels = [];
      const sortedLabels = Array.from(seenLabels).sort((a, b) => a - b).slice(0, 15);

      for (const label of sortedLabels) {
        const data = labelData[label];
        const cx = Math.round(data.sumX / data.count);
        const cy = Math.round(data.sumY / data.count);

        // Convert image coords to canvas coords using view state
        // The formula: canvasCoord = (imageCoord - offset) * scale + canvasCenter
        const canvasCenterX = canvas.width / 2;
        const canvasCenterY = canvas.height / 2;
        const canvasX = Math.round((cx - vs.offsetX) * vs.scale + canvasCenterX);
        const canvasY = Math.round((cy - vs.offsetY) * vs.scale + canvasCenterY);

        let expectedColor = null;
        if (typeof window.getColormapColor === 'function') {
          expectedColor = window.getColormapColor(label);
        }

        labels.push({
          label,
          imageX: cx,
          imageY: cy,
          canvasX,
          canvasY,
          pixelCount: data.count,
          expectedColor,
        });
      }

      return {
        canvasSize: { width: canvas.width, height: canvas.height },
        viewState: { scale: vs.scale, offsetX: vs.offsetX, offsetY: vs.offsetY },
        imgSize: { width, height },
        totalLabels: seenLabels.size,
        labels,
      };
    });

    console.log(`\n[4] View state: scale=${info.viewState.scale?.toFixed(3)}, offset=(${info.viewState.offsetX?.toFixed(1)}, ${info.viewState.offsetY?.toFixed(1)})`);
    console.log(`  Canvas: ${info.canvasSize.width}x${info.canvasSize.height}`);
    console.log(`  Image: ${info.imgSize.width}x${info.imgSize.height}`);
    console.log(`  Total labels: ${info.totalLabels}`);

    // Take screenshot of just the canvas
    const canvasHandle = await page.$('#canvas');
    const screenshotPath = path.join(OUTPUT_DIR, 'png-v3-canvas.png');
    await canvasHandle.screenshot({ path: screenshotPath });

    // Read the PNG
    const pngData = fs.readFileSync(screenshotPath);
    const png = PNG.sync.read(pngData);
    console.log(`  PNG dimensions: ${png.width}x${png.height}`);

    const scaleX = png.width / info.canvasSize.width;
    const scaleY = png.height / info.canvasSize.height;

    console.log('\n[5] Sampling pixels:');
    console.log('    Label | Pixels | Image XY     | Canvas XY    | Expected        | Actual          | Match');
    console.log('    ' + '-'.repeat(95));

    const actualColors = {};
    let matchCount = 0;

    for (const label of info.labels) {
      // Convert canvas coords to PNG coords
      const pngX = Math.round(label.canvasX * scaleX);
      const pngY = Math.round(label.canvasY * scaleY);

      let actualColor = null;
      if (pngX >= 0 && pngX < png.width && pngY >= 0 && pngY < png.height) {
        const pngIdx = (pngY * png.width + pngX) * 4;
        actualColor = [png.data[pngIdx], png.data[pngIdx + 1], png.data[pngIdx + 2]];
      }

      const expected = label.expectedColor;
      const actual = actualColor;

      // Check match with tolerance for alpha blending
      let match = false;
      if (expected && actual) {
        const tol = 50;
        match = Math.abs(expected[0] - actual[0]) < tol &&
                Math.abs(expected[1] - actual[1]) < tol &&
                Math.abs(expected[2] - actual[2]) < tol;
      }

      if (match) matchCount++;

      const expectedStr = expected ? `(${expected.join(',')})` : 'N/A';
      const actualStr = actual ? `(${actual.join(',')})` : 'N/A';
      const matchStr = match ? 'YES' : 'NO';

      console.log(`    ${String(label.label).padStart(5)} | ${String(label.pixelCount).padStart(6)} | (${String(label.imageX).padStart(3)},${String(label.imageY).padStart(3)}) | (${String(label.canvasX).padStart(4)},${String(label.canvasY).padStart(4)}) | ${expectedStr.padEnd(15)} | ${actualStr.padEnd(15)} | ${matchStr}`);

      actualColors[label.label] = actual;
    }

    console.log(`\n[6] Summary: ${matchCount}/${info.labels.length} colors match expected`);

    // Check for duplicate actual colors
    const colorToLabels = {};
    for (const [label, color] of Object.entries(actualColors)) {
      if (color) {
        const key = color.join(',');
        if (!colorToLabels[key]) colorToLabels[key] = [];
        colorToLabels[key].push(Number(label));
      }
    }

    console.log('\n[7] Checking for duplicate colors in rendered image:');
    let dupCount = 0;
    for (const [color, labels] of Object.entries(colorToLabels)) {
      if (labels.length > 1) {
        dupCount++;
        console.log(`  rgb(${color}) used by labels: ${labels.join(', ')}`);
      }
    }
    if (dupCount === 0) {
      console.log('  SUCCESS: No duplicate colors found!');
    } else {
      console.log(`\n  WARNING: ${dupCount} duplicate color groups found!`);
    }

  } catch (err) {
    console.error('\n[ERROR]', err.message);
    console.error(err.stack);
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
