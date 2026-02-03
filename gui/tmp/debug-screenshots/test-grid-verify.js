#!/usr/bin/env node
/**
 * Verify grid mask values and rendered colors
 */

const { chromium } = require('playwright');
const path = require('path');

const DEFAULT_URL = 'http://127.0.0.1:8000/';
const OUTPUT_DIR = path.resolve(__dirname);

async function runTest() {
  console.log('='.repeat(60));
  console.log('GRID VERIFICATION TEST');
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

    // Create grid and get info
    console.log('\n[1] Creating grid and analyzing...');
    const result = await page.evaluate(() => {
      const maskValues = window.OmniPainting?.__debugGetState?.()?.ctx?.maskValues;
      if (!maskValues) return { error: 'No mask values' };

      const width = window.imgWidth || 392;
      const height = window.imgHeight || 384;

      // Create 10x10 grid
      const gridCols = 10;
      const gridRows = 10;
      const cellWidth = Math.floor(width / gridCols);
      const cellHeight = Math.floor(height / gridRows);

      for (let row = 0; row < gridRows; row++) {
        for (let col = 0; col < gridCols; col++) {
          const label = row * gridCols + col + 1;
          const startX = col * cellWidth;
          const startY = row * cellHeight;
          const endX = Math.min(startX + cellWidth, width);
          const endY = Math.min(startY + cellHeight, height);

          for (let y = startY; y < endY; y++) {
            for (let x = startX; x < endX; x++) {
              const idx = y * width + x;
              maskValues[idx] = label;
            }
          }
        }
      }

      // Rebuild state
      if (window.paintingApi && typeof window.paintingApi.rebuildComponents === 'function') {
        window.paintingApi.rebuildComponents();
      }
      window.paletteTextureDirty = true;
      if (typeof window.markMaskTextureFullDirty === 'function') {
        window.markMaskTextureFullDirty();
      }
      if (typeof window.draw === 'function') {
        window.draw();
      }

      // Verify grid values at cell centers
      const verification = [];
      for (let row = 0; row < 5; row++) {
        for (let col = 0; col < 5; col++) {
          const expectedLabel = row * gridCols + col + 1;
          const centerX = col * cellWidth + Math.floor(cellWidth / 2);
          const centerY = row * cellHeight + Math.floor(cellHeight / 2);
          const idx = centerY * width + centerX;
          const actualLabel = maskValues[idx];

          // Get expected color
          let color = null;
          if (typeof window.getColormapColor === 'function') {
            color = window.getColormapColor(actualLabel);
          }

          verification.push({
            row,
            col,
            expectedLabel,
            actualLabel,
            match: expectedLabel === actualLabel,
            color: color ? color.join(',') : 'null',
          });
        }
      }

      return {
        gridSize: { rows: gridRows, cols: gridCols },
        cellSize: { width: cellWidth, height: cellHeight },
        verification,
      };
    });

    if (result.error) {
      console.error('  Error:', result.error);
      return;
    }

    console.log(`  Grid: ${result.gridSize.rows}x${result.gridSize.cols}`);
    console.log(`  Cell size: ${result.cellSize.width}x${result.cellSize.height}`);

    console.log('\n[2] Verifying mask values (first 25 cells):');
    console.log('    Row | Col | Expected | Actual | Match | Color');
    console.log('    ' + '-'.repeat(55));

    let allMatch = true;
    const colorCounts = {};

    for (const v of result.verification) {
      const matchStr = v.match ? 'YES' : 'NO';
      console.log(`    ${String(v.row).padStart(3)} | ${String(v.col).padStart(3)} | ${String(v.expectedLabel).padStart(8)} | ${String(v.actualLabel).padStart(6)} | ${matchStr.padStart(5)} | ${v.color}`);
      if (!v.match) allMatch = false;

      // Count color occurrences
      if (!colorCounts[v.color]) colorCounts[v.color] = [];
      colorCounts[v.color].push(v.actualLabel);
    }

    console.log(`\n  All mask values correct: ${allMatch ? 'YES' : 'NO'}`);

    // Check for duplicate colors among adjacent cells
    console.log('\n[3] Checking for duplicate colors among first 25 cells:');
    let dupCount = 0;
    for (const [color, labels] of Object.entries(colorCounts)) {
      if (labels.length > 1) {
        dupCount++;
        console.log(`  Color ${color} used by labels: ${labels.join(', ')}`);
      }
    }
    if (dupCount === 0) {
      console.log('  No duplicate colors!');
    }

    // Toggle N-color off and take screenshot
    console.log('\n[4] Toggling N-color OFF...');
    await page.evaluate(() => {
      const toggle = document.getElementById('autoNColorToggle');
      if (toggle && toggle.checked) toggle.click();
    });
    await page.waitForTimeout(3000);

    await page.screenshot({ path: path.join(OUTPUT_DIR, 'grid-verify.png'), fullPage: true });
    console.log('  Screenshot saved: grid-verify.png');

    // Check adjacent colors
    console.log('\n[5] Checking adjacent cell colors in first 5x5:');
    const adjColors = await page.evaluate(() => {
      const colors = [];
      for (let label = 1; label <= 25; label++) {
        const c = window.getColormapColor(label);
        colors.push(c ? c.join(',') : 'null');
      }

      // Check horizontal adjacencies
      const issues = [];
      for (let row = 0; row < 5; row++) {
        for (let col = 0; col < 4; col++) {
          const idx1 = row * 5 + col;
          const idx2 = row * 5 + col + 1;
          const label1 = row * 10 + col + 1;
          const label2 = row * 10 + col + 2;
          const c1 = window.getColormapColor(label1);
          const c2 = window.getColormapColor(label2);

          // Check if colors are very similar (within threshold)
          if (c1 && c2) {
            const diff = Math.abs(c1[0] - c2[0]) + Math.abs(c1[1] - c2[1]) + Math.abs(c1[2] - c2[2]);
            if (diff < 60) { // Very similar colors
              issues.push({
                labels: [label1, label2],
                colors: [c1.join(','), c2.join(',')],
                diff,
              });
            }
          }
        }
      }

      // Check vertical adjacencies
      for (let row = 0; row < 4; row++) {
        for (let col = 0; col < 5; col++) {
          const label1 = row * 10 + col + 1;
          const label2 = (row + 1) * 10 + col + 1;
          const c1 = window.getColormapColor(label1);
          const c2 = window.getColormapColor(label2);

          if (c1 && c2) {
            const diff = Math.abs(c1[0] - c2[0]) + Math.abs(c1[1] - c2[1]) + Math.abs(c1[2] - c2[2]);
            if (diff < 60) {
              issues.push({
                labels: [label1, label2],
                colors: [c1.join(','), c2.join(',')],
                diff,
                direction: 'vertical',
              });
            }
          }
        }
      }

      return { issues, totalChecked: 85 };
    });

    if (adjColors.issues.length === 0) {
      console.log('  No adjacent cells with very similar colors (diff < 60)');
    } else {
      console.log(`  Found ${adjColors.issues.length} adjacent pairs with similar colors:`);
      for (const issue of adjColors.issues) {
        console.log(`    Labels ${issue.labels.join(' & ')}: colors (${issue.colors.join(') & (')}), diff=${issue.diff}`);
      }
    }

  } catch (err) {
    console.error('\n[ERROR]', err.message);
    console.error(err.stack);
  } finally {
    await browser.close();
  }
}

runTest().catch(console.error);
