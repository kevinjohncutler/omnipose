const SECTION_PATHS = [
  '/static/html/left-panel.html',
  '/static/html/viewer.html',
  '/static/html/sidebar.html',
];

const SCRIPT_PATHS = [
  '/static/js/configuration.js',
  '/static/js/painting.js',
  '/static/js/gestures.js',
  '/static/js/affinity-graph.js',
  '/static/js/viewer-rendering.js',
  '/static/js/ui-controls.js',
  '/static/js/app-bootstrap.js',
];

const assetSuffix = typeof window !== 'undefined' && typeof window.__OMNI_ASSET_SUFFIX__ === 'string'
  ? window.__OMNI_ASSET_SUFFIX__
  : '';

function withSuffix(path) {
  return assetSuffix ? `${path}${assetSuffix}` : path;
}

async function fetchText(path) {
  const response = await fetch(withSuffix(path));
  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.status}`);
  }
  return response.text();
}

async function loadLayout() {
  const app = document.getElementById('app');
  if (!app) {
    throw new Error('Missing #app container');
  }
  if (app.querySelector('#viewer')) {
    return;
  }
  const [leftPanel, viewer, sidebar] = await Promise.all(
    SECTION_PATHS.map((path) => fetchText(path))
  );
  app.innerHTML = `${leftPanel}
${viewer}
${sidebar}`;
}

function loadScriptSequential(src) {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = withSuffix(src);
    script.async = false;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Failed to load script: ${src}`));
    document.head.appendChild(script);
  });
}

async function loadScripts() {
  for (const src of SCRIPT_PATHS) {
    await loadScriptSequential(src);
  }
}

async function bootstrap() {
  try {
    await loadLayout();
    await loadScripts();
    if (typeof window.boot === 'function') {
      window.boot();
    } else if (typeof window.initialize === 'function') {
      window.initialize();
    }
  } catch (err) {
    console.error('Failed to initialize viewer shell:', err);
  }
}

bootstrap();
