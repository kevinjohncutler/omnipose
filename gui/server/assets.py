"""Asset loading and HTML template rendering for the Omnipose GUI server."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

# Path constants - relative to this module's location
SERVER_DIR = Path(__file__).resolve().parent
GUI_DIR = SERVER_DIR.parent
WEB_DIR = (GUI_DIR / "web").resolve()
LOG_DIR = GUI_DIR / "logs"
LOG_FILE = LOG_DIR / "omni_gui.log"

INDEX_HTML = WEB_DIR / "index.html"
APP_JS = WEB_DIR / "app.js"
HTML_DIR = WEB_DIR / "html"
CSS_DIR = WEB_DIR / "css"
POINTER_JS = WEB_DIR / "js" / "pointer-state.js"
LOGGING_JS = WEB_DIR / "js" / "logging.js"
HISTORY_JS = WEB_DIR / "js" / "history.js"
BRUSH_JS = WEB_DIR / "js" / "brush.js"
PAINTING_JS = WEB_DIR / "js" / "painting.js"
INTERACTIONS_JS = WEB_DIR / "js" / "interactions.js"
COLORMAP_JS = WEB_DIR / "js" / "colormap.js"
UI_UTILS_JS = WEB_DIR / "js" / "ui-utils.js"
STATE_PERSISTENCE_JS = WEB_DIR / "js" / "state-persistence.js"
FILE_NAVIGATION_JS = WEB_DIR / "js" / "file-navigation.js"

HTML_FRAGMENTS = [
    HTML_DIR / "left-panel.html",
    HTML_DIR / "viewer.html",
    HTML_DIR / "sidebar.html",
]

CSS_FILES = [
    CSS_DIR / "layout.css",
    CSS_DIR / "tools.css",
    CSS_DIR / "controls.css",
    CSS_DIR / "viewer.css",
]

CSS_LINKS = (
    '    <link rel="stylesheet" href="/static/css/layout.css" />',
    '    <link rel="stylesheet" href="/static/css/tools.css" />',
    '    <link rel="stylesheet" href="/static/css/controls.css" />',
    '    <link rel="stylesheet" href="/static/css/viewer.css" />',
)

JS_FILES = [POINTER_JS, LOGGING_JS, HISTORY_JS, COLORMAP_JS, UI_UTILS_JS, STATE_PERSISTENCE_JS, FILE_NAVIGATION_JS, BRUSH_JS, PAINTING_JS, INTERACTIONS_JS, APP_JS]

JS_STATIC_PATHS = (
    "/static/js/pointer-state.js",
    "/static/js/logging.js",
    "/static/js/history.js",
    "/static/js/colormap.js",
    "/static/js/ui-utils.js",
    "/static/js/state-persistence.js",
    "/static/js/file-navigation.js",
    "/static/js/brush.js",
    "/static/js/painting.js",
    "/static/js/interactions.js",
    "/static/app.js",
)

# Caches for template content
_INDEX_HTML_CACHE: dict[str, object] = {"content": "", "mtime": None}
_LAYOUT_MARKUP_CACHE: dict[str, object] = {"markup": "", "mtimes": {}}
_INLINE_CSS_CACHE: dict[str, object] = {"text": "", "mtimes": {}}
_INLINE_JS_CACHE: dict[str, object] = {"text": "", "mtimes": {}}
_CACHE_BUSTER = str(int(time.time()))

CAPTURE_LOG_SCRIPT = """<script>
(function(){
  if (window.__omniLogPush) { return; }
  var queue = [];
  var endpoint = '/api/log';
  var maxBatch = 25;
  var flushTimer = null;
  function flush(){
    if (!queue.length) { return; }
    var payload = queue.slice();
    queue.length = 0;
    try {
      fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ entries: payload })
      }).catch(function(){ });
    } catch (err) {
      console.warn('[log] flush failed', err);
    }
  }
  function schedule(){
    if (flushTimer) { return; }
    flushTimer = setTimeout(function(){ flushTimer = null; flush(); }, 300);
  }
  window.__omniLogPush = function(kind, data){
    try {
      queue.push({ kind: kind, data: data, ts: Date.now() });
      if (queue.length >= maxBatch) {
        if (flushTimer) { clearTimeout(flushTimer); flushTimer = null; }
        flush();
      } else {
        schedule();
      }
    } catch (err) {
      console.warn('[log] push failed', err);
    }
  };
  window.addEventListener('error', function(evt){
    window.__omniLogPush('JS_ERROR', {
      message: evt.message || '',
      filename: evt.filename || '',
      lineno: evt.lineno || 0,
      colno: evt.colno || 0,
      stack: evt.error && evt.error.stack ? String(evt.error.stack) : ''
    });
  });
})();
</script>"""

# Tiny script that restores the accent color from localStorage before the main
# JS bundle runs — eliminates the yellow flash on page reload/navigation.
RESTORE_ACCENT_SCRIPT = """<script>
(function(){
  try {
    var raw = localStorage.getItem('__omni_accent');
    if (raw) {
      var a = JSON.parse(raw);
      if (a && a.c) {
        var s = document.documentElement.style;
        s.setProperty('--accent-color', a.c);
        if (a.h) s.setProperty('--accent-hover', a.h);
        if (a.k) s.setProperty('--accent-ink', a.k);
      }
    }
  } catch(_){}
  /* Hide body until app signals ready — prevents layout flash during init */
  document.documentElement.style.opacity = '0';
  document.documentElement.style.transition = 'opacity 80ms ease-in';
})();
</script>"""


def _load_fragment(path: Path) -> str:
    """Return fragment content with leading extract comments removed."""
    lines = path.read_text(encoding="utf-8").splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].lstrip().startswith("<!--"):
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
    return "\n".join(lines)


def _get_index_template() -> str:
    global _INDEX_HTML_CACHE
    try:
        mtime = INDEX_HTML.stat().st_mtime_ns
    except FileNotFoundError:
        mtime = -1
    cached_content = _INDEX_HTML_CACHE.get("content")
    if cached_content and _INDEX_HTML_CACHE.get("mtime") == mtime:
        return cached_content  # type: ignore[return-value]
    content = INDEX_HTML.read_text(encoding="utf-8")
    _INDEX_HTML_CACHE = {"content": content, "mtime": mtime}
    return content


def _snapshot_mtimes(paths: Sequence[Path]) -> dict[str, int]:
    mtimes: dict[str, int] = {}
    for path in paths:
        try:
            mtimes[str(path)] = path.stat().st_mtime_ns
        except FileNotFoundError:
            mtimes[str(path)] = -1
    return mtimes


def _get_layout_markup() -> str:
    global _LAYOUT_MARKUP_CACHE
    mtimes = _snapshot_mtimes(HTML_FRAGMENTS)
    cached_markup = _LAYOUT_MARKUP_CACHE.get("markup")
    cached_mtimes = _LAYOUT_MARKUP_CACHE.get("mtimes")
    if cached_markup and cached_mtimes == mtimes:
        return cached_markup  # type: ignore[return-value]
    markup = "\n".join(_load_fragment(path) for path in HTML_FRAGMENTS)
    _LAYOUT_MARKUP_CACHE = {"markup": markup, "mtimes": mtimes}
    return markup


def _concat_cached_text(paths: Sequence[Path], cache: dict[str, object]) -> str:
    mtimes = _snapshot_mtimes(paths)
    cached_text = cache.get("text")
    if cached_text and cache.get("mtimes") == mtimes:
        return cached_text  # type: ignore[return-value]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    cache["text"] = text
    cache["mtimes"] = mtimes
    return text


def _prime_static_caches() -> None:
    """Pre-load all static assets into caches."""
    try:
        _INDEX_HTML_CACHE["content"] = INDEX_HTML.read_text(encoding="utf-8")
        _INDEX_HTML_CACHE["mtime"] = INDEX_HTML.stat().st_mtime_ns
    except FileNotFoundError:
        _INDEX_HTML_CACHE["content"] = ""
        _INDEX_HTML_CACHE["mtime"] = None
    _LAYOUT_MARKUP_CACHE["markup"] = "\n".join(_load_fragment(path) for path in HTML_FRAGMENTS)
    _LAYOUT_MARKUP_CACHE["mtimes"] = _snapshot_mtimes(HTML_FRAGMENTS)
    _INLINE_CSS_CACHE["text"] = "\n".join(path.read_text(encoding="utf-8") for path in CSS_FILES)
    _INLINE_CSS_CACHE["mtimes"] = _snapshot_mtimes(CSS_FILES)
    _INLINE_JS_CACHE["text"] = "\n\n".join(path.read_text(encoding="utf-8") for path in JS_FILES)
    _INLINE_JS_CACHE["mtimes"] = _snapshot_mtimes(JS_FILES)


# Prime caches on module load
_prime_static_caches()


def render_index(
    config: dict[str, object],
    *,
    inline_assets: bool,
    cache_buster: str | None = None,
) -> str:
    """Render the index.html template with the given configuration."""
    html = _get_index_template()
    layout_markup = _get_layout_markup()
    placeholder = '    <div id="app"></div>'
    if placeholder in html:
        html = html.replace(
            placeholder,
            f"    <div id=\"app\">\n{layout_markup}\n    </div>",
        )
    config_json = json.dumps(config).replace('</', '<\\/')
    debug_webgl = bool(config.get("debugWebgl"))
    config_script = (
        f"<script>window.__OMNI_CONFIG__ = {config_json}; "
        f"window.__OMNI_WEBGL_LOGGING__ = {json.dumps(debug_webgl)};</script>"
    )
    capture_script = CAPTURE_LOG_SCRIPT.strip()
    capture_script = "    " + capture_script.replace("\n", "\n    ")
    accent_script = RESTORE_ACCENT_SCRIPT.strip()
    accent_script = "    " + accent_script.replace("\n", "\n    ")
    css_links = list(CSS_LINKS)
    script_tag = '    <script src="/static/app.js"></script>'
    keep_order_comment = (
        "<!-- IMPORTANT: Viewer scripts must remain classic scripts in this order. "
        "Switching to type=\"module\" breaks PyWebView image loading. -->"
    )

    if inline_assets:
        css_text = _concat_cached_text(CSS_FILES, _INLINE_CSS_CACHE)
        html = html.replace(css_links[0], f"    <style>{css_text}</style>")
        for link in css_links[1:]:
            html = html.replace(f"{link}\n", "")
            html = html.replace(link, "")
        js_bundle = _concat_cached_text(JS_FILES, _INLINE_JS_CACHE)
        bundled_script = f"<script>\n/* {keep_order_comment[5:-4]} */\n{js_bundle}\n</script>"
        bundled_script = "    " + bundled_script.replace("\n", "\n    ")
        html = html.replace(
            script_tag,
            "\n".join([
                config_script,
                accent_script,
                capture_script,
                bundled_script,
            ]),
        )
    else:
        suffix = f"?v={cache_buster}" if cache_buster else ""
        for link in css_links:
            html = html.replace(
                link,
                link.replace(".css\"", f".css{suffix}\""),
            )
        script_parts = [config_script, accent_script, capture_script, f'    {keep_order_comment}']
        script_parts.extend(
            f'    <script src="{path}{suffix}"></script>'
            for path in JS_STATIC_PATHS
        )
        html = html.replace(script_tag, "\n".join(script_parts))
    return html


def build_html(config: Mapping[str, Any], *, inline_assets: bool = True) -> str:
    """Build the complete HTML page with the given configuration."""
    start = time.perf_counter()
    html = render_index(dict(config), inline_assets=inline_assets, cache_buster=_CACHE_BUSTER)
    total_elapsed = time.perf_counter() - start
    print(
        f"[pywebview] build_html rendered in {total_elapsed*1000:.1f}ms (inline_assets={inline_assets})",
        flush=True,
    )
    return html


def append_gui_log(message: str) -> None:
    """Append a message to the GUI log file."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with LOG_FILE.open("a", encoding="utf-8", errors="ignore") as handle:
            handle.write(f"[{timestamp}] {message}\n")
    except Exception:
        pass
