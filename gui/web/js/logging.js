(function initOmniLogging(global) {
  'use strict';

  const state = {
    pending: [],
    flushTimer: null,
    pywebviewReady: false,
    startTime: (typeof performance !== 'undefined' && performance.now)
      ? performance.now()
      : Date.now(),
  };

  function formatMessage(message) {
    const now = (typeof performance !== 'undefined' && performance.now)
      ? (performance.now() - state.startTime)
      : (Date.now() - state.startTime);
    const timestamp = now.toFixed(1).padStart(7, ' ');
    return '[' + timestamp + ' ms] ' + message;
  }

  function pushToQueue(msg) {
    state.pending.push(msg);
    if (state.pending.length > 200) {
      state.pending.shift();
    }
    scheduleLogFlush();
  }

  function flushLogs() {
    if (state.flushTimer !== null) {
      clearTimeout(state.flushTimer);
      state.flushTimer = null;
    }
    if (!state.pending.length) {
      return;
    }
    const payload = { messages: state.pending.splice(0, state.pending.length) };
    if (state.pywebviewReady) {
      const api = global.pywebview && global.pywebview.api && global.pywebview.api.log
        ? global.pywebview.api
        : null;
      if (api && api.log) {
        for (let i = 0; i < payload.messages.length; i += 1) {
          api.log(payload.messages[i]);
        }
      }
    }
    if (typeof fetch !== 'function') {
      return;
    }
    const body = JSON.stringify(payload);
    try {
      if (global.navigator && typeof global.navigator.sendBeacon === 'function') {
        const ok = global.navigator.sendBeacon(
          '/api/log',
          new Blob([body], { type: 'application/json' }),
        );
        if (!ok) {
          state.pending.unshift(...payload.messages);
          scheduleLogFlush();
        }
      } else {
        fetch('/api/log', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body,
          keepalive: true,
        }).catch(() => {
          state.pending.unshift(...payload.messages);
          scheduleLogFlush();
        });
      }
    } catch (_) {
      state.pending.unshift(...payload.messages);
      scheduleLogFlush();
    }
  }

  function scheduleLogFlush(delay = 200) {
    if (state.flushTimer !== null) {
      return;
    }
    state.flushTimer = global.setTimeout(() => {
      state.flushTimer = null;
      flushLogs();
    }, delay);
  }

  function log(message) {
    const formatted = formatMessage(String(message));
    try {
      if (typeof console !== 'undefined' && console.log) {
        console.log('[pywebview]', formatted);
      }
    } catch (_) {
      /* ignore console errors */
    }
    const api = global.pywebview ? global.pywebview.api : null;
    if (api && api.log) {
      api.log(formatted);
    } else {
      pushToQueue(formatted);
    }
  }

  function setPywebviewReady(value) {
    state.pywebviewReady = Boolean(value);
    if (state.pywebviewReady) {
      flushLogs();
    }
  }

  function clearQueue() {
    state.pending.length = 0;
  }

  const originalWarn = console && console.warn ? console.warn.bind(console) : null;
  const originalError = console && console.error ? console.error.bind(console) : null;
  if (console) {
    console.warn = (...args) => {
      try { log('[warn] ' + args.map((a) => String(a)).join(' ')); } catch (_) {}
      if (originalWarn) { originalWarn(...args); }
    };
    console.error = (...args) => {
      try { log('[error] ' + args.map((a) => String(a)).join(' ')); } catch (_) {}
      if (originalError) { originalError(...args); }
    };
  }

  const api = {
    log,
    flushLogs,
    scheduleLogFlush,
    setPywebviewReady,
    clearQueue,
    isPywebviewReady() {
      return state.pywebviewReady;
    },
  };

  global.OmniLog = Object.assign({}, global.OmniLog, api);

  const pollId = global.setInterval(() => {
    if (state.pywebviewReady) {
      global.clearInterval(pollId);
      return;
    }
    const pyapi = global.pywebview && global.pywebview.api && global.pywebview.api.log
      ? global.pywebview.api
      : null;
    if (pyapi && pyapi.log) {
      setPywebviewReady(true);
      log('pywebview api detected via poll');
      global.clearInterval(pollId);
    }
  }, 50);

  global.addEventListener('error', (evt) => {
    try {
      const message = evt && evt.message ? evt.message : String(evt);
      log('uncaught error: ' + message);
    } catch (_) {
      /* ignore */
    }
  });
})(typeof window !== 'undefined' ? window : globalThis);
