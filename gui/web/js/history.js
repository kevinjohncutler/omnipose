(function initOmniHistory(global) {
  'use strict';

  const state = {
    limit: 200,
    undo: [],
    redo: [],
  };

  function init(options) {
    if (!options || typeof options !== 'object') {
      return;
    }
    if (typeof options.limit === 'number' && options.limit > 0) {
      state.limit = options.limit | 0;
    }
  }

  function push(indices, before, after) {
    state.undo.push({
      indices: indices || null,
      before: before || null,
      after: after || null,
    });
    if (state.limit > 0 && state.undo.length > state.limit) {
      state.undo.shift();
    }
    state.redo.length = 0;
    return state.undo.length;
  }

  function undo() {
    if (!state.undo.length) {
      return null;
    }
    const entry = state.undo.pop();
    state.redo.push(entry);
    return entry;
  }

  function redo() {
    if (!state.redo.length) {
      return null;
    }
    const entry = state.redo.pop();
    state.undo.push(entry);
    return entry;
  }

  function clear() {
    state.undo.length = 0;
    state.redo.length = 0;
  }

  function serialize(encoder) {
    const encode = typeof encoder === 'function'
      ? encoder
      : (value) => value;
    return {
      undo: serializeStack(state.undo, encode),
      redo: serializeStack(state.redo, encode),
    };
  }

  function serializeStack(stack, encode) {
    const result = [];
    const start = Math.max(0, stack.length - state.limit);
    for (let i = start; i < stack.length; i += 1) {
      const entry = stack[i] || {};
      result.push({
        indices: encode(entry.indices || new Uint32Array()),
        before: encode(entry.before || new Uint32Array()),
        after: encode(entry.after || new Uint32Array()),
      });
    }
    return result;
  }

  function restore(serialized, decoder, expectedLength) {
    clear();
    if (!serialized || typeof serialized !== 'object') {
      return;
    }
    const decode = typeof decoder === 'function'
      ? decoder
      : (value) => value;
    fillStack(state.undo, serialized.undo, decode, expectedLength);
    fillStack(state.redo, serialized.redo, decode, expectedLength);
  }

  function fillStack(target, serialized, decode, expectedLength) {
    if (!Array.isArray(serialized)) {
      return;
    }
    const start = Math.max(0, serialized.length - state.limit);
    for (let i = start; i < serialized.length; i += 1) {
      const entry = serialized[i] || {};
      target.push({
        indices: decode(entry.indices, expectedLength),
        before: decode(entry.before),
        after: decode(entry.after),
      });
    }
  }

  function getUndoCount() {
    return state.undo.length;
  }

  function getRedoCount() {
    return state.redo.length;
  }

  function getUndoStack() {
    return state.undo;
  }

  function getRedoStack() {
    return state.redo;
  }

  global.OmniHistory = Object.assign({}, global.OmniHistory, {
    init,
    push,
    undo,
    redo,
    clear,
    serialize,
    restore,
    getUndoCount,
    getRedoCount,
    getUndoStack,
    getRedoStack,
    getLimit() {
      return state.limit;
    },
    setLimit(limit) {
      if (typeof limit === 'number' && limit > 0) {
        state.limit = limit | 0;
      }
    },
  });
})(typeof window !== 'undefined' ? window : globalThis);
