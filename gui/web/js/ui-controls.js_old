(function initOmniControls(global) {
  'use strict';

  const sliderRegistry = new Map();
  const dropdownRegistry = new Map();
  let dropdownOpenId = null;

  function clamp(value, min, max) {
    if (!Number.isFinite(value)) {
      return min;
    }
    return Math.min(max, Math.max(min, value));
  }

  function valueToPercent(input) {
    const min = Number(input.min || 0);
    const max = Number(input.max || 1);
    const span = max - min;
    if (!Number.isFinite(span) || span === 0) {
      return 0;
    }
    const value = Number(input.value || min);
    return clamp((value - min) / span, 0, 1);
  }

  function percentToValue(percent, input) {
    const min = Number(input.min || 0);
    const max = Number(input.max || 1);
    const span = max - min;
    const raw = min + clamp(percent, 0, 1) * span;
    const step = Number(input.step || '1');
    if (!Number.isFinite(step) || step <= 0) {
      return clamp(raw, min, max);
    }
    const snapped = Math.round((raw - min) / step) * step + min;
    const precision = (step.toString().split('.')[1] || '').length;
    const factor = 10 ** precision;
    return clamp(Math.round(snapped * factor) / factor, min, max);
  }

  function pointerPercent(evt, container) {
    const rect = container.getBoundingClientRect();
    if (rect.width <= 0) {
      return 0;
    }
    const ratio = (evt.clientX - rect.left) / rect.width;
    return clamp(ratio, 0, 1);
  }

  function registerSlider(root) {
    const id = root.dataset.sliderId || root.dataset.slider || root.id;
    if (!id) {
      return;
    }
    const type = (root.dataset.sliderType || 'single').toLowerCase();
    const inputs = Array.from(root.querySelectorAll('input[type="range"]'));
    if (!inputs.length) {
      return;
    }
    if (type === 'dual' && inputs.length < 2) {
      console.warn(`slider ${id} configured as dual but only one range input found`);
      return;
    }

    root.innerHTML = '';
    const track = document.createElement('div');
    track.className = 'slider-track';
    root.appendChild(track);
    const fill = document.createElement('div');
    fill.className = 'slider-fill';
    track.appendChild(fill);
    const thumbs = inputs.map(() => {
      const thumb = document.createElement('div');
      thumb.className = 'slider-thumb';
      track.appendChild(thumb);
      return thumb;
    });

    const entry = {
      id,
      root,
      type,
      inputs,
      track,
      fill,
      thumbs,
      activePointer: null,
      activeThumb: null,
    };

    const apply = () => {
      if (entry.type === 'dual') {
        const minInput = entry.inputs[0];
        const maxInput = entry.inputs[1];
        const minPercent = valueToPercent(minInput);
        const maxPercent = valueToPercent(maxInput);
        entry.fill.style.left = `${minPercent * 100}%`;
        entry.fill.style.width = `${Math.max(0, maxPercent - minPercent) * 100}%`;
        entry.thumbs[0].style.left = `${minPercent * 100}%`;
        entry.thumbs[1].style.left = `${maxPercent * 100}%`;
      } else {
        const input = entry.inputs[0];
        const percent = valueToPercent(input);
        entry.fill.style.width = `${percent * 100}%`;
        entry.thumbs[0].style.left = `${percent * 100}%`;
      }
    };

    const setValueFromPercent = (index, percent) => {
      const input = entry.inputs[index];
      if (!input) {
        return;
      }
      const value = percentToValue(percent, input);
      if (entry.type === 'dual') {
        const otherIndex = index === 0 ? 1 : 0;
        const otherInput = entry.inputs[otherIndex];
        if (otherInput) {
          const otherValue = Number(otherInput.value || otherInput.min || 0);
          if (index === 0 && value > otherValue) {
            input.value = otherValue;
          } else if (index === 1 && value < otherValue) {
            input.value = otherValue;
          } else {
            input.value = String(value);
          }
        }
      } else {
        input.value = String(value);
      }
      input.dispatchEvent(new Event('input', { bubbles: true }));
      apply();
    };

    const pickThumb = (percent) => {
      if (entry.type !== 'dual') {
        return 0;
      }
      const distances = entry.inputs.map((input) => Math.abs(percent - valueToPercent(input)));
      let bestIndex = 0;
      let bestDistance = distances[0];
      for (let i = 1; i < distances.length; i += 1) {
        if (distances[i] < bestDistance) {
          bestDistance = distances[i];
          bestIndex = i;
        }
      }
      return bestIndex;
    };

    const onPointerDown = (evt) => {
      evt.preventDefault();
      const percent = pointerPercent(evt, entry.root);
      const thumbIndex = entry.type === 'dual' ? pickThumb(percent) : 0;
      entry.activePointer = evt.pointerId;
      entry.activeThumb = thumbIndex;
      entry.root.setPointerCapture(entry.activePointer);
      entry.root.dataset.active = 'true';
      const targetInput = entry.inputs[thumbIndex];
      if (targetInput) {
        targetInput.focus();
      }
      setValueFromPercent(thumbIndex, percent);
    };

    const onPointerMove = (evt) => {
      if (entry.activePointer === null || evt.pointerId !== entry.activePointer) {
        return;
      }
      const percent = pointerPercent(evt, entry.root);
      setValueFromPercent(entry.activeThumb ?? 0, percent);
    };

    const onPointerRelease = (evt) => {
      if (entry.activePointer === null || evt.pointerId !== entry.activePointer) {
        return;
      }
      try {
        entry.root.releasePointerCapture(entry.activePointer);
      } catch (_) {
        /* ignore */
      }
      const percent = pointerPercent(evt, entry.root);
      setValueFromPercent(entry.activeThumb ?? 0, percent);
      entry.activePointer = null;
      entry.activeThumb = null;
      entry.root.dataset.active = 'false';
    };

    entry.root.addEventListener('pointerdown', onPointerDown);
    entry.root.addEventListener('pointermove', onPointerMove);
    entry.root.addEventListener('pointerup', onPointerRelease);
    entry.root.addEventListener('pointercancel', onPointerRelease);

    inputs.forEach((input) => {
      input.addEventListener('focus', () => {
        entry.root.dataset.focused = 'true';
      });
      input.addEventListener('blur', () => {
        entry.root.dataset.focused = 'false';
      });
      input.addEventListener('input', apply);
      input.addEventListener('change', apply);
    });

    entry.apply = apply;
    apply();
    sliderRegistry.set(id, entry);
  }

  function refreshSlider(id) {
    const entry = sliderRegistry.get(id);
    if (entry && typeof entry.apply === 'function') {
      entry.apply();
    }
  }

  function attachNumberInputStepper(input, onAdjust) {
    if (!input || typeof onAdjust !== 'function') {
      return;
    }
    input.addEventListener('keydown', (evt) => {
      if (evt.key !== 'ArrowUp' && evt.key !== 'ArrowDown') {
        return;
      }
      evt.preventDefault();
      const base = Number(input.step || '1');
      const step = Number.isFinite(base) && base > 0 ? base : 1;
      const factor = evt.shiftKey ? 5 : 1;
      const direction = evt.key === 'ArrowUp' ? 1 : -1;
      onAdjust(step * factor * direction);
    });
  }

  function closeDropdown(entry) {
    if (!entry) {
      return;
    }
    entry.root.dataset.open = 'false';
    entry.button.setAttribute('aria-expanded', 'false');
    if (entry.menu) {
      entry.menu.setAttribute('aria-hidden', 'true');
      entry.menu.scrollTop = 0;
    }
    dropdownOpenId = null;
  }

  function openDropdown(entry) {
    if (!entry) {
      return;
    }
    if (dropdownOpenId && dropdownOpenId !== entry.id) {
      closeDropdown(dropdownRegistry.get(dropdownOpenId));
    }
    entry.root.dataset.open = 'true';
    positionDropdown(entry);
    entry.button.setAttribute('aria-expanded', 'true');
    if (entry.menu) {
      entry.menu.setAttribute('aria-hidden', 'false');
    }
    dropdownOpenId = entry.id;
  }

  function toggleDropdown(entry) {
    if (!entry) {
      return;
    }
    const isOpen = entry.root.dataset.open === 'true';
    if (isOpen) {
      closeDropdown(entry);
    } else {
      openDropdown(entry);
    }
  }

  function positionDropdown(entry) {
    if (!entry || !entry.menu) {
      return;
    }
    entry.menu.style.minWidth = '100%';
  }

  function registerDropdown(root) {
    const select = root.querySelector('select');
    if (!select) {
      return;
    }
    const id = root.dataset.dropdownId || select.id || `dropdown-${dropdownRegistry.size}`;
    root.dataset.dropdownId = id;
    root.dataset.open = root.dataset.open || 'false';

    const originalOptions = Array.from(select.options).map((opt) => ({
      value: opt.value,
      label: opt.textContent || opt.value,
      disabled: opt.disabled,
    }));

    select.classList.add('dropdown-input');
    root.innerHTML = '';
    root.appendChild(select);

    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'dropdown-toggle';
    button.setAttribute('aria-haspopup', 'listbox');
    button.setAttribute('aria-expanded', 'false');
    const menu = document.createElement('div');
    menu.className = 'dropdown-menu';
    menu.setAttribute('role', 'listbox');
    menu.setAttribute('aria-hidden', 'true');
    menu.id = `${id}-menu`;
    button.setAttribute('aria-controls', menu.id);
    root.appendChild(button);
    const menuWrapper = document.createElement('div');
    menuWrapper.className = 'dropdown-menu-wrap';
    menuWrapper.appendChild(menu);
    root.appendChild(menuWrapper);

    const entry = {
      id,
      root,
      select,
      button,
      menu,
      menuWrapper,
      options: originalOptions,
    };

    const applySelection = () => {
      const selectedOption = select.options[select.selectedIndex];
      button.textContent = selectedOption ? selectedOption.textContent : 'Select';
      menu.querySelectorAll('.dropdown-option').forEach((child) => {
        child.dataset.selected = child.dataset.value === select.value ? 'true' : 'false';
      });
    };

    const buildMenu = () => {
      menu.innerHTML = '';
      entry.options.forEach((opt) => {
        const item = document.createElement('div');
        item.className = 'dropdown-option';
        item.dataset.value = opt.value;
        item.textContent = opt.label;
        item.setAttribute('role', 'option');
        if (opt.disabled) {
          item.setAttribute('aria-disabled', 'true');
          item.style.opacity = '0.45';
          item.style.pointerEvents = 'none';
        }
        item.addEventListener('pointerdown', (evt) => {
          evt.preventDefault();
          if (opt.disabled) {
            return;
          }
          select.value = opt.value;
          select.dispatchEvent(new Event('change', { bubbles: true }));
          applySelection();
          closeDropdown(entry);
        });
        menu.appendChild(item);
      });
      applySelection();
    };

    button.addEventListener('click', () => {
      toggleDropdown(entry);
    });

    select.addEventListener('change', () => {
      applySelection();
    });

    buildMenu();
    entry.applySelection = applySelection;
    entry.buildMenu = buildMenu;
    positionDropdown(entry);
    dropdownRegistry.set(id, entry);
  }

  function refreshDropdown(id) {
    const entry = dropdownRegistry.get(id);
    if (entry && typeof entry.applySelection === 'function') {
      entry.applySelection();
    }
  }

  function closeOpenDropdown() {
    if (!dropdownOpenId) {
      return;
    }
    const entry = dropdownRegistry.get(dropdownOpenId);
    if (entry) {
      closeDropdown(entry);
    } else {
      dropdownOpenId = null;
    }
  }

  function positionOpenDropdown() {
    if (!dropdownOpenId) {
      return;
    }
    const entry = dropdownRegistry.get(dropdownOpenId);
    if (entry) {
      positionDropdown(entry);
    } else {
      dropdownOpenId = null;
    }
  }

  function hasOpenDropdown() {
    if (!dropdownOpenId) {
      return false;
    }
    const entry = dropdownRegistry.get(dropdownOpenId);
    if (!entry) {
      dropdownOpenId = null;
      return false;
    }
    return true;
  }

  document.addEventListener('pointerdown', (evt) => {
    if (!dropdownOpenId) {
      return;
    }
    const entry = dropdownRegistry.get(dropdownOpenId);
    if (!entry) {
      dropdownOpenId = null;
      return;
    }
    if (!entry.root.contains(evt.target)) {
      closeDropdown(entry);
    }
  });

  const api = {
    registerSlider,
    refreshSlider,
    attachNumberInputStepper,
    registerDropdown,
    refreshDropdown,
    closeOpenDropdown,
    positionOpenDropdown,
    hasOpenDropdown,
  };

  global.OmniControls = Object.assign({}, global.OmniControls, api);
})(typeof window !== 'undefined' ? window : globalThis);
