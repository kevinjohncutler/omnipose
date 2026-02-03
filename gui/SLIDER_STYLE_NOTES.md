# Slider / Toggle / Segmented Control Notes

These are the styling + math rules that keep the controls visually consistent and aligned.

## Core variables (CSS)
- `--slider-track-height`: base height for all tracks.
- `--slider-track-radius`: should be `track-height / 2`.
- `--slider-track-border`: gray outline for tracks.
- `--segmented-inset`: how much the accent pill reveals the track outline.
- `--control-inset`: should equal `--segmented-inset`.
- `--control-inset-radius`: `calc(var(--slider-track-radius) - var(--control-inset))`.
- `--toggle-width`: fixed width (currently `42px`).

## Segmented controls (2/3/4 stops)
- Use `.kernel-toggle.segmented-toggle` for **all** segmented selectors (tools, label style, segmentation mode).
- The accent fill is drawn on `.kernel-option::before` with:
  - `inset: var(--control-inset)`
  - `border-radius: var(--control-inset-radius)`
- Do **not** set `background` directly on `.kernel-option[data-active='true']` or you will lose the outline reveal.
- If the outline disappears, check for selector-specific overrides (e.g. `.seg-mode-toggle` or `.mask-style-toggle`).

## Sliders
- The accent fill is **one element**: `.slider-track::before`.
- It should be clipped, not offset by toggle values.
- Correct fill rule:
  - `left: 0; top: 0; bottom: 0;`
  - `width: calc(var(--slider-fill-px, 0px) + 2 * var(--slider-track-radius));`
  - `border-radius: var(--slider-track-radius);`
  - `clip-path: inset(var(--control-inset) round var(--control-inset-radius));`
- **Never** use toggle-specific vars (e.g. `--toggle-knob-offset`) inside slider rules.

## Slider JS math (web/app.js)
- Knob center should be `trackRadius + fillPx`.
- `trackRadius` is derived from the track height, **not** the inset radius.
- The fill clipping handles the inset; JS should not shift the knob for inset.

## Toggles
- Track uses gray outline (`--slider-track-border`).
- Accent fill is `toggle-switch::before` with inset/radius:
  - `inset: var(--control-inset)`
  - `border-radius: var(--control-inset-radius)`
- Knob center alignment:
  - `left: calc(var(--slider-track-radius) - (var(--slider-knob-size) / 2))`
  - checked translate: `calc(var(--toggle-width) - (2 * var(--slider-track-radius)))`

## Common pitfalls
- **Self-referential CSS var** breaks layout (e.g. `--toggle-width: var(--toggle-width);`).
- Segmented controls will look inconsistent if any selector adds its own `background` for the active state.
- Slider fill offsets if any toggle variable leaks into slider rules.

## Quick checklist for new sliders/toggles
- Uses shared vars (`--control-inset`, `--control-inset-radius`).
- Fill drawn via `::before` with clip-path (sliders) or inset (segmented/toggles).
- No selector-specific overrides on active background unless necessary.
