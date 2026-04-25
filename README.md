# Omnipose Refactor (WIP)

![coverage](badges/coverage.svg)

This folder hosts a clean, minimal re-implementation of Omnipose with two separable packages:

- `omnipose`: the core Omnipose package (ND-first, backwards-compatible API)
- `omnitools`: shared utilities (plotting, file management, transforms)

The goal is to preserve all current model behaviors and outputs while simplifying structure and removing unused Cellpose code paths.

## Goals
- Preserve model logic and argument semantics for backwards compatibility.
- Ensure identical outputs for existing models.
- Prioritize ND implementations; port 2D-only code only for outline plotting utilities.
- Separate generic plotting/file/transforms into `omnitools`.
- Keep tests and GUI functional.
- Avoid nested class overrides (e.g., `UnetModel`/`CellposeModel`/`Cellpose`).

## Progress
- [x] Define target package structure and module mapping.
- [x] Create parity test harness for side-by-side comparisons.
- [ ] Identify and extract shared utilities into `omnitools`.
- [ ] Port ND-centric core and model definitions.
- [ ] Migrate GUI bindings to new package entrypoints.
- [ ] Verify model output parity and update tests.

## Module Mapping (Draft)
Goal: keep ND-first logic intact and avoid behavior changes. This is a starting map.

| Current module | Target module | Notes |
| --- | --- | --- |
| `src/omnipose/core.py` | `refactor/src/omnipose/core.py` | ND-centric core + model logic; preserve function signatures |
| `src/omnipose/data.py` | `refactor/src/omnipose/data.py` | Data I/O and preprocessing; confirm ND handling |
| `src/omnipose/loss.py` | `refactor/src/omnipose/loss.py` | Model losses; keep outputs identical |
| `src/omnipose/gpu.py` | `refactor/src/omnipose/gpu.py` | Device selection/helpers |
| `src/omnipose/measure.py` | `refactor/src/omnipose/measure.py` | Metrics/measurements; ND-first |
| `src/omnipose/stacks.py` | `refactor/src/omnipose/stacks.py` | Volume stack helpers |
| `src/omnipose/cli.py` | `refactor/src/omnipose/cli.py` | CLI entry; keep flags/args |
| `src/omnipose/__main__.py` | `refactor/src/omnipose/__main__.py` | CLI entrypoint |
| `src/omnipose/dependencies.py` | `refactor/src/omnipose/dependencies.py` | Dependency checks (keep lazy imports) |
| `src/omnipose/logger.py` | `refactor/src/omnipose/logger.py` | Logging setup |
| `src/omnipose/profiling.py` | `refactor/src/omnipose/profiling.py` | Profiling utilities |
| `src/omnipose/color.py` | `refactor/src/omnitools/color.py` | Generic color utilities |
| `src/omnipose/plot.py` | `refactor/src/omnitools/plot.py` | Plotting utils; retain 2D outline plotting |
| `src/omnipose/utils.py` | `refactor/src/omnitools/utils.py` | Generic helpers (file mgmt, transforms) |
| `src/omnipose/transforms.py` | `refactor/src/omnitools/transforms.py` | Placeholder for shared transforms |
| `src/omnipose/misc.py` | `refactor/src/omnitools/misc.py` | Dev helpers (may move subset) |
| `src/omnipose/experimental_plot/*` | `refactor/src/omnitools/experimental_plot/*` | Keep optional/experimental |

## ND Priority Notes
- ND implementations are the default; avoid porting 2D-only logic except outline plotting utilities.
- Any 2D-only modules should be isolated under `omnitools` if still needed.
- Preserve all function signatures and argument semantics for model logic and preprocessing.

## Initial `omnitools` API Targets (Draft)
- `omnitools.color`: `sinebow`
- `omnitools.plot`: `setup`, 2D outline plotting helpers
- `omnitools.utils`: `get_module`, `safe_divide`, `rescale`, `find_files`, `getname`, `to_16_bit`
- `omnitools.transforms`: shared ND transforms (placeholder)
- `omnitools.misc`: ND grid/label helpers (subset from `misc.py`)

## Notes
- Temporary package name: `omnipose` to allow co-install with existing `omnipose`.
- Parity tests live in `refactor/tests_parity/`.

## Parity Runs
- `Sample000033.png` with `bact_phase_affinity`: masks identical (CPU).
