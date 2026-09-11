# Loss / flow / GT verification — 2D vs 3D

Confirms that omnipose's distance field, flow ground truth, divergence and every
loss term behave identically (and correctly) in 2D and 3D, using two cubes /
squares that share exactly one face / edge.

## Files
- `_pipeline.py` — builds the test object, runs the *shipped* GT pipeline
  (`masks_to_flows_batch` + `batch_labels`), constructs predicted `y` for a set
  of controlled error scenarios, and computes per-voxel (pre-reduction) maps of
  every loss term.
- `_validate.py` — gate: calls the shipped `omnipose.core.loss.loss` and asserts
  each per-voxel map reduces to the authoritative scalar (worst rel err ~2e-7).
- `make_figures.py` — writes `gt_fields_{2,3}d.png` and `loss_maps_{2,3}d.png`.
- `generate_report.py` — writes `REPORT.md` (GT checks, per-voxel analytic
  verification, expected-vs-computed scalar tables, label-equivariance table).
- `four_cubes.py` — four cubes/squares in a 2x2 grid (labels 1..4). Proves label
  equivariance: a symmetric perturbation lights all four tiles with identical
  per-label loss; a label-1-only perturbation lights only the label-1 tile.
  Writes `four_cubes_{2,3}d.png`.

## Reproduce
```
# run from a Python environment with omnipose installed
cd figures/loss_verification_3d
python _validate.py        # numeric gate
python make_figures.py     # figures
python generate_report.py  # REPORT.md
```

## Scenarios (predicted error injected on top of perfect GT)
- `perfect` — every loss ~0 (sanity).
- `flow_zero_cube2` — flow set to 0 in cube 2.
- `flow_flip_cube2` — flow negated in cube 2 (SSL/norm are sign-blind; the
  flow-following losses lossE/lossA and lossDC catch it).
- `dist_offset` — constant +2 added to predicted distance inside cells
  (dist_loss fires; the derivative term bd_loss only at the cell rim).
- `dist_bump` — localized gaussian added to predicted distance.
