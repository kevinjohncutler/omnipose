# Parity Tests

This folder is for side-by-side tests that compare current `omnipose` behavior to the new `omnirefactor` package.

Planned coverage:
- Model forward pass output parity (same inputs/weights).
- Pre/post-processing parity for ND pipelines.
- CLI/API argument compatibility checks.

## Current Script
Run the first segmentation parity check (CPU):

```bash
python refactor/tests_parity/run_segmentation_parity.py
```
