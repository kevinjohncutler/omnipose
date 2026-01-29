# Determinism note (not in toctree)

CUDA strict determinism is blocked by:
- `reflection_pad2d_backward_cuda` (no deterministic implementation).

Workaround used:
- `torch.use_deterministic_algorithms(True, warn_only=True)` when `--deterministic`
  is used with GPU.

Observed outcome (kevin-tower, CUDA, batch size 5, 1 epoch):
- Weights were identical across repeated runs within each repo.
- Weights were identical across omnipose vs omnirefactor.

This file is intentionally not included in any Sphinx toctree.
