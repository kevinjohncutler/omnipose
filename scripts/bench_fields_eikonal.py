import os
import time
import statistics as stats
from typing import List

import torch

from omnirefactor.core import fields


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _make_inputs(device):
    # Small but non-trivial shapes; adjust via env if needed.
    n = int(os.getenv("EIKONAL_N", "128"))
    k = int(os.getenv("EIKONAL_K", "8"))

    Tneigh = torch.rand((k, n), device=device)
    r = torch.arange(n, device=device)
    d = torch.tensor(2, device=device)
    # index_list[0] is unused in function; keep placeholder
    index_list = [torch.tensor([0, 1], device=device)]
    index_list.append(torch.tensor([0, 1], device=device))
    factors = torch.tensor([0.0, 1.0], device=device)
    return Tneigh, r, d, index_list, factors


def _bench(fn, args, iters=100, warmup=20, label=""):
    for _ in range(warmup):
        _ = fn(*args)
    _sync(args[0].device)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = fn(*args)
        _sync(args[0].device)
        times.append((time.perf_counter() - t0) * 1000)
    mean = stats.mean(times)
    stdev = stats.pstdev(times)
    print(f"{label:>16}: {mean:.3f} ms ± {stdev:.3f} (n={iters})")


def main():
    device_name = os.getenv("DEVICE", "cpu")
    device = torch.device(device_name)

    print("torch", torch.__version__)
    print("device", device)

    args = _make_inputs(device)

    def eikonal_update_torch(
        Tneigh: torch.Tensor,
        r: torch.Tensor,
        d: torch.Tensor,
        index_list: List[torch.Tensor],
        factors: torch.Tensor,
    ) -> torch.Tensor:
        geometric = 1
        phi_total = torch.ones_like(Tneigh[0, :]) if geometric else torch.zeros_like(Tneigh[0, :])

        n = len(factors) - 1

        for inds, f, fsq in zip(index_list[1:], factors[1:], factors[1:] ** 2):
            npair = len(inds) // 2

            mins = torch.stack([
                torch.minimum(Tneigh[inds[i], :], Tneigh[inds[-(i + 1)], :])
                for i in range(npair)
            ])

            update = fields.update_torch(mins, f, fsq)

            if geometric:
                phi_total *= update
            else:
                phi_total += update

        phi_total = torch.pow(phi_total, 1 / n) if geometric else phi_total / n
        return phi_total

    def eikonal_update_torch_vec(
        Tneigh: torch.Tensor,
        r: torch.Tensor,
        d: torch.Tensor,
        index_list: List[torch.Tensor],
        factors: torch.Tensor,
    ) -> torch.Tensor:
        geometric = 1
        phi_total = torch.ones_like(Tneigh[0, :]) if geometric else torch.zeros_like(Tneigh[0, :])

        n = len(factors) - 1

        for inds, f, fsq in zip(index_list[1:], factors[1:], factors[1:] ** 2):
            npair = len(inds) // 2
            left = inds[:npair]
            right = torch.flip(inds, dims=[0])[:npair]
            mins = torch.minimum(Tneigh[left], Tneigh[right])

            update = fields.update_torch(mins, f, fsq)

            if geometric:
                phi_total *= update
            else:
                phi_total += update

        phi_total = torch.pow(phi_total, 1 / n) if geometric else phi_total / n
        return phi_total

    # Base Python function
    _bench(eikonal_update_torch, args, label="python")
    _bench(eikonal_update_torch_vec, args, label="python vec")

    # Correctness check against python baseline
    base = eikonal_update_torch(*args)
    vec = eikonal_update_torch_vec(*args)
    max_abs = (base - vec).abs().max().item()
    print(f"match python vs vec: max_abs={max_abs:.6g}")
    assert torch.allclose(base, vec, atol=1e-6, rtol=1e-6)

    # TorchScript
    try:
        scripted = torch.jit.script(eikonal_update_torch)
        _bench(scripted, args, label="torchscript")
        ts_out = scripted(*args)
        assert torch.allclose(base, ts_out, atol=1e-6, rtol=1e-6)
    except Exception as e:
        print("torchscript failed:", type(e).__name__, e)
    try:
        scripted_vec = torch.jit.script(eikonal_update_torch_vec)
        _bench(scripted_vec, args, label="ts vec")
        ts_vec_out = scripted_vec(*args)
        assert torch.allclose(base, ts_vec_out, atol=1e-6, rtol=1e-6)
    except Exception as e:
        print("torchscript vec failed:", type(e).__name__, e)

    # torch.compile (default)
    if hasattr(torch, "compile"):
        try:
            compiled = torch.compile(eikonal_update_torch)
            _bench(compiled, args, label="torch.compile")
            comp_out = compiled(*args)
            assert torch.allclose(base, comp_out, atol=1e-6, rtol=1e-6)
        except Exception as e:
            print("torch.compile failed:", type(e).__name__, e)
        try:
            compiled_vec = torch.compile(eikonal_update_torch_vec)
            _bench(compiled_vec, args, label="compile vec")
            comp_vec_out = compiled_vec(*args)
            assert torch.allclose(base, comp_vec_out, atol=1e-6, rtol=1e-6)
        except Exception as e:
            print("torch.compile vec failed:", type(e).__name__, e)

        try:
            compiled_ro = torch.compile(eikonal_update_torch, mode="reduce-overhead")
            _bench(compiled_ro, args, label="compile+reduce")
            comp_ro_out = compiled_ro(*args)
            assert torch.allclose(base, comp_ro_out, atol=1e-6, rtol=1e-6)
        except Exception as e:
            print("torch.compile(reduce) failed:", type(e).__name__, e)
        try:
            compiled_ro_vec = torch.compile(eikonal_update_torch_vec, mode="reduce-overhead")
            _bench(compiled_ro_vec, args, label="reduce vec")
            comp_ro_vec_out = compiled_ro_vec(*args)
            assert torch.allclose(base, comp_ro_vec_out, atol=1e-6, rtol=1e-6)
        except Exception as e:
            print("torch.compile(reduce vec) failed:", type(e).__name__, e)


if __name__ == "__main__":
    main()
