"""API route handlers and DebugAPI for the Omnipose GUI server."""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from .assets import WEB_DIR, append_gui_log
from .segmentation import _SEGMENTER, run_segmentation, run_mask_update
from .session import SESSION_MANAGER, SESSION_COOKIE_NAME, SessionState

# WebGL log path
WEBGL_LOG_PATH = Path("/tmp/webgl_log.txt")
try:
    WEBGL_LOG_PATH.write_text("", encoding="utf-8")
except OSError:
    pass


class DebugAPI:
    """API class for debugging and advanced operations."""

    def __init__(self, log_path: Path | None = None) -> None:
        self._log_path = log_path or WEBGL_LOG_PATH

    def log(self, message: str) -> None:
        message = str(message)
        try:
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(message + "\n")
        except OSError:
            return

    def segment(self, settings: Mapping[str, Any] | None = None) -> dict[str, object]:
        mode = None
        if isinstance(settings, Mapping):
            mode = settings.get("mode")
        state = SESSION_MANAGER.get_or_create(None)
        if mode == "recompute":
            return run_mask_update(settings, state=state)
        return run_segmentation(settings, state=state)

    def resegment(self, settings: Mapping[str, Any] | None = None) -> dict[str, object]:
        state = SESSION_MANAGER.get_or_create(None)
        return run_mask_update(settings, state=state)

    def get_ncolor(self) -> dict[str, object]:
        """Return only the current N-color mask from the cached segmentation, if available."""
        try:
            ncolor_mask = _SEGMENTER.get_ncolor_mask()
        except Exception as exc:
            return {"error": str(exc)}
        if ncolor_mask is None:
            return {"nColorMask": None}
        encoded = base64.b64encode(np.ascontiguousarray(ncolor_mask).tobytes()).decode("ascii")
        try:
            print(f"[api] get_ncolor groups={int(np.unique(ncolor_mask[ncolor_mask>0]).size)}")
        except Exception:
            pass
        return {"nColorMask": encoded}

    def relabel_from_affinity(self, payload: Mapping[str, Any]) -> dict[str, object]:
        try:
            mask_b64 = payload.get("mask")
            width = int(payload.get("width"))
            height = int(payload.get("height"))
        except Exception:
            return {"error": "invalid payload"}
        if not mask_b64 or width <= 0 or height <= 0:
            return {"error": "missing mask/shape"}
        try:
            raw = base64.b64decode(mask_b64)
            arr = np.frombuffer(raw, dtype=np.uint32)
            if arr.size != width * height:
                return {"error": "mask size mismatch"}
            mask = arr.reshape((height, width)).astype(np.int32, copy=False)
        except Exception as exc:
            return {"error": f"decode failed: {exc}"}

        ag = payload.get("affinityGraph")
        if not isinstance(ag, Mapping):
            return {"error": "affinityGraph required"}
        try:
            w = int(ag.get("width"))
            h = int(ag.get("height"))
            steps_list = ag.get("steps")
            enc = ag.get("encoded")
            if w <= 0 or h <= 0:
                return {"error": "invalid affinityGraph size"}
            if not isinstance(steps_list, list) or not isinstance(enc, str):
                return {"error": "invalid affinityGraph payload"}
            raw_aff = base64.b64decode(enc)
            arr_aff = np.frombuffer(raw_aff, dtype=np.uint8)
            s = len(steps_list)
            if arr_aff.size != s * h * w:
                return {"error": "affinityGraph data size mismatch"}
            spatial = arr_aff.reshape((s, h, w))
            steps = np.asarray(steps_list, dtype=np.int16)
        except Exception as exc:
            return {"error": f"invalid affinityGraph: {exc}"}

        try:
            print(f"[relabel_from_affinity] using provided spatial affinity S={spatial.shape[0]}")
            before = int(np.unique(mask[mask > 0]).size)
            new_labels = _SEGMENTER.relabel_from_affinity(mask, spatial, steps)
            after = int(np.unique(new_labels[new_labels > 0]).size)
            print(f"[relabel_from_affinity] labels_before={before} labels_after={after}")
        except Exception as exc:
            import traceback
            print("[relabel_from_affinity] EXCEPTION:", file=sys.stderr)
            traceback.print_exc()
            return {"error": f"{type(exc).__name__}: {exc}"}

        cache = _SEGMENTER._cache or {}
        cache["mask"] = np.ascontiguousarray(new_labels.astype(np.uint32, copy=False))
        cache["ncolor_mask"] = _SEGMENTER.get_ncolor_mask()
        try:
            cache["affinity_graph"] = (
                np.ascontiguousarray(steps.astype(np.int8, copy=False)),
                np.ascontiguousarray(spatial.astype(np.uint8, copy=False)),
            )
        except Exception:
            pass
        _SEGMENTER._cache = cache
        encoded_mask = base64.b64encode(cache["mask"].tobytes()).decode("ascii")
        ncolor_mask = cache.get("ncolor_mask")
        encoded_ncolor = None
        if ncolor_mask is not None:
            encoded_ncolor = base64.b64encode(np.ascontiguousarray(ncolor_mask).tobytes()).decode("ascii")
        affinity_payload = _SEGMENTER.get_affinity_graph_payload()
        return {
            "mask": encoded_mask,
            "nColorMask": encoded_ncolor,
            "affinityGraph": affinity_payload,
            "width": int(width),
            "height": int(height),
        }

    def ncolor_from_mask(self, payload: Mapping[str, Any]) -> dict[str, object]:
        try:
            mask_b64 = payload.get("mask")
            width = int(payload.get("width"))
            height = int(payload.get("height"))
        except Exception:
            return {"error": "invalid payload"}
        if not mask_b64 or width <= 0 or height <= 0:
            return {"error": "missing mask/shape"}
        try:
            raw = base64.b64decode(mask_b64)
            arr = np.frombuffer(raw, dtype=np.uint32)
            if arr.size != width * height:
                return {"error": "mask size mismatch"}
            mask = arr.reshape((height, width)).astype(np.int32, copy=False)
        except Exception as exc:
            return {"error": f"decode failed: {exc}"}
        expand = bool(payload.get("expand", True))
        ncm = _SEGMENTER._compute_ncolor_mask(mask, expand=expand)
        if ncm is None:
            return {"nColorMask": None}
        try:
            ngroups = int(np.unique(ncm[ncm > 0]).size)
            print(f"[ncolor-from-mask] groups={ngroups}")
        except Exception:
            pass
        encoded = base64.b64encode(np.ascontiguousarray(ncm.astype(np.uint32, copy=False)).tobytes()).decode("ascii")
        return {"nColorMask": encoded, "width": int(width), "height": int(height)}

    def format_labels(self, payload: Mapping[str, Any]) -> dict[str, object]:
        try:
            mask_b64 = payload.get("mask")
            width = int(payload.get("width"))
            height = int(payload.get("height"))
        except Exception:
            return {"error": "invalid payload"}
        if not mask_b64 or width <= 0 or height <= 0:
            return {"error": "missing mask/shape"}
        try:
            raw = base64.b64decode(mask_b64)
            arr = np.frombuffer(raw, dtype=np.uint32)
            if arr.size != width * height:
                return {"error": "mask size mismatch"}
            mask = arr.reshape((height, width)).astype(np.int32, copy=False)
        except Exception as exc:
            return {"error": f"decode failed: {exc}"}
        try:
            import ncolor  # type: ignore
            formatted = ncolor.format_labels(mask, clean=False, min_area=1, despur=False, verbose=False)
        except Exception as exc:
            return {"error": f"format_labels failed: {exc}"}
        encoded = base64.b64encode(np.ascontiguousarray(formatted.astype(np.uint32, copy=False)).tobytes()).decode("ascii")
        return {"mask": encoded, "width": int(width), "height": int(height)}


def _choose_path_osascript(kind: str) -> str | None:
    """Use macOS osascript to choose a file or folder."""
    try:
        import subprocess
    except Exception:
        return None
    if kind == "file":
        script = 'POSIX path of (choose file with prompt "Select image")'
    else:
        script = 'POSIX path of (choose folder with prompt "Select image folder")'
    result = subprocess.run([
        "osascript",
        "-e",
        script,
    ], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    path_value = (result.stdout or "").strip()
    return path_value or None
