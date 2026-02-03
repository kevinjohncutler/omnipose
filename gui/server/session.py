"""Session management for the Omnipose GUI server."""

from __future__ import annotations

import base64
import io
import json
import secrets
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from imageio import v2 as imageio

from .sample_image import (
    get_preload_image_path,
    load_image_uint8,
    get_instance_color_table,
    _ensure_spatial_last,
    _normalize_uint8,
)

SUPPORTED_IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
}

SESSION_COOKIE_NAME = "OMNISESSION"


def _session_path_key(path: Optional[Path]) -> str:
    return str(path.resolve()) if path else "__sample__"


@dataclass
class SessionState:
    session_id: str
    current_path: Optional[Path]
    directory: Optional[Path]
    files: list[Path] = field(default_factory=list)
    saved_states: dict[str, Any] = field(default_factory=dict)
    current_image: Optional[np.ndarray] = None
    image_is_rgb: bool = False
    encoded_image: Optional[str] = None

    def path_key(self, path: Optional[Path] = None) -> str:
        return _session_path_key(path if path is not None else self.current_path)


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def _create_session_unlocked(self) -> SessionState:
        session_id = secrets.token_urlsafe(16)
        initial_path = get_preload_image_path()
        if initial_path and initial_path.exists():
            image, is_rgb = self._load_image_from_path(initial_path)
            directory = initial_path.parent
            files = self._list_directory_images(directory)
        else:
            image = load_image_uint8(as_rgb=True)
            is_rgb = image.ndim == 3 and image.shape[-1] >= 3
            directory = None
            files: list[Path] = []
            initial_path = None
        state = SessionState(
            session_id=session_id,
            current_path=initial_path,
            directory=directory,
            files=files,
            current_image=np.ascontiguousarray(image, dtype=np.uint8),
            image_is_rgb=is_rgb,
            encoded_image=None,
        )
        self._sessions[session_id] = state
        state.encoded_image = self._encode_image(state.current_image, is_rgb=is_rgb)
        return state

    def get_or_create(self, session_id: Optional[str]) -> SessionState:
        with self._lock:
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]
            return self._create_session_unlocked()

    def get(self, session_id: str) -> SessionState:
        with self._lock:
            return self._sessions[session_id]

    def clear_saved_states(self, state: SessionState) -> None:
        with self._lock:
            existing = self._sessions.get(state.session_id)
            if existing:
                existing.saved_states.clear()

    def _load_image_from_path(self, path: Path) -> tuple[np.ndarray, bool]:
        arr = imageio.imread(path)
        arr = _ensure_spatial_last(arr)
        arr = _normalize_uint8(arr)
        is_rgb = arr.ndim == 3 and arr.shape[-1] >= 3
        return arr, is_rgb

    def _list_directory_images(self, directory: Path) -> list[Path]:
        try:
            files = [
                p for p in sorted(directory.iterdir())
                if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
            ]
        except FileNotFoundError:
            files = []
        return files

    def set_image(self, state: SessionState, path: Optional[Path]) -> None:
        if path is not None:
            path = path.expanduser().resolve()
        if path is not None and not path.exists():
            raise FileNotFoundError(path)
        if path is None:
            image = load_image_uint8(as_rgb=True)
            is_rgb = image.ndim == 3 and image.shape[-1] >= 3
            directory = None
            files: list[Path] = []
        else:
            image, is_rgb = self._load_image_from_path(path)
            directory = path.parent
            files = self._list_directory_images(directory)
        state.current_path = path
        state.directory = directory
        state.files = files
        state.current_image = np.ascontiguousarray(image, dtype=np.uint8)
        state.image_is_rgb = is_rgb
        state.encoded_image = self._encode_image(state.current_image, is_rgb=is_rgb)

    def build_config(self, state: SessionState) -> dict[str, Any]:
        image = state.current_image if state.current_image is not None else load_image_uint8(as_rgb=True)
        is_rgb = state.image_is_rgb
        height, width = image.shape[:2]
        if not state.encoded_image:
            state.encoded_image = self._encode_image(image, is_rgb=is_rgb)
        encoded = state.encoded_image
        directory_entries: list[dict[str, Any]] = []
        index = None
        if state.current_path and state.files:
            for i, item in enumerate(state.files):
                is_current = item == state.current_path
                if is_current:
                    index = i
                directory_entries.append(
                    {
                        "name": item.name,
                        "path": str(item),
                        "isCurrent": is_current,
                    },
                )
        config: dict[str, Any] = {
            "sessionId": state.session_id,
            "width": int(width),
            "height": int(height),
            "imageDataUrl": encoded,
            "colorTable": get_instance_color_table().tolist(),
            "maskOpacity": 0.8,
            "maskThreshold": -2.0,
            "flowThreshold": 0.0,
            "cluster": True,
            "affinitySeg": True,
            "imagePath": str(state.current_path) if state.current_path else None,
            "imageName": state.current_path.name if state.current_path else "Sample Image",
            "directoryEntries": directory_entries,
            "directoryIndex": index,
            "directoryPath": str(state.directory) if state.directory else None,
            "hasPrev": bool(index is not None and index > 0),
            "hasNext": bool(index is not None and index < len(state.files) - 1),
            "isRgb": is_rgb,
            "useWebglPipeline": True,
        }
        saved_state = state.saved_states.get(state.path_key())
        if saved_state:
            try:
                sanitized = json.loads(json.dumps(saved_state))
            except Exception:
                sanitized = saved_state
            config["savedViewerState"] = sanitized
            state.saved_states[state.path_key()] = sanitized
        return config

    def _encode_image(self, array: np.ndarray, *, is_rgb: bool) -> str:
        buffer = io.BytesIO()
        if is_rgb and array.ndim == 3 and array.shape[-1] == 2:
            # promote 2-channel images to 3 for PNG compatibility
            rgb = np.empty((*array.shape[:-1], 3), dtype=array.dtype)
            rgb[..., :2] = array
            rgb[..., 2] = 0
            imageio.imwrite(buffer, rgb, format="png")
        else:
            imageio.imwrite(buffer, array, format="png")
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{data}"

    def navigate(self, state: SessionState, delta: int) -> Optional[Path]:
        if not state.files or state.current_path is None:
            return None
        try:
            idx = state.files.index(state.current_path)
        except ValueError:
            return None
        target = idx + delta
        if target < 0 or target >= len(state.files):
            return None
        return state.files[target]

    def save_viewer_state(self, state: SessionState, image_path: Optional[Path], viewer_state: dict[str, Any]) -> None:
        key = _session_path_key(image_path if image_path is not None else state.current_path)
        try:
            state.saved_states[key] = json.loads(json.dumps(viewer_state))
        except Exception:
            state.saved_states[key] = viewer_state


# Global singleton
SESSION_MANAGER = SessionManager()
