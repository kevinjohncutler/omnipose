"""Minimal PyQt6 viewer replicating core Omnipose image interactions."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import math
import numpy as np
from PyQt6.QtCore import QPoint, QRectF, Qt
from PyQt6.QtGui import QCursor, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QMainWindow,
    QSlider,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
)

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from sample_image import (  # type: ignore
        DEFAULT_BRUSH_RADIUS,
        apply_gamma,
        get_instance_color_table,
        load_image_uint8,
    )
else:
    from .sample_image import DEFAULT_BRUSH_RADIUS, apply_gamma, get_instance_color_table, load_image_uint8


class ImageView(QGraphicsView):
    """Graphics view with scroll/zoom support and mask painting."""

    def __init__(self, image: np.ndarray, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        self._mask_item = QGraphicsPixmapItem()
        self._mask_item.setZValue(1)
        self._mask_item.setOpacity(1.0)
        self._mask_item.setVisible(True)
        self._mask_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._scene.addItem(self._mask_item)

        self.setScene(self._scene)
        self.setBackgroundBrush(Qt.GlobalColor.black)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

        self._base_image = image
        self._mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        self._color_table = get_instance_color_table()
        self._mask_visible = True
        self._current_label = 1
        self._brush_radius = DEFAULT_BRUSH_RADIUS
        self._brush_offsets = self._make_brush_offsets(self._brush_radius)
        self._painting = False
        self._last_paint_point = None
        self._pan_cursor = QCursor(Qt.CursorShape.OpenHandCursor)
        self._paint_cursor = QCursor(Qt.CursorShape.CrossCursor)
        self.viewport().setCursor(self._pan_cursor)

        self.update_image(image)
        self._update_mask_pixmap()

    @staticmethod
    def _make_brush_offsets(radius: int):
        offsets = []
        r2 = radius * radius
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= r2:
                    offsets.append((dy, dx))
        return offsets

    @staticmethod
    def _numpy_to_qimage(array: np.ndarray) -> QImage:
        array = np.ascontiguousarray(array)
        if array.ndim == 2:
            h, w = array.shape
            bytes_per_line = w
            return QImage(array.tobytes(), w, h, bytes_per_line, QImage.Format.Format_Grayscale8).copy()

        if array.ndim != 3 or array.shape[-1] not in (3, 4):
            raise ValueError("expected HxW or HxWx{3,4} array")

        h, w, c = array.shape
        if c == 4:
            fmt = QImage.Format.Format_RGBA8888
            bytes_per_line = 4 * w
        else:
            fmt = QImage.Format.Format_RGB888
            bytes_per_line = 3 * w
        return QImage(array.tobytes(), w, h, bytes_per_line, fmt).copy()

    def update_image(self, array: np.ndarray) -> None:
        pixmap = QPixmap.fromImage(self._numpy_to_qimage(array))
        self._pixmap_item.setPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))

    def wheelEvent(self, event):  # type: ignore[override]
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.0015 ** delta
        current_scale = self.transform().m11()
        min_scale, max_scale = 0.05, 40.0
        new_scale = current_scale * factor
        if new_scale < min_scale:
            factor = min_scale / current_scale
        elif new_scale > max_scale:
            factor = max_scale / current_scale
        self.scale(factor, factor)

    def mousePressEvent(self, event):  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton and event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self._painting = True
            self.viewport().setCursor(self._paint_cursor)
            self._last_paint_point = None
            self._paint_stroke(self._scene_pos(event))
            event.accept()
            return
        self.viewport().setCursor(self._pan_cursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # type: ignore[override]
        if self._painting:
            self._paint_stroke(self._scene_pos(event))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # type: ignore[override]
        if self._painting and event.button() == Qt.MouseButton.LeftButton:
            self._painting = False
            self._last_paint_point = None
            self.viewport().setCursor(self._pan_cursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _scene_pos(self, event) -> QPoint:
        position = event.position().toPoint() if hasattr(event, "position") else event.pos()
        return self.mapToScene(position).toPoint()

    def _paint_stroke(self, scene_point: QPoint) -> None:
        end_point = scene_point
        start_point = self._last_paint_point
        stroke_points = self._stroke_points(start_point, end_point)
        if not stroke_points:
            self._last_paint_point = end_point
            return
        height, width = self._mask.shape
        label = int(self._current_label)
        for py, px in stroke_points:
            if px < 0 or py < 0 or px >= width or py >= height:
                continue
            for dy, dx in self._brush_offsets:
                yy = py + dy
                xx = px + dx
                if 0 <= yy < height and 0 <= xx < width:
                    if label == 0:
                        self._mask[yy, xx] = 0
                    else:
                        self._mask[yy, xx] = label
        self._update_mask_pixmap()
        self._last_paint_point = end_point

    def _stroke_points(self, start: Optional[QPoint], end: QPoint):
        if start is None:
            return [(int(end.y()), int(end.x()))]
        sx, sy = start.x(), start.y()
        ex, ey = end.x(), end.y()
        dx = ex - sx
        dy = ey - sy
        distance = math.hypot(dx, dy)
        if distance == 0:
            return [(int(round(ey)), int(round(ex)))]
        step = max(1.0, self._brush_radius * 0.5)
        steps = int(distance / step) + 1
        points = []
        for i in range(steps + 1):
            t = i / max(steps, 1)
            x = int(round(sx + dx * t))
            y = int(round(sy + dy * t))
            pt = (y, x)
            if not points or points[-1] != pt:
                points.append(pt)
        return points

    def _update_mask_pixmap(self) -> None:
        if not self._mask_visible:
            self._mask_item.setVisible(False)
            return
        lut = self._color_table
        max_index = len(lut) - 1
        mask = np.clip(self._mask, 0, max_index)
        overlay = lut[mask]
        overlay = np.ascontiguousarray(overlay)
        h, w, _ = overlay.shape
        qimage = QImage(overlay.tobytes(), w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
        self._mask_item.setPixmap(QPixmap.fromImage(qimage))
        self._mask_item.setVisible(True)

    def set_mask_visible(self, visible: bool) -> None:
        self._mask_visible = bool(visible)
        self._update_mask_pixmap()

    def is_mask_visible(self) -> bool:
        return self._mask_visible

    def set_current_label(self, label: int) -> None:
        self._current_label = int(label)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Omnipose Minimal PyQt6 Viewer")

        base = load_image_uint8(as_rgb=True)
        self._base_image = base
        self._current_image = base

        self._viewer = ImageView(base)

        self._gamma_slider = QSlider(Qt.Orientation.Vertical)
        self._gamma_slider.setMinimum(10)
        self._gamma_slider.setMaximum(300)
        self._gamma_slider.setValue(100)
        self._gamma_slider.setTickPosition(QSlider.TickPosition.TicksRight)
        self._gamma_slider.valueChanged.connect(self._on_gamma_changed)

        self._gamma_label = QLabel("Gamma: 1.00")
        self._gamma_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self._mask_label = QLabel("Mask Label: 1")
        self._mask_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self._mask_visibility = QLabel("Mask Layer: On (toggle with 'M')")
        self._mask_visibility.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        instructions = QLabel("Hold Shift and drag to paint. Digits 0-9 set label. 'M' toggles mask.")
        instructions.setWordWrap(True)
        instructions.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        side_panel = QVBoxLayout()
        side_panel.addWidget(self._gamma_label)
        side_panel.addWidget(self._gamma_slider)
        side_panel.addSpacing(16)
        side_panel.addWidget(self._mask_label)
        side_panel.addWidget(self._mask_visibility)
        side_panel.addSpacing(16)
        side_panel.addWidget(instructions)
        side_panel.addStretch()

        side_widget = QWidget()
        side_widget.setLayout(side_panel)

        container_layout = QHBoxLayout()
        container_layout.addWidget(self._viewer, stretch=1)
        container_layout.addWidget(side_widget)

        central = QWidget()
        central.setLayout(container_layout)
        self.setCentralWidget(central)
        self.resize(960, 720)

    def _on_gamma_changed(self, value: int) -> None:
        gamma = max(value, 1) / 100.0
        self._gamma_label.setText(f"Gamma: {gamma:.2f}")
        adjusted = apply_gamma(self._base_image, gamma)
        self._viewer.update_image(adjusted)
        self._current_image = adjusted

    def keyPressEvent(self, event):  # type: ignore[override]
        key = event.key()
        if key == Qt.Key.Key_M:
            visible = not self._viewer.is_mask_visible()
            self._viewer.set_mask_visible(visible)
            state = "On" if visible else "Off"
            self._mask_visibility.setText(f"Mask Layer: {state} (toggle with 'M')")
            event.accept()
            return
        if Qt.Key.Key_0 <= key <= Qt.Key.Key_9:
            label = key - Qt.Key.Key_0
            self._viewer.set_current_label(label)
            self._mask_label.setText(f"Mask Label: {label}")
            event.accept()
            return
        super().keyPressEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
