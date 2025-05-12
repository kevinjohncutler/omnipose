import numpy as np

# for cursor
from PyQt6.QtGui import QBrush, QTransform, QCursor
from PyQt6.QtCore import QPointF
import pyqtgraph as pg

# cursor highlight, maybe more of an annotation thing 
def update_highlight(self, pos=None):

    if not self.PencilCheckBox.isChecked():
        self.highlight_rect.hide()
        return

    if pos is None:
        # Get the current global mouse position
        mouse_pos = QCursor.pos()  # global coordinates
        scene_mouse_pos = self.viewbox.scene().views()[0].mapFromGlobal(mouse_pos)  # to scene coords
        pos = QPointF(scene_mouse_pos.x(), scene_mouse_pos.y())

    # Map from scene to view coords
    view_pos = self.viewbox.mapSceneToView(pos)
    xF, yF = view_pos.x(), view_pos.y()
    x, y = int(xF), int(yF)

    # 1) Check if cursor is within the current visible region of the viewbox
    (xMin, xMax), (yMin, yMax) = self.viewbox.viewRange()
    if not (xMin <= xF <= xMax and yMin <= yF <= yMax):
        self.highlight_rect.hide()
        return

    # # 2) Check if within the actual image shape
    # if self.img.image is None:
    #     self.highlight_rect.hide()
    #     return

    # h, w = self.img.image.shape[:2]
    # if x < 0 or y < 0 or x >= w or y >= h:
    #     self.highlight_rect.hide()
    #     return

    # 3) If we reach here, we’re inside both the visible region & image bounds
    self.highlight_rect.show()

    # pick the kernel if brush mode is on
    px = np.ones((1, 1), dtype=bool)
    kernel = getattr(self.layer, '_kernel', px) if self.PencilCheckBox.isChecked() else px

    # Recompute the path if needed
    if not hasattr(self, 'highlight_path'):
        self.compute_kernel_path(kernel)

    # 4) Position the cached path so it’s centered at (x,y)
    transform = QTransform()
    transform.translate(x - kernel.shape[1] // 2, y - kernel.shape[0] // 2)
    transformed_path = transform.map(self.highlight_path)
    self.highlight_rect.setPath(transformed_path)

    # 5) Style the highlight
    base_hex = "#FFF"
    pen_color = pg.mkColor(base_hex)
    pen_color.setAlpha(100)
    self.highlight_rect.setBrush(QBrush(pen_color))
    pen_color.setAlpha(255)
    self.highlight_rect.setPen(pg.mkPen(color=pen_color, width=1))


from PyQt6 import QtGui, QtCore, QtWidgets
from PyQt6.QtCore import Qt, pyqtSlot, QCoreApplication


def eventFilter(self, obj, event):
    if obj != self.win.viewport():
        return False

    if event.type() == QtCore.QEvent.Type.Leave:
        self.highlight_rect.hide()
        return True

    elif event.type() == QtCore.QEvent.Type.MouseMove:
        widget_pos = self.win.viewport().mapFromGlobal(QCursor.pos())
        # If inside the viewport rect, re-show the highlight.
        if self.win.viewport().rect().contains(widget_pos):
            # Call update_highlight so the cursor reappears
            scene_pos = self.viewbox.mapToScene(widget_pos)
            self.update_highlight(pos=scene_pos)
        else:
            # Hide if out of bounds
            self.highlight_rect.hide()
        return True

    return False