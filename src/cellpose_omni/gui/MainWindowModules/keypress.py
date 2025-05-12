from .. import logger
from PyQt6 import QtGui, QtCore, QtWidgets
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtCore import Qt

def plot_clicked(self, event):
    if event.button()==QtCore.Qt.LeftButton and (event.modifiers() != QtCore.Qt.ShiftModifier and
                event.modifiers() != QtCore.Qt.AltModifier):
        if event.double():
            self.recenter()
        elif self.loaded and not self.in_stroke:
            if self.orthobtn.isChecked():
                items = self.win.scene().items(event.scenePos())
                for x in items:
                    if x==self.viewbox:
                        pos = self.viewbox.mapSceneToView(event.scenePos())
                        x = int(pos.x())
                        y = int(pos.y())
                        if y>=0 and y<self.Ly and x>=0 and x<self.Lx:
                            self.yortho = y 
                            self.xortho = x
                            self.update_ortho()

def keyReleaseEvent(self, event):

    # drag / pan
    if event.key() == QtCore.Qt.Key_Space:
        self.spacePressed = False
    
    # pick and fill 
    elif event.key() == QtCore.Qt.Key_G:
        self.flood_fill_enabled = False
    elif event.key() == QtCore.Qt.Key_P or event.key() == QtCore.Qt.Key_I:
        self.pick_label_enabled = False
    
    # super().keyReleaseEvent(event)
    super(QMainWindow, self).keyReleaseEvent(event)


def keyPressEvent(self, event):
    if not self.loaded:
        return  # Do nothing if not loaded

    modifiers = event.modifiers()
    key = event.key()

    # Modifier-based actions (e.g., Undo/Redo)
    if modifiers & QtCore.Qt.ControlModifier:
        if key == QtCore.Qt.Key_Z:  # Ctrl+Z: Undo
            if modifiers & QtCore.Qt.ShiftModifier:  # Ctrl+Shift+Z: Redo
                self.redo_action()
            else:
                self.undo_action()

    # Actions based on individual keys
    elif key == QtCore.Qt.Key_Space:
        self.spacePressed = True  # Enable pan mode
    elif key == QtCore.Qt.Key_G:
        self.flood_fill_enabled = True  # Enable flood fill
    elif key == QtCore.Qt.Key_B:
        self.PencilCheckBox.toggle()  # Toggle brush tool
    elif key == QtCore.Qt.Key_M:
        self.MCheckBox.toggle()  # Toggle masks
    elif key == QtCore.Qt.Key_O:
        self.OCheckBox.toggle()  # Toggle outlines
    elif key == QtCore.Qt.Key_C or key == QtCore.Qt.Key_N:
        self.NCCheckBox.toggle()  # Toggle ncolor
    elif key == QtCore.Qt.Key_H:
        self.CHCheckBox.toggle()  # Toggle crosshairs

    # Navigation keys (Z-stack navigation)
    elif key == QtCore.Qt.Key_A:
        if self.NZ == 1:
            self.get_prev_image()
        else:
            self.currentZ = max(0, self.currentZ - 1)
            self.scroll.setValue(self.currentZ)
    elif key == QtCore.Qt.Key_D:
        if self.NZ == 1:
            self.get_next_image()
        else:
            self.currentZ = min(self.NZ - 1, self.currentZ + 1)
            self.scroll.wsetValue(self.currentZ)

    # Color cycling
    elif key == QtCore.Qt.Key_W:
        self.color = (self.color - 1) % len(self.cmaps)  # Cycle backward
        self.RGBDropDown.setCurrentIndex(self.color)
    elif key == QtCore.Qt.Key_S:
        self.color = (self.color + 1) % len(self.cmaps)  # Cycle forward
        self.RGBDropDown.setCurrentIndex(self.color)
    elif key == QtCore.Qt.Key_R:
        self.color = 1 if self.color != 1 else 0  # Toggle between 0 and 1
        self.RGBDropDown.setCurrentIndex(self.color)

    # Brush size adjustment
    elif key in {QtCore.Qt.Key_BracketLeft, QtCore.Qt.Key_BracketRight}:
        current_value = self.brush_slider.value()
        
        if key == QtCore.Qt.Key_BracketLeft:
            # Attempt to decrease the brush size
            if current_value > self.brush_slider.minimum():
                new_value = current_value - self.brush_slider.singleStep()
            else:
                # If already at the minimum, turn off drawing
                new_value = self.brush_slider.minimum()
                self.PencilCheckBox.setChecked(False)
        else:  # Key_BracketRight
            # Attempt to increase the brush size
            if not self.PencilCheckBox.isChecked():
                # If the checkbox is unchecked, reset to minimum value and enable
                new_value = self.brush_slider.minimum()
                self.PencilCheckBox.setChecked(True)
            else:
                new_value = min(self.brush_slider.maximum(), current_value + self.brush_slider.singleStep())

        self.brush_slider.setValue(new_value)  # Update the slider directly


    # Active label selection / color picking 
    elif event.key() in range(Qt.Key_0, Qt.Key_9 + 1):  # Numeric keys
        self.current_label = event.key() - Qt.Key_0  # Map keys to numbers 0-9
        self.update_active_label_field() # Sync with input field
        
        print(f"Active label set to: {self.current_label}")
        
    elif key == QtCore.Qt.Key_P or key == QtCore.Qt.Key_I:
        self.pick_label_enabled = True  # Enable label picking
            
    # super().keyPressEvent(event)
    super(QMainWindow, self).keyReleaseEvent(event)
    # QtWidgets.QMainWindow.keyReleaseEvent(self, event)

