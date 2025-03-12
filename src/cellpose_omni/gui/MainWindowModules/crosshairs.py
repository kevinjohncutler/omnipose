import pyqtgraph as pg


def cross_hairs(self):
    if self.CHCheckBox.isChecked():
        self.p0.addItem(self.vLine, ignoreBounds=True)
        self.p0.addItem(self.hLine, ignoreBounds=True)
    else:
        self.p0.removeItem(self.vLine)
        self.p0.removeItem(self.hLine)

def update_crosshairs(self):
    self.yortho = min(self.Ly-1, max(0, int(self.yortho)))
    self.xortho = min(self.Lx-1, max(0, int(self.xortho)))
    self.vLine.setPos(self.xortho)
    self.hLine.setPos(self.yortho)
    self.vLineOrtho[1].setPos(self.xortho)
    self.hLineOrtho[1].setPos(self.dz)
    self.vLineOrtho[0].setPos(self.dz)
    self.hLineOrtho[0].setPos(self.yortho)
        
def set_crosshair_colors(self):
    pen = pg.mkPen(self.accent)
    self.vLine.setPen(pen)
    self.hLine.setPen(pen)
    [l.setPen(pen) for l in self.vLineOrtho]
    [l.setPen(pen) for l in self.hLineOrtho]
        
# this is really just for the crosshairs
def mouse_moved(self, pos):
    items = self.win.scene().items(pos)
    for x in items: #why did this get deleted in CP2?
        if x==self.p0:
            mousePoint = self.p0.mapSceneToView(pos)
            if self.CHCheckBox.isChecked():
                self.vLine.setPos(mousePoint.x())
                self.hLine.setPos(mousePoint.y())
                
                
    #for x in items:
    #    if not x==self.p0:
    #        QtWidgets.QApplication.restoreOverrideCursor()
    #        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.DefaultCursor)

