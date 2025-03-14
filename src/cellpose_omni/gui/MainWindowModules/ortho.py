
import numpy as np
import pyqtgraph as pg

def make_orthoviews(self):
    self.pOrtho, self.imgOrtho, self.layerOrtho = [], [], []
    for j in range(2):
        self.pOrtho.append(pg.ViewBox(
                            lockAspect=True,
                            name=f'plotOrtho{j}',
                            # border=[100, 100, 100],
                            invertY=True,
                            enableMouse=False
                        ))
        self.pOrtho[j].setMenuEnabled(False)

        self.imgOrtho.append(pg.ImageItem(viewbox=self.pOrtho[j], parent=self, levels=(0,255)))
        self.imgOrtho[j].autoDownsample = False

        self.layerOrtho.append(pg.ImageItem(viewbox=self.pOrtho[j], parent=self))
        self.layerOrtho[j].setLevels([0,255])

        #self.pOrtho[j].scene().contextMenuItem = self.pOrtho[j]
        self.pOrtho[j].addItem(self.imgOrtho[j])
        self.pOrtho[j].addItem(self.layerOrtho[j])
        self.pOrtho[j].addItem(self.vLineOrtho[j], ignoreBounds=False)
        self.pOrtho[j].addItem(self.hLineOrtho[j], ignoreBounds=False)
    
    self.pOrtho[0].linkView(self.pOrtho[0].YAxis, self.viewbox)
    self.pOrtho[1].linkView(self.pOrtho[1].XAxis, self.viewbox)
    

def add_orthoviews(self):
    self.yortho = self.Ly//2
    self.xortho = self.Lx//2
    if self.NZ > 1:
        self.update_ortho()

    self.win.addItem(self.pOrtho[0], 0, 1, rowspan=1, colspan=1)
    self.win.addItem(self.pOrtho[1], 1, 0, rowspan=1, colspan=1)

    qGraphicsGridLayout = self.win.ci.layout
    qGraphicsGridLayout.setColumnStretchFactor(0, 2)
    qGraphicsGridLayout.setColumnStretchFactor(1, 1)
    qGraphicsGridLayout.setRowStretchFactor(0, 2)
    qGraphicsGridLayout.setRowStretchFactor(1, 1)
    
    #self.viewbox.linkView(self.viewbox.YAxis, self.pOrtho[0])
    #self.viewbox.linkView(self.viewbox.XAxis, self.pOrtho[1])
    
    self.pOrtho[0].setYRange(0,self.Lx)
    self.pOrtho[0].setXRange(-self.dz/3,self.dz*2 + self.dz/3)
    self.pOrtho[1].setYRange(-self.dz/3,self.dz*2 + self.dz/3)
    self.pOrtho[1].setXRange(0,self.Ly)
    # self.pOrtho[0].setLimits(minXRange=self.dz*2+self.dz/3*2)
    # self.pOrtho[1].setLimits(minYRange=self.dz*2+self.dz/3*2)

    self.viewbox.addItem(self.vLine, ignoreBounds=False)
    self.viewbox.addItem(self.hLine, ignoreBounds=False)
    self.viewbox.setYRange(0,self.Lx)
    self.viewbox.setXRange(0,self.Ly)

    self.win.show()
    self.show()
    
    #self.viewbox.linkView(self.viewbox.XAxis, self.pOrtho[1])
    
def remove_orthoviews(self):
    self.win.removeItem(self.pOrtho[0])
    self.win.removeItem(self.pOrtho[1])
    self.viewbox.removeItem(self.vLine)
    self.viewbox.removeItem(self.hLine)
    
    # restore the layout
    qGraphicsGridLayout = self.win.ci.layout
    qGraphicsGridLayout.setColumnStretchFactor(1, 0)
    qGraphicsGridLayout.setColumnStretchFactor(0, 1)
    qGraphicsGridLayout.setRowStretchFactor(1, 0)
    qGraphicsGridLayout.setRowStretchFactor(0, 1)
    
    #restore scale
    self.recenter()
    
    self.win.show()
    self.show()

def toggle_ortho(self):

    print('\n\n\n\n test\n\n\n\n')
    if self.orthobtn.isChecked():
        self.add_orthoviews()
    else:
        self.remove_orthoviews()

def update_ortho(self):
    if self.NZ>1 and self.orthobtn.isChecked():
        dzcurrent = self.dz
        # self.dz = min(100, max(3,int(self.dzedit.text() )))
        self.dz = min(self.NZ,max(1,int(self.dzedit.text())))
        
        self.zaspect = max(0.01, min(100., float(self.zaspectedit.text())))
        self.dzedit.setText(str(self.dz))
        self.zaspectedit.setText(str(self.zaspect))
        self.update_crosshairs()
        if self.dz != dzcurrent:
            self.pOrtho[0].setXRange(-self.dz/3,self.dz*2 + self.dz/3)
            self.pOrtho[1].setYRange(-self.dz/3,self.dz*2 + self.dz/3)

        y = self.yortho
        x = self.xortho
        z = self.currentZ
        # zmin, zmax = max(0, z-self.dz), min(self.NZ, z+self.dz)
        
        zmin = z-self.dz
        zmax = z+self.dz
        zpad = np.array([0,0])
        if zmin<0:
            zpad[0] = 0-zmin
            zmin = 0 
        if zmax>self.NZ:
            zpad[1] = zmax-self.NZ
            zmax = self.NZ
            
        # to keep ortho view centered on Z, the slice needs to be padded.
        # Zmin and zmax need residuals to padd the array.
        # at present, the cursor on the slice views is orented so that the layer at right/below corresponds to the central view
        b = [0,0]
        if self.view==0:
            for j in range(2):
                if j==0:
                    image = np.pad(self.stack[zmin:zmax, :, x],(zpad,b,b)).transpose(1,0,2)
                else:
                    image = np.pad(self.stack[zmin:zmax, y, :],(zpad,b,b))
                    
                if self.color==0:
                    if self.onechan:
                        # show single channel
                        image = image[...,0]
                    self.imgOrtho[j].setImage(image, autoLevels=False, lut=None)
                elif self.color>0 and self.color<4:
                    image = image[...,self.color-1]
                    self.imgOrtho[j].setImage(image, autoLevels=False, lut=self.cmap[self.color])
                elif self.color==4:
                    image = image.astype(np.float32).mean(axis=-1).astype(np.uint8)
                    self.imgOrtho[j].setImage(image, autoLevels=False, lut=None)
                elif self.color==5:
                    image = image.astype(np.float32).mean(axis=-1).astype(np.uint8)
                    self.imgOrtho[j].setImage(image, autoLevels=False, lut=self.cmap[0])
                # self.imgOrtho[j].setLevels(self.saturation[self.currentZ])
            self.pOrtho[0].setAspectLocked(lock=True, ratio=self.zaspect)
            self.pOrtho[1].setAspectLocked(lock=True, ratio=1./self.zaspect)

        else:
            image = np.zeros((10,10), np.uint8)
            self.img.setImage(image, autoLevels=False, lut=None)
            self.img.setLevels([0.0, 255.0])        
    self.win.show()
    self.show()
