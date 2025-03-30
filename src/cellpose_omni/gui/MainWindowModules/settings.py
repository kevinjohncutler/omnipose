import numpy as np

from cellpose_omni.plot import disk

def chanchoose(self, image):
    if image.ndim > 2 and not self.onechan:
        if self.ChannelChoose[0].currentIndex()==0:
            image = image.astype(np.float32).mean(axis=-1)[...,np.newaxis]
        else:
            chanid = [self.ChannelChoose[0].currentIndex()-1]
            if self.ChannelChoose[1].currentIndex()>0:
                chanid.append(self.ChannelChoose[1].currentIndex()-1)
            image = image[:,:,chanid].astype(np.float32)
    return image
    
def get_channels(self):
    channels = [self.ChannelChoose[0].currentIndex(), self.ChannelChoose[1].currentIndex()]
    if self.current_model=='nuclei':
        channels[1] = 0

    if self.nchan==1:
        channels = None
    return channels




def compute_scale(self):
    # print('deprecate this?')
    self.diameter = float(self.Diameter.text())
    self.pr = int(float(self.Diameter.text()))
    # self.radii_padding = int(self.pr*1.25)
    # self.radii = np.zeros((self.Ly+self.radii_padding,self.Lx,4), np.uint8)
    # yy,xx = disk([self.Ly+self.radii_padding/2-1, self.pr/2+1],
    #                     self.pr/2, self.Ly+self.radii_padding, self.Lx)
    # # rgb(150,50,150)
    # self.radii[yy,xx,0] = 255 # making red to correspond to tooltip
    # self.radii[yy,xx,1] = 0
    # self.radii[yy,xx,2] = 0
    # self.radii[yy,xx,3] = 255
    # # self.update_plot()
    # self.viewbox.setYRange(0,self.Ly+self.radii_padding)
    # self.viewbox.setXRange(0,self.Lx)
    # self.win.show()
    # self.show()
    
    

def toggle_scale(self):
    if self.scale_on:
        self.viewbox.removeItem(self.scale)
        self.scale_on = False
    else:
        self.viewbox.addItem(self.scale)
        self.scale_on = True
    self.recenter()    

def set_nchan(self):
    self.nchan = int(self.ChanNumber.text())


def toggle_affinity(self):
    self.recompute_masks = True
    self.run_mask_reconstruction()
    