from omnipose.utils import sinebow
from matplotlib.colors import rgb2hex

N = 29
c = sinebow(N)
COLORS = [rgb2hex(c[i][:3]) for i in range(1,N+1)] #can only do RBG, not RGBA for stylesheet



import pyqtgraph as pg
import numpy as np



def color_choose(self):
    idx = self.RGBDropDown.currentIndex()
    preset = self.cmaps[idx]
    
    # Ensure consistent item state by saving/restoring on self.hist (the HistogramLUTItem)
    # self.hist.restoreState(self.hist.saveState())
    
    # Load the preset on the histogram's gradient
    self.hist.gradient.loadPreset(preset)
    
    # If it's not the raw image view, apply alpha=0 to the first tick if in default_cmaps
    if self.view != 0 and preset in self.default_cmaps:
        st = self.hist.saveState()  # entire histogram LUT state
        pos, color = st['gradient']['ticks'][0]
        color = list(color)
        color[3] = 0
        st['gradient']['ticks'][0] = [pos, tuple(color)]
        self.hist.restoreState(st)
    
    # Store final histogram LUT state in self.hist.view_states
    if not hasattr(self.hist, 'view_states'):
        self.hist.view_states = {}
    
    st2 = self.hist.saveState()
    if st2['gradient'].get('mode') == 'mono':
        st2['gradient']['mode'] = 'rgb'
    self.hist.view_states[self.view] = st2
    
    # Final styling
    self.set_hist_colors()

def set_hist_colors(self):

    region = self.hist.region
    # c = self.palette().brush(QPalette.ColorRole.Text).color() # selects white or black from palette
    # selecting from the palette can be handy, but the corresponding colors in light and dark mode do not match up well
    color = '#44444450' if self.darkmode else '#cccccc50'
    # c.setAlpha(20)
    region.setBrush(color) # I hate the blue background
    
    c = self.accent
    c.setAlpha(60)
    region.setHoverBrush(c) # also the blue hover
    c.setAlpha(255) # reset accent alpha 
    
    color = '#777' if self.darkmode else '#aaa'
    pen =  pg.mkPen(color=color,width=1.5)
    ph =  pg.mkPen(self.accent,width=2)
    # region.lines[0].setPen(None)
    # region.lines[0].setHoverPen(color='c',width = 5)
    # region.lines[1].setPen('r')
    
    # self.hist.paint(self.hist.plot)
    # print('sss',self.hist.regions.__dict__)
    
    for line in region.lines:
        # c.setAlpha(100)
        line.setPen(pen)
        # c.setAlpha(200)
        line.setHoverPen(ph)
    
    self.hist.gradient.gradRect.setPen(pen)
    # c.setAlpha(100)
    self.hist.gradient.tickPen = pen
    self.set_tick_hover_color() 
    
    ax = self.hist.axis
    ax.setPen(color=(0,)*4) # transparent 
    # ax.setTicks([0,255])
    # ax.setStyle(stopAxisAtTick=(True,True))

    # self.hist = self.img.getHistogram()
    # self.hist.disableAutoHistogramRange()
    # c = self.palette().brush(QPalette.ColorRole.ToolTipBase).color() # selects white or black from palette
    # print(c.getRgb(),'ccc')
    
    # c.setAlpha(100)
    self.hist.fillHistogram(fill=True, level=1.0, color= '#222' if self.darkmode else '#bbb')
    self.hist.axis.style['showValues'] = 0
    self.hist.axis.style['tickAlpha'] = 0
    self.hist.axis.logMode = 1
    self.hist.plot.opts['antialias'] = 1
    # self.hist.plot.opts['stepMode'] = True
    # self.hist.setLevels(min=0, max=255)
    
    # policy = QtWidgets.QSizePolicy()
    # policy.setRetainSizeWhenHidden(True)
    # self.hist.setSizePolicy(policy)
    
    # self.histmap_img = self.hist.saveState()


def set_tick_hover_color(self):
    for tick in self.hist.gradient.ticks:
        tick.hoverPen = pg.mkPen(self.accent,width=2)
        
def set_button_color(self):
    color = '#eeeeee' if self.darkmode else '#888888'
    self.ModelButton.setStyleSheet('border: 2px solid {};'.format(color))
