from omnipose.utils import sinebow
from matplotlib.colors import rgb2hex

N = 29
c = sinebow(N)
COLORS = [rgb2hex(c[i][:3]) for i in range(1,N+1)] #can only do RBG, not RGBA for stylesheet



import pyqtgraph as pg



def color_choose(self):
    # 1) Load the preset gradient:
    index = self.RGBDropDown.currentIndex()
    preset = self.cmaps[index]          # e.g. "magma", "viridis", etc.
    self.hist.gradient.loadPreset(preset)

    # 2) Optionally modify the gradient stops:
    # Only do this if you want "magma" to have transparent at the lower bound:
    if preset in self.default_cmaps:
        st = self.hist.gradient.saveState()
        print('debug:', st['ticks'])  # e.g. [(0.0, (0, 0, 3, 255)), (0.25, (80, 18, 123, 255)), ...]

        # Grab the first tick
        pos, color = st['ticks'][0]          # only two items, not three
        color = list(color)                  # convert tuple -> list so we can mutate alpha
        color[3] = 0                         # set alpha = 0

        # Reassign the modified color back into the tick
        st['ticks'][0] = [pos, tuple(color)] # or use a list for color again

        # Restore the gradient
        self.hist.gradient.restoreState(st)

    # 3) Save the current gradient state and re-apply highlight colors:
    self.states[self.view] = self.hist.saveState()
    self.set_tick_hover_color()
    
    # self.update_plot()  # if you have a call to refresh the plot


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
    # self.hist.plot.opts['antialias'] = 1
    self.hist.setLevels(min=0, max=255)
    
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
        



