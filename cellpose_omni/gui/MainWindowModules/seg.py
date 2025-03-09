from omnipose import utils, misc, core
import numpy as np

from .. import logger, ICON_PATH, io

def initialize_seg(self, compute_affinity=False):
    self.shape = self.masks.shape
    self.dim = len(self.shape) 
    self.steps, self.inds, self.idx, self.fact, self.sign = utils.kernel_setup(self.dim)
    self.supporting_inds = utils.get_supporting_inds(self.steps)
    self.coords = misc.generate_flat_coordinates(self.shape) # for all pixelsom the image 
    # ^ I need to see if I use that anymore, becasue the below method does not use it 
    # but might be needed below to index the WHOLE FoV, not just the masks
    
    self.neighbors = utils.get_neighbors(self.coords, self.steps, self.dim, self.shape)
    self.indexes, self.neigh_inds, self.ind_matrix = utils.get_neigh_inds(tuple(self.neighbors),self.coords,self.shape)
    self.non_self = np.array(list(set(np.arange(len(self.steps)))-{self.inds[0][0]})) 
    
    #ind_matrix could be comouted much more cheaply, or not even deeded? 
    # with the spatial affinity version
    
    logger.info('initializing segmentation')
    logger.info(f'mask shape {self.shape}')
     
    
    if not hasattr(self,'affinity_graph') or compute_affinity:
        logger.info('initializing affinity graph')

        
        # intialize affinity graph with spatial affinity
        S = len(self.steps)
        self.affinity_graph = np.zeros((S,)+self.shape,bool)
        
        if np.any(self.masks):
            coords = np.nonzero(self.masks)
            affinity_graph = core.masks_to_affinity(self.masks, coords, self.steps, 
                                                    self.inds, self.idx, self.fact, 
                                                    self.sign, self.dim)

            # assign to spatial affinity 
            # self.affinity_graph[...,*coords] = affinity_graph
            self.affinity_graph[(Ellipsis,)+coords] = affinity_graph
            
    
    
    logger.info(f'affinity graph shape {self.affinity_graph.shape}')
    # self.pixelGridOverlay.initialize_colors_from_affinity() 

def toggle_masks(self):
    if self.MCheckBox.isChecked():
        self.masksOn = True
        self.restore_masks = True
    else:
        self.masksOn = False
        self.restore_masks = False
        
    if self.OCheckBox.isChecked():
        self.outlinesOn = True
    else:
        self.outlinesOn = False
        
    if not self.masksOn and not self.outlinesOn:
        self.p0.removeItem(self.layer)
        self.layer_off = True
    else:
        if self.layer_off:
            self.p0.addItem(self.layer)
        self.draw_layer()
        self.update_layer()
    if self.loaded:
        # self.update_plot()
        self.update_layer()

def toggle_ncolor(self):
    if self.NCCheckBox.isChecked():
        self.ncolor = True
    else:
        self.ncolor = False
    io._masks_to_gui(self, format_labels=True)
    self.draw_layer()
    if self.loaded:
        # self.update_plot()
        self.update_layer()

def toggle_affinity_graph(self):
    print('yoyo')
    if self.ACheckBox.isChecked():
        self.affinityOn = True
        self.affinityOverlay._generate_lines()
        
        self.affinityOverlay.setVisible(True)
    else:
        self.affinityOn = False
        self.affinityOverlay.setVisible(False)
        
    # self.draw_layer()
    # if self.loaded:
    #     # self.update_plot()
    #     self.update_layer()
    
    

def update_active_label(self):
    """Update self.current_label from the input field."""
    try:
        self.current_label = int(self.LabelInput.text())
        print(f"Active label updated to: {self.current_label}")
    except ValueError:
        print("Invalid label input.")
        
def update_active_label_field(self):
    """Sync the active label input field with the current label."""
    self.LabelInput.setText(str(self.current_label))
    



def draw_layer(self, region=None, z=None):
        """
        Re-colorize the overlay (self.layerz) based on self.cellpix[z].
        If region is None, update the entire image. Otherwise, only update
        the specified sub-region: (x_min, x_max, y_min, y_max).
        """
        
        # if region is None:
        #     print('drawing entire layer')
            
        if z is None:
            z = self.currentZ

        # Default to the entire image if region is None
        if region is None:
            region = (0, self.Lx, 0, self.Ly)
            
        x_min, x_max, y_min, y_max = region

        # Clip the region to image bounds
        x_min = max(0, x_min)
        x_max = min(self.Lx, x_max)
        y_min = max(0, y_min)
        y_max = min(self.Ly, y_max)

        # Ensure self.layerz is allocated and correct shape
        if getattr(self, 'layerz', None) is None or self.layerz.shape[:2] != (self.Ly, self.Lx):
            self.layerz = np.zeros((self.Ly, self.Lx, 4), dtype=np.uint8)
        
        # Extract subarray of cellpix
        sub_cellpix = self.cellpix[z, y_min:y_max, x_min:x_max]

        # Prepare a subarray for color
        sub_h = y_max - y_min
        sub_w = x_max - x_min
        sub_layerz = np.zeros((sub_h, sub_w, 4), dtype=np.uint8)
        # 1) Color + Alpha
        if self.masksOn and self.view == 0:
            # Basic coloring
            sub_layerz[..., :3] = self.cellcolors[sub_cellpix, :] if len(self.cellcolors) > 1 else [255,0,0]
            sub_layerz[..., 3] = self.opacity * (sub_cellpix > 0).astype(np.uint8)

            # Selected cell -> white
            if self.selected > 0:
                mask_sel = (sub_cellpix == self.selected)
                sub_layerz[mask_sel] = np.array([255, 255, 255, self.opacity], dtype=np.uint8)
        else:
            # No masks -> alpha=0
            sub_layerz[..., 3] = 0

        # 2) Outlines
        if self.outlinesOn:
            # there is something weird going on woith initializing the affinity graoh from the npy
            # they need to be deleted I think, or need some workaround to overwrite masks and shape etc. 
            # as they get reset as 512
            
            
            
            # print(self.cellpix[z].shape, self.shape,self.affinity_graph.shape, len(self.coords), self.coords[0].shape)
            # self.outpix = core.affinity_to_boundary( self.cellpix[z], self.affinity_graph, tuple(self.coords))[np.newaxis,:,:]
            sub_outpix = self.outpix[z, y_min:y_max, x_min:x_max]
            sub_layerz[sub_outpix > 0] = np.array(self.outcolor, dtype=np.uint8)

        # Put the subarray back into the main overlay
        self.layerz[y_min:y_max, x_min:x_max] = sub_layerz

        # Finally update the displayed image
        self.layer.setImage(self.layerz, autoLevels=False)
        
        
        # self.pixelGridOverlay

