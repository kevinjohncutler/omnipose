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
    
    logger.info(f'initializing segmentation with shape {self.shape}')
    logger.info(f'mask shape {self.shape}')
     
    # if not hasattr(self,'affinity_graph') or compute_affinity or self.affinity_graph.shape[-2:] != self.shape:
    if (not hasattr(self, 'affinity_graph')
        or self.affinity_graph is None
        or compute_affinity
        or self.affinity_graph.shape[-2:] != self.shape
       ):
        
        logger.info(f'initializing affinity graph {len(self.steps)}')

        
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
            
    self.csum = np.sum(self.affinity_graph,axis=0)            

    
    logger.info(f'[initialize_seg] affinity graph {self.affinity_graph.shape} {self.affinity_graph.sum()}')
    
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
        self.viewbox.removeItem(self.layer)
        self.layer_off = True
    else:
        if self.layer_off:
            self.viewbox.addItem(self.layer)
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
    try:
        self.current_label = int(self.LabelInput.text())
        print(f"Active label updated to: {self.current_label}") 
        self.regenerate_cellcolors_for_active_label()
    except ValueError:
        print("Invalid label input.")
    
    if self.current_label > 0:
        # Example of using a color array/dict; adapt as needed
        # for your actual color lookup
        color = self.cellcolors[self.current_label % len(self.cellcolors)]
        print('color', color)
        self.LabelInput.setStyleSheet(
            f"QLineEdit {{ border: 2px solid {color}; font-weight: bold; }}"
        )
    else:
        # revert to default styling
        self.LabelInput.setStyleSheet("")
        
def update_active_label_field(self):
    """Sync the active label input field with the current label."""
    self.LabelInput.setText(str(self.current_label))
    self.update_active_label()


from omnipose.utils import sinebow
def regenerate_cellcolors_for_active_label(self):
    print('regenerate_cellcolors_for_active_label', len(self.cellcolors), self.cellcolors)

    needed_index = self.current_label  # the active label index
    # Optionally also account for the maximum label in the mask stack if you'd like
    if hasattr(self, 'mask_stack') and self.mask_stack.size > 0:
        needed_index = max(needed_index, self.mask_stack.max())

    # If the current color array is already big enough, do nothing
    if needed_index < len(self.cellcolors):
        return

    # Otherwise, re-generate enough colors for all labels up to needed_index.
    ncolors = needed_index + 1
    # This assumes that io.sinebow(n) returns an (n x 3) float array in [0,1].
    c = sinebow(ncolors)
    colors = (np.array(list(c.values()))[1:,:3] * (2**8-1) ).astype(np.uint8)

    # Convert sinebow from float in [0,1] to uint8 in [0..255]
    self.cellcolors = np.concatenate((np.array([[0]*3]), colors), axis=0).astype(np.uint8)
    # Keep track of how many colors we have
    self.ncellcolors = len(self.cellcolors)
    

def draw_layer(self, region=None, z=None):
        """
        Re-colorize the overlay (self.layerz) based on self.mask_stack[z].
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
        
        # Extract subarray of mask_stack
        sub_mask_stack = self.mask_stack[z, y_min:y_max, x_min:x_max]

        # Prepare a subarray for color
        sub_h = y_max - y_min
        sub_w = x_max - x_min
        sub_layerz = np.zeros((sub_h, sub_w, 4), dtype=np.uint8)
        # 1) Color + alpha channel
        if self.masksOn and self.view == 0:
            # Basic coloring
            sub_layerz[..., :3] = self.cellcolors[sub_mask_stack, :] if len(self.cellcolors) > 1 else [255,0,0]
            sub_layerz[..., 3] = self.opacity * (sub_mask_stack > 0).astype(np.uint8)

            # Selected cell -> white
            if self.selected > 0:
                mask_sel = (sub_mask_stack == self.selected)
                sub_layerz[mask_sel] = np.array([255, 255, 255, self.opacity], dtype=np.uint8)
        else:
            # No masks -> alpha=0
            sub_layerz[..., 3] = 0


        # 2) Outlines
        if self.outlinesOn:
            # We want the boundary pixels to have the same color as their underlying label,
            # even if masks are turned off. So we look up the mask label for those pixels
            # and assign color & full opacity.
            # 
            # If the "flow field" is toggled on (assume we track it with "self.flowOn"),
            # we want the outlines to appear white instead. We'll do a conditional:
            
            outl_region = self.outl_stack[z, y_min:y_max, x_min:x_max]
            outline_pixels = (outl_region > 0)

            if self.view == 1:
                # accentuate the flow edges 
                self.img.setOpacity(.5)
                image = self.flows[self.view-1][self.currentZ, y_min:y_max, x_min:x_max].copy()
                image[outline_pixels, 3] = np.maximum(image[outline_pixels, 3],128)
                sub_layerz[outline_pixels] = image[outline_pixels]
            elif self.view == 2:
            # use gray on distance field
                sub_layerz[outline_pixels] = [255//2]*3+[255]
            else:
                # otherwise, use the cell color
                sub_layerz[outline_pixels, :3] = self.cellcolors[sub_mask_stack[outline_pixels]]
                # fully opaque outline
                sub_layerz[outline_pixels, 3] = 2 * 255 // 3
                
        else:
            self.img.setOpacity(1)
        

        # Put the subarray back into the main overlay
        self.layerz[y_min:y_max, x_min:x_max] = sub_layerz

        # Finally update the displayed image
        self.layer.setImage(self.layerz, autoLevels=False)
        
        
        # self.pixelGridOverlay
