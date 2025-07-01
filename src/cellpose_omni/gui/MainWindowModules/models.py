from cellpose_omni import models, dynamics
from omnipose import core, gpu, misc
from .. import logger  # Imports logger from __init__.py in parent
from .. import io

import time

def timeit(func):
    """Decorator that reports the execution time."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[TIMEIT] {func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper
    

from PyQt6.QtWidgets import QApplication # could do all in init then import * in these 

OMNI_INSTALLED = True
from omnipose.utils import normalize99, to_8_bit
from cellpose_omni.transforms import resize_image
import numpy as np

# print("\n\n\nReloading f.py, version 1.2\n\n\n")  # update the version manually on changes

def calibrate_size(self):
    self.initialize_model()
    diams, _ = self.model.sz.eval(self.stack[self.currentZ].copy(), invert=self.invert.isChecked(),
                                channels=self.get_channels(), progress=self.progress)
    diams = np.maximum(5.0, diams)
    logger.info('estimated diameter of cells using %s model = %0.1f pixels'%
            (self.current_model, diams))
    self.Diameter.setText('%0.1f'%diams)
    self.diameter = diams
    self.compute_scale()
    self.progress.setValue(100)

# def model_choose(self, index):
    # if index > 0:
def model_choose(self):
    logger.info(f'selected model {self.ModelChoose.currentText()}, loading now')
    self.initialize_model()
    # self.diameter = self.model.diam_labels
    
    # only set this when selected, not if user chooses a new value 
    bacterial = 'bact' in self.current_model
    if bacterial:
        self.diameter = 0.
        self.Diameter.setText('%0.1f'%self.diameter)
    else:
        self.diameter = float(self.Diameter.text())
    
    logger.info(f'diameter set to {self.diameter: 0.2f} (but can be changed)')

# two important things: invert size added, and initialize model takes care of selecting a model
def check_gpu(self, use_torch=True):
    # also decide whether or not to use torch
    self.torch = use_torch
    self.useGPU.setChecked(False)
    self.useGPU.setEnabled(False)    
    if self.torch and gpu.get_device(use_torch=True)[-1]:
        self.useGPU.setEnabled(True)
        self.useGPU.setChecked(True)
    else:
        self.useGPU.setStyleSheet("color: rgb(80,80,80);")
        

def get_model_path(self):
    self.current_model = self.ModelChoose.currentText()
    if self.current_model in models.MODEL_NAMES:
        self.current_model_path = models.model_path(self.current_model, 0, self.torch)
    else:
        self.current_model_path = os.fspath(models.MODEL_DIR.joinpath(self.current_model))


def initialize_model(self):
    self.get_model_path()


    if self.current_model in models.MODEL_NAMES:

        # make sure 2-channel models are initialized correctly
        if self.current_model in models.C2_MODEL_NAMES:
            self.nchan = 2
            self.ChanNumber.setText(str(self.nchan))

        # ensure that the boundary/nclasses is set correctly
        self.boundary.setChecked(self.current_model in models.BD_MODEL_NAMES)
        self.nclasses = 2 + self.boundary.isChecked()

        logger.info(f'Initializing model: nchan set to {self.nchan}, nclasses set to {self.nclasses}, dim set to {self.dim}')        

        # if self.SizeModel.isChecked():
        #     self.model = models.Cellpose(gpu=self.useGPU.isChecked(),
        #                                     use_torch=self.torch,
        #                                     model_type=self.current_model,
        #                                     nchan=self.nchan,
        #                                     nclasses=self.nclasses)
        # else:
        self.model = models.CellposeModel(gpu=self.useGPU.isChecked(),
                                            use_torch=self.torch,
                                            model_type=self.current_model,                                             
                                            nchan=self.nchan,
                                            nclasses=self.nclasses)
        
        omni_model = 'omni' in self.current_model
        bacterial = 'bact' in self.current_model
        if omni_model or bacterial:
            self.NetAvg.setCurrentIndex(1) #one run net
            
    else:
        self.nclasses = 2 + self.boundary.isChecked()
        self.model = models.CellposeModel(gpu=self.useGPU.isChecked(), 
                                            use_torch=True,
                                            pretrained_model=self.current_model_path,                                             
                                            nchan=self.nchan,
                                            nclasses=self.nclasses)



def add_model(self):
    io._add_model(self)
    return

def remove_model(self):
    io._remove_model(self)
    return

def new_model(self):
    if self.NZ!=1:
        logger.error('cannot train model on 3D data')
        return
    
    # train model
    image_names = self.get_files()[0]
    self.train_data, self.train_labels, self.train_files = io._get_train_set(image_names)
    TW = guiparts.TrainWindow(self, models.MODEL_NAMES)
    train = TW.exec_()
    if train:
        logger.info(f'training with {[os.path.split(f)[1] for f in self.train_files]}')
        self.train_model()

    else:
        logger.info('training cancelled')

# this probably needs an overhaul 
def train_model(self):
    if self.training_params['model_index'] < len(models.MODEL_NAMES):
        model_type = models.MODEL_NAMES[self.training_params['model_index']]
        logger.info(f'training new model starting at model {model_type}')        
    else:
        model_type = None
        logger.info(f'training new model starting from scratch')     
    self.current_model = model_type   
    
    self.channels = self.get_channels()
    logger.info(f'training with chan = {self.ChannelChoose[0].currentText()}, chan2 = {self.ChannelChoose[1].currentText()}')
        
    self.model = models.CellposeModel(gpu=self.useGPU.isChecked(), 
                                        model_type=model_type)

    save_path = os.path.dirname(self.filename)
    
    logger.info('name of new model:' + self.training_params['model_name'])
    self.new_model_path = self.model.train(self.train_data, self.train_labels, 
                                            channels=self.channels, 
                                            save_path=save_path, 
                                            nimg_per_epoch=8,
                                            learning_rate = self.training_params['learning_rate'], 
                                            weight_decay = self.training_params['weight_decay'], 
                                            n_epochs = self.training_params['n_epochs'],
                                            model_name = self.training_params['model_name'])
    diam_labels = self.model.diam_labels.copy()
    # run model on next image 
    io._add_model(self, self.new_model_path, load_model=False)
    self.new_model_ind = len(self.model_strings)
    self.autorun = True
    if self.autorun:
        channels = self.channels.copy()
        self.clear_all()
        self.get_next_image(load_seg=True)
        # keep same channels
        self.ChannelChoose[0].setCurrentIndex(channels[0])
        self.ChannelChoose[1].setCurrentIndex(channels[1])
        self.diameter = diam_labels
        self.Diameter.setText('%0.2f'%self.diameter)        
        logger.info(f'>>>> diameter set to diam_labels ( = {diam_labels: 0.3f} )')
        self.compute_model()
    logger.info(f'!!! computed masks for {os.path.split(self.filename)[1]} from new model !!!')
    
def get_thresholds(self):
    # the text field version
    # also, the special case for NZ>1 desn't make sense for omnipose
    # which can calculate flows in 3D 
    
#         try:
#             flow_threshold = float(self.flow_threshold.text())
#             cellprob_threshold = float(self.cellprob_threshold.text())
#             if flow_threshold==0.0 or self.NZ>1:
#                 flow_threshold = None
            
#             return flow_threshold, cellprob_threshold

    #The slider version 
    try:
        return self.threshslider.value(), self.probslider.value()
    except Exception as e:
        print('flow threshold or cellprob threshold not a valid number, setting to defaults')
        self.flow_threshold.setText('0.0')
        self.cellprob_threshold.setText('0.0')
        return 0.0, 0.0

# @timeit


# @pyinstrument_profile
def run_mask_reconstruction(self):
    # profiler = Profiler()
    # profiler.start()

    # use_omni = 'omni' in self.current_model
    
    # needed to be replaced with recompute_masks
    # rerun = False
    have_enough_px = self.probslider.value() > self.cellprob # slider moves up
    
    # update thresholds
    self.threshold, self.cellprob = self.get_thresholds()

    
    # if self.cellprob != self.probslider.value():
    #     rerun = True
    #     self.cellprob = self.probslider.value()
        
    # if self.threshold != self.threshslider.value():
    #     rerun = True
    #     self.threshold = self.threshslider.value()
    
    # if not self.recompute_masks:
    #     return
    
    self.threshold, self.cellprob = self.get_thresholds()
    
    if self.threshold is None:
        logger.info('computing masks with cell prob=%0.3f, no flow error threshold'%
                (self.cellprob))
    else:
        logger.info('computing masks with cell prob=%0.3f, flow error threshold=%0.3f'%
                (self.cellprob, self.threshold))

    net_avg = self.NetAvg.currentIndex()==0 and self.current_model in models.MODEL_NAMES
    resample = self.NetAvg.currentIndex()<2
    omni = OMNI_INSTALLED and self.omni.isChecked()
    
    # useful printout for easily copying parameters to a notebook etc. 
    s = ('channels={}, mask_threshold={:.2f}, '
            'flow_threshold={:.2f}, diameter={:.2f}, invert={}, cluster={}, net_avg={},'
            'do_3D={}, omni={}'
        ).format(self.get_channels(),
                    self.cellprob,
                    self.threshold,
                    self.diameter,
                    self.invert.isChecked(),
                    self.cluster.isChecked(),
                    net_avg,
                    False,
                    omni)
    
    self.runstring.setPlainText(s)
        
    if not omni:
        maski = dynamics.compute_masks(dP=self.flows[-1][:-1], 
                                        cellprob=self.flows[-1][-1],
                                        p=self.flows[-2].copy(),  
                                        mask_threshold=self.cellprob,
                                        flow_threshold=self.threshold,
                                        resize=self.mask_stack.shape[-2:],
                                        verbose=self.verbose.isChecked())[0]
    else:
        #self.flows[3] is p, self.flows[-1] is dP, self.flows[5] is dist/prob, self.flows[6] is bd
        
        # must recompute flows trajectory if we add pixels, because p does not contain them
        # an alternate approach would be to compute p for the lowest allowed threshold
        # and then never recompute (the threshold prodces a mask that selects from existing trajectories, see get_masks)
        # seems like the dbscanm method breaks with this, but affinity is fine... 
        # p = self.flows[-2].copy() if have_enough_px  else None 
        p = self.flows[-2].copy() if have_enough_px and self.AffinityCheck.isChecked() else None 
    
        dP = self.flows[-1][:-self.model.dim]
        dist = self.flows[-1][self.model.dim]
        bd = self.flows[-1][self.model.dim+1]
        
        ret = core.compute_masks(dP=dP, 
                                dist=dist, 
                                affinity_graph=None, 
                                bd=bd,
                                p=p, 
                                mask_threshold=self.cellprob,
                                flow_threshold=self.threshold,
                                resize=self.mask_stack.shape[-2:],
                                cluster=self.cluster.isChecked(),
                                verbose=self.verbose.isChecked(),
                                nclasses=self.model.nclasses,
                                affinity_seg=self.AffinityCheck.isChecked(),
                                omni=omni)
        
        maski, p, tr, bounds, augmented_affinity = ret
        
        self.masks = maski
        
        # print('yoyo', augmented_affinity) empty list 

        # repeated logic, factor out
        self.initialize_seg(compute_affinity=True) # may not need to run again 
        if self.AffinityCheck.isChecked():
            self.neighbors = augmented_affinity[:self.dim]
            affinity_graph = augmented_affinity[self.dim]
            coords = np.nonzero(self.masks)
            # self.coords is form generate_flat_coordinates, which is all pixels in the image
            # here, still using the mask coordinates
            self.bounds = core.affinity_to_boundary(self.masks, affinity_graph, coords)
            
            # update the full affinity graph - maybe easiest to use spatical affinity format?
            self.affinity_graph = core.spatial_affinity(affinity_graph, coords, self.shape)
            self.csum = np.sum(self.affinity_graph,axis=0)            
            
             
        else:
            self.bounds = bounds 

    self.pixelGridOverlay.initialize_colors_from_affinity() 


    # self.masksOn = True
    # self.MCheckBox.setChecked(True)
    # self.outlinesOn = True #should not turn outlines back on by default; masks make sense though 
    # self.OCheckBox.setChecked(True)
    if not (self.masksOn or self.outlinesOn):
        self.masksOn = True
        self.MCheckBox.setChecked(True)
    
    if maski.ndim<3:
        maski = maski[np.newaxis,...]
    logger.info('%d cells found'%(len(misc.unique_nonzero(maski))))
    io._masks_to_gui(self) # replace this to show boundary emphasized masks
    self.show()
    # profiler.stop()
    # print(profiler.output_text(unicode=True, color=True))
        
# @timeit
def compute_model(self):
    # self.progress.setValue(10)
    # QApplication.processEvents() 
    logger.info('running compute_model()')

    tic=time.time()
    self.clear_all()
    self.flows = [[],[],[]]
    self.initialize_model()
    logger.info('using model %s'%self.current_model)
    # self.progress.setValue(20)
    # QApplication.processEvents() 
    do_3D = False
    if self.NZ > 1:
        do_3D = True
        data = self.stack.copy()
    else:
        data = self.stack[0].copy() # maybe chanchoose here 
    channels = self.get_channels()
    
    self.diameter = float(self.Diameter.text())
    
    
    ### will either have to put in edge cases for worm etc or just generalize model loading to respect what is there 
    # try:
    omni_model = 'omni' in self.current_model or 'affinity' in self.current_model
    bacterial = 'bact' in self.current_model
    if omni_model or bacterial:
        self.NetAvg.setCurrentIndex(1) #one run net
    # if bacterial:
    #     self.diameter = 0.
    #     self.Diameter.setText('%0.1f'%self.diameter)

    # allow omni to be togged manually or forced by model
    if OMNI_INSTALLED:
        if omni_model:
            logger.info('turning on Omnipose mask reconstruction version for Omnipose models (see menu)')
            if not self.omni.isChecked():
                print('WARNING: Omnipose models require Omnipose mask reconstruction (toggle back on in menu)')
            if not self.cluster.isChecked():
                print(('NOTE: clustering algorithm can help with over-segmentation in thin cells.'
                        'Default is ON with omnipose models (see menu)'))
                
        elif self.omni.isChecked():
            print('NOTE: using Omnipose mask recontruction with built-in cellpose model (toggle in Omnipose menu)')

    net_avg = self.NetAvg.currentIndex()==0 and self.current_model in models.MODEL_NAMES
    resample = self.NetAvg.currentIndex()<2
    omni = OMNI_INSTALLED and self.omni.isChecked()
    
    self.threshold, self.cellprob = self.get_thresholds()

    # useful printout for easily copying parameters to a notebook etc. 
    s = ('channels={}, mask_threshold={}, '
            'flow_threshold={}, diameter={}, invert={}, cluster={}, net_avg={}, '
            'do_3D={}, omni={}'
        ).format(self.get_channels(),
                    self.cellprob,
                    self.threshold,
                    self.diameter,
                    self.invert.isChecked(),
                    self.cluster.isChecked(),
                    net_avg,
                    do_3D,
                    omni)
    self.runstring.setPlainText(s)
    self.progress.setValue(30)
    
    masks, flows = self.model.eval(data, channels=channels,
                                    mask_threshold=self.cellprob,
                                    flow_threshold=self.threshold,
                                    diameter=self.diameter, 
                                    invert=self.invert.isChecked(),
                                    net_avg=net_avg, 
                                    augment=False, 
                                    resample=resample,
                                    do_3D=do_3D, 
                                    progress=self.progress,
                                    verbose=self.verbose.isChecked(),
                                    omni=omni, 
                                    tile=self.tile.isChecked(),
                                    affinity_seg=self.AffinityCheck.isChecked(),
                                    cluster = self.cluster.isChecked(),
                                    transparency=True,
                                    channel_axis=-1
                                    )[:2]
        
    # except Exception as e:
    #     print('GUI.py: NET ERROR: %s'%e)
    #     self.progress.setValue(0)
    #     return
    
    # self.progress.setValue(75)
    # QApplication.processEvents() 
    #if not do_3D:
    #    masks = masks[0][np.newaxis,:,:]
    #    flows = flows[0]
    
    # flows here are [RGB, dP, cellprob, p, bd, tr]
    self.flows[0] = to_8_bit(flows[0]) #RGB flow for plotting
    self.flows[1] = to_8_bit(flows[2]) #dist/prob for plotting
    if self.boundary.isChecked():
        self.flows[2] = to_8_bit(flows[4]) #boundary for plotting
    else:
        self.flows[2] = np.zeros_like(self.flows[1])
        
    self.masks = masks
    
    # boundary and affinity
    bounds = flows[-1]
    augmented_affinity = flows[-2]
    
    
    # repeated logic, factor out
    self.initialize_seg(compute_affinity=True) # may not need to run again now
    
    if self.AffinityCheck.isChecked():
        self.neighbors = augmented_affinity[:self.dim]
        affinity_graph = augmented_affinity[self.dim]
        coords = np.nonzero(self.masks)
        # self.coords is form generate_flat_coordinates, which is all pixels in the image
        # here, still using the mask coordinates
        self.bounds = core.affinity_to_boundary(self.masks, affinity_graph, coords)
        
        # update the full affinity graph - maybe easiest to use spatical affinity format?
        self.affinity_graph = core.spatial_affinity(affinity_graph, coords, self.shape)
        self.csum = np.sum(self.affinity_graph,axis=0)            
    else:
        self.bounds = bounds
        
    self.pixelGridOverlay.initialize_colors_from_affinity() 
    # self.pixelGridOverlay.reset() 
        

    if not do_3D:
        masks = masks[np.newaxis,...]
        for i in range(3):
            self.flows[i] = resize_image(self.flows[i], masks.shape[-2], masks.shape[-1])
        
        #critical line from what I had commended out below
        self.flows = [self.flows[n][np.newaxis,...] for n in range(len(self.flows))]
    
    # I think this is a z-component placeholder. Relaceing with boundary output, will
    # put this back later for the 3D update 
    # if not do_3D:
    #     self.flows[2] = np.zeros(masks.shape[1:], dtype=np.uint8)
    #     self.flows = [self.flows[n][np.newaxis,...] for n in range(len(self.flows))]
    # else:
    #     self.flows[2] = (flows[1][0]/10 * 127 + 127).astype(np.uint8)
        

    # this stores the original flow components for recomputing masks
    if len(flows)>2: 
        self.flows.append(flows[3].squeeze()) #p put in position -2
        flws = [flows[1], #self.flows[-1][:self.dim] is dP
                flows[2][np.newaxis,...]] #self.flows[-1][self.dim] is dist/prob
        if self.boundary.isChecked():
            flws.append(flows[4][np.newaxis,...]) #self.flows[-1][self.dim+1] is bd
        else:
            flws.append(np.zeros_like(flws[-1]))
        
        self.flows.append(np.concatenate(flws))

    logger.info('%d cells found with model in %0.3f sec'%(len(np.unique(masks)[1:]), time.time()-tic))
    # self.progress.setValue(80)
    # QApplication.processEvents() 
    z=0
    self.masksOn = True
    self.MCheckBox.setChecked(True)
    # self.outlinesOn = True #again, this option should persist and not get toggled by another GUI action 
    # self.OCheckBox.setChecked(True)

    # print('masks found, drawing now', self.masks.shape)
    io._masks_to_gui(self)
    self.progress.setValue(100)

    # self.toggle_server(off=True)
    if not do_3D:
        self.threshslider.setEnabled(True)
        self.probslider.setEnabled(True)
        
    # except Exception as e:
    #     print('ERROR in compute_models: %s'%e)

def copy_runstring(self):
    self.clipboard.setText(self.runstring.toPlainText())