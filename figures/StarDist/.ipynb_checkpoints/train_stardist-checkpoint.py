from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.models import Config2D, StarDist2D, StarDistData2D

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from stardist import matching
from stardist.models.model2d import StarDistData2D

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


import sys, os, argparse, pickle
import numpy as np
dm11_root = '/groups/pachitariu/pachitariulab/'
#dm11_root = '/home/carsen/dm11/'
basedir = '/groups/pachitariu/home/stringerc/models' # where to save outputs
#basedir = '/media/carsen/DATA1/datasets/models' # where to save outputs
cellpose_root = os.path.join(dm11_root, 'code/github/cellpose/')
sys.path.insert(0, cellpose_root)
from cellpose import transforms, datasets

def pad_224(img):
    Ly,Lx = img.shape[:2]
    lyp = max(0,int(np.ceil(224-Ly)/2)+1)
    lxp = max(0,int(np.ceil(224-Lx)/2)+1)
    if img.ndim>2:
        pads = np.array([[lyp,lyp],[lxp,lxp],[0,0]])
    else:
        pads = np.array([[lyp,lyp],[lxp,lxp]])
    I = np.pad(img,pads, mode='constant')
    return I


parser = argparse.ArgumentParser(description='stardist training')
parser.add_argument('--LR', default = .0003, type=float, help='initial learning rate')
parser.add_argument('--cyto', default = 1, type=int, help='dataset')
parser.add_argument('--nepochs', default = 500, type=int, help='number of epochs')
parser.add_argument('--specialist', default = 0, type=int, help='specialist training?')
parser.add_argument('--batch_size', default = 8, type=int, help='batch_size')

args = parser.parse_args() 
cyto = np.bool(args.cyto)
specialist = np.bool(args.specialist)
nepochs = args.nepochs
batch_size = args.batch_size
learning_rate = args.LR

root_data = os.path.join(dm11_root, 'datasets/cellpose/')

if cyto:
    data_str = 'cyto'
    filename = os.path.join(root_data, 'vf_cyto5.pickle')
    filecell = os.path.join(root_data,'iscell.npy')
    diam_mean = 27.
else:
    data_str = 'nuclei'
    filename = os.path.join(root_data, 'vf_nuclei.pickle')
    filecell = os.path.join(root_data, 'isnuc.npy')
    diam_mean = 15.

if specialist:
    sp_str = '_sp'
else:
    sp_str = ''

print('dataset: %s , specialist? %d , nepochs: %d, LR: %0.5f'%(data_str, specialist, nepochs, learning_rate))

(train_data, train_labels, train_cell, 
    test_data, test_labels, test_cell) = datasets.load_pickle(filename, filecell, cyto, specialist)

# divide into train and validation
rng = np.random.RandomState(42)
ind = rng.permutation(len(train_data))
n_val = max(1, int(round(0.15 * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]

# make sure minimum size of 224 x 224
X_trn, Y_trn = ([pad_224(np.transpose(train_data[i], (1,2,0))) for i in ind_train]  , 
                [pad_224(train_labels[i][0].astype(np.uint16)) for i in ind_train])
X_val, Y_val = ([pad_224(np.transpose(train_data[i], (1,2,0))) for i in ind_val], 
                [pad_224(train_labels[i][0].astype(np.uint16)) for i in ind_val])
X_test, Y_test = ([np.transpose(test_data[i], (1,2,0)) for i in range(len(test_data))], 
                    [test_labels[i][0].astype(np.uint16) for i in range(len(test_data))])

print(train_data[0].shape)
print(X_trn[0].shape)
print('number of images: %3d' % len(train_data))
print('- training:       %3d' % len(X_trn))
print('- validation:     %3d' % len(X_val))
print('- testing:        %3d' % len(X_test))                                                                    

# 32 is a good default choice (see 1_data.ipynb)
n_rays = 32
use_gpu = False

# Predict on subsampled grid for increased efficiency and larger field of view
grid = (2,2)
n_channel = 1 + int(cyto) # two channels if cytoplasm
conf = Config2D (
    n_rays       = n_rays,
    grid         = grid,
    use_gpu      = use_gpu,
    n_channel_in = n_channel,
    train_batch_size = batch_size,
    train_patch_size = (224, 224),
    train_epochs = nepochs,
    train_steps_per_epoch = int(np.ceil(len(X_trn)/batch_size)),
    train_learning_rate = learning_rate
)
print(conf)
modeldir = 'stardist_%s%s_%0.4f'%(data_str, sp_str, learning_rate)
model = StarDist2D(conf, name=modeldir, basedir=basedir)

augmenter = transforms.random_rotate_and_resize

# train model
model.train(X_trn, Y_trn, validation_data=(X_val, Y_val), augmenter=augmenter)

# after training, get AVERAGE loss over all images
data_kwargs = dict (
            n_rays           = conf.n_rays,
            patch_size       = conf.train_patch_size,
            grid             = conf.grid,
            shape_completion = conf.train_shape_completion,
            b                = conf.train_completion_crop,
            use_gpu          = conf.use_gpu,
        )
data_train = StarDistData2D(X_trn, Y_trn, batch_size=batch_size, 
                            augmenter=augmenter, **data_kwargs)
losses = model.keras_model.evaluate_generator(data_train)
print('>>>>>>>>>>>>>> AVG LOSS: %0.5f <<<<<<<<<<<<<<<<<'%losses[0])

# optimize thresholds for cell prob
model.optimize_thresholds(X_val, Y_val)

# predict test masks
masks = [model.predict_instances(X_test[i])[0] for i in range(len(X_test))]
np.save(os.path.join(basedir, modeldir, 'test_masks.npy'), masks)

# score output
rez = matching.matching_dataset(Y_test, masks, thresh=[0.5,0.75,.9], by_image=True)
print(rez)
