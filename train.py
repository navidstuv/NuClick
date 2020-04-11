# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 12:53:23 2018

@author: Mosi
"""

from __future__ import print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
from skimage.io import imsave
import numpy as np
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K

import matplotlib.pyplot as plt
from data_handler.data import load_data_single
from data_handler.image_segmentation_singleDist_v2 import ImageDataGenerator
from config import config
import h5py
import warnings
from models.models import getModel

warnings.filterwarnings("ignore")

# Setting Parameters
seeddd = 1
np.random.seed(seeddd)
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 128  # 480#640
img_cols = 128  # 768#1024
img_chnls = 3
input_shape = (img_rows, img_cols)

modelType = config.modelType #
cellLoss = 'bce_dice'
batchSize = 32  # set this as large as possible
batchSizeVal = batchSize  # leaave this to 1 anyway

gpus = [x.name for x in K.device_lib.list_local_devices() if x.name[:11] == '/device:GPU']
multi_gpu = True  # NAVID DO THIS!`


class ModelCheckpointMGPU(ModelCheckpoint):
    def __init__(self, original_model, filepath, monitor='val_loss', verbose=0, save_best_only=False,
                 save_weights_only=False, mode='auto', period=1):
        self.original_model = original_model
        super().__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)

    def on_epoch_end(self, epoch, logs=None):
        self.model = self.original_model
        super().on_epoch_end(epoch, logs)



''' Loading and preprocessing train data'''
print('-' * 30)
print('Loading training data...')
print('-' * 30)
imgs, masks, JaccWeights, bceWeights, pointNucs, pointOthers, imgNames = load_data_single('D:/Nuclick project/Data/nuclick_data/Train/')
#for i in range(len(pointNucs)):
#    pointNucs[i,] = binary_dilation(pointNucs[i,],np.ones((3,3)))
JaccWeights = JaccWeights[..., np.newaxis]  # margins = margins.astype('float32')
bceWeights = bceWeights[..., np.newaxis]  # sepBorders = sepBorders.astype('float32')
masks = masks[..., np.newaxis]
pointNucs = pointNucs[..., np.newaxis]
pointOthers = pointOthers[..., np.newaxis]
dists = np.concatenate((pointNucs,pointOthers,JaccWeights),axis=3)
del bceWeights
del JaccWeights
del pointNucs
del pointOthers
print('Train data loading is done.')


''' Loading and preprocessing test data'''
print('-' * 30)
print('Loading test data...')
print('-' * 30)
imgs_test, masks_test, JaccWeights_test, bceWeights_test, pointNucs_test, pointOthers_test, imgNames_test = load_data_single('D:/Nuclick project/Data/nuclick_data/Validation/')
JaccWeights_test = JaccWeights_test[..., np.newaxis]  # margins = margins.astype('float32')
bceWeights_test = bceWeights_test[..., np.newaxis]  # sepBorders = sepBorders.astype('float32')
masks_test = masks_test[..., np.newaxis]
pointNucs_test = pointNucs_test[..., np.newaxis]
pointOthers_test = pointOthers_test[..., np.newaxis]
dists_test = np.concatenate((pointNucs_test,pointOthers_test,JaccWeights_test),axis=3)
del bceWeights_test
del JaccWeights_test
del pointNucs_test
del pointOthers_test
print('Test data loading is done.')


# Initiating data generators
train_gen_args = dict(
    random_click_perturb = 'Train',
#    width_shift_range=0.1,
#    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    zoom_range=(.6, 1.4),  # (0.7, 1.3),
    shear_range=.1,
    fill_mode='constant',  # Applicable to image onlyS
    albumentation=True,
#    channel_shift_range=False,
#    contrast_adjustment=False,
#    illumination_gradient=False,○
#    intensity_scale_range=0.,
#    sharpness_adjustment=False,
#    apply_noise=False,
#    elastic_deformation=False,
    rescale=1. / 255
)

image_datagen = ImageDataGenerator(**train_gen_args)

image_datagen_val = ImageDataGenerator(random_click_perturb = 'Train',
    rescale=1. / 255)

modelBaseName = 'nuclickNuclei_%s_%s' % (modelType, cellLoss)
if not os.path.exists(modelBaseName):
    os.mkdir(modelBaseName)

train_generator = image_datagen.flow(
    imgs, weightMap=dists, mask1=masks,
    shuffle=True,
    batch_size=batchSize,
    color_mode='rgb',  # rgbhsvl
    seed=seeddd)
val_generator = image_datagen_val.flow(
    imgs_test, weightMap=dists_test, mask1=masks_test,
    shuffle=False,
    batch_size=batchSizeVal,
    color_mode='rgb',
    seed=seeddd)

num_train = imgs.shape[0]  # 0
num_val = imgs_test.shape[0]  # 0

print('-' * 30)
print('Creating and compiling model...')
print('-' * 30)
modelName = "%s" % (modelBaseName)
modelSaveName = "./%s/weights-%s.h5" % (modelBaseName, modelName)
modelLogName = "./%s/Log-%s.log" % (modelBaseName, modelName)
logDir = "./%s/log" % (modelBaseName)
csv_logger = CSVLogger(modelLogName, append=True, separator='\t')

model = getModel(modelType, cellLoss, input_shape)
model.load_weights(modelSaveName)

model_checkpoint = ModelCheckpoint(filepath=modelSaveName, monitor='val_loss', save_best_only=True)

print('-' * 30)
print('Fitting model...')
print('-' * 30)

history = model.fit_generator(train_generator, steps_per_epoch=num_train // batchSize, nb_epoch=150,
                              validation_data=val_generator,
                              validation_steps=num_val // batchSizeVal, callbacks=[model_checkpoint, csv_logger],
                              max_queue_size=64, workers=8)

print('*' * 90)
print('Done')
print('*' * 90)
