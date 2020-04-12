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
from data_handler.npyDataOps import loadData
from data_handler.customImageGenerator import ImageDataGenerator
from config import config
import h5py
import warnings
from models.models import getModel

from utils.ModelCheckpointMGPU import ModelCheckpointMGPU
warnings.filterwarnings("ignore")

# Setting Parameters
seeddd = 1
np.random.seed(seeddd)
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = config.img_rows  # 480#640
img_cols = config.img_cols  # 768#1024
img_chnls = config.img_chnls

input_shape = (img_rows, img_cols)

modelType = config.modelType
lossType = config.lossType
batchSize = config.batchSize
batchSizeVal = batchSize  

gpus = [x.name for x in K.device_lib.list_local_devices() if x.name[:11] == '/device:GPU']
multiGPU = config.multiGPU





print('-' * 30)
print('Loading data...')
print('-' * 30)
imgs, masks, weightMaps, objectPoints, pointOthers, imgNames = loadData(config.train_data_path)
weightMaps = weightMaps[..., np.newaxis]
masks = masks[..., np.newaxis]
objectPoints = objectPoints[..., np.newaxis]
pointOthers = pointOthers[..., np.newaxis]
guidingSignals = np.concatenate((objectPoints,pointOthers,weightMaps),axis=3)
del weightMaps
del objectPoints
del pointOthers
print('Train data loading is done.')

if not config.valid_data_path==None:
    imgs_test, masks_test, weightMaps_test, objectPoints_test, pointOthers_test, imgNames_test = loadData(config.valid_data_path)
    weightMaps_test = weightMaps_test[..., np.newaxis]
    masks_test = masks_test[..., np.newaxis]
    objectPoints_test = objectPoints_test[..., np.newaxis]
    pointOthers_test = pointOthers_test[..., np.newaxis]
    guidingSignals_test = np.concatenate((objectPoints_test, pointOthers_test, weightMaps_test),axis=3)
    del weightMaps_test
    del objectPoints_test
    del pointOthers_test
    print('Test data loading is done.')


if config.application=='nucleus':
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
    modelBaseName = 'nuclickNuclei_%s_%s' % (modelType, lossType)
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

if config.application=='cell':
    # Initiating data generators
    train_gen_args = dict(
        random_click_perturb='Train',
        #    width_shift_range=0.1,
        #    height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=20.,
        zoom_range=(.7, 1.3),  # (0.7, 1.3),
        shear_range=.1,
        fill_mode='constant',  # Applicable to image onlyS
        albumentation=True,
        #    channel_shift_range=False,  # This must be in range of 255?
        #    contrast_adjustment=False,  #####MOSI
        #    illumination_gradient=False,
        #    intensity_scale_range=0.,  #####MOSI
        #    sharpness_adjustment=False,
        #    apply_noise=False,
        #    elastic_deformation=False,
        rescale=1. / 255
    )

    image_datagen = ImageDataGenerator(**train_gen_args
                                       )

    image_datagen_val = ImageDataGenerator(random_click_perturb='Train',
                                           rescale=1. / 255)

    '''
    Cross-Validation::: Loop over the different folds and perform train on them.
    Save the best model which has best performance on validation set in each fold.
    '''
    modelBaseName = 'nuclickHemato_%s_%s' % (modelType, lossType)
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

if config.application=='gland':
    # Initiating data generators
    pointMapT = 'Skeleton'
    train_gen_args = dict(
        random_click_perturb='Skeleton',
        pointMapType=pointMapT,
        #    width_shift_range=0.1,
        #    height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=20.,
        zoom_range=(.5, 1.5),  # (0.7, 1.3),T
        shear_range=.2,
        fill_mode='constant',  # Applicable to image onlyS
        albumentation=True,
        #    channel_shift_range=False,  # This must be in range of 255?
        #    contrast_adjustment=False,  #####MOSI
        #    illumination_gradient=False,
        #    intensity_scale_range=0.,  #####MOSI
        #    sharpness_adjustment=False,
        #    apply_noise=False,
        #    elastic_deformation=True,
        rescale=1. / 255
    )

    image_datagen = ImageDataGenerator(**train_gen_args
                                       )

    image_datagen_val = ImageDataGenerator(random_click_perturb='SkeletonValid', pointMapType=pointMapT,
                                           rescale=1. / 255)

    '''
    Cross-Validation::: Loop over the different folds and perform train on them.
    Save the best model which has best performance on validation set in each fold.
    '''
    modelBaseName = 'nuclickGland_%s_%s_%s' % (pointMapT, modelType, lossType)
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

model = getModel(modelType, lossType, input_shape)
model.load_weights(modelSaveName)

if config.multiGPU:
    model_checkpoint = ModelCheckpointMGPU(model, filepath=modelSaveName, monitor='val_loss', mode='min', save_best_only=True)
else:
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
