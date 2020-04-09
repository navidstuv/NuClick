# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 12:53:23 2018

@author: Mosi
"""

from __future__ import print_function
import glob
import gc
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from skimage.transform import resize
from skimage.io import imsave, imread
from scipy.misc import imresize
import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, Dropout, \
    Activation, UpSampling2D, add, Dense, Flatten, AveragePooling2D, Add
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, RemoteMonitor, TensorBoard, CSVLogger, TerminateOnNaN, \
    LearningRateScheduler
from keras import backend as K
from keras.utils import multi_gpu_model
import tensorflow as tf
from PIL import Image, ImageOps
import random
from math import floor, ceil, pow
# from nasUnet import NASUNet
# from se import squeeze_excite_block
# from data import load_train_data, load_test_data
# from elastic_functions import perform_elastic_3image
import matplotlib.pyplot as plt
from data import load_data_single
from image_segmentation_singleDist_v2 import ImageDataGenerator
from skimage import exposure
# %matplotlib inline
import h5py
import cv2
import warnings
from model_factory1 import getModel
from skimage.morphology import binary_dilation

warnings.filterwarnings("ignore")

# Setting Parameters
seeddd = 1
np.random.seed(seeddd)
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 256  # 480#640
img_cols = 256  # 768#1024
img_chnls = 3
input_shape = (img_rows, img_cols)

modelType = 'MultiScaleResUnet'
cellLoss = 'complexBCEweighted'
marginLoss = 'jaccard'
batchSize = 4  # set this as large as possible
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
scalingFactor = 64.0
imgs, masks, JaccWeights, bceWeights, pointNucs, pointOthers, imgNames = load_data_single('E:/Nuclick project_Hemato/Data/nuclick_data/Train/')
#for i in range(len(pointNucs)):
#    pointNucs[i,] = binary_dilation(pointNucs[i,],np.ones((3,3)))

JaccWeights = JaccWeights[..., np.newaxis]  # margins = margins.astype('float32')
bceWeights = bceWeights[..., np.newaxis]  # sepBorders = sepBorders.astype('float32')
masks = masks[..., np.newaxis]
pointNucs = pointNucs[..., np.newaxis]
pointOthers = pointOthers[..., np.newaxis]
#nucDists/=scalingFactor;
#othersDists/=scalingFactor;
dists = np.concatenate((pointNucs,pointOthers,bceWeights),axis=3)
#dists = np.concatenate((pointNucs,JaccWeights),axis=3)#
del bceWeights
del JaccWeights
del pointNucs
del pointOthers
print('Train data loading is done.')


''' Loading and preprocessing test data'''
print('-' * 30)
print('Loading test data...')
print('-' * 30)
imgs_test, masks_test, JaccWeights_test, bceWeights_test, pointNucs_test, pointOthers_test, imgNames_test = load_data_single('E:/Nuclick project_Hemato/Data/nuclick_data/Validation/')
#for i in range(len(pointNucs_test)):
#    pointNucs_test[i,] = binary_dilation(pointNucs_test[i,],np.ones((3,3)))

JaccWeights_test = JaccWeights_test[..., np.newaxis]  # margins = margins.astype('float32')
bceWeights_test = bceWeights_test[..., np.newaxis]  # sepBorders = sepBorders.astype('float32')
masks_test = masks_test[..., np.newaxis]
pointNucs_test = pointNucs_test[..., np.newaxis]
pointOthers_test = pointOthers_test[..., np.newaxis]
#nucDists_test /= scalingFactor;
#othersDists_test /= scalingFactor;
dists_test = np.concatenate((pointNucs_test,pointOthers_test,bceWeights_test),axis=3)
#dists_test = np.concatenate((pointNucs_test,JaccWeights_test),axis=3)

del bceWeights_test
del JaccWeights_test
del pointNucs_test
del pointOthers_test
print('Test data loading is done.')




''' Creating Train and validation lists for cross-validation experiments based on image list'''


# Initiating data generators
train_gen_args = dict(
    random_click_perturb = 'Train',
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

image_datagen_val = ImageDataGenerator(random_click_perturb = 'Train',
    rescale=1. / 255)

'''
Cross-Validation::: Loop over the different folds and perform train on them.
Save the best model which has best performance on validation set in each fold.
'''
modelBaseName = 'nuclickHemato_%s_%s' % (modelType, cellLoss)
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

model = getModel(modelType, cellLoss, marginLoss, input_shape)
#model.load_weights(modelSaveName)
#if multi_gpu:
#    with tf.device("/cpu:0"):
#        model = getModel(modelType, cellLoss, marginLoss, input_shape) 
#else:
#    model = getModel(modelType, cellLoss, marginLoss, input_shape)
#
#if multi_gpu:
#    model = multi_gpu_model(model, len(gpus))
    
# model.load_weights(modelSaveName)
#model_checkpoint = ModelCheckpointMGPU(model, filepath=modelName + '/'+'weights.{epoch:02d}-{val_loss:.2f}.h5', save_weights_only =True)
#model_checkpoint = ModelCheckpointMGPU(model, filepath=modelSaveName, monitor='val_loss', mode='min', save_best_only=True)
model_checkpoint = ModelCheckpoint(filepath=modelSaveName, monitor='val_loss', save_best_only=True)

print('-' * 30)
print('Fitting model...')
print('-' * 30)
# model.load_weights(modelSaveName)
history = model.fit_generator( train_generator,steps_per_epoch=num_train//batchSize,nb_epoch=30,validation_data = val_generator,
                               validation_steps=num_val//batchSizeVal , callbacks=[model_checkpoint,csv_logger], max_queue_size=64, workers=8)
history = model.fit_generator(train_generator, steps_per_epoch=num_train // batchSize, nb_epoch=300,
                              validation_data=val_generator,
                              validation_steps=num_val // batchSizeVal, callbacks=[model_checkpoint, csv_logger],
                              max_queue_size=64, workers=8)
#
# print('-' * 30)
# print('Predicting on validation...')
# print('-' * 30)
model.load_weights(modelSaveName)
batchSizeVal = 1
val_generator = image_datagen_val.flow(
    imgs_test, weightMap=dists_test, mask1=masks_test,
    shuffle=False,
    batch_size=batchSizeVal,
    color_mode='rgb',
    seed=seeddd)

val_predicts  = model.predict_generator(val_generator, steps=num_val // batchSizeVal)
pred_dir = "./%s/valPred_%s" % (modelBaseName, modelBaseName)
imgs_mask_test = np.matrix.squeeze(val_predicts, axis=3)
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)
for image_id in range(0, len(imgs_mask_test)):
    mask = np.uint8(imgs_mask_test[image_id, :, :] * 255)
    imsave(os.path.join(pred_dir, imgNames_test[image_id] + '_mask.png'), mask)
#
# del model
# K.clear_session()
# gc.collect()

print('*' * 90)
print('Done, Go and enjoy life.')
print('*' * 90)
