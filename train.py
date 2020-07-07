"""
This code can be used to train NuClick on your own data. 
You can train from scratch or fine-tuning one of the existing models.
Generally, training models follows the scheme of one of following applications:
    Nucleus, Cell, Glands
However, you can control the training process by adjusting the parameters in the 
config.py files.

For more information, read our arXiv paper:
    NuClick: From Clicks in the Nuclei to Nuclear Boundaries
    https://arxiv.org/abs/1909.03253
"""

from __future__ import print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
from skimage.io import imsave
import numpy as np
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K
from data_handler.npyDataOps import loadData
from data_handler.customImageGenerator import ImageDataGenerator
from config import config
import h5py
import warnings
from models.models import getModel
from utils.ModelCheckpointMGPU import ModelCheckpointMGPU
warnings.filterwarnings("ignore")

# Setting Parameters
SEED = 1
np.random.seed(SEED)
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


# Defining train/validation image generators and their argumants
train_gen_args = dict(
    RandomizeGuidingSignalType = config.guidingSignalType,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    zoom_range=(.5, 1.5) if config.application=='Gland' else (.7, 1.3),  
    shear_range=0.2 if config.application=='Gland' else 0.1,  
    fill_mode='constant',  # Applicable to image onlyS
    albumentation=True,
    rescale=1. / 255)
image_datagen = ImageDataGenerator(**train_gen_args)
image_datagen_val = ImageDataGenerator(rescale=1./255)
    
if not config.valid_data_path==None:
    num_train = imgs.shape[0]  # 0
    num_val = imgs_test.shape[0]
    train_generator = image_datagen.flow( imgs, weightMap=guidingSignals, mask=masks,
        shuffle=True, batch_size=batchSize, color_mode='rgb', seed=SEED)
    val_generator = image_datagen_val.flow(imgs_test, weightMap=guidingSignals_test, mask=masks_test,
        shuffle=False, batch_size=batchSizeVal, color_mode='rgb', seed=SEED)
else:
    num_train = np.round((1-config.valPrec)*imgs.shape[0])
    num_val = imgs.shape[0]-num_train
    train_generator = image_datagen.flow(imgs[:num_train,], weightMap=guidingSignals[:num_train,], mask=masks[:num_train,],
        shuffle=True, batch_size=batchSize, color_mode='rgb', seed=SEED)
    val_generator = image_datagen_val.flow(imgs[num_train:,], weightMap=guidingSignals[num_train:,], mask=masks[num_train:,],
        shuffle=False, batch_size=batchSizeVal, color_mode='rgb', seed=SEED)
    imgNames_test = imgNames[num_train:,]

print('-' * 30)
print('Creating and compiling model...')
print('-' * 30)
# defining names and adresses
modelBaseName = 'NuClick_%s_%s_%s' % (config.application, modelType, lossType)
if not os.path.exists(config.weights_path):
    os.mkdir(config.weights_path)
modelSaveName = "%s/weights-%s.h5" % (config.weights_path, modelBaseName)
modelLogName = "%s/Log-%s.log" % (config.weights_path, modelBaseName)
csv_logger = CSVLogger(modelLogName, append=True, separator='\t')

# creating model instance
model = getModel(modelType, lossType, input_shape)
if config.resumeTraining:
    model.load_weights(modelSaveName)
    
if config.multiGPU:
    model_checkpoint = ModelCheckpointMGPU(model, filepath=modelSaveName, monitor='val_loss', mode='min', save_best_only=True)
else:
    model_checkpoint = ModelCheckpoint(filepath=modelSaveName, monitor='val_loss', save_best_only=True)

print('-' * 30)
print('Fitting model...')
print('-' * 30)
history = model.fit_generator(train_generator, steps_per_epoch=num_train//batchSize, nb_epoch=5,
                              validation_data=val_generator, validation_steps=num_val//batchSizeVal, 
                              callbacks=[model_checkpoint, csv_logger], max_queue_size=64, workers=8)
history = model.fit_generator(train_generator, steps_per_epoch=num_train//batchSize, nb_epoch=150,
                              validation_data=val_generator, validation_steps=num_val//batchSizeVal, 
                              callbacks=[model_checkpoint, csv_logger], max_queue_size=64, workers=8)

print('*' * 90)
print('Training is Done')
print('*' * 90)

if config.outputValPreds:
    print('-' * 30)
    print('Predicting on validation...')
    print('-' * 30)
    model.load_weights(modelSaveName)
    batchSizeVal = 1
    val_generator.shuffle=False
    val_generator.batch_size=1
    val_predicts  = model.predict_generator(val_generator, steps=num_val//batchSizeVal)
    
    pred_dir = "%s/valPred_%s" % (config.preds_path, modelBaseName)
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    
    imgs_mask_test = np.matrix.squeeze(val_predicts, axis=3)
    for image_id in range(0, len(imgs_mask_test)):
        mask = np.uint8(imgs_mask_test[image_id, :, :] * 255)
        imsave(os.path.join(pred_dir, imgNames_test[image_id] + '_pred.png'), mask)