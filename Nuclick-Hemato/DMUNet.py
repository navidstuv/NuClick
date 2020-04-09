
"""
This Network has connection from all level of decending path to the current level in the acending path!!!!
It interplates the feature maps to be able to caoncatenating them with the desired level feature map size.
It is a self feeding network!!! train quickly.

"""
from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Dropout, AlphaDropout, BatchNormalization , Activation
from keras.layers import UpSampling2D, Lambda, add, AveragePooling2D
from keras.optimizers import Adam, SGD
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint,CSVLogger
from keras.regularizers import l2
from keras import backend as K
import tensorflow as tf
import h5py
#from losses import getLoss


import warnings
warnings.filterwarnings("ignore")



K.set_image_data_format('channels_last')  # TF dimension ordering in this code

#img_rows = 1024
#img_cols = 1024
img_chls = 3
#input_shape = (img_rows,img_cols)

weight_decay=5e-5
smooth = 1.
bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
def dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def weighted_dice_loss(weights,reps=1):
    def loss(y_true, y_pred):
        if reps>1:
            weights_r = K.repeat_elements(weights,reps,axis=K.ndim(weights)-1)
        else:
            weights_r = weights
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        weight_f = K.flatten(weights_r)
        intersection = K.sum(y_true_f * y_pred_f)
        return 1-(2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f*weight_f) + smooth)
    return loss

def jaccard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    y_true_f_s = K.pow(y_true_f,2)
    y_pred_f_s = K.pow(y_pred_f,2)
    return (intersection + smooth) / (K.sum(y_true_f_s) + K.sum(y_pred_f_s) - intersection + smooth)
    
def jaccard_loss(y_true, y_pred):
    return 1-jaccard_coef(y_true, y_pred)
	
def weighted_jaccard_loss(weights,reps=1):
    def loss(y_true, y_pred):
        if reps>1:
            weights_r = K.repeat_elements(weights,reps,axis=K.ndim(weights)-1)
        else:
            weights_r = weights
        weight_f = K.flatten(weights_r)
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        y_true_f_s = K.pow(y_true_f,2)
        y_pred_f_s = K.pow(y_pred_f,2)
        return 1 - (intersection + smooth) / (K.sum(y_true_f_s) + K.sum(y_pred_f_s*weight_f) - intersection + smooth)
    return loss

def weighted_binary_crossentropy(y_true, y_pred,weights,reps=1):
    if reps>1:
        weights_r = K.repeat_elements(weights,reps,axis=K.ndim(weights)-1)
    else:
        weights_r = weights
    weight_f = K.flatten(weights_r)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    bce = K.binary_crossentropy(y_true_f, y_pred_f)
    return K.mean(bce*weight_f, axis=-1)
    
def complex_loss(weights,a=1.,b=1.): #'''*** dice_loss can be replaced with jaccard_loss***'''
    def loss(y_true, y_pred):
        cmplxLoss = a*jaccard_loss(y_true, y_pred) + b*weighted_binary_crossentropy(y_true, y_pred,weights,1)
        return cmplxLoss
    return loss
    
'''    
##################### DEFINING MAIN BLOCKS #######################################
'''      
def _conv_bn_relu(input, features=32, kernelSize=(3,3), strds=(1, 1), actv = 'relu', useBias=False, useRegulizer=False, dilatationRate=(1, 1), doBatchNorm =True):
    if useRegulizer:
        kernelRegularizer=l2(weight_decay)
    else:
        kernelRegularizer=None
    if actv=='selu':
        kernel_init='lecun_normal'
    else:
        kernel_init='glorot_uniform'
    convB1 = Conv2D(features, kernelSize, strides=strds,  padding='same', use_bias=useBias, kernel_regularizer=kernelRegularizer, kernel_initializer=kernel_init, dilation_rate=dilatationRate)(input)
    if actv!='selu' and doBatchNorm:
        convB1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(convB1)
    if actv!='None':
        convB1 = Activation(actv)(convB1)
    return convB1
    
def multiScaleConv_block (input_map,features, sizes, dilatationRates, strds=(1, 1), actv = 'relu', useBias=False, useRegulizer=False):
    conv0 = _conv_bn_relu(input_map, 4*features, 1, strds, actv, useBias, useRegulizer)
    conv1 = _conv_bn_relu(conv0, features, sizes[0], strds, actv, useBias, useRegulizer, (dilatationRates[0],dilatationRates[0]))
    conv2 = _conv_bn_relu(conv0, features, sizes[1], strds, actv, useBias, useRegulizer, (dilatationRates[1],dilatationRates[1]))
    conv3 = _conv_bn_relu(conv0, features, sizes[2], strds, actv, useBias, useRegulizer, (dilatationRates[2],dilatationRates[2]))
    conv4 = _conv_bn_relu(conv0, features, sizes[3], strds, actv, useBias, useRegulizer, (dilatationRates[3],dilatationRates[3]))
    output_map = concatenate([conv1, conv2, conv3, conv4], axis=bn_axis)
    output_map = _conv_bn_relu(output_map, features, 3, strds, actv, useBias, useRegulizer)
    output_map = concatenate([input_map, output_map], axis=bn_axis)
    return output_map

def conv_block (x,features, size, strds=(1, 1), actv = 'relu', useBias=False, useRegulizer=False):
    x1 = _conv_bn_relu(x, 4*features, 1, strds, actv, useBias, useRegulizer)
    x1 = _conv_bn_relu(x1, features, 3, strds, actv, useBias, useRegulizer)
    x = concatenate([x, x1], axis=bn_axis)
    return x
    
def transition_block(x, reduction):
    x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1, use_bias=False, padding='same')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation('relu')(x)
    xp = AveragePooling2D(2, strides=2)(x)
    return xp, x

def transition_block_up(x, reduction):
    x = Conv2DTranspose(int(K.int_shape(x)[bn_axis] * reduction), 2, strides=(2, 2), use_bias=False, padding='same')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation('relu')(x)
    return x

def dense_multiscale_block(x, blocks, sizes=[3,5,7,9], dilatationRates=[1,1,1,1], growth=32):
    for i in range(blocks):
        x = multiScaleConv_block(x, growth, sizes, dilatationRates)
    return x

def dense_block(x, blocks, sizes=3, growth=32):
    for i in range(blocks):
        x = conv_block (x,growth, sizes)
    return x

'''    
##################### DEFINING NETWORKS #######################################
'''    
def get_Dense_MultiScale_UNet(inputs,input_shape):
# ALL DECONV IN DECODING PATH
    k = 32 # growth rate
#    inputs = Input(input_shape+(img_chls,), name='main_input') # 7 channel input?!
    input_shape = K.int_shape(inputs) 
    ## ENCODING PATH
    conv1 =  _conv_bn_relu(inputs, 64, 7)
    #level1
    conv1 = dense_block(conv1, 1, sizes=3, growth=k) 
    conv1 = dense_multiscale_block(conv1, 1, sizes=[3,3,5,7], dilatationRates=[1,2,2,1], growth=k) # FOV = [3,5,9,7]
    pool1, conv1 = transition_block(conv1, .5) # number of outputs = [64]
    conv1 = concatenate([inputs, conv1], axis=bn_axis)
    toConcat1 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0]//2,input_shape[1]//2),method=tf.image.ResizeMethod.BILINEAR))(inputs)
    pool1 = concatenate([pool1, toConcat1], axis=bn_axis)
    
    #level2
    conv2 = dense_block(pool1, 2, sizes=3, growth=k) 
    conv2 = dense_multiscale_block(conv2, 1, sizes=[3,5,3,5], dilatationRates=[1,1,4,4], growth=k) # FOV = [3,5,9,17]
    pool2, conv2 = transition_block(conv2, .5) # number of outputs = [96]
    conv2 = concatenate([toConcat1, conv2], axis=bn_axis)
    toConcat2 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0]//4,input_shape[1]//4),method=tf.image.ResizeMethod.BILINEAR))(inputs)
    pool2 = concatenate([pool2, toConcat2], axis=bn_axis)
    
    #level3
    conv3 = dense_block(pool2, 2, sizes=3, growth=k) 
    conv3 = dense_multiscale_block(conv3, 1, sizes=[3,5,5,5], dilatationRates=[1,1,2,3], growth=k) # FOV = [3,5,9,13]
    pool3, conv3 = transition_block(conv3, .5) # number of outputs = [176]
    conv3 = concatenate([toConcat2, conv3], axis=bn_axis)
    toConcat3 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0]//8,input_shape[1]//8),method=tf.image.ResizeMethod.BILINEAR))(inputs)
    pool3 = concatenate([pool3, toConcat3], axis=bn_axis)
    
    #level4
    conv4 = dense_multiscale_block(pool3, 2, sizes=[3,5,5,5], dilatationRates=[1,1,2,3], growth=k) # FOV = [3,5,9,13]# number of outputs = [368]
#    pool4, conv4 = transition_block(conv4, .5) # number of outputs = [368]
    conv4 = concatenate([toConcat3, conv4], axis=bn_axis)
#    toConcat4 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0]//16,input_shape[1]//16),method=tf.image.ResizeMethod.BILINEAR))(inputs)
#    pool4 = concatenate([pool4, toConcat4], axis=bn_axis)


    ## DECODING PATH
    #level3
    up3 = transition_block_up(conv4, .5)
    up3 = concatenate([up3, conv3], axis=bn_axis)
    conv5 = dense_block(up3, 2, sizes=3, growth=k) 
    conv5 = dense_multiscale_block(conv5, 2, sizes=[3,5,3,5], dilatationRates=[1,1,6,6], growth=k) # FOV = [3,5,13,25] #number of outputs = [552]
    
    #level2
    up2 = transition_block_up(conv5, .5)
    up2 = concatenate([up2, conv2], axis=bn_axis)
    conv6 = dense_multiscale_block(up2, 2, sizes=[3,3,5,7], dilatationRates=[1,4,6,8], growth=k) # FOV = [3,9,25,49] #number of outputs = [500]
    conv6 = dense_block(conv6, 2, sizes=3, growth=k) 
    conv6 = dense_multiscale_block(conv6, 2, sizes=[3,3,5,7], dilatationRates=[1,4,6,8], growth=k) # FOV = [3,9,25,49] #number of outputs = [500]
    
    #level1
    up1 = transition_block_up(conv6, .5)
    up1 = concatenate([up1, conv1], axis=bn_axis)
    conv7 = dense_multiscale_block(up1, 2, sizes=[3,3,5,7], dilatationRates=[1,4,6,8], growth=k) # FOV = [3,9,25,49] #number of outputs = [378]
    conv7 = dense_block(conv7, 1, sizes=3, growth=k) 
    conv7 = dense_multiscale_block(conv7, 1, sizes=[3,3,5,7], dilatationRates=[1,4,6,8], growth=k) # FOV = [3,9,25,49] #number of outputs = [378]
    
    conv8 =  _conv_bn_relu(conv7, 64, 3)
#    conv8 = concatenate([inputs, conv8], axis=bn_axis) # NVAID -- Be or not to be!!! that's the problem!
    cell_output = Conv2D(6, (1, 1), activation='sigmoid', name='cell_output', padding='same', use_bias=False)(conv8)

    model = Model(inputs=[inputs], outputs=[cell_output])
 
    return model
