
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
from keras.utils import multi_gpu_model
import h5py
from losses import getLoss
from AdamAccumulate import AdamAccumulate
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

bn_axis = -1 if K.image_data_format() == 'channels_last' else 1


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
    
def multiScaleConv_block (input_map,features, sizes, dilatationRates, strds=(1, 1), actv = 'relu', useBias=False, useRegulizer=False, isDense=True):
    if isDense:
        conv0 = _conv_bn_relu(input_map, 4*features, 1, strds, actv, useBias, useRegulizer)
    else:
        conv0 = input_map
    conv1 = _conv_bn_relu(conv0, features, sizes[0], strds, actv, useBias, useRegulizer, (dilatationRates[0],dilatationRates[0]))
    conv2 = _conv_bn_relu(conv0, features, sizes[1], strds, actv, useBias, useRegulizer, (dilatationRates[1],dilatationRates[1]))
    conv3 = _conv_bn_relu(conv0, features, sizes[2], strds, actv, useBias, useRegulizer, (dilatationRates[2],dilatationRates[2]))
    conv4 = _conv_bn_relu(conv0, features, sizes[3], strds, actv, useBias, useRegulizer, (dilatationRates[3],dilatationRates[3]))
    output_map = concatenate([conv1, conv2, conv3, conv4], axis=bn_axis)
    if isDense:
        output_map = _conv_bn_relu(output_map, features, 3, strds, actv, useBias, useRegulizer)
        output_map = concatenate([input_map, output_map], axis=bn_axis)
    return output_map

def conv_block (x,features, size, strds=(1, 1), actv = 'relu', useBias=False, useRegulizer=False, dilatationRate=(1, 1)):
    x1 = _conv_bn_relu(x, 4*features, 1, strds, actv, useBias, useRegulizer, dilatationRate=(1, 1))
    x1 = _conv_bn_relu(x1, features, 3, strds, actv, useBias, useRegulizer, dilatationRate)
    x = concatenate([x, x1], axis=bn_axis)
    return x
    
def transition_block(x, reduction):
    x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1, use_bias=False, padding='same')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = Activation('relu')(x)
    xp = MaxPooling2D(pool_size=(2, 2))(x)
#    xp = AveragePooling2D(2, strides=2)(x)
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

def dense_block(x, blocks, sizes=3, growth=32, dilatationRate=(1,1)):
    for i in range(blocks):
        x = conv_block (x,growth, sizes, dilatationRate=dilatationRate)
    return x
    
def residual_conv(input, features=32, kernelSize=(3,3), strds=(1, 1), actv = 'relu', useBias=False, useRegulizer=False, dilatationRate=(1, 1)):
    if actv == 'selu':
        conv1 = _conv_bn_relu(input, features, kernelSize, strds, actv = 'None', useBias=useBias, useRegulizer=useRegulizer, dilatationRate=dilatationRate, doBatchNorm=False)
        conv2 = _conv_bn_relu(conv1, features, kernelSize, strds, actv = 'None', useBias=useBias, useRegulizer=useRegulizer, dilatationRate=dilatationRate, doBatchNorm=False)
    else:
        conv1 = _conv_bn_relu(input, features, kernelSize, strds, actv = 'None', useBias=useBias, useRegulizer=useRegulizer, dilatationRate=dilatationRate, doBatchNorm=True)
        conv2 = _conv_bn_relu(conv1, features, kernelSize, strds, actv = 'None', useBias=useBias, useRegulizer=useRegulizer, dilatationRate=dilatationRate, doBatchNorm=True)
    out = add([conv1, conv2])
    out = Activation(actv)(out)
    return out

def residual_conv_identity(input, features=32, kernelSize=(3,3), strds=(1, 1), actv = 'relu', useBias=False, useRegulizer=False, dilatationRate=(1, 1)):
    conv1 = _conv_bn_relu(input, features, kernelSize, strds, actv = 'None', useBias=useBias, useRegulizer=useRegulizer, dilatationRate=dilatationRate, doBatchNorm=True)
    conv2 = _conv_bn_relu(conv1, features, kernelSize, strds, actv = 'None', useBias=useBias, useRegulizer=useRegulizer, dilatationRate=dilatationRate, doBatchNorm=True)
    out = add([input, conv2])
    out = Activation(actv)(out)
    return out


'''    
##################### DEFINING NETWORKS #######################################
'''    
def get_Dense_MultiScale_UNet(input_shape, cellLoss):
    with tf.device("/cpu:0"):
    # ALL DECONV IN DECODING PATH
        k = 32 # growth rate
        img = Input(input_shape + (img_chls,), name='main_input')  # size: 1024
        auxInput = Input(input_shape + (3,), name='dists_input')
        weights = Lambda(lambda x : x[:,:,:,2])(auxInput)
        dists = Lambda(lambda x : x[:,:,:,0:2])(auxInput)
        
        inputs = concatenate([img, dists], axis=bn_axis)
        
        ## ENCODING PATH
        conv1 =  _conv_bn_relu(inputs, 64, 7) #128x128
        conv1 =  _conv_bn_relu(conv1, 64, 7) #128x128
        #level1
    #    conv1 = dense_block(conv1, 2, sizes=3, growth=k) 
    #    conv1 = dense_multiscale_block(conv1, 1, sizes=[3,3,5,7], dilatationRates=[1,2,2,1], growth=k) # FOV = [3,5,9,7]
    #    pool1, conv1 = transition_block(conv1, .5) # number of outputs = [64]
    #    toConcat1 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0]//2,input_shape[1]//2),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))(dists)
        conv1 = concatenate([dists, conv1], axis=bn_axis)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        toConcat1 = MaxPooling2D(pool_size=(2, 2))(dists)
    #    pool1 = concatenate([pool1, toConcat1], axis=bn_axis)
        
        #level2
        conv2 = dense_block(pool1, 2, sizes=3, growth=k)  #64x64
        conv2 = dense_block(conv2, 2, sizes=3, growth=k,dilatationRate=(3,3)) 
        conv2 = dense_block(conv2, 2, sizes=5, growth=k,dilatationRate=(2,2)) 
    #    conv2 = dense_multiscale_block(conv2, 2, sizes=[3,5,3,5], dilatationRates=[1,1,4,4], growth=k) # FOV = [3,5,9,17]
        pool2, conv2 = transition_block(conv2, .5) # number of outputs = [128]
        conv2 = concatenate([toConcat1, conv2], axis=bn_axis)
    #    toConcat2 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0]//4,input_shape[1]//4),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))(dists)
        toConcat2 = MaxPooling2D(pool_size=(2, 2))(toConcat1)
        pool2 = concatenate([pool2, toConcat2], axis=bn_axis)
        
        #level3
        conv3 = dense_block(pool2, 4, sizes=3, growth=k) #32x32
        conv3 = dense_block(conv3, 2, sizes=5, growth=k) #32x32    
        conv3 = dense_block(conv3, 2, sizes=5, growth=k,dilatationRate=(2,2)) #32x32   
        conv3 = dense_block(conv3, 2, sizes=5, growth=k,dilatationRate=(3,3)) #32x32   
        conv3 = dense_block(conv3, 2, sizes=3, growth=k) #32x32    
    #    conv3 = dense_multiscale_block(conv3, 3, sizes=[3,5,5,5], dilatationRates=[1,1,2,3], growth=k) # FOV = [3,5,9,13]
        pool3, conv3 = transition_block(conv3, .5) # number of outputs = [256]
        conv3 = concatenate([toConcat2, conv3], axis=bn_axis)
    #    toConcat3 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0]//8,input_shape[1]//8),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))(dists)
        toConcat3 = MaxPooling2D(pool_size=(2, 2))(toConcat2)
        pool3 = concatenate([pool3, toConcat3], axis=bn_axis)
        
        #level4
        conv4 = dense_block(pool3, 4, sizes=3, growth=k) #16x16
        conv4 = dense_block(conv4, 4, sizes=5, growth=k,dilatationRate=(1,1))
        conv4 = dense_block(conv4, 2, sizes=3, growth=k,dilatationRate=(1,1))
        conv4 = dense_block(conv4, 4, sizes=5, growth=k,dilatationRate=(2,2))
        conv4 = dense_block(conv4, 2, sizes=3, growth=k,dilatationRate=(1,1))
        conv4 = dense_block(conv4, 4, sizes=5, growth=k,dilatationRate=(3,3))
        conv4 = dense_block(conv4, 4, sizes=3, growth=k,dilatationRate=(1,1))
    #    conv4 = dense_multiscale_block(pool3, 6, sizes=[3,5,5,5], dilatationRates=[1,1,2,3], growth=k) # FOV = [3,5,9,13]# number of outputs = [368]
        pool4, conv4 = transition_block(conv4, .5) # number of outputs = [512]
        conv4 = concatenate([toConcat3, conv4], axis=bn_axis)
    #    toConcat3 = Lambda(lambda image: tf.image.resize_images(image,(input_shape[0]//8,input_shape[1]//8),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))(dists)
        toConcat4 = MaxPooling2D(pool_size=(2, 2))(toConcat3)
        pool4 = concatenate([pool4, toConcat4], axis=bn_axis)
        
        #level5
        conv4_ = dense_block(pool4, 16, sizes=3, growth=k) #8x8
    #    conv4_ = dense_multiscale_block(conv4_, 7, sizes=[3,5,7,5], dilatationRates=[1,1,1,1], growth=k) # FOV = [3,5,9,13]# number of outputs = [368]
        # number of outputs = [1024]
        
        ## DECODING PATH
        #level4
        up6 = concatenate([transition_block_up(conv4_, .5), conv4], axis=3)
        conv6 = residual_conv(up6, features=512) #16*16
        conv6 = multiScaleConv_block (conv6, 128, sizes=[3,3,5,5], dilatationRates=[1,3,2,3], isDense=False) # FOV = [3,7,9,13] 
        conv6 = residual_conv(conv6, features=512)
    
        up7 = concatenate([transition_block_up(conv6, .5), conv3], axis=3)
        conv7 = residual_conv(up7, features=256) #32*32
        conv7 = residual_conv(conv7, features=256)
        conv7 = residual_conv(conv7, features=256)
    
        up8 = concatenate([transition_block_up(conv7, .5), conv2], axis=3)
        conv8 = residual_conv(up8, features=128) #64*64
        conv8 = multiScaleConv_block (conv8, 32, sizes=[3,3,5,7], dilatationRates=[1,3,3,6], isDense=False) # FOV = [3,7,13,37] 
        conv8 = residual_conv(conv8, features=128)
        conv8 = residual_conv(conv8, features=128)
    
        up9 = concatenate([transition_block_up(conv8, .5), conv1], axis=3)
        conv9 = _conv_bn_relu(up9,64,7)
        conv9 = _conv_bn_relu(conv9,64,5)
        conv9 = _conv_bn_relu(conv9,32)
    
        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    
        model = Model(inputs=[img, auxInput], outputs=[conv10])
        
        
    model = multi_gpu_model(model , 2)
    model.compile(optimizer=Adam(lr=4e-4), loss=getLoss(cellLoss, weightMap=weights), metrics=[dice_coef]) # adding the momentum 
    return model


def get_unet(input_shape, cellLoss):
#    with tf.device("/cpu:0"):
    img = Input(input_shape + (img_chls,), name='main_input')  # size: 1024
    auxInput = Input(input_shape + (3,), name='dists_input')
    weights = Lambda(lambda x : x[:,:,:,2])(auxInput)
    dists = Lambda(lambda x : x[:,:,:,0:2])(auxInput)

    
    inputs = concatenate([img, dists], axis=bn_axis)
    
    conv1 = _conv_bn_relu(inputs,32)
    conv1 = _conv_bn_relu(conv1,32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = _conv_bn_relu(pool1,64)
    conv2 = _conv_bn_relu(conv2,64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = _conv_bn_relu(pool2,128)
    conv3 = _conv_bn_relu(conv3,128)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = _conv_bn_relu(pool3,256)
    conv4 = _conv_bn_relu(conv4,256)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = _conv_bn_relu(pool4,512)
    conv5 = _conv_bn_relu(conv5,512)

    up6 = concatenate([Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = _conv_bn_relu(up6,256)
    conv6 = _conv_bn_relu(conv6,256)

    up7 = concatenate([Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = _conv_bn_relu(up7,256)
    conv7 = _conv_bn_relu(conv7,256)

    up8 = concatenate([Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = _conv_bn_relu(up8,128)
    conv8 = _conv_bn_relu(conv8,128)

    up9 = concatenate([Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = _conv_bn_relu(up9,32)
    conv9 = _conv_bn_relu(conv9,32)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[img, auxInput], outputs=[conv10])
    
    
#    model = multi_gpu_model(model , 2)

    model.compile(optimizer=Adam(lr=3e-3), loss=getLoss(cellLoss, weightMap=weights), metrics=[dice_coef]) # adding the momentum

    return model


    
def get_MultiScale_ResUnet (input_shape, cellLoss):
#    with tf.device("/cpu:0"):
    img = Input(input_shape + (img_chls,), name='main_input')  # size: 1024
    auxInput = Input(input_shape + (3,), name='dists_input')
    weights = Lambda(lambda x : x[:,:,:,2])(auxInput)
    dists = Lambda(lambda x : x[:,:,:,0:2])(auxInput)

    
    inputs = concatenate([img, dists], axis=bn_axis)
    
    conv1 = _conv_bn_relu(inputs,64,7) #128
    conv1 = _conv_bn_relu(conv1,32,5)
    conv1 = _conv_bn_relu(conv1,32,3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = residual_conv(pool1, features=64) #64*64
    conv2 = residual_conv(conv2, features=64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = residual_conv(pool2, features=128) #32*32
    conv3 = multiScaleConv_block (conv3, 32, sizes=[3,3,5,5], dilatationRates=[1,3,3,6], isDense=False)  # FOV = [3,7,13,25] 
    conv3 = residual_conv(conv3, features=128)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = residual_conv(pool3, features=256) #16*16
    conv4 = residual_conv(conv4, features=256)
    conv4 = residual_conv(conv4, features=256)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = residual_conv(pool4, features=512) #8*8
    conv5 = residual_conv(conv5, features=512)
    conv5 = residual_conv(conv5, features=512)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    
    conv51 = residual_conv(pool5, features=1024) #8*8
    conv51 = residual_conv(conv51, features=1024)
    
    up61 = concatenate([Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv51), conv5], axis=3)
    conv61 = residual_conv(up61, features=512) #16*16
    conv61 = residual_conv(conv61, features=256)
    
    up6 = concatenate([Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv61), conv4], axis=3)
    conv6 = residual_conv(up6, features=256) #16*16
    conv6 = multiScaleConv_block (conv6, 64, sizes=[3,3,5,5], dilatationRates=[1,3,2,3], isDense=False) # FOV = [3,7,9,13] 
    conv6 = residual_conv(conv6, features=256)

    up7 = concatenate([Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = residual_conv(up7, features=128) #32*32
    conv7 = residual_conv(conv7, features=128)

    up8 = concatenate([Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = residual_conv(up8, features=64) #64*64
    conv8 = multiScaleConv_block (conv8, 16, sizes=[3,3,5,7], dilatationRates=[1,3,3,6], isDense=False) # FOV = [3,7,13,37] 
    conv8 = residual_conv(conv8, features=64)

    up9 = concatenate([Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = _conv_bn_relu(up9,64)
    conv9 = _conv_bn_relu(conv9,32)
    conv9 = _conv_bn_relu(conv9,32)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[img, auxInput], outputs=[conv10])

#    model = multi_gpu_model(model , 2)
    model.compile(optimizer=Adam(lr=1e-5), loss=getLoss(cellLoss, weightMap=weights), metrics=[dice_coef]) # adding the momentum

    return model

def get_ResUnet_ASPP (input_shape, cellLoss):
#    with tf.device("/cpu:0"):
    img = Input(input_shape + (img_chls,), name='main_input')  # size: 1024
    auxInput = Input(input_shape + (3,), name='dists_input')
    weights = Lambda(lambda x : x[:,:,:,2])(auxInput)
    dists = Lambda(lambda x : x[:,:,:,0:2])(auxInput)

    
    inputs = concatenate([img, dists], axis=bn_axis)
    
    conv1 = _conv_bn_relu(inputs,64,3,useRegulizer=True) #320x512
    conv1 = _conv_bn_relu(conv1,64,3,useRegulizer=True)
    conv1 = _conv_bn_relu(conv1,64,3,useRegulizer=True)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = residual_conv(pool1, features=128,useRegulizer=True) #160x256
    conv2 = residual_conv(conv2, features=128,useRegulizer=True)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = residual_conv(pool2, features=256,useRegulizer=True) #80x128
    conv3 = residual_conv(conv3, features=256,useRegulizer=True)
    conv3 = residual_conv(conv3, features=256,useRegulizer=True)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = residual_conv(pool3, features=512,useRegulizer=True) #40x64
    conv4 = residual_conv(conv4, features=512,useRegulizer=True)
    conv4 = residual_conv(conv4, features=512,dilatationRate=(2, 2),useRegulizer=True)
    conv4 = residual_conv(conv4, features=512,dilatationRate=(2, 2),useRegulizer=True)
    conv41 = residual_conv(conv4, features=512,dilatationRate=(4, 4),useRegulizer=True)
    conv41 = residual_conv(conv41, features=512,dilatationRate=(4, 4),useRegulizer=True)
    conv41 = multiScaleConv_block (conv41, 256, sizes=[3,3,5,5], dilatationRates=[1,5,5,9], isDense=False,useRegulizer=True)
    
    up7 = concatenate([Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv41), conv3], axis=3)
    conv7 = _conv_bn_relu(up7, features=256,useRegulizer=True) #32*32
    conv7 = _conv_bn_relu(conv7, features=256,useRegulizer=True)

    up8 = concatenate([Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = _conv_bn_relu(up8, features=128,useRegulizer=True) #64*64
    conv8 = _conv_bn_relu(conv8, features=128,useRegulizer=True)

    up9 = concatenate([Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = _conv_bn_relu(up9,64,useRegulizer=True)
    conv9 = _conv_bn_relu(conv9,64,useRegulizer=True)
    conv9 = Dropout(0.5)(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid',name='cell_output')(conv9)
    
    aux_out = UpSampling2D(size=(8, 8))(conv4)
    aux_out = _conv_bn_relu(aux_out,256,useRegulizer=True)
    aux_out = _conv_bn_relu(aux_out,64,useRegulizer=True)
    aux_out = Conv2D(1, (1, 1), activation='sigmoid',name='aux_output')(aux_out)

    model = Model(inputs=[img, auxInput], outputs=[conv10, aux_out])

#    model = multi_gpu_model(model , 2)
    model.compile(optimizer=AdamAccumulate(lr=8e-4,accum_iters=10), loss={'cell_output': getLoss(cellLoss, weightMap=weights), 'aux_output': getLoss(cellLoss,weightMap=weights)},
              loss_weights={'cell_output': 1., 'aux_output': 0.5}, metrics=[dice_coef]) # adding the momentum

    return model
    