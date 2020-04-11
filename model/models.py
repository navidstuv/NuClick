
"""
NuClick Network Architecture
It is using MultiScale and Residual convolutional blocks. 
Design paradigm follows Unet model which comprises Encoder and Decoder paths.

"""
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import BatchNormalization , Activation
from keras.layers import Lambda, add
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
from keras.utils import multi_gpu_model
from losses import getLoss, dice_coef
import tensorflow as tf
from .config import config


multiGPU = config.multiGPU
learningRate = config.LearningRate

K.set_image_data_format('channels_last')  # TF dimension ordering in this code
img_chls = 3

weight_decay=5e-5
bn_axis = -1 if K.image_data_format() == 'channels_last' else 1

    
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



'''    
##################### DEFINING NETWORKS #######################################
'''
    
def get_MultiScale_ResUnet (input_shape, lossType):
    if multiGPU:
        with tf.device("/cpu:0"):
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
        
            model = multi_gpu_model(model , 2)
            model.compile(optimizer=Adam(lr=learningRate), loss=getLoss(lossType, weightMap=weights), metrics=[dice_coef]) # adding the momentum
        
            return model

