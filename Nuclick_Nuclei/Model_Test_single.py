'''
Testing time:
Perform each of model variations on the test images and average their output to
form the final output (Bagging style).
Test predictions can be made on the image crops or its full domain.
we also can use augomentation in testing phase to make better predictions.
'''
from __future__ import print_function
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from skimage.io import imsave, imread
import numpy as np
from keras import backend as K
# from elastic_functions import perform_elastic_3image
import matplotlib.pyplot as plt
# from data import load_test_data_crops,load_train_data_crops,load_train_data,load_test_data
from skimage import exposure
from skimage.filters import gaussian
from image_segmentation1 import ImageDataGenerator
import time
from keras.applications import imagenet_utils
from model_factory1 import getModel

''' Setting parameter '''
# folder containing trained model (5 folds) must be presented in the current working directory
modelType = 'spagettiNet_twoHeaded_multiscale_bottleneck'
cellLoss = 'weightedJaccard'
marginLoss = 'jaccard'
test_batchSize = 1
kFold = 5
seeddd = 1
Row = 1024
Col = 1024
numChnls = 3
predChnls = 1
input_shape = (1024, 1024)

modelBaseName = 'kumar_%s_%s-%s' % (modelType, cellLoss, marginLoss)

dataPath = '.\\kumar-dataset\\images'
test_data_path = os.path.join(dataPath, 'Test', 'Test_diff')  # Name of the test set folder
test_data_path = '.\\kumar-dataset\\test\\images'
pred_dir = os.path.join(modelBaseName, 'Mask_Test')  # MarkerMarginPrediction must be already created in the dataPath
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)


def _unsharp_mask_single_channel(image, radius, amount):
    """Single channel implementation of the unsharp masking filter."""
    blurred = gaussian(image, sigma=radius, mode='reflect')
    result = image + (image - blurred) * amount
    result = np.clip(result, 0, 1)
    return result


def sharpnessEnhancement(imgs):  # needs the input to be in range of [0,1]
    imgs_out = imgs.copy()
    for channel in range(imgs_out.shape[-1]):
        imgs_out[..., channel] = 255 * _unsharp_mask_single_channel(imgs_out[..., channel] / 255., 2, .8)
    return imgs_out


def contrastEnhancement(imgs):  # needs the input to be in range of [0,255]
    imgs_out = imgs.copy()
    p2, p98 = np.percentile(imgs_out, (2, 98))  #####
    if p2 == p98:
        p2, p98 = np.min(imgs_out), np.max(imgs_out)
    if p98 > p2:
        imgs_out = exposure.rescale_intensity(imgs_out, in_range=(p2, p98), out_range=(0., 255.))
    return imgs_out


def adaptiveIntensityScaling(imgs):
    imgs_out = np.float32(imgs) / 255.
    m = 0.6 / (np.mean(imgs_out) + 1e-7)
    if m < 0.9:  # ORIGINALLY .93
        s = .9
    else:
        s = np.min([np.max([m, 1.3]), 1.6])
    return np.uint8(np.clip(imgs * s, 0., 255.))


# from skimage.color import rgb2hsv, rgb2lab
#
#
# def addAuxialaryChannels(img):
#     img = np.uint8(img)
#     HSV = np.array(rgb2hsv(img))
#     Lab = np.array(rgb2lab(img))
#     output = img
#     output = np.append(img, np.uint8(255 * HSV), axis=2)
#     L = np.uint8(255 * Lab[:, :, 0] / 100)
#     L = L[..., np.newaxis]
#     output = np.append(output, L, axis=2)
#     return output.astype(np.float32)


image_datagen_test = ImageDataGenerator(

    # preprocessing_function=addAuxialaryChannels,
    rescale=1. / 255)

# CROP-WISE testing: Crop each image to image sizes with overlaps and test each network on each crop
print('*' * 90)
print('Test phase: predict the output using all trained networks')
print('*' * 90)

fileFormat = '.tif'
imagesTemp = os.listdir(test_data_path)
images = []
for i, image in zip(range(len(imagesTemp)), imagesTemp):
    if fileFormat in image:
        images.append(image)
totalImages = int(len(images))

# Creating instance of base model
model = getModel(modelType, cellLoss, marginLoss, input_shape)
modelName = "%s" % (modelBaseName)
modelSaveName = "weights-%s.h5" % (modelName)
model.load_weights(os.path.join(modelBaseName, 'SWA_nocycle.h5'))
# model.load_weights('weights.97-0.70.h5')
print('-' * 30)
print('Loading and preprocessing test data (ONE BY ONE APPROACH)...')
print('-' * 30)

iter = 0
im_iter = 0
for image_name in images:
    start_time = time.time()
    if not (fileFormat in image_name):
        continue
    img = imread(os.path.join(test_data_path, image_name))
    img = img[:,:,:3]
    imgRow = img.shape[0]
    imgCol = img.shape[1]

    # NEEDED PREPRCESSINGs
    img = np.array([img])
    rowPad = (Row - imgRow) // 2
    colPad = (Col - imgCol) // 2
    img = np.pad(img, ((0, 0), (rowPad, rowPad), (colPad, colPad), (0, 0)), 'reflect')
    dummyWeight = np.float32(img[:, :, :, 0:1])

    numAug = 6  # original+fliplr+flipud+contrast(fliplr(flipud))+sharppen
    numOutputs = np.float(numAug)
    num_test = numAug

    predInput = np.ndarray((numAug, Row, Col, 3), dtype=np.uint8)
    outputMarker = np.zeros((1, Row, Col, 1), dtype='float64')
    outputMargin = np.zeros((1, Row, Col, 1), dtype='float64')

    # Creating input tensor
    predInput[0, :, :, :] = img[0, :, :, :]  # original image
    predInput[1, :, :, :] = img[0, ::-1, ::-1, :]
    predInput[2, :, :, :] = adaptiveIntensityScaling(img[0, ::-1, :, :])
    predInput[3, :, :, :] = contrastEnhancement(img[0, ::-1, ::-1, :])  # contrastEnhancing(fliplr(flipud))
    predInput[4, :, :, :] = sharpnessEnhancement(img[0, :, ::-1, :])  # sharpenning(original image)
    predInput[5, :, :, :] = sharpnessEnhancement(adaptiveIntensityScaling(img[0, :, :, :]))

    predDummyWeights = np.repeat(dummyWeight, numAug, axis=0)

    # Predict the output for each crop: Aggregated from different different models and augomentations

    # iterate on the number of validation models (kFold)



    # Make predictions
    test_generator = image_datagen_test.flow(predInput, weightMap1=predDummyWeights,
                                             weightMap2=predDummyWeights,
                                             shuffle=False,
                                             batch_size=test_batchSize, color_mode='rgb', seed=seeddd)
    temp, temp2 = model.predict_generator(test_generator, steps=num_test // test_batchSize)
    temp[1, :, :, :] = temp[1, ::-1, ::-1, :]
    temp[2, :, :, :] = temp[2, ::-1, :, :]
    temp[3, :, :, :] = temp[3, ::-1, ::-1, :]
    temp[4, :, :, :] = temp[4, :, ::-1, :]
    outputMarker = outputMarker + np.asarray([np.sum(temp, axis=0)]) / numOutputs
    temp2[1, :, :, :] = temp2[1, ::-1, ::-1, :]
    temp2[2, :, :, :] = temp2[2, ::-1, :, :]
    temp2[3, :, :, :] = temp2[3, ::-1, ::-1, :]
    temp2[4, :, :, :] = temp2[4, :, ::-1, :]
    outputMargin = outputMargin + np.asarray([np.sum(temp2, axis=0)]) / numOutputs

    # Construct the final Full-Resolution image from crop predictions
    marker = np.uint8(outputMarker[0, rowPad:Row - rowPad, colPad:Col - colPad, 0] * 255)
    margin = np.uint8(outputMargin[0, rowPad:Row - rowPad, colPad:Col - colPad, 0] * 255)
    elapsed_time = time.time() - start_time
    imsave(os.path.join(pred_dir, image_name.split('.')[0] + '_marker' + '.png'), marker)
    imsave(os.path.join(pred_dir, image_name.split('.')[0] + '_mask' + '.png'), margin)
    im_iter += 1
    print('Done: %d/%d images in %03f Secs.' % (im_iter, totalImages, elapsed_time))
   # print('Done: {0}/{1} images with loss = {2}'.format(im_iter, totalImages, loss))
