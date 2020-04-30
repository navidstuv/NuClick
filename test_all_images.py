# -*- coding: utf-8 -*-
"""
Nuclick Prediction

Consists functions to used for nuclick prediction

"""

import numpy as np
from skimage.io import imsave, imread
import pandas as pd
import os
import matplotlib.pyplot as plt
from data_handler.customImageGenerator import ImageDataGenerator
from keras.models import Model
from skimage.morphology import remove_small_objects, remove_small_holes, reconstruction, disk
from models.models import getModel
from skimage.color import label2rgb
from skimage import exposure
from skimage.filters import gaussian
import cv2
import tkinter
from tkinter import filedialog
from PIL import Image
from scipy.io import loadmat
import warnings

bb = 128
seeddd = 1
modelEnsembling = True
testTimeAug = True


def _unsharp_mask_single_channel(image, radius, amount):
    """Single channel implementation of the unsharp masking filter."""
    blurred = gaussian(image, sigma=radius, mode='reflect')
    result = image + (image - blurred) * amount
    result = np.clip(result, 0, 1)
    return result


def sharpnessEnhancement(imgs):  # needs the input to be in range of [0,1]
    imgs_out = imgs.copy()
    for channel in range(imgs_out.shape[-1]):
        imgs_out[..., channel] = 255 * _unsharp_mask_single_channel(imgs_out[..., channel] / 255., 2, .5)
    return imgs_out


def contrastEnhancement(imgs):  # needs the input to be in range of [0,255]
    imgs_out = imgs.copy()
    p2, p98 = np.percentile(imgs_out, (2, 98))  #####
    if p2 == p98:
        p2, p98 = np.min(imgs_out), np.max(imgs_out)
    if p98 > p2:
        imgs_out = exposure.rescale_intensity(imgs_out, in_range=(p2, p98), out_range=(0., 255.))
    return imgs_out


def readImageFromPathAndGetClicks(path, name, ext='.bmp'):
    refPt = []

    def getClickPosition(event, x, y, flags, param):
        #        global refPt
        if event == cv2.EVENT_LBUTTONUP:
            refPt.append((x, y))
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            cv2.imshow("image", image)

    # load the image, clone it, and setup the mouse callback function
    image = cv2.imread(os.path.join(path, name + ext))
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", getClickPosition)
    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
        # if the 'r' key is pressed, reset the clicked region
        if key == ord("r"):
            image = clone.copy()
            refPt = []
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break
            # close all open windows
    cv2.destroyAllWindows()
    refPt = np.array(refPt)
    cx = refPt[:, 0]
    cy = refPt[:, 1]
    img = clone[:, :, ::-1]
    return img, cx, cy


def readImageAndGetClicks(currdir=os.getcwd()):
    refPt = []

    def getClickPosition(event, x, y, flags, param):
        #        global refPt
        if event == cv2.EVENT_LBUTTONUP:
            refPt.append((x, y))
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            cv2.imshow("image", image)

    # load the image, clone it, and setup the mouse callback function
    root = tkinter.Tk()
    root.withdraw()  # use to hide tkinter window
    root.wm_attributes('-topmost', 1)
    imgPath = filedialog.askopenfilename(
        filetypes=(("PNG", "*.png"), ("JPG", "*.jpg"), ("BMP", "*.bmp"), ("TIF", "*.tif"), ("All files", "*")),
        parent=root, initialdir=currdir, title='Please select an image')
    image = cv2.imread(imgPath)
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", getClickPosition)
    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
        # if the 'r' key is pressed, reset the clicked region
        if key == ord("r"):
            image = clone.copy()
            refPt = []
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break
            # close all open windows
    cv2.destroyAllWindows()
    refPt = np.array(refPt)
    cx = refPt[:, 0]
    cy = refPt[:, 1]
    img = clone[:, :, ::-1]
    return img, cx, cy





def getClickMapAndBoundingBox(cx, cy, m, n):
    clickMap = np.zeros((m, n), dtype=np.uint8)

    # Removing points out of image dimension (these points may have been clicked unwanted)
    cx_out = [x for x in cx if x >= n]
    cx_out_index = [cx.index(x) for x in cx_out]

    cy_out = [x for x in cy if x >= m]
    cy_out_index = [cy.index(x) for x in cy_out]

    indexes = cx_out_index + cy_out_index
    cx = np.delete(cx, indexes)
    cx = cx.tolist()
    cy = np.delete(cy, indexes)
    cy = cy.tolist()

    clickMap[cy, cx] = 1
    boundingBoxes = []
    for i in range(len(cx)):
        xStart = cx[i] - bb // 2
        yStart = cy[i] - bb // 2
        if xStart < 0:
            xStart = 0
        if yStart < 0:
            yStart = 0
        xEnd = xStart + bb - 1
        yEnd = yStart + bb - 1
        if xEnd > n - 1:
            xEnd = n - 1
            xStart = xEnd - bb + 1
        if yEnd > m - 1:
            yEnd = m - 1
            yStart = yEnd - bb + 1
        boundingBoxes.append([xStart, yStart, xEnd, yEnd])
    return clickMap, boundingBoxes


def getPatchs(img, clickMap, boundingBoxes, cx, cy, m, n):
    total = len(boundingBoxes)
    img = np.array([img])
    clickMap = np.array([clickMap])
    clickMap = clickMap[..., np.newaxis]
    patchs = np.ndarray((total, bb, bb, 3), dtype=np.uint8)
    nucPoints = np.ndarray((total, bb, bb, 1), dtype=np.uint8)
    otherPoints = np.ndarray((total, bb, bb, 1), dtype=np.uint8)
    cx_out = [x for x in cx if x >= n]
    cx_out_index = [cx.index(x) for x in cx_out]

    cy_out = [x for x in cy if x >= m]
    cy_out_index = [cy.index(x) for x in cy_out]

    indexes = cx_out_index + cy_out_index
    cx = np.delete(cx, indexes)
    cx = cx.tolist()
    cy = np.delete(cy, indexes)
    cy = cy.tolist()
    for i in range(len(boundingBoxes)):
        boundingBox = boundingBoxes[i]
        xStart = boundingBox[0]
        yStart = boundingBox[1]
        xEnd = boundingBox[2]
        yEnd = boundingBox[3]
        patchs[i] = img[0, yStart:yEnd + 1, xStart:xEnd + 1, :]
        thisClickMap = np.zeros((1, m, n, 1), dtype=np.uint8)
        thisClickMap[0, cy[i], cx[i], 0] = 1
        othersClickMap = np.uint8((clickMap - thisClickMap) > 0)
        nucPoints[i] = thisClickMap[0, yStart:yEnd + 1, xStart:xEnd + 1, :]
        otherPoints[i] = othersClickMap[0, yStart:yEnd + 1, xStart:xEnd + 1, :]
    return patchs, nucPoints, otherPoints


def predictPatchs(model, patchs, dists, clickPrtrb='Test'):
    num_val = len(patchs)
    image_datagen_val = ImageDataGenerator(random_click_perturb=clickPrtrb, rescale=1. / 255)
    batchSizeVal = 1
    val_generator = image_datagen_val.flow(
        patchs, weightMap=dists,
        shuffle=False,
        batch_size=batchSizeVal,
        color_mode='rgb',
        seed=seeddd)
    preds = model.predict_generator(val_generator, steps=num_val // batchSizeVal)
    preds = np.matrix.squeeze(preds, axis=3)
    return preds


def postProcessing(preds, thresh=0.33, minSize=10, minHole=30, doReconstruction=False, nucPoints=None):
    masks = preds > thresh
    masks = remove_small_objects(masks, min_size=minSize)
    masks = remove_small_holes(masks, min_size=minHole)
    if doReconstruction:
        for i in range(len(masks)):
            thisMask = masks[i]
            thisMarker = nucPoints[i, :, :, 0] > 0
            try:
                thisMask = reconstruction(thisMarker, thisMask, selem=disk(1))
                masks[i] = np.array([thisMask])
            except:
                warnings.warn('Nuclei reconstruction error #' + str(i))
    return masks


def generateInstanceMap(masks, boundingBoxes, m, n):
    instanceMap = np.zeros((m, n), dtype=np.uint16)
    for i in range(len(masks)):
        thisBB = boundingBoxes[i]
        thisMaskPos = np.argwhere(masks[i] > 0)
        thisMaskPos[:, 0] = thisMaskPos[:, 0] + thisBB[1]
        thisMaskPos[:, 1] = thisMaskPos[:, 1] + thisBB[0]
        instanceMap[thisMaskPos[:, 0], thisMaskPos[:, 1]] = i + 1
    return instanceMap


modelNames = ['MultiScaleResUnet']
losses = ['complexBCEweighted']
suffixes = ['']

# loading models
models = []
tt = len(modelNames) if modelEnsembling else 1
for i in range(len(modelNames)):
    print('Loading model %s_%s%s into the memory' % (modelNames[i], losses[i], suffixes[i]))
    modelBaseName = 'nuclickNew_%s_%s%s' % (modelNames[i], losses[i], suffixes[i])
    modelName = 'nuclickNew_%s_%s' % (modelNames[i], losses[i])
    modelSaveName = "./%s/weights-%s.h5" % (modelBaseName, modelName)
    model = getModel(modelNames[i], losses[i], losses[i], (bb, bb))
    model.load_weights(modelSaveName)
    models.append(model)

##Reading images
path = "test"
name = "91"
Dot_path = '/media/navid/HDD1/Back_up/PanNuke/asap/Categories/Stomach/mats'
image_path = '/media/navid/HDD1/Back_up/PanNuke/asap/Categories/Stomach/images'
save_path = '/media/navid/HDD1/Back_up/git-files/Nuclick/NuClick/Stomach121212/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
all_centroid_files = os.listdir(Dot_path)

for image_name in all_centroid_files:
    if os.path.exists(os.path.join(save_path, image_name[:-4] + '_instances.png')):
        print('image {} has already been processed'.format(image_name))
        pass
    else:

        print('processing image :{}'.format(image_name))

        out = readImageAndCentroids(image_path, Dot_path, image_name)
        if len(out) == 1:
            print('this image has no nuclei')
            imsave(os.path.join(save_path, image_name[:-4] + '_instances.png'), out[0])
            pass
        else:
            img, cx, cy = out
            m, n = img.shape[0:2]
            cx = [int(np.round(i)) for i in cx]
            cy = [int(np.round(i)) for i in cy]
            # img, cx, cy = readImageFromPathAndGetClicks (path,name,ext='.bmp')
            # img, cx, cy = readImageAndGetClicks(path)
            m, n = img.shape[0:2]
            clickMap, boundingBoxes = getClickMapAndBoundingBox(cx, cy, m, n)
            patchs, nucPoints, otherPoints = getPatchs(img, clickMap, boundingBoxes, cx, cy, m, n)
            # the last one is only dummy!
            dists = np.float32(np.concatenate((nucPoints, otherPoints, otherPoints), axis=3))
            # sharpenning the image
            patchs_shappened = patchs.copy()
            for i in range(len(patchs)):
                patchs_shappened[i] = sharpnessEnhancement(patchs[i])
            # contrast enhancing the image
            patchs_contrasted = patchs.copy()
            for i in range(len(patchs)):
                patchs_contrasted[i] = contrastEnhancement(patchs[i])

            # prediction with model ensembling and test time augmentation
            numModel = len(models)
            augNum = 3
            predNum = 1  # augNum*numModel
            preds = np.zeros((len(patchs), bb, bb), dtype=np.float32)
            for i in range(len(models)):
                print('-----Working on model := %s_%s%s-----' % (modelNames[i], losses[i], suffixes[i]))
                model = models[i]
                preds += predictPatchs(model, patchs, dists)
                predNum += 1
                print("Original images prediction, DONE!")
                if testTimeAug:
                    print("Test Time Augmentation Started")
                    temp = predictPatchs(model, patchs_shappened[:, ::-1, :], dists[:, ::-1, :], clickPrtrb='Test')
                    preds += temp[:, ::-1, :]
                    predNum += 1
                    print("Flipped images prediction, DONE!")

                    temp = predictPatchs(model, patchs_shappened[:, :, ::-1], dists[:, :, ::-1], clickPrtrb='Test')
                    preds += temp[:, :, ::-1]
                    predNum += 1
                    print("Sharpenned images prediction, DONE!")

                    temp = predictPatchs(model, patchs_contrasted[:, ::-1, ::-1], dists[:, ::-1, ::-1],
                                         clickPrtrb='Test')
                    preds += temp[:, ::-1, ::-1]
                    predNum += 1
                    print("Contrasted images prediction, DONE!")
            preds /= predNum
            masks = postProcessing(preds, thresh=0.2, minSize=5, minHole=30, doReconstruction=True, nucPoints=nucPoints)
            instanceMap = generateInstanceMap(masks, boundingBoxes, m, n)
            # instanceMap_RGB  = label2rgb(instanceMap, image=img, alpha=0.3,
            #  bg_label=0, bg_color=(0, 0, 0), image_alpha=1, kind='overlay')
            # plt.figure(),plt.imshow(instanceMap_RGB)
            imsave(os.path.join(save_path, image_name[:-4] + '_instances.png'), instanceMap)
            # plt.figure(),plt.imshow(img)