from skimage import exposure
from skimage.filters import gaussian
import cv2
import tkinter
from tkinter import filedialog
from skimage.morphology import remove_small_objects, remove_small_holes, reconstruction, disk
import numpy as np
from skimage.io import imread
import os
from data_handler.customImageGenerator import ImageDataGenerator
import pandas as pd
import random
from config import config
from PIL import Image
from scipy.io import loadmat, savemat
import warnings
from scipy.ndimage.measurements import center_of_mass

seeddd = 1
bb = config.img_rows
testTimeAug = config.testTimeAug

colorPallete = [(random.randrange(0, 240), random.randrange(0, 240), random.randrange(0, 240)) for i in range(1000)]

def _unsharp_mask_single_channel(image, radius, amount):
    """Single channel implementation of the unsharp masking filter."""
    blurred = gaussian(image, sigma=radius, mode='reflect')
    result = image + (image - blurred) * amount
    result = np.clip(result, 0, 1)
    return result


def sharpnessEnhancement(imgs):  # needs the input to be in range of [0,1]
    imgs_out = imgs.copy()
    for channel in range(imgs_out.shape[-1]):
        imgs_out[..., channel] = 255 * _unsharp_mask_single_channel(imgs_out[..., channel] / 255., 3, 1)
    return imgs_out


def contrastEnhancement(imgs):  # needs the input to be in range of [0,255]
    imgs_out = imgs.copy()
    p2, p98 = np.percentile(imgs_out, (1, 99))  #####
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
    #    image = rescale(image,.75)
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
    return img, cx, cy, imgPath


def readImageAndCentroids(path, name, ext='.bmp'):
    img = imread(os.path.join(path, name + ext))
    cents = pd.read_csv(os.path.join(path, name + '.txt'), sep=',', header=None)
    cents = np.array(cents)
    cx = cents[:, 0]
    cy = cents[:, 1]
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

def getPatchs_gland(img, clickMap):
    uniqueSignals = np.unique(clickMap)
    uniqueSignals = uniqueSignals[1:]
    total = len(uniqueSignals)
    img = np.array([img])
    clickMap = np.array([clickMap])
    clickMap = clickMap[..., np.newaxis]
    patchs=np.ndarray((total,)+img.shape[1:], dtype=np.uint8)
    nucPoints=np.ndarray((total,)+clickMap.shape[1:], dtype=np.uint8)
    otherPoints=np.zeros((total,)+clickMap.shape[1:], dtype=np.uint8)
    for i in range(total):
        patchs[i] = img[0]
        thisClickMap = np.uint8((clickMap==uniqueSignals[i]))
        nucPoints[i] = thisClickMap
        othersClickMap = np.uint8(((1-(clickMap==uniqueSignals[i]))*clickMap))
        thisUniqueValues = np.unique(othersClickMap)
        thisUniqueValues = thisUniqueValues[1:]
        cx=np.ndarray((total-1), dtype=np.uint16)
        cy=np.ndarray((total-1), dtype=np.uint16)
        for j in range(len(thisUniqueValues)):
            thisOtherMask = othersClickMap[0,:,:,0]==thisUniqueValues[j]
            cyCandids, cxCandids = np.where(thisOtherMask==1)
            rndIdx = np.random.randint(0,high=len(cyCandids))
            cx[j] = cxCandids[rndIdx]
            cy[j] = cyCandids[rndIdx]
#            print((np.floor(thisCent[0]).astype('uint16'),np.floor(thisCent[1]).astype('uint16')))
        thisOtherPoints = np.zeros(clickMap.shape[1:3],dtype='uint8')
        thisOtherPoints[cy,cx] = 1
        otherPoints[i,:,:,0] = thisOtherPoints
    return patchs, nucPoints, otherPoints

def predictPatchs(model, patchs, dists, clickPrtrb=config.testTimeJittering):
    num_val = len(patchs)
    image_datagen_val = ImageDataGenerator(RandomizeGuidingSignalType=clickPrtrb, rescale=1. / 255)
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
    masks = remove_small_holes(masks, area_threshold=minHole)
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


def generateInstanceMap_gland(masks):
    instanceMap = np.zeros(masks.shape[1:3],dtype=np.uint8)
    for i in range(len(masks)):
        thisMask = masks[i]
        instanceMap[thisMask] = i+1
    return instanceMap

def readImageAndGetSignals(currdir=os.getcwd()):
    drawing = False  # true if mouse is pressed
    mode = True  # if True, draw rectangle. Press 'm' to toggle to curve

    # mouse callback function
    def begueradj_draw(event, former_x, former_y, flags, param):
        global current_former_x, current_former_y, drawing, mode, color
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            current_former_x, current_former_y = former_x, former_y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                cv2.line(im, (current_former_x, current_former_y), (former_x, former_y), colorPallete[2 * ind], 5)
                cv2.line(signal, (current_former_x, current_former_y), (former_x, former_y), (ind, ind, ind), 1)
                current_former_x = former_x
                current_former_y = former_y
                # print former_x,former_y
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.line(im, (current_former_x, current_former_y), (former_x, former_y), colorPallete[2 * ind], 5)
            current_former_x = former_x
            current_former_y = former_y
        return signal

        # load the image, clone it, and setup the mouse callback function

    root = tkinter.Tk()
    root.withdraw()  # use to hide tkinter window
    root.wm_attributes('-topmost', 1)
    imgPath = filedialog.askopenfilename(
        filetypes=(("PNG", "*.png"), ("JPG", "*.jpg"), ("BMP", "*.bmp"), ("TIF", "*.tif"), ("All files", "*")),
        parent=root, initialdir=currdir, title='Please select an image')
    im = cv2.imread(imgPath)
    # im = cv2.resize(im, (512,512))
    signal = np.zeros(im.shape, dtype='uint8')
    clone = im.copy()

    ind = 1
    cv2.namedWindow("i=increase object - d=decrease object - esc=ending annotation")
    cv2.setMouseCallback('i=increase object - d=decrease object - esc=ending annotation', begueradj_draw)
    while (1):
        cv2.imshow('i=increase object - d=decrease object - esc=ending annotation', im)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == ord("i"):
            ind += 1
        elif k == ord("d"):
            ind -= 1
    cv2.destroyAllWindows()
    img = clone[:, :, ::-1]
    signal = signal[:, :, 0]
    return img, signal, imgPath


def predictSingleImage(model, img, markups):
    patchs, includeMap, excludeMap = getPatchs_gland(img, markups)
    # patchs, nucPoints, otherPoints = getPatchs(img, clickMap, boundingBoxes, cx, cy, m, n)
    dists = np.float32(np.concatenate((includeMap, excludeMap, excludeMap), axis=3))  # the last one is only dummy!
    predNum = 0  # len(models)
    # dists = clickMap[np.newaxis,...,np.newaxis]
    if testTimeAug:
        # sharpenning the image
        patchs_shappened = patchs.copy()
        for i in range(len(patchs)):
            patchs_shappened[i] = sharpnessEnhancement(patchs[i])
        # contrast enhancing the image
        patchs_contrasted = patchs.copy()
        for i in range(len(patchs)):
            patchs_contrasted[i] = contrastEnhancement(patchs[i])

            # prediction with model ensambling and test time augmentation
    preds = np.zeros(patchs.shape[:3], dtype=np.float32)

    preds += predictPatchs(model, patchs, dists)
    predNum += 1
    #        print("Original images prediction, DONE!")
    if testTimeAug:
        #            print("Test Time Augmentation Started")
        temp = predictPatchs(model, patchs_shappened[:, :, ::-1], dists[:, :, ::-1])
        preds += temp[:, :, ::-1]
        predNum += 1
        #            print("Sharpenned images prediction, DONE!")
        temp = predictPatchs(model, patchs_contrasted[:, ::-1, ::-1], dists[:, ::-1, ::-1])
        preds += temp[:, ::-1, ::-1]
        predNum += 1
#            print("Contrasted images prediction, DONE!")
    preds /= predNum
    masks = postProcessing(preds, thresh=config.Thresh, minSize=config.minSize, minHole=config.minHole, doReconstruction=False)
    instanceMap = generateInstanceMap_gland(masks)
    return instanceMap

def readImageAndCentroids(img_path, dot_path, name):
    all_cents = []
    try:
        # img = imread(os.path.join(img_path, name[:-4]+'.tif'))
        img = Image.open(os.path.join(img_path, name[:-9] + '.tif'))
        img = np.asarray(img)
    except:
        print('image {} has some problem in reading'.format(name))
        # img = imread(os.path.join(img_path, name[:-4] + '.png'))
        img = Image.open(os.path.join(img_path, name[:-9] + '.png'))
        img = np.asarray(img)

    cents = loadmat(os.path.join(dot_path, name))
    for key in cents.keys():
        if key not in ['__header__', '__version__', '__globals__', 'NoNuclei']:
            all_cents = np.ndarray.tolist(cents[key]) + all_cents
    all_cents = [[x[0], x[1]] for x in all_cents]
    if len(all_cents):
        all_cents = np.array(all_cents)
        cx = all_cents[:, 1]
        cy = all_cents[:, 0]
        return [img, cx, cy]
    else:
        return [np.zeros((img.shape[0], img.shape[1]))]

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

def extract_centroids(mask):
    '''
    Extrac centroids of instances in instance-wise segmentation mask
    :param mask:
    :return: return coordinates
    '''
    unique_labels = np.unique(mask)
    unique_labels = list(unique_labels)
    unique_labels.remove(0)
    bin_mask = mask > 0
    centroids = center_of_mass(bin_mask, labels=mask, index=unique_labels)
    return centroids

if __name__=='__main__':
    path = 'E:\Back_up\git-files\\Nuclick--\monuseg-data\masks'
    path_list = os.listdir(path)
    for img in path_list:
        mask = imread(os.path.join(path, img))
        centroids = extract_centroids(mask)
        savemat(os.path.join(path, img[:-4]+'.mat'), {'centers':centroids})
