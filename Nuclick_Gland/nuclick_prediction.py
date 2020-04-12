import numpy as np
from skimage.io import imsave
import pandas as pd
import os, glob
import matplotlib.pyplot as plt
from data_handler.customImageGenerator import ImageDataGenerator
from keras.models import Model
from skimage.morphology import remove_small_objects, remove_small_holes, reconstruction
from model_factory1 import getModel
from skimage.color import label2rgb
from skimage import exposure
from skimage.filters import gaussian
import cv2
import tkinter
from tkinter import filedialog
import random
from skimage.measure import regionprops

bb=512
seeddd=1
modelEnsembling = True
testTimeAug = True
augNum = 3
colorPallete = [(random.randrange(0, 240), random.randrange(0, 240), random.randrange(0, 240)) for i in range(1000)]

def _unsharp_mask_single_channel(image, radius, amount):
    """Single channel implementation of the unsharp masking filter."""
    blurred = gaussian(image,sigma=radius,mode='reflect')
    result = image + (image - blurred) * amount
    result = np.clip(result, 0, 1)
    return result
    
def sharpnessEnhancement(imgs): # needs the input to be in range of [0,1]
    imgs_out = imgs.copy()
    for channel in range(imgs_out.shape[-1]):
        imgs_out[..., channel] = 255*_unsharp_mask_single_channel(imgs_out[..., channel]/255., 2, .5)
    return imgs_out

def contrastEnhancement (imgs):# needs the input to be in range of [0,255]
    imgs_out = imgs.copy()
    p2, p98 = np.percentile(imgs_out, (2, 98)) #####
    if p2==p98:
        p2, p98 = np.min(imgs_out), np.max(imgs_out)
    if p98>p2:
        imgs_out = exposure.rescale_intensity(imgs_out, in_range=(p2, p98), out_range=(0.,255.))
    return imgs_out

def readImageAndGetSignals (currdir=os.getcwd()):
    drawing=False # true if mouse is pressed
    mode=True # if True, draw rectangle. Press 'm' to toggle to curve
    
    # mouse callback function
    def begueradj_draw(event,former_x,former_y,flags,param):
        global current_former_x,current_former_y,drawing, mode, color
        if event==cv2.EVENT_LBUTTONDOWN:
            drawing=True
            current_former_x,current_former_y=former_x,former_y
    
        elif event==cv2.EVENT_MOUSEMOVE:
            if drawing==True:
                cv2.line(im,(current_former_x,current_former_y),(former_x,former_y),colorPallete[2*ind],5)
                cv2.line(signal,(current_former_x,current_former_y),(former_x,former_y),(ind,ind,ind),1)
                current_former_x = former_x
                current_former_y = former_y
                    #print former_x,former_y
        elif event==cv2.EVENT_LBUTTONUP:
            drawing=False
            cv2.line(im,(current_former_x,current_former_y),(former_x,former_y),colorPallete[2*ind],5)
            current_former_x = former_x
            current_former_y = former_y
        return signal    
    
    # load the image, clone it, and setup the mouse callback function
    root = tkinter.Tk()
    root.withdraw() #use to hide tkinter window
    root.wm_attributes('-topmost', 1)
    imgPath = filedialog.askopenfilename(filetypes = (("PNG", "*.png"), ("JPG", "*.jpg"), ("BMP", "*.bmp"), ("TIF", "*.tif"), ("All files", "*")), parent=root, initialdir=currdir, title='Please select an image')
    im = cv2.imread(imgPath)
    signal = np.zeros(im.shape,dtype='uint8')
    clone = im.copy()
    
    ind=1
    cv2.namedWindow("i=increase object - d=decrease object - esc=ending annotation")
    cv2.setMouseCallback('i=increase object - d=decrease object - esc=ending annotation',begueradj_draw)
    while(1):
        cv2.imshow('i=increase object - d=decrease object - esc=ending annotation',im)
        k=cv2.waitKey(1)&0xFF
        if k==27:
            break
        elif k==ord("i"):
            ind+=1
        elif k==ord("d"):
            ind-=1
    cv2.destroyAllWindows()
    img = clone[:, :, ::-1]
    signal = signal[:,:,0]
    return img, signal, imgPath
    
def getPatchs(img, clickMap):
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
        thisClickMap = np.uint8(255*(clickMap==uniqueSignals[i]))
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
    
def predictPatchs(model, patchs, dists, random_click_perturb=None, pointMapType=None):
    num_val = len(patchs)
    image_datagen_val = ImageDataGenerator(random_click_perturb = random_click_perturb,pointMapType = pointMapType,rescale=1. / 255)
    batchSizeVal = 1
    val_generator = image_datagen_val.flow(
        patchs, weightMap=dists,
        shuffle=False,
        batch_size=batchSizeVal,
        color_mode='rgb',
        seed=seeddd)
    preds  = model.predict_generator(val_generator, steps=num_val // batchSizeVal)
    preds = np.matrix.squeeze(preds, axis=3)
    return preds

def postProcessing(preds,thresh=0.33,minSize=10,minHole=30,doReconstruction=False,nucPoints=None):
    masks = preds>thresh
    masks = remove_small_objects(masks,min_size=minSize)
    masks = remove_small_holes(masks,min_size=minHole)
    if doReconstruction:
        for i in range(len(masks)):
            thisMask = masks[i]
            thisMarker = nucPoints[i,:,:,0]>0
            thisMask = reconstruction(thisMarker, thisMask)
            masks[i] = np.array([thisMask])  
    return masks

def generateInstanceMap(masks):
    instanceMap = np.zeros(masks.shape[1:3],dtype=np.uint8)
    for i in range(len(masks)):
        thisMask = masks[i]
        instanceMap[thisMask] = i+1
    return instanceMap

def predictSingleImage (models,img, markups):
    patchs, includeMap, excludeMap =  getPatchs(img, markups)
    # patchs, nucPoints, otherPoints = getPatchs(img, clickMap, boundingBoxes, cx, cy, m, n)
    dists = np.float32(np.concatenate((includeMap, excludeMap, excludeMap),axis=3)) # the last one is only dummy!
    predNum = 0#len(models)
    # dists = clickMap[np.newaxis,...,np.newaxis]
    if testTimeAug:
        #sharpenning the image
        patchs_shappened = patchs.copy()
        for i in range(len(patchs)):
            patchs_shappened[i] = sharpnessEnhancement(patchs[i])
        #contrast enhancing the image
        patchs_contrasted = patchs.copy()
        for i in range(len(patchs)):
            patchs_contrasted[i] = contrastEnhancement(patchs[i])   
    
    # prediction with model ensambling and test time augmentation
    preds = np.zeros(patchs.shape[:3],dtype=np.float32)
    for i in range(len(models)):
#        print('-----Working on model := %s_%s%s-----' % (modelNames[i], losses[i],suffixes[i]))
        model = models[i]
        preds += predictPatchs(model, patchs, dists)
        predNum+=1
#        print("Original images prediction, DONE!")
        if testTimeAug:
#            print("Test Time Augmentation Started")
            temp = predictPatchs(model, patchs_shappened[:, :, ::-1], dists[:, :, ::-1])
            preds += temp[:, :, ::-1]
            predNum+=1
#            print("Sharpenned images prediction, DONE!")
            temp = predictPatchs(model, patchs_contrasted[:, ::-1, ::-1], dists[:, ::-1, ::-1])
            preds += temp[:, ::-1, ::-1]
            predNum+=1
#            print("Contrasted images prediction, DONE!")
    preds /= predNum 
    masks = postProcessing(preds,thresh=0.5,minSize=1000,minHole=1000,doReconstruction=False)
    instanceMap = generateInstanceMap(masks)
    return instanceMap

modelNames = ['MultiScaleResUnet']
losses = ['bce_dice']
suffixes = ['']
pointMapTypes = ['Skeleton']

#loading models
models = []
tt = len(modelNames) if modelEnsembling else 1
for i in range(len(modelNames)):
    print('Loading model %s_%s%s into the memory' % (modelNames[i], losses[i],suffixes[i]))
    modelBaseName = 'nuclickGland_%s_%s_%s%s' % (pointMapTypes[i], modelNames[i], losses[i],suffixes[i])
    modelName = 'nuclickGland_%s_%s_%s' % (pointMapTypes[i], modelNames[i], losses[i])
    modelSaveName = "./%s/weights-%s.h5" % (modelBaseName, modelName)
    model = getModel(modelNames[i], losses[i], losses[i], (bb,bb))
    model.load_weights(modelSaveName)
    models.append(model)

path = 'E:/Nuclick project_Gland/Data/test/';
img, markups, imgPath = readImageAndGetSignals(path)

instanceMap = predictSingleImage(models,img, markups)
instanceMap_RGB  = label2rgb(np.uint8(instanceMap), image=img, alpha=0.3, bg_label=0, bg_color=(0, 0, 0), image_alpha=1, kind='overlay')
plt.figure(),plt.imshow(instanceMap_RGB), plt.show()
imsave(imgPath[:-4]+'_instances.png',instanceMap)
imsave(imgPath[:-4]+'_signals.png',markups)
# imsave(os.path.join(path, fileName.split('.')[0]+'_instances.png'),instanceMap)



##Reading images
# path = "F:/Jahanifar/crops/"
# os.chdir(path)
# fileNames = []
# for file in os.listdir(path):
#     if file.endswith(".png"):
#         fileNames.append(file)
#
# i=0
# for fileName in fileNames:
#     print('-----Working on Image := %d \t %s----' % (i, fileName))
#     txtName = fileName.split('.')[0]+'.txt'
#     if not os.path.isfile(path+txtName):
#         instanceMap = np.zeros((256,256),dtype=np.uint16)
#         imsave(os.path.join(path, fileName.split('.')[0]+'_instances.png'),instanceMap)
#         i+=1
#         continue
#
#     # Select one image input paradigm
#     #     img, cx, cy = readImageFromPathAndGetClicks (path,name,ext='.bmp')
#     img, cx, cy = readImageAndGetClicks(path)
#     # img, cx, cy = readImageAndCentroids(path,fileName.split('.')[0],ext='.png')
#     m,n = img.shape[0:2]
#     instanceMap = prediictSingleImage (img, cx, cy, m, n)
# #    instanceMap_RGB  = label2rgb(instanceMap, image=img, alpha=0.3, bg_label=0, bg_color=(0, 0, 0), image_alpha=1, kind='overlay')
# #    plt.imshow(instanceMap_RGB)
#     imsave(os.path.join(path, fileName.split('.')[0]+'_instances.png'),instanceMap)
#     i+=1