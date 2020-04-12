import numpy as np
from skimage.io import imsave
import matplotlib.pyplot as plt
from image_segmentation import ImageDataGenerator
from model_factory1 import getModel
from skimage.color import label2rgb
from config import config


from utils.utils import readImageAndGetClicks, getClickMapAndBoundingBox,\
    getPatchs, sharpnessEnhancement, contrastEnhancement,\
    predictPatchs, postProcessing, generateInstanceMap


bb = 256
seeddd = 1
modelEnsembling = True
testTimeAug = True



def main():


    modelNames = config.modelType #['MultiScaleResUnet']
    losses = ['bce_dice']
    suffixes = ['']
    # loading models
    print('Loading model %s_%s%s into the memory' % (modelNames, losses, suffixes))
    modelBaseName = 'nuclickNuclei_%s_%s%s' % (modelNames, losses, suffixes)  # nuclickWithPoints_
    modelName = 'nuclickNuclei_%s_%s' % (modelNames, losses)
    modelSaveName = "./%s/weights-%s.h5" % (modelBaseName, modelName)
    model = getModel(modelNames, losses, (bb, bb))
    model.load_weights(modelSaveName)
    ##Reading images
    path = "E:/Nuclick project/Data/test"
    # Select one image input paradigm
    # img, cx, cy = readImageAndCentroids(path,name)
    # img, cx, cy = readImageFromPathAndGetClicks (path,name,ext='.bmp')
    if config.application in ['cell', 'nucleus']:
        img, cx, cy, imgPath = readImageAndGetClicks(path)
        m, n = img.shape[0:2]
        clickMap, boundingBoxes = getClickMapAndBoundingBox(cx, cy, m, n)
        patchs, nucPoints, otherPoints = getPatchs(img, clickMap, boundingBoxes, cx, cy, m, n)
        # for i in range(len(nucPoints)):
        #    nucPoints[i,:,:,0] =  binary_dilation(nucPoints[i,:,:,0],disk(1))
        dists = np.float32(np.concatenate((nucPoints, otherPoints, otherPoints), axis=3))  # the last one is only dummy!
        # sharpenning the image
        patchs_shappened = patchs.copy()
        for i in range(len(patchs)):
            patchs_shappened[i] = sharpnessEnhancement(patchs[i])
        # contrast enhancing the image
        patchs_contrasted = patchs.copy()
        for i in range(len(patchs)):
            patchs_contrasted[i] = contrastEnhancement(patchs[i])

        # prediction with model ensambling and test time augmentation
        augNum = 3
        predNum = 0  # augNum*numModel
        preds = np.zeros((len(patchs), bb, bb), dtype=np.float32)
        print('-----Working on model := %s_%s%s-----' % (modelNames[i], losses[i], suffixes[i]))
        preds += predictPatchs(model, patchs, dists)
        predNum += 1
        print("Original images prediction, DONE!")
        if testTimeAug:
            print("Test Time Augmentation Started")
            temp = predictPatchs(model, patchs_shappened[:, :, ::-1], dists[:, :, ::-1])
            preds += temp[:, :, ::-1]
            predNum += 1
            print("Sharpenned images prediction, DONE!")
            temp = predictPatchs(model, patchs_contrasted[:, ::-1, ::-1], dists[:, ::-1, ::-1])
            preds += temp[:, ::-1, ::-1]
            predNum += 1
            print("Contrasted images prediction, DONE!")
        preds /= predNum
        masks = postProcessing(preds, thresh=0.8, minSize=10, minHole=30, doReconstruction=True, nucPoints=nucPoints)
        instanceMap = generateInstanceMap(masks, boundingBoxes, m, n)
        instanceMap_RGB = label2rgb(instanceMap, image=img, alpha=0.3, bg_label=0, bg_color=(0, 0, 0), image_alpha=1,
                                    kind='overlay')
        plt.figure(), plt.imshow(instanceMap_RGB)
        plt.show()
        # imsave(imgPath[:-4]+'_overlay.png',instanceMap_RGB)
        imsave(imgPath[:-4] + '_instances.png', instanceMap * 255)
        imsave(imgPath[:-4] + '_points.png', np.uint8(255 * np.sum(nucPoints, axis=(0, 3))))
        # plt.figure(),plt.imshow(img)

    if config.application=='gland':

        pointMapTypes = ['Skeleton']
        # loading models
        models = []
        tt = len(modelNames) if modelEnsembling else 1
        for i in range(len(modelNames)):
            print('Loading model %s_%s%s into the memory' % (modelNames[i], losses[i], suffixes[i]))
            modelBaseName = 'nuclickGland_%s_%s_%s%s' % (pointMapTypes[i], modelNames[i], losses[i], suffixes[i])
            modelName = 'nuclickGland_%s_%s_%s' % (pointMapTypes[i], modelNames[i], losses[i])
            modelSaveName = "./%s/weights-%s.h5" % (modelBaseName, modelName)
            model = getModel(modelNames[i], losses[i], losses[i], (bb, bb))
            model.load_weights(modelSaveName)
            models.append(model)

        path = 'E:/Nuclick project_Gland/Data/test/';
        img, markups, imgPath = readImageAndGetSignals(path)

        instanceMap = predictSingleImage(models, img, markups)
        instanceMap_RGB = label2rgb(np.uint8(instanceMap), image=img, alpha=0.3, bg_label=0, bg_color=(0, 0, 0),
                                    image_alpha=1, kind='overlay')
        plt.figure(), plt.imshow(instanceMap_RGB), plt.show()
        imsave(imgPath[:-4] + '_instances.png', instanceMap)
        imsave(imgPath[:-4] + '_signals.png', markups)

if __name__=='__name__':
    main()