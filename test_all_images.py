"""
Nuclick Prediction
Consists functions to be used for NuClick prediction

"""
import numpy as np
from skimage.io import imsave, imread
import os
from models.models import getModel
from utils.utils import readImageAndCentroids, getClickMapAndBoundingBox, getPatchs, sharpnessEnhancement,\
                            contrastEnhancement, predictPatchs, postProcessing, generateInstanceMap
from config import config
import matplotlib.pyplot as plt
from skimage.color import label2rgb


def main():
    Dot_path = config.mat_path
    image_path = config.images_path
    save_path = config.save_path
    testTimeAug = config.testTimeAug
    modelNames = [config.modelType]
    losses = [config.lossType]
    suffixes = ['']
    input_shape = (config.img_rows, config.img_cols)
    modelBaseName = 'NuClick_%s_%s_%s' % (config.application, config.modelType, config.lossType)
    modelSaveName = "%s/weights-%s.h5" % (config.weights_path, modelBaseName)


    model = getModel(config.modelType, config.lossType, input_shape)
    model.load_weights(modelSaveName)
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
                cx = [int(np.round(i)) for i in cx]
                cy = [int(np.round(i)) for i in cy]
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

                # prediction with  test time augmentation
                augNum = 3
                predNum = 1  # augNum*numModel
                preds = np.zeros((len(patchs), config.img_rows, config.img_cols), dtype=np.float32)
                print('-----Working on model := %s_%s%s-----' % (modelNames[0], losses[0], suffixes[0]))
                preds += predictPatchs(model, patchs, dists)
                print("Original images prediction, DONE!")
                if testTimeAug:
                    print("Test Time Augmentation Started")
                    temp = predictPatchs(model, patchs_shappened[:, ::-1, :], dists[:, ::-1, :])
                    preds += temp[:, ::-1, :]
                    predNum += 1
                    print("Flipped images prediction, DONE!")

                    temp = predictPatchs(model, patchs_shappened[:, :, ::-1], dists[:, :, ::-1])
                    preds += temp[:, :, ::-1]
                    predNum += 1
                    print("Sharpenned images prediction, DONE!")

                    temp = predictPatchs(model, patchs_contrasted[:, ::-1, ::-1], dists[:, ::-1, ::-1])
                    preds += temp[:, ::-1, ::-1]
                    predNum += 1
                    print("Contrasted images prediction, DONE!")
                preds /= predNum
                try:
                    masks = postProcessing(preds, thresh=0.2, minSize=5, minHole=30, doReconstruction=True, nucPoints=nucPoints)
                except:
                    masks = postProcessing(preds, thresh=0.2, minSize=5, minHole=30, doReconstruction=False, nucPoints=nucPoints)
                instanceMap = generateInstanceMap(masks, boundingBoxes, m, n)
                instanceMap_RGB  = label2rgb(instanceMap, image=img, alpha=0.3,
                 bg_label=0, bg_color=(0, 0, 0), image_alpha=1, kind='overlay')
                plt.figure()
                plt.imshow(instanceMap_RGB)
                plt.show()
                imsave(os.path.join(save_path, image_name[:-4] + '_instances.png'), instanceMap)
                # plt.figure(),plt.imshow(img)
if __name__=='__main__':
    main()
    # aa = imread('E:\Back_up\git-files\\Nuclick--\here\TCGA-RD-A8N9-01A-01-TS1_mask_instances.png')
    # aa