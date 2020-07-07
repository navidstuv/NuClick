import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import scipy.ndimage as ndi
from skimage.morphology import skeletonize_3d

def generateGuidingSignal (mask,RandomizeGuidingSignalType):
    if RandomizeGuidingSignalType == 'Point':
        binaryMask = np.uint8(mask[:,:,0]>=(0.99*np.max(mask[:,:,0])))
        temp = distance_transform_edt(binaryMask)
        tempMean = ndi.mean(temp, labels=binaryMask)
        tempStd = ndi.standard_deviation(temp, labels=binaryMask)
        tempThresh = np.random.uniform(tempMean-tempStd, tempMean+tempStd)
        tempThresh = np.max([tempThresh,0])
        tempThresh = np.min([tempThresh,np.max(temp)-1])
        newMask = np.float32(temp>tempThresh)
        if np.sum(newMask)==0:
            newMask = np.float32(temp>(tempMean/2))
        if np.sum(newMask)==0:
            newMask = np.float32(binaryMask)
        indices = np.argwhere(newMask==1) #
        if len(indices)>=2:
            rndIdx = np.random.randint(np.floor(0.05*len(indices)),np.floor(0.95*len(indices)))
            rndX = indices[rndIdx,1]
            rndY = indices[rndIdx,0]
            pointMask = np.zeros_like(mask)   
            pointMask[rndY,rndX,0] = 1
        elif len(indices)==1:
            rndIdx = 0
            rndX = indices[rndIdx,1]
            rndY = indices[rndIdx,0]
            pointMask = np.zeros_like(mask)   
            pointMask[rndY,rndX,0] = 1
        return pointMask
    if RandomizeGuidingSignalType == 'Skeleton':
        binaryMask = np.uint8(mask[:,:,0]>(0.9*np.max(mask[:,:,0])))
        if np.sum(binaryMask)>100:
            temp = distance_transform_edt(binaryMask)
            tempMean = ndi.mean(temp, labels=binaryMask)
            tempStd = ndi.standard_deviation(temp, labels=binaryMask)
            tempThresh = np.random.uniform(tempMean-tempStd, tempMean+tempStd)
            if tempThresh<0:
                tempThresh = np.random.uniform(tempMean/2, tempMean+tempStd/2)
            newMask = temp>(tempThresh)
            if np.sum(newMask)==0:
                newMask = temp>(tempThresh/2)
            if np.sum(newMask)==0:
                newMask = binaryMask
            skel = skeletonize_3d(newMask)
            skel = skel[...,np.newaxis]
            skel = np.float32(skel)
        else:
            skel = np.zeros(binaryMask.shape+(1,),dtype='float32')
        return skel
                    
def jitterClicks (weightMap):
    pointPos = np.argwhere(weightMap[:,:,0]>0)
    if len(pointPos)>0:
        xPos = pointPos[0,1] + np.random.randint(-3,3)
        xPos = np.min([xPos,weightMap.shape[1]-1])
        xPos = np.max([xPos,0])
        yPos = pointPos[0,0] + np.random.randint(-3,3)
        yPos = np.min([yPos,weightMap.shape[0]-1])
        yPos = np.max([yPos,0])
        pointMask = np.zeros_like(weightMap)
        pointMask[yPos,xPos,0] = 1
        return pointMask