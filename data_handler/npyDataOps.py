from __future__ import print_function

import os
import numpy as np
import scipy.io
from config import config


img_rows = config.img_rows
img_cols = config.img_cols


def infosToNumpyData(data_path):
    train_data_path = os.path.join(data_path, 'infos')
    images = os.listdir(train_data_path)
    total = len(images)
    

    imgs = np.ndarray((total, img_rows, img_cols,3), dtype=np.uint8)
    masks = np.ndarray((total, img_rows, img_cols), dtype=np.uint8)
    weightMaps = np.ndarray((total, img_rows, img_cols), dtype=np.float32)
    objectPoints = np.ndarray((total, img_rows, img_cols), dtype=np.uint8)
    pointOthers = np.ndarray((total, img_rows, img_cols), dtype=np.uint8)
    image_names = []

    

    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    i=0
    for image_name in images:
        mat = scipy.io.loadmat(os.path.join(train_data_path, image_name))
        img = mat['thisImg']
        mask = mat['thisObject']
        weightMap = mat['thisWeight']
        objectPoint = mat['thisPoint']
        pointOther = mat['otherPoints']

        img = np.array([img])
        mask = np.array([mask])
        weightMap = np.array([weightMap])
        objectPoint = np.array([objectPoint])
        pointOther = np.array([pointOther])

        imgs[i] = img
        masks[i] = mask
        weightMaps[i] = weightMap
        objectPoints[i] = objectPoint
        pointOthers[i] = pointOther
        image_names.append(image_name[:-9])
        i+=1
        if i % 1000 == 0:
            print('Done: {0}/{1} images'.format(i, total))
            
        
    print('Done: Converting info files to npy files.')
    
    npy_data_path = os.path.join(data_path, 'npyfiles')
    if not os.path.exists(npy_data_path):
        os.mkdir(npy_data_path)
    np.save(os.path.join(npy_data_path,'imgs.npy'), imgs)
    np.save(os.path.join(npy_data_path,'masks.npy'), masks)
    np.save(os.path.join(npy_data_path,'weightMaps.npy'), weightMaps)
    np.save(os.path.join(npy_data_path,'objectPoints.npy'), objectPoints)
    np.save(os.path.join(npy_data_path,'pointOthers.npy'), pointOthers)
    np.save(os.path.join(npy_data_path,'image_names.npy'), image_names)

    print('Done: Saving to .npy files into [npyfiles] filder.')


def loadData (data_path):
    npy_data_path = os.path.join(data_path, 'npyfiles')
    imgs = np.load(os.path.join(npy_data_path,'imgs.npy'))
    masks = np.load(os.path.join(npy_data_path,'masks.npy'))
    weightMaps = np.load(os.path.join(npy_data_path,'weightMaps.npy'))
    objectPoints = np.load(os.path.join(npy_data_path,'objectPoints.npy'))
    pointOthers = np.load(os.path.join(npy_data_path,'pointOthers.npy'))
    image_names = np.load(os.path.join(npy_data_path,'image_names.npy'))
    return imgs, masks, weightMaps, objectPoints, pointOthers, image_names
    