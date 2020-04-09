from __future__ import print_function

import os
import numpy as np
import scipy.io
import pandas as pd

from skimage.io import imsave, imread
from skimage.transform import resize
from skimage.morphology import binary_dilation


SET = 'Train'
data_path = 'F:/Nuclick project_Hemato/Data/nuclick_data/'+SET+'/'
#data_path = 'F:\Nuclick project\test/'

image_rows = 256
image_cols = 256


def create_train_data():
#    ''' Reading images based on the image list (filePropos) '''
#    df = pd.read_csv(data_path+'fileProps.txt',sep='\t',header=None)
#    arr = np.array(df)
#    image_names = arr[:,0]
#    image_names = image_names.astype(np.str)
#    image_numbers = arr[:,1]
#    image_numbers = image_numbers.astype(np.int64)
#    total = int(len(image_names))
    # Here, image cluster or category can be added as well, for experiments like what Allen suggested.
    
    train_data_path = os.path.join(data_path, 'infos')
    images = os.listdir(train_data_path)
    total = len(images)
    

    imgs = np.ndarray((total, image_rows, image_cols,3), dtype=np.uint8)
    masks = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    JaccWeights = np.ndarray((total, image_rows, image_cols), dtype=np.float32)
    bceWeights = np.ndarray((total, image_rows, image_cols), dtype=np.float32)
    pointNucs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    pointOthers = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    image_names = []

    

    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    i=0
    for image_name in images:
        mat = scipy.io.loadmat(os.path.join(train_data_path, image_name))
        img = mat['thisImg']
        mask = mat['thisMask']
        JaccWeight = mat['thisWeightJacc']
        bceWeight = mat['thisWeightBCE']
        pointNuc = mat['thisNucPoint']
        pointOther = mat['otherNucsPoints']

        img = np.array([img])
        mask = np.array([mask])
        JaccWeight = np.array([JaccWeight])
        bceWeight = np.array([bceWeight])
        pointNuc = np.array([pointNuc])
        pointOther = np.array([pointOther])

        imgs[i] = img
        masks[i] = mask
        JaccWeights[i] = JaccWeight
        bceWeights[i] = bceWeight
        pointNucs[i] = pointNuc
        pointOthers[i] = pointOther
        image_names.append(image_name[:-9])
        i+=1
        if i % 1000 == 0:
            print('Done: {0}/{1} images'.format(i, total))
            
        
    print('Loading done.')
    
    npy_data_path = os.path.join(data_path, 'npyfiles')
    if not os.path.exists(npy_data_path):
        os.mkdir(npy_data_path)
    np.save(os.path.join(npy_data_path,'imgs.npy'), imgs)
    np.save(os.path.join(npy_data_path,'masks.npy'), masks)
    np.save(os.path.join(npy_data_path,'JaccWeights.npy'), JaccWeights)
    np.save(os.path.join(npy_data_path,'bceWeights.npy'), bceWeights)
    np.save(os.path.join(npy_data_path,'pointNucs.npy'), pointNucs)
    np.save(os.path.join(npy_data_path,'pointOthers.npy'), pointOthers)
    np.save(os.path.join(npy_data_path,'image_names.npy'), image_names)

    print('Saving to .npy files done.')


def load_train_data():
    npy_data_path = os.path.join(data_path, 'npyfiles')
    imgs = np.load(os.path.join(npy_data_path,'train_imgs.npy'))
    masks = np.load(os.path.join(npy_data_path,'train_masks.npy'))
    margins = np.load(os.path.join(npy_data_path,'train_margins.npy'))
    sepBorders = np.load(os.path.join(npy_data_path,'train_sepBorders.npy'))
    sepMasks = np.load(os.path.join(npy_data_path,'train_sepMasks.npy'))
    weights = np.load(os.path.join(npy_data_path,'train_weights.npy'))
    residuals = np.load(os.path.join(npy_data_path,'train_residuals.npy'))
    image_numbers = np.load(os.path.join(npy_data_path,'train_image_numbers.npy'))
    image_names = np.load(os.path.join(npy_data_path,'train_image_names.npy'))
    return imgs, masks, margins, sepBorders, sepMasks, weights, residuals, image_names, image_numbers

def load_data_single(path):
    npy_data_path = os.path.join(path, 'npyfiles')
    imgs = np.load(os.path.join(npy_data_path,'imgs.npy'))
    masks = np.load(os.path.join(npy_data_path,'masks.npy'))
    JaccWeights = np.load(os.path.join(npy_data_path,'JaccWeights.npy'))
    bceWeights = np.load(os.path.join(npy_data_path,'bceWeights.npy'))
    pointNucs = np.load(os.path.join(npy_data_path,'pointNucs.npy'))
    pointOthers = np.load(os.path.join(npy_data_path,'pointOthers.npy'))
    image_names = np.load(os.path.join(npy_data_path,'image_names.npy'))
    return imgs, masks, JaccWeights, bceWeights, pointNucs, pointOthers, image_names

def create_test_data(test_path):
    images = os.listdir(test_path)
    total = len(images)
    
    imgs = np.ndarray((total, image_rows, image_cols,3), dtype=np.uint8)
    imgs_shape = np.ndarray((total, 3), dtype=np.int32)
    imgs_id = np.ndarray((total,), dtype=object)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        img_id = image_name.split('.')[0]
        img = imread(os.path.join(test_path, image_name))
        img_shape = img.shape
        img = resize(img, (image_rows, image_cols), preserve_range=True)
        img = np.array([img])
        img_shape = np.array([img_shape])

        imgs[i] = img
        imgs_id[i] = img_id
        imgs_shape[i] = img_shape

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_id_test.npy', imgs_id)
    np.save('imgs_shape_test.npy', imgs_shape)
    print('Saving to .npy files done.')
    

def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    imgs_shape = np.load('imgs_shape_test.npy')
    return imgs_test, imgs_id, imgs_shape

if __name__ == '__main__':
    create_train_data()
#    create_test_data()
