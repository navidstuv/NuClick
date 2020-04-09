"""Fairly basic set of tools for real-time data augmentation on image data.
Can easily be extended to include new transformations,
new preprocessing methods, etc...
EDITOR: MOSI
This version can be used for iimage segmetation. The generator.flow method takes two inputs ann return two images (img, mask) for the training.
Augomentations are done separately to each of outputs. 
This property works justs for numpy inputs (FOR NOW)
"""
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import threading
import warnings
import multiprocessing.pool
from functools import partial
from skimage import exposure
from skimage.filters import gaussian
from skimage.util import random_noise
from scipy.ndimage.morphology import distance_transform_edt
from skimage.transform import PiecewiseAffineTransform, warp
from albumentations import (RGBShift, HueSaturationValue, RandomBrightness, RandomContrast, CLAHE, RandomGamma, 
                            GaussianBlur, IAASharpen, IAAEmboss, GaussNoise, JpegCompression, OneOf, Compose)

from keras import backend as K
from keras.utils.data_utils import Sequence


try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

def random_elastic_field_generator (IMAGE_HIGHT, IMAGE_WIDTH, IMAGE_HIGHTSP=16, IMAGE_WIDTHSP=16, scale = 0.01):
    src_IMAGE_WIDTH = np.linspace(0, IMAGE_WIDTH, IMAGE_WIDTHSP)
    src_IMAGE_HIGHT = np.linspace(0, IMAGE_HIGHT, IMAGE_HIGHTSP)
    src_IMAGE_HIGHT, src_IMAGE_WIDTH = np.meshgrid(src_IMAGE_HIGHT, src_IMAGE_WIDTH)
    src = np.dstack([src_IMAGE_WIDTH.flat, src_IMAGE_HIGHT.flat])[0]
    # add random oscillation to row and column coordinates
    rndDst_IMAGE_HIGHT = np.random.normal(0,scale,src[:, 1].shape)*IMAGE_HIGHT
    rndDst_IMAGE_WIDTH = np.random.normal(0,scale,src[:, 0].shape)*IMAGE_WIDTH
    dst_IMAGE_HIGHT = src[:, 1] + rndDst_IMAGE_HIGHT
    dst_IMAGE_WIDTH = src[:, 0] + rndDst_IMAGE_WIDTH
    dst = np.vstack([dst_IMAGE_WIDTH, dst_IMAGE_HIGHT]).T
    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)
    return tform

def _unsharp_mask_single_channel(image, radius, amount):
    """Single channel implementation of the unsharp masking filter."""
    blurred = gaussian(image,sigma=radius,mode='reflect')
    result = image + (image - blurred) * amount
    result = np.clip(result, 0, 1)
    return result

if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
    if hasattr(pil_image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS


def random_rotation(x, rg, row_axis=1, col_axis=2, channel_axis=0,
                    fill_mode='nearest', cval=0.):
    """Performs a random rotation of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        rg: Rotation range, in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Rotated Numpy image tensor.
    """
    theta = np.pi / 180 * np.random.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_shift(x, wrg, hrg, row_axis=1, col_axis=2, channel_axis=0,
                 fill_mode='nearest', cval=0.):
    """Performs a random spatial shift of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        wrg: Width shift range, as a float fraction of the width.
        hrg: Height shift range, as a float fraction of the height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Shifted Numpy image tensor.
    """
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_shear(x, intensity, row_axis=1, col_axis=2, channel_axis=0,
                 fill_mode='nearest', cval=0.):
    """Performs a random spatial shear of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        intensity: Transformation intensity.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Sheared Numpy image tensor.
    """
    shear = np.random.uniform(-intensity, intensity)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_zoom(x, zoom_range, row_axis=1, col_axis=2, channel_axis=0,
                fill_mode='nearest', cval=0.):
    """Performs a random spatial zoom of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        zoom_range: Tuple of floats; zoom range for width and height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Zoomed Numpy image tensor.

    # Raises
        ValueError: if `zoom_range` isn't a tuple.
    """
    if len(zoom_range) != 2:
        raise ValueError('`zoom_range` should be a tuple or list of two floats. '
                         'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_channel_shift(x, intensity, channel_axis=0):
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x

def random_intensity_scaling(x,scale):
    x = np.clip(x * np.random.uniform(1.-scale, 1.+scale), 0., 255.)
    return x

def random_contrast_adjustment(x):
    if np.random.random() < .5: # Lowering contrast for 75% of times
        min_x, max_x = np.min(x), np.max(x)
        if max_x>min_x:
            low_out = min_x+(np.random.uniform(.1,.3)*max_x)
            high_out = max_x-(np.random.uniform(.1,.3)*max_x)
            x = exposure.rescale_intensity(x, in_range=(min_x, max_x), out_range=(low_out, high_out)) 
#    elif np.random.random() < 0.5:
#        x = 255. * exposure.equalize_adapthist(x/255., clip_limit=0.005)
    else: # Enhancing the contrast for the rest.
        p2, p98 = np.percentile(x, (2, 98)) #####
        if p2==p98:
            p2, p98 = np.min(x), np.max(x)
        if p98>p2:
            x = exposure.rescale_intensity(x, in_range=(p2, p98), out_range=(0.,255.))
    return np.clip(x, 0., 255., out=x)

def random_channel_contrast_adjustment(x):
    if x.shape[2] !=3:
        return x
    else:
        chnlIdx = np.random.randint(0, 2 + 1)
        p2, p98 = np.percentile(x[:,:,chnlIdx], (2, 98))
        if p2==p98:
            p2, p98 = np.min(x), np.max(x)
        x[:,:,chnlIdx] = 255. * exposure.rescale_intensity(x[:,:,chnlIdx], in_range=(p2, p98))
    return x
    
def random_illumination_gradient(x):
    if np.random.random()<0.35:
        centerY = x.shape[0] // 2
        centerX = x.shape[1] // 2
        rndY = np.int64(np.random.uniform(centerY-0.1*centerY,centerY+0.1*centerY))
        rndX = np.int64(np.random.uniform(centerX-0.2*centerX,centerX+0.2*centerX))
        c = np.zeros((x.shape[0:2]))
        c[rndY,rndX] = 1
        c = distance_transform_edt(1-c)
        c = np.max(c)-c
        low = np.random.uniform(.4, .6)
        high = np.random.uniform(1.1, 1.3)
        c = exposure.rescale_intensity(c, in_range=(np.min(c), np.max(c)), out_range=(low,high))
        c = c[..., np.newaxis]
        c = np.repeat(c,3,axis=2)
#        plt.figure()
#        plt.imshow(c/np.max(c))     
    else: # Horizontal llumination gradient
        low = np.random.uniform(.6, .75)
        high = np.random.uniform(1.1, 1.2)
        if np.random.random()<0.75:
            c = np.linspace(low, high, x.shape[1])[None, :, None]
            c = np.repeat(c,x.shape[0],axis=0)
            c = np.repeat(c,3,axis=2)
            if np.random.random()<0.5:
                c = flip_axis(c, 1)
        else:
            c = np.linspace(low, high, x.shape[0])[:, None, None]
            c = np.repeat(c,x.shape[1],axis=1)
            c = np.repeat(c,3,axis=2)
            if np.random.random()<0.5:
                c = flip_axis(c, 0)
    x = np.clip(x * c, 0., 255.)
    return x

def random_sharpness_adjustment(x):
    if np.random.random()<0.7: # Bluring the mage for 40% of times
        sig = np.random.uniform(.5, 1.) # (.8,1.2)
        x=gaussian(x,sig,multichannel=True,preserve_range=True,mode='reflect')
        x=np.clip(x, 0., 255.)
    else: # Sharping the image using unsharp_filtering  !!! ITS MAY BE BETTER TO APPLY SHARPPENING ON v CHANNEL FROM hsv
        for channel in range(x.shape[-1]):
            x[..., channel] = 255. * _unsharp_mask_single_channel(x[..., channel]/255., 2, 1)
    return x

def random_apply_noise(x):
    noiseType =  np.random.randint(0,3,1)
    if noiseType==0:
        x = 255.*random_noise(x/255., mode='speckle', var=np.random.uniform(.001,.004),clip=True)
    if noiseType==1:
        x = 255.*random_noise(x/255., mode='gaussian', var=np.random.uniform(.001,.0015),clip=True)
    if noiseType==2:
        x = 255.*random_noise(x/255., mode='s&p', amount=np.random.uniform(.005,.02) ,clip=True)
    return np.clip(x, 0., 255., out=x)
    
def random_hair_occlusion(x, h):
    rndIdx = np.random.randint(0,len(h))
    thisHairMask = h[rndIdx,]
    x = x * np.float32(thisHairMask)
    return x

def albumentation_transform(x):
    x = x.astype(np.uint8)
    aug = Compose([
            OneOf([
                RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.5), #.7
                HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(-60,20), val_shift_limit=0, always_apply=False, p=.5),#.8
                HueSaturationValue(hue_shift_limit=20, sat_shift_limit=5, val_shift_limit=0, always_apply=False, p=.5),
                ],
                p=1.),
            OneOf([
                GaussianBlur(blur_limit=7, p=.5),
                IAASharpen(alpha=(0.2, 0.4), lightness=(1.0, 1.0), p=0.5),
                IAAEmboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.5),
                ],
                p=1.),
            OneOf([
                RandomBrightness(limit=.25, p=0.5),#.75
                RandomContrast(limit=0.2, p=0.3),
                RandomGamma(gamma_limit=(75, 125), p=0.5),#.7
                CLAHE(clip_limit=3.0, tile_grid_size=(4, 4), p=0.3),
                ], 
                p=1.),
            OneOf([
                GaussNoise(var_limit=(10.0, 30.0), p=.5),
                JpegCompression(quality_lower=80, quality_upper=90, p=0.5),
                ], 
                p=1),
        ], p=1.)
    
    augmented_img = aug(image=x)['image']
    while(np.array_equal(augmented_img,np.zeros(augmented_img.shape)) == True):  #Avoid black images
        augmented_img=aug(image=x)['image']

    return augmented_img.astype(np.float32)
       
def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.):
    """Apply the image transformation specified by a matrix.

    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        The transformed version of the input.
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    if cval=='random':
        if np.random.random()<0.5: # 50% of cases use black images
            cval = np.zeros((len(x),))
        else:
            cval = np.random.randint(np.min(x),np.max(x),(len(x),))
        channel_images = [ndi.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=0,
            mode=fill_mode,
            cval=i) for (x_channel,i) in zip(x,cval)]
    else:
        channel_images = [ndi.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=0,
            mode=fill_mode,
            cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def array_to_img(x, data_format=None, scale=True):
    """Converts a 3D Numpy array to a PIL Image instance.

    # Arguments
        x: Input Numpy array.
        data_format: Image data format.
        scale: Whether to rescale image values
            to be within [0, 255].

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = np.asarray(x, dtype=K.floatx())
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape:', x.shape)
    if x.shape[2] == 7:
        x = x[:,:,:3]

    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format:', data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: ', x.shape[2])


def img_to_array(img, data_format=None):
    """Converts a PIL Image instance to a Numpy array.

    # Arguments
        img: PIL Image instance.
        data_format: Image data format.

    # Returns
        A 3D Numpy array.

    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=K.floatx())
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x


def load_img(path, grayscale=False, target_size=None,
             interpolation='nearest'):
    """Loads an image into PIL format.

    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return img


def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f)]


class ImageDataGenerator(object):
    """Generate minibatches of image data with real-time data augmentation.

    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channel.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided. This is
            applied after the `preprocessing_function` (if any provided)
            but before any other transformation.
        preprocessing_function: function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
            (the depth) is at index 1, in 'channels_last' mode it is at index 3.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    """

    def __init__(self,
                 albumentation=False,
                 elastic_deformation=False,
                 apply_noise=False,
                 sharpness_adjustment=False,
                 hair_occlusion=False,
                 illumination_gradient=False,#####MOSI
                 channel_contrast_adjustment=False,#####MOSI
                 contrast_adjustment=False, #####MOSI
                 intensity_scale_range = 0.,#####MOSI
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None):
        if data_format is None:
            data_format = K.image_data_format()
        
        self.albumentation = albumentation
        self.elastic_deformation = elastic_deformation
        self.apply_noise = apply_noise
        self.sharpness_adjustment = sharpness_adjustment
        self.hair_occlusion = hair_occlusion
        self.illumination_gradient = illumination_gradient
        self.channel_contrast_adjustment = channel_contrast_adjustment
        self.contrast_adjustment = contrast_adjustment
        self.intensity_scale_range = intensity_scale_range
        self.featurewise_center = featurewise_center
        self.samplewise_center = samplewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_std_normalization = samplewise_std_normalization
        self.zca_whitening = zca_whitening
        self.zca_epsilon = zca_epsilon
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function

        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError('`data_format` should be `"channels_last"` (channel after row and '
                             'column) or `"channels_first"` (channel before row and column). '
                             'Received arg: ', data_format)
        self.data_format = data_format
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
        if data_format == 'channels_last':
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2

        self.mean = None
        self.std = None
        self.principal_components = None

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('`zoom_range` should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', zoom_range)

    def flow(self, x, weightMap=None, mask1=None, mask2=None, y=None, h=None, batch_size=32, shuffle=True, seed=None,color_mode='rgb',
             save_to_dir=None, save_prefix='', save_format='png'):
        return NumpyArrayIterator(
            x, weightMap, mask1, mask2, y, h, self,
            batch_size=batch_size,
            color_mode = color_mode,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)

    def flow_from_directory(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            interpolation='nearest'):
        return DirectoryIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            interpolation=interpolation)

    def standardize(self, x):
        """Apply the normalization configuration to a batch of inputs.

        # Arguments
            x: batch of inputs to be normalized.

        # Returns
            The inputs, normalized.
        """
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.rescale:
            x *= self.rescale
        if self.samplewise_center:
            x -= np.mean(x, keepdims=True)
        if self.samplewise_std_normalization:
            x /= np.std(x, keepdims=True) + 1e-7

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_center`, but it hasn\'t '
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + 1e-7)
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_std_normalization`, but it hasn\'t '
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.zca_whitening:
            if self.principal_components is not None:
                flatx = np.reshape(x, (-1, np.prod(x.shape[-3:])))
                whitex = np.dot(flatx, self.principal_components)
                x = np.reshape(whitex, x.shape)
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening`, but it hasn\'t '
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        return x

    def random_transform(self, x,weightMap,mask1,mask2,hm, seed=None):
        """Randomly augment a single image tensor.

        # Arguments
            x: 3D tensor, single image.
            seed: random seed.

        # Returns
            A randomly transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        if seed is not None:
            np.random.seed(seed)
        
        
        isDeformed = False
        if self.elastic_deformation:
            if np.random.random() < .5:
                isDeformed=True
                tScale = 0.0075
                hstp = 16
                wstp = 16
                IMAGE_HIGHT, IMAGE_WIDTH = x.shape[0], x.shape[1]
                tform = random_elastic_field_generator (IMAGE_HIGHT, IMAGE_WIDTH, IMAGE_HIGHTSP=hstp, IMAGE_WIDTHSP=wstp, scale = tScale)
                x = warp(x, tform, output_shape=(IMAGE_HIGHT, IMAGE_WIDTH),preserve_range=True)
                if len(np.unique(weightMap))>1:
                    weightMap = warp(weightMap, tform, output_shape=(IMAGE_HIGHT, IMAGE_WIDTH),preserve_range=True,order=0)
                if len(np.unique(mask1))>1:
                    mask1 = warp(mask1, tform, output_shape=(IMAGE_HIGHT, IMAGE_WIDTH),preserve_range=True,order=0) 
                if len(np.unique(mask2))>1:
                    mask2 = warp(mask2, tform, output_shape=(IMAGE_HIGHT, IMAGE_WIDTH),preserve_range=True,order=0) 

        if self.hair_occlusion and hm is not None:
            if np.random.random() < 0.5:
                x = random_hair_occlusion(x, hm)    
                
        illuminated = False
        if self.illumination_gradient:
            if np.random.random() < 0.5:
                x = random_illumination_gradient(x)
                illuminated = True
        
        if self.albumentation:
            x = albumentation_transform(x)
            
        if self.sharpness_adjustment and not self.albumentation:
            if np.random.random() < 0.5:
                x = random_sharpness_adjustment(x)
                
    
        if self.apply_noise  and not self.albumentation:
            if np.random.random() < 0.5:
                x = random_apply_noise(x)
        
        channelShifted = False
        if self.channel_shift_range != 0  and not self.albumentation:
            if np.random.random() < 0.5:
                x = random_channel_shift(x,
                                         self.channel_shift_range,
                                         img_channel_axis)
                channelShifted = True
        
        if self.channel_contrast_adjustment and not channelShifted  and not self.albumentation:
            if np.random.random() < 0.5:
                x = random_channel_contrast_adjustment(x)
         
        contrasted = False
        if self.contrast_adjustment  and not self.albumentation: #####
            if np.random.random() < 0.5: ##### Do contrast adjustment with more probability if it was enabled
                x = random_contrast_adjustment(x)
                contrasted = True
            
        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range and not isDeformed:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                    [0, np.cos(shear), 0],
                                    [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            x = apply_transform(x, transform_matrix, img_channel_axis,
                                fill_mode=self.fill_mode, cval=self.cval)
            mask1 = apply_transform(mask1, transform_matrix, img_channel_axis,
                                fill_mode="constant", cval=0)
            mask2 = apply_transform(mask2, transform_matrix, img_channel_axis,
                                fill_mode="constant", cval=0)
            weightMap = apply_transform(weightMap, transform_matrix, img_channel_axis,
                                fill_mode="constant", cval=0.) ### MOSSSIIII ### THIS VALUE MUST HANDEDLED BASED ON THE WEIGHT GENERATION FILE IN MATLAB
                
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_axis)
                mask1 = flip_axis(mask1, img_col_axis)
                mask2 = flip_axis(mask2, img_col_axis)
                weightMap = flip_axis(weightMap, img_col_axis)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_axis)
                mask1 = flip_axis(mask1, img_row_axis)
                mask2 = flip_axis(mask2, img_row_axis)
                weightMap = flip_axis(weightMap, img_row_axis)

        if self.intensity_scale_range != 0 and not illuminated and not contrasted:
            if np.random.random() < 0.5:
                x = random_intensity_scaling(x,self.intensity_scale_range)
                

        return x, weightMap, mask1, mask2

    def fit(self, x,
            augment=False,
            rounds=1,
            seed=None):
        """Fits internal statistics to some sample data.

        Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.

        # Arguments
            x: Numpy array, the data to fit on. Should have rank 4.
                In case of grayscale data,
                the channels axis should have value 1, and in case
                of RGB data, it should have value 3.
            augment: Whether to fit on randomly augmented samples
            rounds: If `augment`,
                how many augmentation passes to do over the data
            seed: random seed.

        # Raises
            ValueError: in case of invalid input `x`.
        """
        x = np.asarray(x, dtype=K.floatx())
        if x.ndim != 4:
            raise ValueError('Input to `.fit()` should have rank 4. '
                             'Got array with shape: ' + str(x.shape))
        if x.shape[self.channel_axis] not in {1, 3, 4}:
            warnings.warn(
                'Expected input to be images (as Numpy array) '
                'following the data format convention "' + self.data_format + '" '
                '(channels on axis ' + str(self.channel_axis) + '), i.e. expected '
                'either 1, 3 or 4 channels on axis ' + str(self.channel_axis) + '. '
                'However, it was passed an array with shape ' + str(x.shape) +
                ' (' + str(x.shape[self.channel_axis]) + ' channels).')

        if seed is not None:
            np.random.seed(seed)

        x = np.copy(x)
        if augment:
            ax = np.zeros(tuple([rounds * x.shape[0]] + list(x.shape)[1:]), dtype=K.floatx())
            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.random_transform(x[i])
            x = ax

        if self.featurewise_center:
            self.mean = np.mean(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.mean = np.reshape(self.mean, broadcast_shape)
            x -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.std = np.reshape(self.std, broadcast_shape)
            x /= (self.std + K.epsilon())

        if self.zca_whitening:
            flat_x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
            sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
            u, s, _ = linalg.svd(sigma)
            self.principal_components = np.dot(np.dot(u, np.diag(1. / np.sqrt(s + self.zca_epsilon))), u.T)


class Iterator(Sequence):
    """Base class for image data iterators.

    Every `Iterator` must implement the `_get_batches_of_transformed_samples`
    method.

    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size  # round up

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[current_index:
                                   current_index + self.batch_size]

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        # Arguments
            index_array: array of sample indices to include in batch.

        # Returns
            A batch of transformed samples.
        """
        raise NotImplementedError


class NumpyArrayIterator(Iterator):
    """Iterator yielding data from a Numpy array.

    # Arguments
        x: Numpy array of input data.
        y: Numpy array of targets data.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, x, weightMap, mask1, mask2, y, h, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 data_format=None, color_mode='rgb',
                 save_to_dir=None, save_prefix='', save_format='png'):
        if y is not None and len(x) != len(y):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))

        if data_format is None:
            data_format = K.image_data_format()
        self.x = np.asarray(x, dtype=K.floatx())
        self.weightMap=np.asarray(weightMap,dtype=K.floatx()) if weightMap is not None else None
        self.mask1=np.asarray(mask1,dtype=K.floatx()) if mask1 is not None else None
        self.mask2=np.asarray(mask2,dtype=K.floatx()) if mask2 is not None else None
        self.y=np.asarray(y,dtype=K.floatx()) if y is not None else None
        self.h=np.asarray(h,dtype=K.floatx()) if h is not None else None

        if self.x.ndim != 4:
            raise ValueError('Input data in `NumpyArrayIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.x.shape)
        channels_axis = 3 if data_format == 'channels_last' else 1
        if self.x.shape[channels_axis] not in {1, 3, 4, 7}:
            warnings.warn('NumpyArrayIterator is set to use the '
                          'data format convention "' + data_format + '" '
                          '(channels on axis ' + str(channels_axis) + '), i.e. expected '
                          'either 1, 3 or 4 channels on axis ' + str(channels_axis) + '. '
                          'However, it was passed an array with shape ' + str(self.x.shape) +
                          ' (' + str(self.x.shape[channels_axis]) + ' channels).')
						  
        if color_mode not in {'rgb', 'grayscale','rgbhsvl'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale" or "rgbhsvl".')
        self.color_mode = color_mode
        if self.color_mode == 'rgb':
            if data_format == 'channels_last':
                self.image_shape = self.x.shape[1:3] + (3,)
            else:
                self.image_shape = (3,) + self.x.shape[1:3]
        else:
            if data_format == 'channels_last':
                self.image_shape = self.x.shape[1:3] + (1,)
            else:
                self.image_shape = (1,) + self.x.shape[1:3]
        if self.color_mode == 'rgbhsvl':
            if data_format == 'channels_last':
                self.image_shape = self.x.shape[1:3] + (7,)
            else:
                self.image_shape = (7,) + self.x.shape[1:3]
        
        if data_format == 'channels_last':
            self.mask_shape = self.x.shape[1:3] + (1,)
            self.dist_shape = self.x.shape[1:3] + (2,)
        else:
            self.mask_shape = (1,) + self.x.shape[1:3]
		
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(NumpyArrayIterator, self).__init__(x.shape[0], batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx()) #np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:]), dtype=K.floatx())
        batch_weightMap = np.zeros((len(index_array),) + self.dist_shape, dtype=K.floatx()) #np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:]), dtype=K.floatx())
        batch_mask1 = np.zeros((len(index_array),) + self.mask_shape, dtype=K.floatx()) #np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:]), dtype=K.floatx())
        batch_mask2 = np.zeros((len(index_array),) + self.mask_shape, dtype=K.floatx()) #np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:]), dtype=K.floatx())
        for i, j in enumerate(index_array):
            x = self.x[j]
            mask1 = self.mask1[j] if self.mask1 is not None else np.zeros(shape=self.mask_shape)
            weightMap = self.weightMap[j] if self.weightMap is not None else np.zeros(shape=self.dist_shape)
            mask2 = self.mask2[j] if self.mask2 is not None else np.zeros_like(mask1)
            h = self.h if self.h is not None else np.zeros_like(mask1)
            x,weightMap,mask1,mask2 = self.image_data_generator.random_transform(x.astype(K.floatx()),weightMap.astype(K.floatx()),mask1.astype(K.floatx()),mask2.astype(K.floatx()),h)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            batch_mask1[i] = mask1
            batch_mask2[i] = mask2
            batch_weightMap[i] = weightMap
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=j,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        if self.mask1 is not None:
            if self.weightMap is None and self.mask2 is not None:
                return batch_x, [batch_mask1, batch_mask2]
            elif self.weightMap is None and self.mask2 is None:
                return batch_x, batch_mask1
            elif self.weightMap is not None and self.mask2 is None:
                return [batch_x, batch_weightMap], batch_mask1
            else:
                return [batch_x, batch_weightMap], [batch_mask1, batch_mask2]
        else:
            if self.weightMap is not None:
                return [batch_x, batch_weightMap]
            else:
                return batch_x

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


def _count_valid_files_in_directory(directory, white_list_formats, follow_links):
    """Count files with extension in `white_list_formats` contained in a directory.

    # Arguments
        directory: absolute path to the directory containing files to be counted
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.

    # Returns
        the count of files with extension in `white_list_formats` contained in
        the directory.
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])

    samples = 0
    for root, _, files in _recursive_list(directory):
        for fname in files:
            is_valid = False
            for extension in white_list_formats:
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                samples += 1
    return samples


def _list_valid_filenames_in_directory(directory, white_list_formats,
                                       class_indices, follow_links):
    """List paths of files in `subdir` relative from `directory` whose extensions are in `white_list_formats`.

    # Arguments
        directory: absolute path to a directory containing the files to list.
            The directory name is used as class label and must be a key of `class_indices`.
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        class_indices: dictionary mapping a class name to its index.

    # Returns
        classes: a list of class indices
        filenames: the path of valid files in `directory`, relative from
            `directory`'s parent (e.g., if `directory` is "dataset/class1",
            the filenames will be ["class1/file1.jpg", "class1/file2.jpg", ...]).
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])

    classes = []
    filenames = []
    subdir = os.path.basename(directory)
    basedir = os.path.dirname(directory)
    for root, _, files in _recursive_list(directory):
        for fname in sorted(files):
            is_valid = False
            for extension in white_list_formats:
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                classes.append(class_indices[subdir])
                # add filename relative to directory
                absolute_path = os.path.join(root, fname)
                filenames.append(os.path.relpath(absolute_path, basedir))
    return classes, filenames


class DirectoryIterator(Iterator):
    """Iterator capable of reading images from a directory on disk.

    # Arguments
        directory: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        classes: Optional list of strings, names of subdirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
    """

    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None, save_to_dir=None,
                 save_prefix='', save_format='png',
                 follow_links=False, interpolation='nearest'):
        if data_format is None:
            data_format = K.image_data_format()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale','rgbhsvl'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale" or "rgbhsvl".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        if self.color_mode == 'rgbhsvl':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (7,)
            else:
                self.image_shape = (7,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm'}

        # first, count the number of samples and classes
        self.samples = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        pool = multiprocessing.pool.ThreadPool()
        function_partial = partial(_count_valid_files_in_directory,
                                   white_list_formats=white_list_formats,
                                   follow_links=follow_links)
        self.samples = sum(pool.map(function_partial,
                                    (os.path.join(directory, subdir)
                                     for subdir in classes)))

        print('Found %d images belonging to %d classes.' % (self.samples, self.num_classes))

        # second, build an index of the images in the different class subfolders
        results = []

        self.filenames = []
        self.classes = np.zeros((self.samples,), dtype='int32')
        i = 0
        for dirpath in (os.path.join(directory, subdir) for subdir in classes):
            results.append(pool.apply_async(_list_valid_filenames_in_directory,
                                            (dirpath, white_list_formats,
                                             self.class_indices, follow_links)))
        for res in results:
            classes, filenames = res.get()
            self.classes[i:i + len(classes)] = classes
            self.filenames += filenames
            i += len(classes)
        pool.close()
        pool.join()
        super(DirectoryIterator, self).__init__(self.samples, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(os.path.join(self.directory, fname),
                           grayscale=grayscale,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=j,
                                                                  hash=np.random.randint(1e7),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_classes), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)
