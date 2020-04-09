#pip install albumentations
#Webpage: https://albumentations.readthedocs.io/en/latest/
import numpy as np

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose
)

class augmentation_clss():
    
    def __init__(self, mode): #4 MODES, IAAPerspective, ShiftScaleRotate, MediumAugmentation y StrongAugmentation
        try:
            self.augment_img = {
                'IAAPerspective': self.IAAP,
                'ShiftScaleRotate': self.SSR,
                'MediumAug': self.MediumAug,
                'StrongAug': self.StrongAug,
            }[mode]
        except:
            raise ValueError('Mode must be \'IAAPerspective\', \'ShiftScaleRotate\', \'MediumAug\', or \'StrongAug\'')     
        self.mode = mode
     
    def IAAP(self, image, scale=0.2, p=1):
        image = image.astype(np.uint8)
        aug = IAAPerspective(scale=scale, p=p)
        output = aug(image=image)['image']
        return output.astype(np.float32)
    
    def SSR (self, image, p=1):
        image = image.astype(np.uint8)
        aug = ShiftScaleRotate(p=1)
        output = aug(image=image)['image']
        return output.astype(np.float32)
    
    def MediumAug(self, image, p=1):
        image = image.astype(np.uint8)
        aug = Compose([
            CLAHE(),
            RandomRotate90(),
            Transpose(),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
            Blur(blur_limit=3),
            OpticalDistortion(),
            GridDistortion(),
            HueSaturationValue()
        ], p=p)
        output = aug(image=image)['image']
        return output.astype(np.float32)
    
    def StrongAug(self, image, p=1):
        image = image.astype(np.uint8)
        aug = Compose([
            RandomRotate90(),
            Flip(),
            Transpose(),
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p=0.2),
            OneOf([
                MotionBlur(p=.2),
                MedianBlur(blur_limit=3, p=.1),
                Blur(blur_limit=3, p=.1),
            ], p=0.2),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.2),
            OneOf([
                OpticalDistortion(p=0.3),
                GridDistortion(p=.1),
                IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            OneOf([
                CLAHE(clip_limit=2),
                IAASharpen(),
                IAAEmboss(),
                RandomContrast(),
                RandomBrightness(),
            ], p=0.3),
            HueSaturationValue(p=0.3),
        ], p=p)
        output = aug(image=image)['image']
        return output.astype(np.float32)
    
    
    
        
        




        