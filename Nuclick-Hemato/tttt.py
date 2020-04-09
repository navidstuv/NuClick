#from image import ImageDataGenerator
from image_segmentation import ImageDataGenerator

imgs_train = imgs[1001:1005,]
masks_train= masks[1001:1005,]
#margins= margins[:100,]
#sepBorders= sepBorders[:100,]
#sepMasks= sepMasks[:100,]
weights_train= dists[1001:1005,]


train_gen_args = dict(
    random_click_perturb = 'Test',
    pointMapType='',
#    width_shift_range=0,
#    height_shift_range=0,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    zoom_range=(.7, 1.3),
#    shear_range=0.2,
    fill_mode='constant',  # Applicable to image onlyS
    albumentation=True,
#    channel_shift_range=False,  # This must be in range of 255?
#    contrast_adjustment=False,  #####MOSI
#    illumination_gradient=False,
#    intensity_scale_range=0.1,  #####MOSI
#    sharpness_adjustment=False,
#    apply_noise=False,
#    elastic_deformation=False,
    rescale=1. / 255
)


image_datagen = ImageDataGenerator(**train_gen_args
                                   ) 

train_generator = image_datagen.flow(
    imgs_train, weightMap=weights_train, mask1=masks_train,
    shuffle=False,
    batch_size=4,
    color_mode='rgb',  # rgbhsvl
    seed=seeddd)



for  [x_batch,w_batch], mask1_batch in train_generator:
    # Show the first 9 images
    plt.figure()
    for i in range(0, 4):
        temp = x_batch[i].reshape(img_rows, img_cols, 3)
        img_color = temp[:,:,:3]
        temp = w_batch[i].reshape(img_rows, img_cols, 3)
        w_pred = temp[:,:,:]
        temp = mask1_batch[i].reshape(img_rows, img_cols, 1)
        mask1_pred = temp[:,:,0]
        mosi = img_color.copy()
        mosi[:,:,0] = (1-w_pred[:,:,0])*mosi[:,:,0]
        mosi[:,:,1] = (1-w_pred[:,:,0])*mosi[:,:,1]
        mosi[:,:,2] = (1-w_pred[:,:,0])*mosi[:,:,2]
#        temp = mask2_batch[i].reshape(img_rows, img_cols, 1)
#        mask2_pred = temp[:,:,0]
        plt.subplot(4,4,i*4+1)
        plt.imshow(mosi)
        plt.subplot(4,4, i*4+2)
        plt.imshow(w_pred[:,:,0])
        plt.subplot(4,4, i*4+3)
        plt.imshow(w_pred[:,:,1])
        plt.subplot(4,4, i*4+4)
        plt.imshow(mask1_pred)
    plt.show()    
#    break