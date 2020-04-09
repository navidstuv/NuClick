#from image import ImageDataGenerator
from image_segmentation_singleDist_v2 import ImageDataGenerator
idxs = np.arange(6,10,1)
imgs_train = imgs_test[idxs,]
masks_train= masks_test[idxs,]
#margins= margins[:100,]
#sepBorders= sepBorders[:100,]
#sepMasks= sepMasks[:100,]
weights_train= dists_test[idxs,]


train_gen_args = dict(
     random_click_perturb = 'Skeleton',
#     pointMapType='Polygon',
#    width_shift_range=0.1,
#    height_shift_range=0.1,
#    horizontal_flip=True,
#    vertical_flip=True,
#    rotation_range=20.,
#    zoom_range=(.8, 1.2),  # (0.7, 1.3),
#    shear_range=.2,
#    fill_mode='constant',  # Applicable to image onlyS
    albumentation=False,
#    elastic_deformation=True,
    rescale=1. / 255
)

image_datagen = ImageDataGenerator(**train_gen_args
                                   ) 
batchSize = 4
train_generator = image_datagen.flow(
    imgs_train, weightMap=weights_train, mask1=masks_train,
    shuffle=False,
    batch_size=batchSize,
    color_mode='rgb',  # rgbhsvl
    seed=seeddd)



t = 0
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
    
    plt.figure()
    plt.imshow(mosi)
    plt.show()   
    t+=1
#    if t>= len(idxs):
#        break