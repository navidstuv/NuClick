# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 18:34:56 2019

@author: Nrp-PC
"""
path = 'E:/Jahanifar/Gland_512_data_all/CRAG_valid/'
imgs_test, masks_test, JaccWeights_test, bceWeights_test, pointNucs_test, pointOthers_test, imgNames_test, imgNumbers_test = load_data_single(path)
#for i in range(len(pointNucs_test)):
#    pointNucs_test[i] = binary_dilation(pointNucs_test[i],np.ones((5,5),dtype=np.uint8))
JaccWeights_test = JaccWeights_test[..., np.newaxis]  # margins = margins.astype('float32')
bceWeights_test = bceWeights_test[..., np.newaxis]  # sepBorders = sepBorders.astype('float32')
masks_test = masks_test[..., np.newaxis]
pointNucs_test = pointNucs_test[..., np.newaxis]
pointOthers_test = pointOthers_test[..., np.newaxis]
dists_test = np.concatenate((pointNucs_test,pointOthers_test,bceWeights_test),axis=3)#dists_test = np.concatenate((pointNucs_test,pointOthers_test,JaccWeights_test),axis=3)
del bceWeights_test
del JaccWeights_test
del pointNucs_test
del pointOthers_test

num_val = imgs_test.shape[0]  # 0

image_datagen_val = ImageDataGenerator(random_click_perturb = 'Skeleton', pointMapType=pointMapT,
    rescale=1. / 255)

#model.load_weights(modelSaveName)
batchSizeVal = 1
val_generator = image_datagen_val.flow(
    imgs_test, weightMap=dists_test, mask1=masks_test,
    shuffle=False,
    batch_size=batchSizeVal,
    color_mode='rgb',
    seed=seeddd)

val_predicts  = model.predict_generator(val_generator, steps=num_val // batchSizeVal)
pred_dir = "%s/GoodCondition_%s" % (path, 'preds')
imgs_mask_test = np.matrix.squeeze(val_predicts, axis=3)
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)
for image_id in range(0, len(imgs_mask_test)):
    mask = np.uint8(imgs_mask_test[image_id, :, :] * 255)
    imsave(os.path.join(pred_dir, imgNames_test[image_id] + '_mask.png'), mask)