class DefaultConfigs(object):
    application = 'Nucleus' # either: 'Nucleus', 'Cell' (for WBC segmentation), 'Gland'
    multiGPU = False
    LearningRate = 4e-4
    modelType = 'MultiScaleResUnet'
    
    if application=='Nucleus':
        img_rows = 128
        img_cols = 128
        img_chnls = 3
    elif application=='Cell':
        img_rows = 256
        img_cols = 256
        img_chnls = 3
    elif application=='Gland':
        img_rows = 512
        img_cols = 512
        img_chnls = 3
    else: # define your custom sizes
        img_rows = 128
        img_cols = 128
        img_chnls = 3

    #path to train folder comprising info folders and npy folders
    train_data_path = 'F:/Nuclick project_Hemato/Data/nuclick_data/train/' 
    valid_data_path = None



config = DefaultConfigs()