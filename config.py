class DefaultConfigs(object):
    application = 'Nucleus' # either: 'Nucleus', 'Cell' (for WBC segmentation), 'Gland'
    multiGPU = False
    LearningRate = 4e-4
    modelType = 'MultiScaleResUnet'
    lossType = 'complexBCEweighted'
    batchSize = 32  # set this as large as possible
    
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
        
    if application=='Gland':
        guidingSignalType = 'Skeleton'
    else:
        guidingSignalType = 'Point'

    #path to train folder comprising info folders and npy folders
    train_data_path = ''
    valid_data_path = None
    weights_path = './weights'
    preds_path = './preds'

    # for processing images with their coressponding dots
    mat_path = ''
    images_path = ''
    save_path = ''
    ##########################################################

    resumeTraining = False
    outputValPreds = True # whether to run on validation set when training ends
    if valid_data_path is None:
        valPrec = 0.2 # if no validation folder specified, this part of training set would be used for validation

    testTimeAug = True
    if application=='Gland':
        testTimeJittering = None
    else:
        testTimeJittering = 'PointJiterring'
     #None
    if application=='Gland':
        Thresh = 0.5
        minSize=1000
        minHole=1000
    elif application=='Cell':
        Thresh = 0.8
        minSize=100
        minHole=100
    else:
        Thresh = 0.5
        minSize=10
        minHole=30
        
        
config = DefaultConfigs()