class DefaultConfigs(object):
    application = 'Nucleus' # either: 'Nucleus', 'Cell' (for WBC segmentation), 'Gland'
    multiGPU = False
    LearningRate = 4e-4
    modelType = 'MultiScaleResUnet'


config = DefaultConfigs()