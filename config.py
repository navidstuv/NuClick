class DefaultConfigs(object):
    multiGPU = False
    #Mixture parametrs
    m = 1
    c = 2
    num_class = 5
    grid_size = 16

    model_name = 'unet_mdn16_class'
    logs = 'logs'
    dataset= 'PanNuke'
    fold_number = 1

    base_save_folder= 'E:\Back_up\experiments_log\MDNexperiments\\'
    if not os.path.exists(base_save_folder):
        os.mkdir(base_save_folder)

    weights = base_save_folder+"/fold" + str(fold_number)+dataset+str(grid_size)
    best_models = weights + "/best_model/"

    # data_path ='E:\Back_up\experiments_log\crchistophenotypes_2016_04_28\CRCHistoPhenotypes_2016_04_28\Detection\patches\images'
    # target_path = 'E:\Back_up\experiments_log\crchistophenotypes_2016_04_28\CRCHistoPhenotypes_2016_04_28\Detection\patches\gt_maps'
    # fold_path = 'E:\Back_up\git-files\MDN-detection\dataloader\\folds\\fold' + str(fold_number) +'.pickle'
    #
    # data_path_class ='E:\Back_up\experiments_log\crchistophenotypes_2016_04_28\CRCHistoPhenotypes_2016_04_28\Detection\patches_class\images'
    # target_path_class = 'E:\Back_up\experiments_log\crchistophenotypes_2016_04_28\CRCHistoPhenotypes_2016_04_28\Detection\patches_class\gt_maps'
    # fold_path_class = 'E:\Back_up\git-files\MDN-detection\dataloader\\folds\\fold' + str(fold_number) +'.pickle'

    data_path_class_train ='E:\Back_up\experiments_log\pannuke\\train\images'
    target_path_class_train = 'E:\Back_up\experiments_log\pannuke\\train\dots'

    data_path_class_test ='E:\Back_up\experiments_log\pannuke\\test\images'
    target_path_class_test = 'E:\Back_up\experiments_log\pannuke\\test\dots'

    data_path_class_valid ='E:\Back_up\experiments_log\pannuke\\validation\images'
    target_path_class_valid = 'E:\Back_up\experiments_log\pannuke\\validation\dots'



    #metric parameters
    imgSize = (256,256)
    r = 10
    alpha = 4


    seed = 30
    lr = 1e-3
    lr_decay = 1e-4
    weight_decay = 1e-4
    train_batch_size = 32
    test_batch_size = 64

    epochs = 500

    gpus = '0'




config = DefaultConfigs()