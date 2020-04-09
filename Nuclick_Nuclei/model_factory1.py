from MoNuSeg_Models1 import get_UNET_twoHead_multiscale_residual, get_UNET_twoHead_multiscale_residual_shallow
from MoNuSeg_Models1 import get_spagettiNet_singleHead_multiscale_residual_deep, get_spagettiNet_singleHead_multiscale_residual
from MoNuSeg_Models1 import get_spagettiNet_twoHeaded_multiscale_residual, get_spagettiNet_twoHeaded_multiscale_residual_deep
from MoNuSeg_Models1 import get_spagettiNet_twoHeaded_multiscale_deep, get_UNET_twoHead_multiscale_residual_deep , get_spagettiNet_twoHeaded_multiscale_bottleneck
from MoNuSeg_Models1 import get_spagettiNet_twoHeaded_multiscale_residual_veryDeep,get_spagettiNet_ThreeHead_noshit_multiscale_deep_sephead
from MoNuSeg_Models1 import get_spagettiNet_twoHeaded_mask_multiscale_residual_deep, unet,get_spagettiNet_ThreeHead_noshit_multiscale_deep, get_spagettiNet_ThreeHeaded_noshit_multiscale_veryDeep
from MoNuSeg_Models1 import get_spagettiNet_threeHeaded_multiscale_residual_deep,get_spagettiNet_singleHead_noshit_multiscale_deep,get_UNET_singleHead_multiscale_residual , get_UNET_ThreeHead_multiscale_residual_deep
from DMUNet_edited import get_Dense_MultiScale_UNet, get_unet, get_MultiScale_ResUnet


def getModel(network, cellLoss, marginLoss,input_shape):
    if network == 'spagetti-multiscale-residual':
        return get_spagettiNet_twoHeaded_multiscale_residual(input_shape,cellLoss, marginLoss)
    elif network == 'spagetti-multiscale-residual-deep':
        return get_spagettiNet_twoHeaded_multiscale_residual_deep(input_shape,cellLoss, marginLoss)
    elif network == 'mask-spagetti-multiscale-residual-deep':
        return get_spagettiNet_twoHeaded_mask_multiscale_residual_deep(input_shape,cellLoss, marginLoss)
    elif network == 'spagetti-multiscale-residual-veryDeep':
        return get_spagettiNet_twoHeaded_multiscale_residual_veryDeep(input_shape,cellLoss, marginLoss)
    elif network == 'spagetti-singleHead-multiscale-residual-deep':
        return get_spagettiNet_singleHead_multiscale_residual_deep(input_shape,cellLoss)
    elif network == 'spagetti-singleHead-multiscale-residual':
        return get_spagettiNet_singleHead_multiscale_residual(input_shape,cellLoss)

    elif network == 'spagettiNet_twoHeaded_multiscale_bottleneck':
        return get_spagettiNet_twoHeaded_multiscale_bottleneck(input_shape,cellLoss, marginLoss)
    elif network== 'spagettiNet_singleHead_noshit_multiscale_deep':
        return get_spagettiNet_singleHead_noshit_multiscale_deep(input_shape,cellLoss, marginLoss)
    elif network == 'spagettiNet_ThreeHead_noshit_multiscale_deep':
        return get_spagettiNet_ThreeHead_noshit_multiscale_deep(input_shape,cellLoss, marginLoss)
    elif network == 'spagettiNet_ThreeHead_noshit_multiscale_verydeep':
        return get_spagettiNet_ThreeHeaded_noshit_multiscale_veryDeep(input_shape,cellLoss, marginLoss)
    elif network == 'spagettiNet_ThreeHead_noshit_multiscale_deep_sephead':
        return get_spagettiNet_ThreeHead_noshit_multiscale_deep_sephead(input_shape,cellLoss, marginLoss)
    elif network == 'spagetti-multiscale-deep':
        return get_spagettiNet_twoHeaded_multiscale_deep(input_shape,cellLoss, marginLoss)
    elif network == 'unet-multiscale-residual':
        return get_UNET_twoHead_multiscale_residual(input_shape,cellLoss, marginLoss)
    elif network == 'unet-singleHead-multiscale-residual':
        return get_UNET_singleHead_multiscale_residual(input_shape,cellLoss, marginLoss)
    elif network == 'unet-multiscale-residual-deep':
        return get_UNET_twoHead_multiscale_residual_deep(input_shape,cellLoss, marginLoss)
    elif network == 'unet-multiscale-residual-shallow':
        return get_UNET_twoHead_multiscale_residual_shallow(input_shape,cellLoss, marginLoss)
    elif network == 'spagetti-Threehead-multiscale-residual-deep':
        return get_spagettiNet_threeHeaded_multiscale_residual_deep(input_shape,cellLoss, marginLoss)
    elif    network == 'Unet-Threehead-multiscale-residual-deep':
        return get_UNET_ThreeHead_multiscale_residual_deep(input_shape,cellLoss, marginLoss)
    elif network == 'unet':
        return unet(input_shape, cellLoss, marginLoss)
    elif network == 'deeplabv3+':
        return Deeplabv3(input_shape + (3,), cellLoss, num_classes=1, last_activation=True, OS=16)
    elif network == 'segnet':
        return segnet(
        input_shape + (3,),
        cellLoss,
        n_labels = 1,
        kernel=3,
        pool_size=(2, 2))
    elif network in {'DMUNet','DMSUNet','Dense_MultiScale_UNet'}:
        return get_Dense_MultiScale_UNet(input_shape,cellLoss)
    elif network=='UNet':
        return get_unet(input_shape,cellLoss)
    elif network in {'msresunet','MultiScaleResUnet','MSRUNet'}:
        return get_MultiScale_ResUnet(input_shape,cellLoss)
    else:
        raise ValueError('unknown network ' + network)