import keras.backend as K
from keras.losses import categorical_crossentropy


def hard_dice_coef(y_true, y_pred, smooth=1e-3):
    y_true_f = K.flatten(K.round(y_true[..., 0]))
    y_pred_f = K.flatten(K.round(y_pred[..., 0]))
    intersection = K.sum(y_true_f * y_pred_f)
    return 100. * (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def hard_dice_coef_ch1(y_true, y_pred, smooth=1e-3):
    y_true_f = K.flatten(K.round(y_true[..., 1]))
    y_pred_f = K.flatten(K.round(y_pred[..., 1]))
    intersection = K.sum(y_true_f * y_pred_f)
    return 100. * (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef(y_true, y_pred,a=1.,b=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return K.mean((2. * intersection + a) / (K.sum(y_true_f) + K.sum(y_pred_f) + b))


def dice_coef_loss(y_true, y_pred,a=1.,b=1.):
    return 1 - dice_coef(y_true, y_pred,a=a,b=b)
    
def weighted_dice_loss(y_true, y_pred,weights,reps=1):
    if reps>1:
        weights_r = K.repeat_elements(weights,reps,axis=K.ndim(weights)-1)
    else:
        weights_r = weights
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    weight_f = K.flatten(weights_r)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1-(2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f*weight_f) + 1.)



def dice_coef_loss_bce(y_true, y_pred, dice=0.5, bce=0.5):
    return binary_crossentropy(y_true, y_pred) * bce + dice_coef_loss(y_true, y_pred) * dice


def binary_crossentropy(y, p):
    return K.mean(K.binary_crossentropy(y, p))


def double_head_loss(y_true, y_pred):
    mask_loss = dice_coef_loss_bce(y_true[..., 0], y_pred[..., 0])
    contour_loss = dice_coef_loss_bce(y_true[..., 1], y_pred[..., 1])
    return mask_loss + contour_loss


def mask_contour_mask_loss(y_true, y_pred):
    mask_loss = dice_coef_loss_bce(y_true[..., 0], y_pred[..., 0])
    contour_loss = dice_coef_loss_bce(y_true[..., 1], y_pred[..., 1])
    full_mask = dice_coef_loss_bce(y_true[..., 2], y_pred[..., 2])
    return mask_loss + 2 * contour_loss + full_mask


def softmax_dice_loss(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred) * 0.6 + dice_coef_loss(y_true[..., 0], y_pred[..., 0]) * 0.2 + dice_coef_loss(y_true[..., 1], y_pred[..., 1]) * 0.2

def jaccard_coef(y_true, y_pred,a=1.,b=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    y_true_f_s = K.pow(y_true_f,2)
    y_pred_f_s = K.pow(y_pred_f,2)
    return (intersection+ a) / (K.sum(y_true_f_s) + K.sum(y_pred_f_s) - intersection + b)
    
def jaccard_loss(y_true, y_pred,a=1.,b=1): 
    return 1-jaccard_coef(y_true, y_pred,a=a,b=b)

def jaccard(y_true, y_pred,a=1.,b=1,): 
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1. - K.mean((intersection + a) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + b))

def jaccard_loss_bce(y_true, y_pred, a=1., b=1.):
    return binary_crossentropy(y_true, y_pred) * 0.5 + jaccard(y_true, y_pred,a ,b) * 0.5

def sampleWeighted_jaccard_bce(sample_weights):
    def loss(y_true, y_pred):
        weightedLoss = sample_weights*jaccard_loss_bce(y_true, y_pred)
        return weightedLoss
    return loss
	
def weighted_jaccard_loss(weights,reps=1):
    def loss(y_true, y_pred):
        if reps>1:
            weights_r = K.repeat_elements(weights,reps,axis=K.ndim(weights)-1)
        else:
            weights_r = weights
        weight_f = K.flatten(weights_r)
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        y_true_f_s = K.pow(y_true_f,2)
        y_pred_f_s = K.pow(y_pred_f,2)
        return 1 - (intersection + 1.) / (K.sum(y_true_f_s) + K.sum(y_pred_f_s*weight_f) - intersection + 1.)
    return loss

def sampleWeighted_jaccard_loss(sample_weights,a=1.,b=1.):
    def loss(y_true, y_pred):
        weightedLoss = sample_weights*jaccard_loss(y_true, y_pred,a=a,b=b)
        return weightedLoss
    return loss
	
def weighted_binary_crossentropy(y_true, y_pred,weights,reps=1):
    if reps>1:
        weights_r = K.repeat_elements(weights,reps,axis=K.ndim(weights)-1)
    else:
        weights_r = weights
    weight_f = K.flatten(weights_r)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    bce = K.binary_crossentropy(y_true_f, y_pred_f)
    return K.mean(bce*weight_f, axis=-1)
    
def complex_loss(y_true, y_pred,weights): #'''*** dice_loss can be replaced with jaccard_loss***'''
    Dice = weighted_dice_loss(y_true, y_pred,weights,1)
    BCE = weighted_binary_crossentropy(y_true, y_pred,weights,1)
    cmplxLoss = Dice + BCE
    return cmplxLoss
	
def complex_loss_bceWeighted(y_true, y_pred,weights): #'''*** dice_loss can be replaced with jaccard_loss***'''
    Dice = dice_coef_loss(y_true, y_pred)
    BCE = weighted_binary_crossentropy(y_true, y_pred,weights,1)
    cmplxLoss = Dice + BCE
    return cmplxLoss

def complex_loss_diceWeighted(y_true, y_pred,weights): #'''*** dice_loss can be replaced with jaccard_loss***'''
    Dice = weighted_dice_loss(y_true, y_pred,weights,1)
    BCE = binary_crossentropy(y_true, y_pred)
    cmplxLoss = Dice + BCE
    return cmplxLoss
    
def getLoss(loss_name, weightMap=None, sampleWeight=None, a=1., b=1.):
    if loss_name == 'bce_dice':
        def loss(y, p):
            return dice_coef_loss_bce(y, p, dice=0.5, bce=0.5)
        return loss

    elif loss_name == 'bce':
        def loss(y, p):
            return binary_crossentropy(y, p)
        return loss
    
    elif loss_name == 'categorical_dice':
        return softmax_dice_loss
    
    elif loss_name == 'double_head_loss':
        return double_head_loss
    
    elif loss_name == 'mask_contour_mask_loss':
        return mask_contour_mask_loss
    
    elif loss_name == 'dice':
        def loss(y,p):
            return dice_coef_loss(y,p,a=a,b=b)
        return loss
    
    elif loss_name == 'weightedDice' and weightMap is not None:
        def loss(y,p):
            return weighted_dice_loss(y,p,weightMap)
        return loss
    
    elif loss_name == 'jaccard':
        def loss(y,p):
            return jaccard_loss(y,p,a=a,b=b)
        return loss
    
    elif loss_name == 'bce_jaccard':
        def loss(y, p):
            return jaccard_loss_bce(y, p, a=a, b=b)
        return loss
    
    elif loss_name == 'weightedJaccard' and weightMap is not None:
        return weighted_jaccard_loss(weightMap)
		
    elif loss_name == 'sampleWeightedJaccard' and sampleWeight is not None:
        return sampleWeighted_jaccard_loss(sampleWeight,a=a,b=b)
    
    elif loss_name == 'sampleWeightedJaccardBCE' and sampleWeight is not None:
        return sampleWeighted_jaccard_bce(sampleWeight)
    
    elif loss_name in {'complex','cmplxLoss','complexDiceBCEweighted'}:
        def loss(y,p):
            return complex_loss(y,p,weightMap)
        return loss
		
    elif loss_name in {'complex_loss_bceWeighted','complexBCEweighted'}:
        def loss(y,p):
            return complex_loss_bceWeighted(y,p,weightMap)
        return loss
    
    elif loss_name in {'complex_loss_diceWeighted','complexDiceWeighted'}:
        def loss(y,p):
            return complex_loss_diceWeighted(y,p,weightMap)
        return loss
		
    else:
        ValueError("Unknown loss.")
