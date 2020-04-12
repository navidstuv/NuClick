import keras.backend as K

def dice_coef(y_true, y_pred,a=1.,b=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return K.mean((2. * intersection + a) / (K.sum(y_true_f) + K.sum(y_pred_f) + b))


def dice_coef_loss(y_true, y_pred,a=1.,b=1.):
    return 1 - dice_coef(y_true, y_pred,a=a,b=b)
    

def dice_coef_loss_bce(y_true, y_pred, dice=0.5, bce=0.5):
    return binary_crossentropy(y_true, y_pred) * bce + dice_coef_loss(y_true, y_pred) * dice


def binary_crossentropy(y, p):
    return K.mean(K.binary_crossentropy(y, p))

def weighted_binary_crossentropy(y_true, y_pred,weights):
    weight_f = K.flatten(weights)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    bce = K.binary_crossentropy(y_true_f, y_pred_f)
    return K.mean(bce*weight_f, axis=-1)
 
def complex_loss_bceWeighted(y_true, y_pred,weights): #'''*** dice_loss can be replaced with jaccard_loss***'''
    Dice = dice_coef_loss(y_true, y_pred)
    BCE = weighted_binary_crossentropy(y_true, y_pred,weights,1)
    cmplxLoss = Dice + BCE
    return cmplxLoss
    
def getLoss(loss_name, weightMap=None, a=1., b=1.):

    if loss_name == 'bce_dice':
        def loss(y, p):
            return dice_coef_loss_bce(y, p, dice=0.5, bce=0.5)
        return loss
    elif loss_name == 'dice':
        def loss(y,p):
            return dice_coef_loss(y,p,a=a,b=b)
        return loss
    elif loss_name in ['complex_loss_bceWeighted','complexBCEweighted']:
        def loss(y, p):
            return complex_loss_bceWeighted(y,p,weightMap)
        return loss
    else:
        ValueError("Unknown loss.")
