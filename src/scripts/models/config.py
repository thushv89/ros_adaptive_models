import models_utils

CONTEXT_WINDOW_SIZE = 1

DRIVE_ANGLE_LOG = 'drive_angle_log.log'
BUMP_DRIVE_ANGLE_LOG = 'bump_drive_angle_log.log'

#Features TFRecord
FEAT_IMG_RAW = 'image_raw'
FEAT_IMG_HEIGHT = 'height'
FEAT_IMG_WIDTH = 'width'
FEAT_IMG_CH = 'channels'
FEAT_LABEL = 'label'
FEAT_IMG_ID = 'id'

# INPUT SIZE (ORIGINAL) (128,96)
# NEW INPUT SIZE  (RESIZED) (96,56)
USE_GRAYSCALE = False
TF_INPUT_SIZE = [96,128,3] # Original
TF_INPUT_AFTER_RESIZE = [64,64,3]
use_square_input = (TF_INPUT_AFTER_RESIZE[0]==TF_INPUT_AFTER_RESIZE[1])

if not USE_GRAYSCALE:
    TF_RESIZE_TO = [56, 128, 3]
else:
    TF_RESIZE_TO = [56, 128, 1]

TF_NONCOL_STR = 'non-collision'
TF_COL_STR = 'collision'

TF_WEIGHTS_STR = 'weights'
TF_BIAS_STR = 'bias'
TF_MOMENTUM_STR = 'CustomMomentum'
TF_COL_MOMENTUM_STR = 'CustomBumpMomentum'
TF_SCOPE_DIVIDER = '/'

FIVE_WAY_EXPERIMENT = False
if not FIVE_WAY_EXPERIMENT:
    TF_NUM_CLASSES = 3
    TF_DIRECTION_LABELS = ['left','straight','right']
else:
    TF_NUM_CLASSES = 5
    TF_DIRECTION_LABELS = ['hard-left', 'soft-left', 'straight', 'soft-right', 'hard-right']
    #TF_DIRECTION_LABELS_GENERAL = ['left', 'left', 'straight', 'right', 'right']

ENABLE_MASKING = True # drop out logits of zero elements on the one-hot vectors so that they are not optimized for that step
ENABLE_SOFT_CLASSIFICATION = False # use 0.9 and 0.1 instead of 0 and 1 in the one-hot vectors
SOFT_NONCOLLISION_LABEL = 0.95
SOFT_COLLISION_LABEL = 0.05


FC1_WEIGHTS = 600
FC1_WEIGHTS_DETACHED = 96

BATCH_SIZE = 50
USE_DROPOUT = False
IN_DROPOUT = 0.1
LAYER_DROPOUT = 0.2

USE_CONV_STRIDE_WITHOUT_POOLING = False

ACTIVATION = 'lrelu'


FACTOR_OF_TRAINING_TO_USE = 1.0

USE_DILATION = False

if USE_CONV_STRIDE_WITHOUT_POOLING:
    TF_ANG_SCOPES = ['conv1', 'conv2', 'conv3','conv4','conv5','fc1', 'out']
    TF_ANG_STRIDES = {'conv1': [1, 1, 1, 1], 'conv2': [1, 1, 2, 1], 'conv3': [1, 2, 2, 1], 'conv4': [1, 1, 1, 1],
                      'conv5': [1, 1, 1, 1]}

else:
    TF_ANG_SCOPES = ['conv1', 'pool1','conv2', 'pool2','conv3', 'pool3','conv4','fc1', 'out']
    #TF_ANG_STRIDES = {'conv1': [1, 1, 1, 1], 'conv2': [1, 1, 1, 1], 'conv3': [1, 1, 1, 1],'conv4':[1,1,1,1],
    #                  'pool1': [1, 1, 2, 1], 'pool2': [1, 2, 2, 1], 'pool3': [1, 2, 4, 1]}
    if use_square_input:
        if USE_DILATION:
            TF_ANG_SCOPES = ['conv1', 'conv2', 'conv4', 'pool1', 'fc1', 'out']
            TF_DILATION = {'conv1': [2, 2], 'conv2': [4, 4], 'conv3': [8, 8], 'conv4': [8, 8]}
            TF_ANG_STRIDES = {'conv1': [1, 1, 1, 1], 'conv2': [1, 1, 1, 1], 'conv3': [1, 1, 1, 1],
                              'conv4': [1, 1, 1, 1],
                              'pool1': [1, 8, 8, 1]}
        else:
            TF_ANG_STRIDES = {'conv1': [1, 1, 1, 1], 'conv2': [1, 1, 1, 1], 'conv3': [1, 1, 1, 1], 'conv4': [1, 1, 1, 1],
                                                'pool1': [1, 2, 2, 1], 'pool2': [1, 2, 2, 1], 'pool3': [1, 2, 2, 1]}

    else:
        TF_ANG_STRIDES = {'conv1': [1, 1, 1, 1], 'conv2': [1, 1, 1, 1], 'conv3': [1, 1, 1, 1], 'conv4': [1, 1, 1, 1],
                                            'pool1': [1, 2, 2, 1], 'pool2': [1, 2, 2, 1], 'pool3': [1, 2, 2, 1]}
        TF_DILATION = {}

fc_h, fc_w = models_utils.get_fc_height_width(TF_INPUT_AFTER_RESIZE, TF_ANG_SCOPES, TF_ANG_STRIDES)


# Best performing model from model search

if use_square_input:
    TF_ANG_VAR_SHAPES_NAIVE = {
        'conv1': [4, 4, 3, 96], 'pool1':[1,2,2,1], 'conv2': [4, 4, 96, 96], 'pool2':[1,2,2,1],
        'conv3': [4, 4, 96, 96],'pool3':[1,2,2,1], 'conv4': [4,4,96,96],
        'fc1': [fc_h * fc_w * 96, FC1_WEIGHTS],
        'out': [FC1_WEIGHTS, TF_NUM_CLASSES//3]
    }

    if not FIVE_WAY_EXPERIMENT:
        TF_ANG_VAR_SHAPES_MULTIPLE = {'conv1': [4, 4, 3, 32], 'pool1': [1, 2, 2, 1], 'conv2': [4, 4, 96, 32],
                                      'pool2': [1, 2, 2, 1],
                                      'conv3': [4, 4, 96, 32], 'pool3': [1, 2, 2, 1], 'conv4': [4, 4, 96, 32],
                                      'conv5': [6, 6, 96, 32],
                                      'fc1': [fc_h * fc_w * 96, FC1_WEIGHTS // 3],
                                      'out': [FC1_WEIGHTS, 1]}
    else:
        TF_ANG_VAR_SHAPES_MULTIPLE = {'conv1': [4, 4, 3, 20], 'pool1': [1, 2, 2, 1], 'conv2': [4, 4, 100, 20],
                                      'pool2': [1, 2, 2, 1],
                                      'conv3': [4, 4, 100, 20], 'pool3': [1, 2, 2, 1], 'conv4': [4, 4, 100, 20],
                                      'fc1': [fc_h * fc_w * 100, FC1_WEIGHTS // 5],
                                      'out': [FC1_WEIGHTS, 1]}

else:
    TF_ANG_VAR_SHAPES_NAIVE = {
        'conv1': [4, 8, 3, 32], 'pool1': [1, 6, 6, 1], 'conv2': [6, 6, 32, 32], 'pool2': [1, 3, 3, 1],
        'conv3': [6, 6, 32, 32], 'pool3': [1, 6, 6, 1], 'conv4': [6, 6, 32, 32], 'conv5': [6, 6, 32, 32],
        'fc1': [fc_h * fc_w * 32, FC1_WEIGHTS],
        'out': [FC1_WEIGHTS, TF_NUM_CLASSES//3]
    }

    TF_ANG_VAR_SHAPES_MULTIPLE = {'conv1': [4, 8, 3, 12], 'pool1': [1, 6, 6, 1], 'conv2': [6, 6, 36, 24],
                               'pool2': [1, 2, 4, 1],
                               'conv3': [6, 6, 72, 24], 'pool3': [1, 6, 6, 1], 'conv4': [6, 6, 72, 24],
                               'conv5': [6, 6, 72, 24],
                               'fc1': [fc_h * fc_w * 72, FC1_WEIGHTS//3],
                               'out': [FC1_WEIGHTS//3, 1]}

TF_ANG_VAR_SHAPES_DETACHED = {'conv1': [4, 8, 3, 32], 'pool1':[1,4,8,1], 'conv2': [5, 5, 32, 48], 'pool2':[1,3,3,1],
                           'conv3': [3, 3, 48, 64],'pool3':[1,3,3,1],'conv4': [3,3,64,64],
                           'fc1': [fc_h * fc_w * 48, FC1_WEIGHTS_DETACHED],
                           'out': [FC1_WEIGHTS_DETACHED, 1]}


TF_FIRST_FC_ID = 'fc1'

TF_SDAE_ANG_SCOPES = ['fc1','fc2','fc3','out']
TF_SDAE_ANG_VAR_SHAPES = {'fc1':[TF_RESIZE_TO[0]*TF_RESIZE_TO[1]*TF_RESIZE_TO[2],512],'fc2':[512,256],'fc3':[256,128],'out':[128,TF_NUM_CLASSES]}

OPTIMIZE_HYBRID_LOSS = False



ACTIVATION_MAP_DIR = 'activation_maps'
WEIGHT_SAVE_DIR = 'model_weights'