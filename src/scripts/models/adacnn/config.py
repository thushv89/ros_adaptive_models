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

TF_DROPOUT_STR = 'dropout'
# INPUT SIZE (ORIGINAL) (128,96)
# NEW INPUT SIZE  (RESIZED) (96,56)

TF_INPUT_SIZE = [96,160,3] # Original
TF_INPUT_AFTER_RESIZE = [64,128,3]

TF_NONCOL_STR = 'non-collision'


TF_WEIGHTS_STR = 'weights'
TF_BIAS_STR = 'bias'

TF_SCOPE_DIVIDER = '/'


FIVE_WAY_EXPERIMENT = False

TF_NUM_CLASSES = 3
TF_DIRECTION_LABELS = ['left','straight','right']

ENABLE_MASKING = True # drop out logits of zero elements on the one-hot vectors so that they are not optimized for that step
ENABLE_SOFT_CLASSIFICATION = False # use 0.9 and 0.1 instead of 0 and 1 in the one-hot vectors
SOFT_NONCOLLISION_LABEL = 0.95
SOFT_COLLISION_LABEL = 0.05


FC1_WEIGHTS = 600
FC1_WEIGHTS_DETACHED = 200

BATCH_SIZE = 10
USE_DROPOUT = True
IN_DROPOUT = 0.1
LAYER_DROPOUT = 0.5
L2_BETA = 0.0001

START_LR = 0.00001

USE_CONV_STRIDE_WITHOUT_POOLING = True

ACTIVATION = 'lrelu'

FACTOR_OF_TRAINING_TO_USE = 1.0

USE_CONV_STRIDE_WITHOUT_POOLING = True
fc_h, fc_w = -1,-1
TF_ANG_VAR_SHAPES_DETACHED = None
TF_ANG_STRIDES, TF_ANG_SCOPES = None, None

def setup_user_dependent_hyperparameters(no_pooling,square_input):

    global USE_CONV_STRIDE_WITHOUT_POOLING,fc_h, fc_w
    global TF_ANG_VAR_SHAPES_DETACHED, TF_FIRST_FC_ID, TF_INPUT_SIZE
    global TF_ANG_STRIDES, TF_ANG_SCOPES
    global TF_INPUT_AFTER_RESIZE

    if square_input:
        TF_INPUT_AFTER_RESIZE = [64, 64, 3]
        use_square_input = (TF_INPUT_SIZE[0] == TF_INPUT_SIZE[1])

    USE_CONV_STRIDE_WITHOUT_POOLING = no_pooling

    if USE_CONV_STRIDE_WITHOUT_POOLING:
        TF_ANG_SCOPES = ['conv1', 'conv2', 'conv3','conv4','fc1', 'out']
        TF_ANG_STRIDES = {'conv1': [1, 2, 2, 1], 'conv2': [1, 2, 2, 1], 'conv3': [1, 2, 2, 1], 'conv4': [1, 2, 2, 1],
                          'conv5': [1, 1, 1, 1]}

    else:
        TF_ANG_SCOPES = ['conv1', 'pool1','conv2', 'pool2','conv3', 'pool3','conv4','fc1', 'out']
        #TF_ANG_STRIDES = {'conv1': [1, 1, 1, 1], 'conv2': [1, 1, 1, 1], 'conv3': [1, 1, 1, 1],'conv4':[1,1,1,1],
        #                  'pool1': [1, 1, 2, 1], 'pool2': [1, 2, 2, 1], 'pool3': [1, 2, 4, 1]}
        if use_square_input:
            TF_ANG_STRIDES = {'conv1': [1, 1, 1, 1], 'conv2': [1, 1, 1, 1], 'conv3': [1, 1, 1, 1], 'conv4': [1, 1, 1, 1],
                                                'pool1': [1, 2, 2, 1], 'pool2': [1, 2, 2, 1], 'pool3': [1, 2, 2, 1]}

        else:
            TF_ANG_STRIDES = {'conv1': [1, 1, 1, 1], 'conv2': [1, 1, 1, 1], 'conv3': [1, 1, 1, 1], 'conv4': [1, 1, 1, 1],
                                                'pool1': [1, 2, 2, 1], 'pool2': [1, 2, 2, 1], 'pool3': [1, 2, 2, 1]}

    fc_h, fc_w = models_utils.get_fc_height_width(TF_INPUT_SIZE, TF_ANG_SCOPES, TF_ANG_STRIDES)

    # Best performing model from model search

    TF_ANG_VAR_SHAPES_DETACHED = {'conv1': [4, 8, 3, 32], 'pool1':[1,4,8,1], 'conv2': [5, 5, 32, 64], 'pool2':[1,3,3,1],
                               'conv3': [3, 3, 64, 128],'pool3':[1,3,3,1],'conv4': [3,3,128,128],
                               'fc1': [fc_h * fc_w * 128, FC1_WEIGHTS_DETACHED],
                               'out': [FC1_WEIGHTS_DETACHED, 1]}

    TF_FIRST_FC_ID = 'fc1'


def setup_best_user_dependent_hyperparameters():
    global USE_CONV_STRIDE_WITHOUT_POOLING, fc_h, fc_w
    global TF_ANG_VAR_SHAPES_DETACHED, TF_FIRST_FC_ID, TF_INPUT_SIZE
    global TF_ANG_STRIDES, TF_ANG_SCOPES
    global TF_INPUT_AFTER_RESIZE
    global BATCH_SIZE, L2_BETA, IN_DROPOUT, LAYER_DROPOUT, START_LR
    global FC1_WEIGHTS_DETACHED

    BATCH_SIZE = 25
    L2_BETA = 0.0001
    IN_DROPOUT = 0.0
    LAYER_DROPOUT = 0.5
    START_LR = 0.00005
    FC1_WEIGHTS_DETACHED = 100

    TF_ANG_SCOPES = ['conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'out']
    TF_ANG_STRIDES = {'conv1': [1, 2, 2, 1], 'conv2': [1, 2, 2, 1], 'conv3': [1, 2, 2, 1], 'conv4': [1, 1, 1, 1],
                      'conv5': [1, 1, 1, 1]}

    fc_h, fc_w = models_utils.get_fc_height_width(TF_INPUT_SIZE, TF_ANG_SCOPES, TF_ANG_STRIDES)

    # Best performing model from model search

    TF_ANG_VAR_SHAPES_DETACHED = {'conv1': [2, 4, 3, 64], 'pool1': [1, 4, 8, 1], 'conv2': [4, 8, 64, 64],
                                  'pool2': [1, 3, 3, 1],
                                  'conv3': [2, 4, 64, 128], 'pool3': [1, 3, 3, 1], 'conv4': [2, 2, 128, 128],
                                  'fc1': [fc_h * fc_w * 128, FC1_WEIGHTS_DETACHED],
                                  'out': [FC1_WEIGHTS_DETACHED, 1]}

    TF_FIRST_FC_ID = 'fc1'


ACTIVATION_MAP_DIR = 'activation_maps'
WEIGHT_SAVE_DIR = 'model_weights'