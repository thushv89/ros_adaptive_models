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
TF_INPUT_AFTER_RESIZE = [64,128,3]

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
TF_NUM_CLASSES = 3
ENABLE_MASKING = True # drop out logits of zero elements on the one-hot vectors so that they are not optimized for that step
ENABLE_SOFT_CLASSIFICATION = False # use 0.9 and 0.1 instead of 0 and 1 in the one-hot vectors
SOFT_NONCOLLISION_LABEL = 0.95
SOFT_COLLISION_LABEL = 0.05

FC1_WEIGHTS = 256
FC1_WEIGHTS_DETACHED = 96

BATCH_SIZE = 50
USE_DROPOUT = True
IN_DROPOUT = 0.1
LAYER_DROPOUT = 0.25

USE_CONV_STRIDE_WITHOUT_POOLING = False

ACTIVATION = 'lrelu'
OUTPUT_ACTIVATION = 'sigmoid'

FACTOR_OF_TRAINING_TO_USE = 2


if USE_CONV_STRIDE_WITHOUT_POOLING:
    TF_ANG_SCOPES = ['conv1', 'conv2', 'conv3','conv4','conv5','fc1', 'out']
    TF_ANG_STRIDES = {'conv1': [1, 1, 1, 1], 'conv2': [1, 1, 2, 1], 'conv3': [1, 2, 2, 1], 'conv4': [1, 1, 1, 1],
                      'conv5': [1, 1, 1, 1]}
else:
    TF_ANG_SCOPES = ['conv1', 'pool1','conv2', 'pool2','conv3', 'pool3','conv4','fc1', 'out']
    TF_ANG_STRIDES = {'conv1': [1, 1, 1, 1], 'conv2': [1, 1, 1, 1], 'conv3': [1, 1, 1, 1],'conv4':[1,1,1,1],
                      'pool1': [1, 1, 2, 1], 'pool2': [1, 2, 2, 1], 'pool3': [1, 2, 4, 1]}

fc_h, fc_w = models_utils.get_fc_height_width(TF_INPUT_AFTER_RESIZE, TF_ANG_SCOPES, TF_ANG_STRIDES)

if USE_CONV_STRIDE_WITHOUT_POOLING:

    TF_ANG_VAR_SHAPES_NAIVE = {'conv1': [4, 8, 3, 16], 'conv2': [4, 8, 16, 32],'conv3': [5,5,32,48],'conv4':[3,3,48,64],'conv5':[3,3,64,64],
                         'fc1': [fc_h * fc_w * 64, FC1_WEIGHTS],
                         'out': [FC1_WEIGHTS, TF_NUM_CLASSES]}

    TF_ANG_VAR_SHAPES_DETACHED = {'conv1': [4, 8, 3, 16], 'conv2': [4, 8, 16, 32],'conv3': [5,5,32,48],'conv4':[3,3,48,64],
                         'fc1': [fc_h * fc_w * 48, FC1_WEIGHTS_DETACHED],
                         'out': [FC1_WEIGHTS_DETACHED, 1]}

    TF_VAR_SHAPES_DUAL_DETACHED_NONCOL = {'conv1': [4, 8, 3, 16], 'conv2': [4, 8, 16, 32],'conv3': [5,5,32,48],'conv4':[3,3,48,64],'conv5':[3,3,64,64],
                         'fc1': [fc_h * fc_w * 64, FC1_WEIGHTS],
                         'out': [FC1_WEIGHTS, 1]}

    TF_VAR_SHAPES_DUAL_DETACHED_COL = {'conv2': [4, 8, 16, 16], 'conv3': [3, 6, 16, 24],
                                          'fc1': [fc_h * fc_w * 24, FC1_WEIGHTS//2],
                                          'out': [FC1_WEIGHTS//2, 1]}

    TF_VAR_SHAPES_DUAL_NAIVE_NONCOL = {'conv1': [4, 8, 3, 32], 'conv2': [3, 6, 32, 48],
                                          'conv3': [3, 6, 48, 64],
                                          'fc1': [fc_h * fc_w * 64, FC1_WEIGHTS],
                                          'out': [FC1_WEIGHTS, 3]}

    TF_VAR_SHAPES_DUAL_NAIVE_COL = {'conv1': [4, 8, 3, 32], 'conv2': [3, 6, 32, 24],
                                       'conv3': [3, 6, 24, 32],
                                       'fc1': [fc_h * fc_w * 32, FC1_WEIGHTS // 2],
                                       'out': [FC1_WEIGHTS // 2, 3]}

else:
    # Best performing model from model search
    TF_ANG_VAR_SHAPES_NAIVE = {'conv1': [4, 8, 3, 32], 'pool1':[1,6,6,1], 'conv2': [6, 6, 32, 64], 'pool2':[1,2,4,1],
                               'conv3': [6, 6, 64, 64],'pool3':[1,6,6,1], 'conv4': [6,6,64,64], 'conv5':[6,6,64,64],
                               'fc1': [fc_h * fc_w * 64, FC1_WEIGHTS],
                               'out': [FC1_WEIGHTS, TF_NUM_CLASSES]}

    TF_ANG_VAR_SHAPES_MULTIPLE = {'conv1': [4, 8, 3, 12], 'pool1': [1, 6, 6, 1], 'conv2': [6, 6, 12, 24],
                               'pool2': [1, 2, 4, 1],
                               'conv3': [6, 6, 24, 24], 'pool3': [1, 6, 6, 1], 'conv4': [6, 6, 24, 24],
                               'conv5': [6, 6, 24, 24],
                               'fc1': [fc_h * fc_w * 24, FC1_WEIGHTS//3],
                               'out': [FC1_WEIGHTS//3, 1]}

    TF_ANG_VAR_SHAPES_DETACHED = {'conv1': [4, 8, 3, 32], 'pool1':[1,4,8,1], 'conv2': [5, 5, 32, 48], 'pool2':[1,3,3,1],
                               'conv3': [3, 3, 48, 64],'pool3':[1,3,3,1],'conv4': [3,3,64,64],
                               'fc1': [fc_h * fc_w * 48, FC1_WEIGHTS_DETACHED],
                               'out': [FC1_WEIGHTS_DETACHED, 1]}

    TF_VAR_SHAPES_DUAL_NAIVE_NONCOL = {'conv1': [4, 8, 3, 16], 'pool1': [1, 6, 6, 1], 'conv2': [6, 6, 32, 48],
                                          'pool2': [1, 2, 4, 1],
                                          'conv3': [6, 6, 64, 48], 'pool3': [1, 3, 3, 1],'conv4':[6,6,64,48],'conv5':[6,6,64,48],
                                          'fc1': [fc_h * fc_w * 64, FC1_WEIGHTS],
                                          'out': [FC1_WEIGHTS, 3]}

    TF_VAR_SHAPES_DUAL_NAIVE_COL = {'conv1': [4, 8, 3, 16], 'pool1': [1, 6, 6, 1], 'conv2': [6, 6, 32, 16], 'pool2': [1, 3, 3, 1],
                                       'conv3': [6, 6, 64, 16], 'pool3': [1, 6, 6, 1], 'conv4':[6,6,64,16], 'conv5':[6,6,64,16]
                                       }

TF_FIRST_FC_ID = 'fc1'

TF_SDAE_ANG_SCOPES = ['fc1','fc2','fc3','out']
TF_SDAE_ANG_VAR_SHAPES = {'fc1':[TF_RESIZE_TO[0]*TF_RESIZE_TO[1]*TF_RESIZE_TO[2],512],'fc2':[512,256],'fc3':[256,128],'out':[128,TF_NUM_CLASSES]}

OPTIMIZE_HYBRID_LOSS = False



ACTIVATION_MAP_DIR = 'activation_maps'
WEIGHT_SAVE_DIR = 'model_weights'