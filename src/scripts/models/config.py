CONTEXT_WINDOW_SIZE = 1

DRIVE_ANGLE_LOG = 'drive_angle_log.log'
BUMP_DRIVE_ANGLE_LOG = 'bump_drive_angle_log.log'

#Features TFRecord
FEAT_IMG_RAW = 'image_raw'
FEAT_IMG_HEIGHT = 'height'
FEAT_IMG_WIDTH = 'width'
FEAT_IMG_CH = 'channels'
FEAT_LABEL = 'label'

#INPUT SIZE (128,96)

USE_GRAYSCALE = True
TF_INPUT_SIZE = [96,128,3]

if not USE_GRAYSCALE:
    TF_RESIZE_TO = [56, 128, 3]
else:
    TF_RESIZE_TO = [56, 128, 1]

TF_WEIGHTS_STR = 'weights'
TF_BIAS_STR = 'bias'
TF_MOMENTUM_STR = 'Momentum'
TF_NUM_CLASSES = 3
ENABLE_MASKING = True # drop out logits of zero elements on the one-hot vectors so that they are not optimized for that step
ENABLE_SOFT_CLASSIFICATION = True # use 0.9 and 0.1 instead of 0 and 1 in the one-hot vectors
SOFT_NONCOLLISION_LABEL = 0.95
SOFT_COLLISION_LABEL = 0.05
TF_ANG_SCOPES = ['conv1','conv2','fc1','out']

if not USE_GRAYSCALE:
    TF_ANG_VAR_SHAPES = {'conv1':[7,7,3,16],'conv2':[5,5,16,32],
                         'fc1':[16*12*32,256*TF_NUM_CLASSES],
                         'out_0':[256,TF_NUM_CLASSES]}
else:
    TF_ANG_VAR_SHAPES = {'conv1': [7, 7, 1, 16], 'conv2': [5, 5, 16, 32], 'fc1':[16 * 12 * 32, 256],
                         'out': [256, TF_NUM_CLASSES]}
TF_ANG_STRIDES = {'conv1':[1,4,4,1],'conv2':[1,2,2,1],'conv3':[1,2,2,1]}
TF_FIRST_FC_ID = 'fc1'

TF_SDAE_ANG_SCOPES = ['fc1','fc2','fc3','out']
TF_SDAE_ANG_VAR_SHAPES = {'fc1':[TF_RESIZE_TO[0]*TF_RESIZE_TO[1]*TF_RESIZE_TO[2],512],'fc2':[512,256],'fc3':[256,128],'out':[128,TF_NUM_CLASSES]}