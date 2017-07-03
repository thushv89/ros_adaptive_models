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
TF_INPUT_SIZE = [128,96,3]
TF_RESIZE_INPUT_SIZE = [96,96,3]

TF_DECONV_STR = 'deconv'
TF_WEIGHTS_STR = 'weights'
TF_BIAS_STR = 'bias'

TF_SCOPES = ['conv1','conv2','fc1']
TF_VAR_SHAPES = {'conv1':[5,5,3,32],'conv2':[3,3,32,64],'fc1':[49152,128]}
TF_STRIDES = {'conv1':[1,2,2,1],'conv2':[1,2,2,1]}
TF_OUTPUT_SHAPES = {'conv1':[5,128,96,3],'conv2':[5,64,48,32]}

TF_NUM_CLASSES = 3
ENABLE_RANDOM_MASKING = False # drop out logits of zero elements on the one-hot vectors so that they are not optimized for that step
ENABLE_SOFT_CLASSIFICATION = True # use 0.9 and 0.1 instead of 0 and 1 in the one-hot vectors
USE_GRAYSCALE = False

TF_ANG_SCOPES = ['conv1','conv2','conv3','out']

if not USE_GRAYSCALE:
    TF_ANG_VAR_SHAPES = {'conv1':[7,7,3,64],'conv2':[5,5,64,128],'conv3':[3,3,128,256],'out':[16*12*256,TF_NUM_CLASSES]}
else:
    TF_ANG_VAR_SHAPES = {'conv1': [7, 7, 1, 64], 'conv2': [5, 5, 64, 128], 'conv3': [3, 3, 128, 256],
                         'out': [16 * 12 * 256, TF_NUM_CLASSES]}
TF_ANG_STRIDES = {'conv1':[1,2,2,1],'conv2':[1,2,2,1],'conv3':[1,2,2,1]}

