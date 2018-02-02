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

#INPUT SIZE (128,96)

USE_GRAYSCALE = False
TF_INPUT_SIZE = [96,128,3]


if not USE_GRAYSCALE:
    TF_RESIZE_TO = [56, 128, 3]
else:
    TF_RESIZE_TO = [56, 128, 1]


TF_NUM_CLASSES = 3

