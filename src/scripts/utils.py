

############# THE MOST IMPORTANT SETTING ################

TYPE = None #SIM or REAL

# DATA PERSISTING
IMG_DIR = None
IMG_UPDOWN = False

# we dump all the image and pose info once the buffer is full
IMG_BUFFER_CAP = 50
IMG_BUFFER = []


IMG_CHANNELS = 3 if TYPE =='REAL' else 4 # /camera/image gives out a sequence of numbers and has to be formed into a matrix. This value specifiy the number of channels
IMG_SAVE_SKIP = 3 # how many images skipped when saving sequence
BUMP_IMG_SAVE_SKIP = 3

BUMP_IMG_BUFFER_CAP = 25
BUMP_IMG_BUFFER = []
# SINGLE ITEM = (index,[pose,pose,pose,ori,ori,ori,ori])
POSE_BUFFER = []

LABEL_BUFFER = []
BUMP_LABEL_BUFFER = []

PERSIST_INDEX = 0
#########################################################

CURR_POSI, CURR_ORI = None,None
PREV_POSI, PREV_ORI = None,None
CURR_YAW, PREV_YAW = None,None
YAW_THRESHOLD = 0.1
MAX_YAW,MIN_YAW = 3.142,-3.142


#COMMON TOPICS
CAMERA_IMAGE_TOPIC = "/camera/image_color" if TYPE == 'REAL' else "/camera/image/" # sim /camera/image real /camera/image_color
LASER_SCAN_TOPIC = "/scan" # sim and real /scan
ODOM_TOPIC = "/odom"
CMD_VEL_TOPIC = "/wombot/cmd_vel" if TYPE == 'REAL' else "/cmd_vel" #sim /cmd_vel real /wombot/cmd_vel

#BUMP DETECTION
BUMP_1_THRESH = 0.25 if TYPE=='REAL' else 0.6 #sim 0.6 real 0.2
BUMP_02_THRESH = 0.18 if TYPE=='REAL' else 0.7 #sim 0.7 real 0.25
NO_RETURN_THRESH = 0.05
REVERSE_PUBLISH_DELAY = 0.1 # real 0.12
ZERO_VEL_PUBLISH_DELAY = 0.01 # publish 0 valued cmd_vel data
BUMP_LABEL_LENGTH = 3

LASER_SKIP = 4
LASER_FREQUENCY = 40 if TYPE=='REAL' else 10
LASER_POINT_COUNT = 1080 if TYPE=='REAL' else 180
LASER_ANGLE = 270 if TYPE=='REAL' else 180 #

#ODOM
NO_MOVE_COUNT_THRESH = 5

# CAMERA
CAMERA_FREQUENCY = 30 if TYPE=='REAL' else 10
IMG_W, IMG_H = 320, 240
THUMBNAIL_W, THUMBNAIL_H = 128, 96

REVERSE_TIME_OUT = 10
#Parallelization
THREADS = 50
