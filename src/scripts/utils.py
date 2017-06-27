

############# THE MOST IMPORTANT SETTING ################

TYPE = None #SIM or REAL

# DATA PERSISTING
IMG_DIR = None
IMG_UPDOWN = False

# we dump all the image and pose info once the buffer is full
IMG_BUFFER_CAP = 10
IMG_BUFFER = []

IMG_CHANNELS = 3 if TYPE =='REAL' else 4 # /camera/image gives out a sequence of numbers and has to be formed into a matrix. This value specifiy the number of channels
IMG_SAVE_SKIP = 5 # how many images skipped when saving sequence
# SINGLE ITEM = (index,[pose,pose,pose,ori,ori,ori,ori])
POSE_BUFFER = []

LABEL_BUFFER = []
PERSIST_INDEX = 0
#########################################################

CURR_POSI, CURR_ORI = None,None
PREV_POSI, PREV_ORI = None,None
CURR_YAW, PREV_YAW = None,None
#COMMON TOPICS
CAMERA_IMAGE_TOPIC = "/camera/image_color" if TYPE == 'REAL' else "/camera/image/" # sim /camera/image real /camera/image_color
LASER_SCAN_TOPIC = "/scan" # sim and real /scan
ODOM_TOPIC = "/odom"
CMD_VEL_TOPIC = "/wombot/cmd_vel" if TYPE == 'REAL' else "/cmd_vel" #sim /cmd_vel real /wombot/cmd_vel

#MY TOPICS
ACTION_STATUS_TOPIC = "/action_status"
EPISODE_STATUS_TOPIC = "/current_episode"
DATA_SENT_STATUS = "/data_sent_status"
DATA_INPUT_TOPIC = "/data_inputs"
DATA_LABEL_TOPIC = "/data_labels"
OBSTACLE_STATUS_TOPIC = "/obstacle_status"
RESTORE_AFTER_BUMP_TOPIC = "/restored_bump"
INITIAL_RUN_TOPIC = "/initial_run"
GOAL_TOPIC = "/move_base_simple/goal"

#ODOM
NO_MOVE_COUNT_THRESH = 5

# CAMERA
CAMERA_FREQUENCY = 30 if TYPE=='REAL' else 10
IMG_W, IMG_H = 320, 240
THUMBNAIL_W, THUMBNAIL_H = 128, 96


#Parallelization
THREADS = 50
