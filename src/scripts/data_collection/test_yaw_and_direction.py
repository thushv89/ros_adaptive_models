import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool,Int16
from rospy_tutorials.msg import Floats
import sys
import logging
import tf
import getopt
import os.path
import utils

logger,poselogger,driveAngleLogger = None,None,None

logging_level = logging.DEBUG
logging_format = '[%(name)s] [%(funcName)s] %(message)s'

isMoving = False # check if robot is moving
image_seq_idx = 0 # Indexing used for storing images in sequential order

buffer_images = []
buffer_labels = []
buffer_pose = []


image_count = -1 # when image_count%image_skip==0 we add that to buffer

persist_index = 0 # This is the main-index


def get_direction_from_yaw_5_way(current_angle, rel_angle):
    utils.YAW_THRESHOLD = 0.01
    if (current_angle > (utils.MAX_YAW - utils.YAW_THRESHOLD) and  current_angle < utils.MAX_YAW) or \
        (current_angle < (utils.MIN_YAW + utils.YAW_THRESHOLD) and current_angle > utils.MIN_YAW):
        return 2
    elif rel_angle > utils.YAW_THRESHOLD and rel_angle > 0.14:
        return 0
    elif rel_angle > utils.YAW_THRESHOLD and rel_angle <= 0.14:
        return 1
    elif rel_angle < - utils.YAW_THRESHOLD and rel_angle > -0.14:
        return 3
    elif rel_angle < - utils.YAW_THRESHOLD and rel_angle <= -0.14:
        return 4
    else:
        return 2

def get_direction_from_yaw(current_angle, rel_angle):

    if (current_angle > (utils.MAX_YAW - utils.YAW_THRESHOLD) and  current_angle < utils.MAX_YAW) or \
        (current_angle < (utils.MIN_YAW + utils.YAW_THRESHOLD) and current_angle > utils.MIN_YAW):
        return 1
    elif rel_angle > utils.YAW_THRESHOLD:
        return 0

    elif rel_angle < - utils.YAW_THRESHOLD:
        return 2
    else:
        return 1

def callback_cam(msg):


    global isMoving
    global image_count
    global logger

    image_count += 1
    if image_count % utils.IMG_SAVE_SKIP != 0:
        return

    if isMoving:

        try:
            (utils.CURR_POSI, utils.CURR_ORI) = listener.lookupTransform("map", "base_link", rospy.Time(0));

            utils.IMG_BUFFER.append((msg.data, utils.PERSIST_INDEX))
            # currently using absolute YAW
            # used to use the relative YAW w.r.t previous image,
            # but that makes the labels dependent on the previous image making labels dependent on the path it took
            (_,_,utils.CURR_YAW) = tf.transformations.euler_from_quaternion(utils.CURR_ORI)
            if utils.PREV_YAW is None:
                utils.PREV_YAW = utils.CURR_YAW

            utils.LABEL_BUFFER.append([utils.PERSIST_INDEX, utils.CURR_YAW,get_direction_from_yaw_5_way(utils.CURR_YAW,utils.CURR_YAW - utils.PREV_YAW)])
            utils.POSE_BUFFER.append([utils.PERSIST_INDEX, list(utils.CURR_POSI) + list(utils.CURR_ORI)])

            logger.info("Collecting data (Buffer: %d) Last Direction (%d) ..",len(utils.IMG_BUFFER),get_direction_from_yaw_5_way(utils.CURR_YAW,utils.CURR_YAW - utils.PREV_YAW))
            logger.info('Current Yaw: %.5f, Rel Yaw: %.5f',utils.CURR_YAW,utils.CURR_YAW - utils.PREV_YAW)
            utils.PREV_YAW = utils.CURR_YAW
            utils.PERSIST_INDEX += 1

            assert len(utils.LABEL_BUFFER)==len(utils.IMG_BUFFER)

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            logger.info("No transform detected. Not persisting data")


def callback_odom(msg):
    '''
      we use this call back to detect the first ever move after termination of move_exec_robot script
      after that we use callback_action_status
      the reason to prefer callback_action_status is that we can make small adjustments to robots pose without adding data
    :param msg:
    :return:
    '''
    global isMoving

    data = msg
    utils.CURR_POSI = data.pose.pose.position  # has position and orientation
    utils.CURR_ORI = data.pose.pose.orientation

    x = float(utils.CURR_POSI.x)
    prevX = float(utils.PREV_POSI.x) if not utils.PREV_POSI == None else 0.0
    y = float(utils.CURR_POSI.y)
    prevY = float(utils.PREV_POSI.y) if not utils.PREV_POSI == None  else 0.0
    z = float(utils.CURR_POSI.z)
    prevZ = float(utils.PREV_POSI.z) if not utils.PREV_POSI == None else 0.0

    xo = float(utils.CURR_ORI.x)
    prevXO = float(utils.PREV_ORI.x) if not utils.PREV_ORI == None else 0.0
    yo = float(utils.CURR_ORI.y)
    prevYO = float(utils.PREV_ORI.y) if not utils.PREV_ORI == None else 0.0
    zo = float(utils.CURR_ORI.z)
    prevZO = float(utils.PREV_ORI.z) if not utils.PREV_ORI == None else 0.0
    wo = float(utils.CURR_ORI.w)
    prevWO = float(utils.PREV_ORI.w) if not utils.PREV_ORI == None  else 0.0

    pose_tol = 0.001
    ori_tol = 0.001

    # and abs(xo - prevXO) < ori_tol and abs(yo - prevYO) < ori_tol and abs(zo - prevZO) < ori_tol and abs(
    #        wo - prevWO) < ori_tol)
    if (abs(x - prevX) < pose_tol and abs(y - prevY) < pose_tol and abs(z - prevZ) < pose_tol):
        isMoving = False
    else:
        isMoving = True

    utils.PREV_POSI = data.pose.pose.position
    utils.PREV_ORI = data.pose.pose.orientation


def pose_buffer_to_string(pose_data):
    string = ''
    for index,p_list in pose_data:
        string += '%d:'%index
        for item in p_list:
            string += '%.7f,'%item
        string += '\n'
    return string


if __name__=='__main__':

    try:
        opts, args = getopt.getopt(
            sys.argv[1:], "", ["experiment-type="])
    except getopt.GetoptError as error:
        print('<filename>.py --data-dir=<dirname> --experiment-type=<sim or wombot>')
        print(error)
        sys.exit(2)

    if len(opts) != 0:
        for opt, arg in opts:

            if opt == '--experiment-type':
                if str(arg).lower() == 'sim':
                    utils.TYPE = 'SIM'
                    utils.IMG_UPDOWN = False
                elif str(arg).lower() == 'real':
                    utils.TYPE = 'REAL'
                    utils.IMG_UPDOWN = True
                else:
                    raise NotImplementedError

    logger = logging.getLogger('Logger')
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(logging_format))
    console.setLevel(logging_level)
    logger.addHandler(console)

    logger.info('Data recieved as args: ')
    logger.info('\t experiment-type: %s',utils.TYPE)

    rospy.init_node("data_collect_node")

    listener = tf.TransformListener()

    #sent_input_pub = rospy.Publisher(utils.DATA_INPUT_TOPIC, numpy_msg(Floats), queue_size=10)
    #sent_label_pub = rospy.Publisher(utils.DATA_LABEL_TOPIC, numpy_msg(Floats), queue_size=10)
    #obstacle_status_pub = rospy.Publisher(utils.OBSTACLE_STATUS_TOPIC, Bool, queue_size=10)

    rospy.Subscriber(utils.CAMERA_IMAGE_TOPIC, Image, callback_cam)
    rospy.Subscriber(utils.ODOM_TOPIC, Odometry, callback_odom)

    # rate.sleep()
    rospy.spin()  # this will block untill you hit Ctrl+C