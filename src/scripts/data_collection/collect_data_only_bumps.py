import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool,Int16
from rospy_tutorials.msg import Floats
import numpy as np
from PIL import Image as PILImage
import math
import sys
import scipy.misc as sm
import logging
import pickle
from multiprocessing import Pool
from copy import deepcopy
import tf
import getopt
import os.path
import utils
import os
import time
import threading

logger,poselogger,driveAngleLogger = None,None,None

logging_level = logging.DEBUG
logging_format = '[%(name)s] [%(funcName)s] %(message)s'

isMoving = False # check if robot is moving
image_seq_idx = 0 # Indexing used for storing images in sequential order

buffer_images = []
buffer_labels = []
buffer_pose = []


image_count = -1 # when image_count%image_skip==0 we add that to buffer
laser_count = -1
persist_index = 0 # This is the main-index

LASER_RANGE_LEFT,LASER_RANGE_STRAIGHT,LASER_RANGE_RIGHT = None,None,None

reverse_lock = threading.Lock()
reversing = False

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
    global isMoving, reversing
    global image_count
    global logger

    image_count += 1
    if image_count % utils.BUMP_IMG_SAVE_SKIP != 0:
        return

    if utils.IMG_DIR and isMoving:
        try:
            (utils.CURR_POSI, utils.CURR_ORI) = listener.lookupTransform("map", "base_link", rospy.Time(0));

            if not reversing:
                utils.IMG_BUFFER.append((msg.data, utils.PERSIST_INDEX))
            # currently using absolute YAW
            # used to use the relative YAW w.r.t previous image,
            # but that makes the labels dependent on the previous image making labels dependent on the path it took
            (_,_,utils.CURR_YAW) = tf.transformations.euler_from_quaternion(utils.CURR_ORI)

            if not reversing:
                if utils.PREV_YAW is None:
                    utils.PREV_YAW = utils.CURR_YAW
                utils.LABEL_BUFFER.append([utils.PERSIST_INDEX, utils.CURR_YAW,get_direction_from_yaw(utils.CURR_YAW,utils.CURR_YAW - utils.PREV_YAW)])
                utils.PREV_YAW = utils.CURR_YAW

            utils.POSE_BUFFER.append([utils.PERSIST_INDEX, list(utils.CURR_POSI) + list(utils.CURR_ORI)])

            logger.info("Collecting data (Buffer: %d) ..",len(utils.IMG_BUFFER))
            utils.PERSIST_INDEX += 1

            assert len(utils.LABEL_BUFFER)==len(utils.IMG_BUFFER)

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            logger.info("No transform detected. Not persisting data")


def get_laser_ranges():
    # laser range slicing algorithm
    if utils.LASER_ANGLE <= 180:
        laser_slice = int(utils.LASER_POINT_COUNT / 6.)
        laser_range_left = (0, int(laser_slice))
        laser_range_straight = (
        int((utils.LASER_POINT_COUNT / 2.) - laser_slice), int((utils.LASER_POINT_COUNT / 2.) + laser_slice))
        laser_range_right = (-int(laser_slice), -1)

    # if laser exceeds 180 degrees
    else:
        laser_slice = int(((utils.LASER_POINT_COUNT * 1.0 / utils.LASER_ANGLE) * 180.) / 6.)
        cutoff_angle_per_side = int((utils.LASER_ANGLE - 180) / 2.)
        ignore_points_per_side = (utils.LASER_POINT_COUNT * 1.0 / utils.LASER_ANGLE) * cutoff_angle_per_side
        laser_range_left = (int(ignore_points_per_side), int(laser_slice + ignore_points_per_side))
        laser_range_straight = (
        int((utils.LASER_POINT_COUNT / 2.) - laser_slice), int((utils.LASER_POINT_COUNT / 2.) + laser_slice))
        laser_range_right = (-int(laser_slice - ignore_points_per_side), -int(ignore_points_per_side))

    return [laser_range_left,laser_range_straight,laser_range_right]


def callback_laser(msg):
    global reverse_lock, reversing
    global laser_count
    global LASER_RANGE_LEFT, LASER_RANGE_STRAIGHT, LASER_RANGE_RIGHT

    laser_count += 1
    #if laser_count % utils.LASER_SKIP != 0:
    #    return

    rangesTup = msg.ranges
    rangesNum = [float(r) for r in rangesTup]
    rangesNum.reverse()

    bump_thresh_1 = utils.BUMP_1_THRESH
    bump_thresh_0_2 = utils.BUMP_02_THRESH

    # print "took laser %s,%s,%s"%(got_action,isMoving,reversing)
    filtered_ranges = np.asarray(rangesNum)
    filtered_ranges[filtered_ranges < utils.NO_RETURN_THRESH] = 1000

    logger.debug("Min range recorded:%.4f", np.min(filtered_ranges))

    # middle part of laser [45:75]
    if (not reversing) and (np.min(filtered_ranges[LASER_RANGE_STRAIGHT[0]:LASER_RANGE_STRAIGHT[1]]) < bump_thresh_1 or \
                    np.min(filtered_ranges[LASER_RANGE_LEFT[0]:LASER_RANGE_LEFT[1]]) < bump_thresh_0_2 or \
                    np.min(filtered_ranges[LASER_RANGE_RIGHT[0]:LASER_RANGE_RIGHT[1]]) < bump_thresh_0_2):

        logger.debug("setting Obstacle to True")

        # If there are any labels that arent 1, we make all of them to that direction
        # e.g. labe sequence 1,1,1,2,2 will turn to be 2,2,2,2,2
        temp_label_seq = []
        for buf_i in range(len(utils.IMG_BUFFER)-utils.BUMP_LABEL_LENGTH,len(utils.IMG_BUFFER)):
            utils.BUMP_IMG_BUFFER.append(utils.IMG_BUFFER[buf_i])
            temp_label_seq.append(utils.LABEL_BUFFER[buf_i])

        utils.BUMP_LABEL_BUFFER.extend(correct_labels(temp_label_seq))

        utils.IMG_BUFFER = []
        utils.LABEL_BUFFER = []
        if len(utils.BUMP_IMG_BUFFER)>=utils.BUMP_IMG_BUFFER_CAP:
            save_img_sequence_pose()

        reversing = True
        logger.debug('Reverse lock acquired ...')

        time.sleep(utils.REVERSE_TIME_OUT)
        logger.debug('Releasing the reverse lock ...')
        reversing = False


def correct_labels(label_seq):

    label_to_assign = 1
    for lbl in label_seq[len(label_seq)//2:]:
        if lbl != 1:
            print('Turning all labels to %d',lbl)
            label_to_assign = lbl
            break
    return [label_to_assign for _ in range(len(label_seq))]


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
    if (abs(x - prevX) < pose_tol and abs(y - prevY) < pose_tol and abs(z - prevZ) < pose_tol
        and abs(xo - prevXO) < ori_tol and abs(yo - prevYO) < ori_tol and abs(zo - prevZO) < ori_tol and abs(
            wo - prevWO) < ori_tol):
        isMoving = False
    else:
        isMoving = True

    utils.PREV_POSI = data.pose.pose.position
    utils.PREV_ORI = data.pose.pose.orientation


def save_image(img_data_with_id):

    img_data,id = img_data_with_id

    img_np_2=np.empty((utils.IMG_H,utils.IMG_W,3),dtype=np.int16)

    for i in range(0,len(img_data),utils.IMG_CHANNELS):
        r_idx,c_idx=(i//utils.IMG_CHANNELS)//utils.IMG_W,(i//utils.IMG_CHANNELS)%utils.IMG_W
        img_np_2[r_idx,c_idx,(i%utils.IMG_CHANNELS)]=int(ord(img_data[i]))
        img_np_2[r_idx,c_idx,(i%utils.IMG_CHANNELS)+1]=int(ord(img_data[i+1]))
        img_np_2[r_idx,c_idx,(i%utils.IMG_CHANNELS)+2]=int(ord(img_data[i+2]))

    if utils.IMG_UPDOWN:
        img_np_2 = np.fliplr(img_np_2)
        img_np_2 = np.flipud(img_np_2)

    im = PILImage.fromarray(img_np_2.astype(np.uint8))
    im = im.resize((utils.THUMBNAIL_W,utils.THUMBNAIL_H))
    sm.imsave(utils.IMG_DIR + os.sep + 'bump_img_' +str(id) + ".png",im)


def save_img_sequence_pose():

    copy_cam_data = deepcopy(utils.BUMP_IMG_BUFFER)
    copy_pose_data = deepcopy(utils.POSE_BUFFER)
    copy_label_data = deepcopy(utils.BUMP_LABEL_BUFFER)
    utils.BUMP_IMG_BUFFER = []
    utils.POSE_BUFFER = []
    utils.BUMP_LABEL_BUFFER = []

    logger.info("Storage summary for episode %s", utils.PERSIST_INDEX)
    logger.info('\tImage count: %s\n', len(copy_cam_data))
    logger.info("\tPose count: %s", len(copy_pose_data))

    if utils.IMG_DIR is not None:
        try:
            pool = Pool(utils.THREADS)
            pool.map(save_image, copy_cam_data)
            pool.close()
            pool.join()
            pose_string = pose_buffer_to_string(copy_pose_data)
            poselogger.info(pose_string)

            for index,rel_yaw,direction in copy_label_data:
                driveAngleLogger.info('%d:%.5f:%d', index,rel_yaw,direction)

        except Exception as e:
            logger.info(e)


def pose_buffer_to_string(pose_data):
    string = ''
    for index,p_list in pose_data:
        string += '%d:'%index
        for item in p_list:
            string += '%.7f,'%item
        string += '\n'
    return string


if __name__=='__main__':
    global LASER_RANGE_LEFT, LASER_RANGE_STRAIGHT, LASER_RANGE_RIGHT

    try:
        opts, args = getopt.getopt(
            sys.argv[1:], "", ["experiment-type=", "data-dir="])
    except getopt.GetoptError as error:
        print('<filename>.py --data-dir=<dirname> --experiment-type=<sim or wombot>')
        print(error)
        sys.exit(2)

    if len(opts) != 0:
        for opt, arg in opts:
            if opt == '--data-dir':
                utils.IMG_DIR = '.'+ os.sep + str(arg)
            if opt == '--experiment-type':
                if str(arg).lower() == 'sim':
                    utils.TYPE = 'SIM'
                    utils.set_parameters_with_type(utils.TYPE)
                elif str(arg).lower() == 'real':
                    utils.TYPE = 'REAL'
                    utils.set_parameters_with_type(utils.TYPE)
                else:
                    raise NotImplementedError

    LASER_RANGE_LEFT, LASER_RANGE_STRAIGHT, LASER_RANGE_RIGHT = get_laser_ranges()

    logger = logging.getLogger('Logger')
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(logging_format))
    console.setLevel(logging_level)
    logger.addHandler(console)

    logger.info('Data recieved as args: ')
    logger.info('\t data-dir: %s',utils.IMG_DIR)
    logger.info('\t experiment-type: %s',utils.TYPE)
    if utils.IMG_DIR and not os.path.exists(utils.IMG_DIR):
        os.mkdir(utils.IMG_DIR)

    poselogger = logging.getLogger('PoseLogger')
    poselogger.setLevel(logging.INFO)
    fh = logging.FileHandler(utils.IMG_DIR + os.sep + 'bump_pose_log.log', mode='w')
    fh.setFormatter(logging.Formatter('%(message)s'))
    fh.setLevel(logging.INFO)
    poselogger.addHandler(fh)

    driveAngleLogger = logging.getLogger('DriveAngleLogger')
    driveAngleLogger.setLevel(logging.INFO)
    anglefh = logging.FileHandler(utils.IMG_DIR + os.sep + 'bump_drive_angle_log.log', mode='w')
    anglefh.setFormatter(logging.Formatter('%(message)s'))
    anglefh.setLevel(logging.INFO)
    driveAngleLogger.addHandler(anglefh)

    rospy.init_node("data_collect_node")

    listener = tf.TransformListener()

    #sent_input_pub = rospy.Publisher(utils.DATA_INPUT_TOPIC, numpy_msg(Floats), queue_size=10)
    #sent_label_pub = rospy.Publisher(utils.DATA_LABEL_TOPIC, numpy_msg(Floats), queue_size=10)
    #obstacle_status_pub = rospy.Publisher(utils.OBSTACLE_STATUS_TOPIC, Bool, queue_size=10)

    rospy.Subscriber(utils.LASER_SCAN_TOPIC, LaserScan, callback_laser)
    rospy.Subscriber(utils.CAMERA_IMAGE_TOPIC, Image, callback_cam)
    rospy.Subscriber(utils.ODOM_TOPIC, Odometry, callback_odom)

    # rate.sleep()
    rospy.spin()  # this will block untill you hit Ctrl+C