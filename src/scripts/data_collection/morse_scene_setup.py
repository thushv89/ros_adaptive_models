__author__ = 'thushv89'
import getopt

from morse.builder import *


if __name__ == '__main__':
    environment_str = ''
    # A 'naked' PR2 robot to the scene
    atrv = ATRV()
    atrv.translate(x=2.5, y=3.2, z=0.0)

    # An odometry sensor to get odometry information
    odometry = Odometry()
    atrv.append(odometry)
    odometry.add_interface('ros', topic="/odom",frame_id="odom",child_frame_id='base_link')

    # Keyboard control
    keyboard = Keyboard()
    atrv.append(keyboard)

    # Camera causes trouble cause delays and screwing up AMCL
    cam_frequency = 10
    camera = VideoCamera()
    #camera.translate(x = -1.5, z= 0.9)
    camera.translate(x = 0.7, z= 0.5)
    camera.rotate(y = -0.0)
    #camera.properties(cam_width=256,cam_height=192,cam_far=500,cam_near=2.15)
    camera.properties(cam_width=320,cam_height=240,cam_focal=6.,cam_near=0.1,cam_far=500)
    camera.frequency(cam_frequency)
    atrv.append(camera)
    camera.add_interface('ros',topic='/camera')


    # for localization
    scan = Hokuyo()
    scan.translate(x=0.275, z=0.252)
    atrv.append(scan)
    scan.properties(Visible_arc = False)
    scan.properties(laser_range = 30.0)
    scan.properties(resolution = 1)
    scan.properties(scan_window = 180.0) #angle of laser
    scan.frequency(10)
    scan.create_laser_arc()
    scan.add_interface('ros', topic='/scan',frame_id='base_scan')

    motion = MotionXYW()
    atrv.append(motion)
    motion.add_interface('ros', topic='/cmd_vel')

    # Set the environment
    # MY environments my-environments
    #apartment-my1.blend  apartment-my3.blend
    #apartment-my2.blend  indoor-1-my1.blend

    env = Environment('sandbox')

    #environment_str = 'indoors-1/indoor-1'
    #environment_str = 'my-environments/indoor-1-my1'
    #environment_str = 'my-environments/indoor-1-my2'
    #environment_str = 'my-environments/apartment-my1'
    #environment_str = 'my-environments/apartment-my2'
    #environment_str = 'my-environments/apartment-my3'
    #environment_str = 'my-environments/grande_salle-my1'
    #environment_str = 'my-environments/grande_salle-my2'
    #environment_str = 'my-environments/grande_salle-my3'
    #env = Environment(environment_str)

