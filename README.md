## Navigation (Predicting Driving Angles) based on monocular images

## Dependencies
* ROS (Indigo)
* Morse (1.4)
* Tensorflow (1.0.0)

## ROS Packages
* map_launcher
* multi_map_server 
* robots/sim - Robot launch files for Morse simulator
* robots/wombot - Robot launch files for Wombot
* scripts - Various python scripts

## How to Run
* Run ```roscore``` (new terminal)
* Run ```morse run <morse-scene-file>``` (new terminal with python 3.4 virtualenv activated)
* Run ```roslaunch map_launcher <map-launch-file>``` (new terminal)
* Run ```roslaunch sim amcl.launch``` (new terminal)
* Run ```roslaunch robot_state.launch``` (new terminal)
* Run ```export LIBGL_ALWAYS_SOFTWARE=1``` (new terminal)
* Run ```rosrun rviz rviz``` 

## Tips
* Make sure you source ```/opt/ros/indigo/setup.bash``` and ```<project-folder>/devel/setup.bash``` before running above things
