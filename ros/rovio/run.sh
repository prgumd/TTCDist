#!/bin/bash

rocker --volume ../../../../time_to_contact_depth --nvidia --x11 --name rovio ros:rovio bash

# docker run -it --rm ros:vins-mono \
#   /bin/bash -c \
#   "cd /root/catkin_ws/; \
#      source devel/setup.bash; \
#      roslaunch vins_estimator ${1}"
