###############################################################################
#
# File: ros_record_vins_mono.py
#
# Record vins mono output to a npy file for comparison
#
# History:
# 11-09-20 - Levi Burner - Created file
#
###############################################################################

import argparse
import json
import os
import glob
import pickle
import time

import numpy as np

import rospy
from nav_msgs.msg import Odometry
import rosbag
import pickle

last_timestamp = None
all_recorded = False

def odometry_callback(odom):
    global last_timestamp
    global all_recorded

    if all_recorded:
        print('All recorded')
        return

    p = [odom.pose.pose.position.x,
         odom.pose.pose.position.y,
         odom.pose.pose.position.z]
    t = odom.header.stamp.to_sec()

    if last_timestamp is None:
        last_timestamp = t

    if t < last_timestamp:
        all_recorded = True

    print(t, p)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rec', dest='recording', type=str, help='Bag file')
    #parser.add_argument('--allspeed', dest='allspeed', type=str, help='Run over all to calculate average speed')
    parser.add_argument('--allspeed', dest='allspeed', action='store_true', help='Run over all to calculate average speed')

    args = parser.parse_args()


    if not args.allspeed:
        record_dir = os.path.dirname(args.recording)
        recording  = os.path.basename(args.recording)

        bag = rosbag.Bag(os.path.join(record_dir, recording))

        poses = []
        message_t = []
        for topic, odom, t in bag.read_messages(topics=['/vins_estimator/odometry']):
            x = [odom.header.stamp.to_sec(),
                 odom.pose.pose.position.x,
                 odom.pose.pose.position.y,
                 odom.pose.pose.position.z]
            poses.append(x)
            message_t.append(t.to_sec())

        poses = np.array(poses)
        message_t = np.array(message_t)

        print('================================')
        print(1.0 / np.mean(np.gradient(message_t)))
        print(1.0 / np.mean(np.gradient(poses[:, 0])))


        with open(os.path.join(record_dir, 'results.pickle'), 'wb') as file:
            pickle.dump({'poses': np.array(poses)}, file, protocol=2)

        bag.close()
    else:
        BASE_RESULTS_PATH = 'results'
        bags = sorted(glob.glob('results/record_*/vins_mono/*.bag'))

        print(bags)

        total_poses = 0.0
        total_pose_time_span = 0.0
        total_message_time_span = 0.0
        for bag_name in bags:
            bag = rosbag.Bag(bag_name)

            poses = []
            message_t = []
            for topic, odom, t in bag.read_messages(topics=['/vins_estimator/odometry']):
                x = [odom.header.stamp.to_sec(),
                     odom.pose.pose.position.x,
                     odom.pose.pose.position.y,
                     odom.pose.pose.position.z]
                poses.append(x)
                message_t.append(t.to_sec())
            bag.close()

            poses = np.array(poses)
            message_t = np.array(message_t)

            total_poses += poses.shape[0]
            total_pose_time_span += poses[-1,0]-poses[0,0]
            total_message_time_span += message_t[-1] - message_t[0]

            print('================================')
            #print('Pose HZ'.format((poses[-1,0]-poses[0,0]) / poses.shape[0]))
            #print('Pose HZ'.format((poses[-1,0]-poses[0,0]) / poses.shape[0]))
            #print(1.0 / np.mean(np.gradient(message_t)))
            #print(1.0 / np.mean(np.gradient(poses[:, 0])))


        print('============ Overall ==============')
        print('Total pose messages {}'.format(total_poses))
        print('Total wall time of poses {}'.format(total_pose_time_span))
        print('Total time to generate poses {}'.format(total_message_time_span))
        print('Times realtime {}'.format(total_pose_time_span/total_message_time_span))
        #print(1.0 / np.mean(np.gradient(message_t)))
        #print(1.0 / np.mean(np.gradient(poses[:, 0])))