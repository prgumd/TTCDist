###############################################################################
#
# File: ros_publish_folder_recording.py
#
# Publish a recording to ROS
#
# History:
# 11-08-20 - Levi Burner - Created file
#
###############################################################################

import argparse
import json
import os
import glob
import pickle
import time

import numpy as np
from scipy import interpolate

from utilities import latest_recording_dir

import rospy
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge, CvBridgeError

class RecordedIMUSource(object):
    def __init__(self, record_dir):
        sensor_data = pickle.load(open(os.path.join(record_dir, 'imu.pickle'), 'rb'))
        self._accel_samples = sensor_data['accel']
        self._gyro_samples = sensor_data['gyro']
        self._sample_index = 0

        # TODO resampling like this should occur live as well
        interpolater = interpolate.interp1d(self._accel_samples[:, 0],
                                            self._accel_samples[:, 1:4],
                                            axis=0,
                                            bounds_error=False,
                                            fill_value='extrapolate')
        accel_samples_interpolated = interpolater(self._gyro_samples[:, 0])

        self._imu_samples = np.hstack((np.atleast_2d(self._gyro_samples[:, 0]).transpose(),
                                       accel_samples_interpolated,
                                       self._gyro_samples[:, 1:4]))

    def earliest_timestamp(self):
        if self._sample_index >= len(self._imu_samples):
            return None
        return self._imu_samples[0, 0]

    def latest_timestamp(self):
        if self._sample_index >= len(self._imu_samples):
            return None
        return self._imu_samples[-1, 0]

    def next_sample(self):
        if self._sample_index >= len(self._imu_samples):
            return None

        sample = self._imu_samples[self._sample_index]
        self._sample_index += 1
        return sample

def preprocess_image(frame):
    return frame

class RecordedFrameSource(object):
    def __init__(self, record_dir, preload=False, preprocess=False):
        self._preload = preload
        self._preprocess = preprocess

        self._frame_names = sorted(list(glob.glob(os.path.join(record_dir, 'images/frame_*.npy'))))

        frame_metadata = pickle.load(open(os.path.join(record_dir, 'frame_metadata.pickle'), 'rb'))
        self._frame_ts = frame_metadata['ts']

        self._sample_index = 0

        if self._preload:
            if self._preprocess:
                self._preloaded_frames = [preprocess_image(np.load(frame_name)) for frame_name in self._frame_names]
            else:
                self._preloaded_frames = [np.load(frame_name) for frame_name in self._frame_names]

    def earliest_timestamp(self):
        return self._frame_ts[0]

    def latest_timestamp(self):
        if self._sample_index >= len(self._frame_ts):
            return None
        return self._frame_ts[-1]

    def next_sample(self):
        if self._sample_index >= len(self._frame_ts):
            return None

        if not self._preload:
            frame = np.load(self._frame_names[self._sample_index])
            frame_gray = preprocess_image(frame)
            sample = (self._frame_ts[self._sample_index], frame_gray)
        else:
            if not self._preprocess:
                sample = (self._frame_ts[self._sample_index], preprocess_image(self._preloaded_frames[self._sample_index]))
            else:
                sample = (self._frame_ts[self._sample_index], self._preloaded_frames[self._sample_index])

        self._sample_index += 1
        return sample

    def free_sample(self):
        pass

class ROSFrameIMUPublisher(object):
    def __init__ (self, frame_source, imu_source):
        self._frame_source = frame_source
        self._imu_source = imu_source

        self._image_pub = rospy.Publisher('frames', Image, queue_size=10)
        self._imu_pub = rospy.Publisher('imu', Imu, queue_size=10)

        frame_first_timestamp = self._frame_source.earliest_timestamp()
        imu_first_timestamp   = self._imu_source.earliest_timestamp()
        self._first_timestamp = min(frame_first_timestamp, imu_first_timestamp)


        frame_last_timestamp = self._frame_source.latest_timestamp()
        imu_last_timestamp   = self._imu_source.latest_timestamp()
        self._last_timestamp = max(frame_last_timestamp, imu_last_timestamp)

        self._first_ros_time  = rospy.get_rostime().to_sec()
        self._time_delta = self._first_ros_time - self._first_timestamp
        self._rate_limiter = rospy.Rate(1000)

        self._next_frame = None
        self._next_imu = None

    def run(self):
        while not rospy.is_shutdown():
            now = rospy.get_rostime().to_sec()
            sensor_time = now - self._time_delta

            if sensor_time > self._last_timestamp:
                break

            if self._next_frame is None:
                self._next_frame = self._frame_source.next_sample()

            if self._next_frame is not None:
                if self._next_frame[0] < sensor_time:
                    image_msg = CvBridge().cv2_to_imgmsg(self._next_frame[1])
                    image_msg.header.stamp = rospy.Time.from_sec(self._next_frame[0])
                    self._image_pub.publish(image_msg)

                    self._next_frame = None

            if self._next_imu is None:
                self._next_imu = self._imu_source.next_sample()

            if self._next_imu is not None:
                if self._next_imu[0] < sensor_time:
                    imu_msg = Imu()
                    imu_msg.header.stamp = rospy.Time.from_sec(self._next_imu[0])
                    imu_msg.angular_velocity.x = self._next_imu[4]
                    imu_msg.angular_velocity.y = self._next_imu[5]
                    imu_msg.angular_velocity.z = self._next_imu[6]
                    imu_msg.linear_acceleration.x = self._next_imu[1]
                    imu_msg.linear_acceleration.y = self._next_imu[2]
                    imu_msg.linear_acceleration.z = self._next_imu[3]

                    self._imu_pub.publish(imu_msg)

                    self._next_imu = None

            self._rate_limiter.sleep()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', dest='dir', type=str, help='Directory to load data from. If not specified the latest in the default location is used')
    parser.add_argument('--visualize', dest='visualize', action='store_true', help='Visualize')
    parser.add_argument('--wait', dest='wait', action='store_true', help='Wait for key press when visualizing')
    parser.add_argument('--april', dest='april', action='store_true', help='Use apriltag for ground truth')
    parser.add_argument('--preload', dest='preload', action='store_true', help='Read all images before processing for benchmarking')
    parser.add_argument('--preprocess', dest='preprocess', action='store_true', help='Preprocess all images when preloading')
    parser.add_argument('--nopublish', dest='nopublish', action='store_true', help='Dont publish with ZMQ')
    parser.add_argument('--bench', dest='bench', action='store_true', help='Activate preload nopublish and some other things')
    parser.add_argument('--realtime', dest='realtime', action='store_true', help='Test ability to run in realtime with april feedback')

    args = parser.parse_args()

    if not args.dir:
        record_dir = latest_recording_dir()
    else:
        record_dir = args.dir

    rospy.init_node('ttc_recording_pub')

    frame_source = RecordedFrameSource(record_dir, preload=args.preload, preprocess=args.preprocess)
    imu_source = RecordedIMUSource(record_dir)
    ros_publisher = ROSFrameIMUPublisher(frame_source, imu_source)

    ros_publisher.run()
