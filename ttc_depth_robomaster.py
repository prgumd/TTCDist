###############################################################################
#
# File: ttc_depth_robomaster.py
# Available under MIT license
#
# Run experiments with RoboMaster robot for TTCDist paper
#
# History:
# 09-02-22 - Levi Burner - Adapted file from ttc_depth_realsense.py
# 09-26-22 - Levi Burner - Open source release
#
###############################################################################

import argparse
import json
import glob
from multiprocessing import (Process,
                             Queue,
                             resource_tracker,
                             shared_memory,
                             Value)
import queue
import os
import pickle
import time
import traceback

import cv2
import numpy as np

from ttc_depth import TTCDepth

from zmq_vector_publisher import ZMQVectorPublisher

import robomaster
from robomaster import robot

BASE_RECORD_PATH = '../robomaster_recordings'

def remove_shm_from_resource_tracker():
    """Monkey-patch multiprocessing.resource_tracker so SharedMemory won't be tracked

    More details at: https://bugs.python.org/issue38119
    """

    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.register(self, name, rtype)
    resource_tracker.register = fix_register

    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.unregister(self, name, rtype)
    resource_tracker.unregister = fix_unregister

    if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
        del resource_tracker._CLEANUP_FUNCS["shared_memory"]

class RoboMasterIMUSource(object):
    def __init__(self, no_gyro, record_dir = None, b=1.0, efference=False):
        self._no_gyro = no_gyro
        self._record_dir = record_dir

        self._queue = Queue()
        self._pose_hat_queue = Queue()
        self._earliest_timestamp = Value('d', -1.0)
        self._earliest_timestamp_set = False
        self._latest_timestamp = Value('d', -1.0)
        self._exit_everything = Value('b', False)
        self._backup = []

        self._last_accel_sample = np.array((0.0, 0.0, 0.0), dtype=np.float32)
        self._last_gyro_sample = np.array((0.0, 0.0, 0.0), dtype=np.float32)

        self._accel_list = []
        self._gyro_list = []
        self._gyro_b = None

        self._u_list = []
        self._u_t_list = []
        self._int_u_list = []
        self._x_t_list = []
        self._x_hat_list = []

        self._E_dt = 1/50.0
        self._ts_hat = None
        self._time_filter_alpha = 0.01
        self._t0 = None

        self.int_u = np.zeros((3,))
        self.last_t = None
        self.last_pose = None

        self.first_pose_t = None
        self.stop = False

        if b is None:
            self._b = 1.0
        else:
            self._b = b
        self._efference = efference

    def add_pose_hat(self, t, pose_hat):
        self._pose_hat_queue.put((t, pose_hat))

    def get_latest_pose(self):
        sample = None
        while True:
            try:
                sample = self._pose_hat_queue.get_nowait()
            except queue.Empty:
                break
        return sample

    def add_sample(self, imu_sample):
        self._ts_hat = time.time()
        imu_sample = np.array(imu_sample)
        acc_x_r, acc_y_r, acc_z_r, gyro_x_r, gyro_y_r, gyro_z_r = imu_sample
        sample = np.array((self._ts_hat, acc_y_r, acc_z_r, acc_x_r, gyro_y_r, gyro_z_r, gyro_x_r))

        sample[1:4] *= 9.81 # G to m/s^2

        if self._gyro_b is not None:
            sample[4:7] -= self._gyro_b

        if self._no_gyro:
            sample[4:7] = 0

        self._accel_list.append(np.concatenate(((self._ts_hat,), sample[1:4])))
        self._gyro_list.append(np.concatenate(((self._ts_hat,), sample[4:7])))

        if len(self._gyro_list) < 100:
            return
        elif len(self._gyro_list) == 100:
            gyro_arr = np.array(self._gyro_list)
            self._gyro_b = np.mean(gyro_arr[:, 1:4], axis=0)

        if self._t0 is None:
            self._t0 = time.time()

        t_abs = time.time()
        t = t_abs - self._t0

        if self.last_t is None:
            self.last_t = t - (1.0 / 50.0)

        dt = t - self.last_t
        self.last_t = t

        l_over_d_w = (0.2 / 1.0)

        w_t = np.pi
        x_t = np.array([[0.0, 0.0,    -1.0*l_over_d_w / w_t * np.cos(w_t*t) - 5.0*l_over_d_w],
                        [0.0, 0.0,     1.0*l_over_d_w * np.sin(w_t*t)]])
        u_t = np.array( [0.0, 0.0, w_t*1.0*l_over_d_w * np.cos(w_t*t)])

        try:
            pose = self.get_latest_pose()

            if pose is None and self.last_pose is None:
                u = u_t
            elif pose is None and self.last_pose is not None:
                pose = (self.last_pose_t, self.last_pose)

            if pose is not None:
                pose_t, pose = pose

                if self.first_pose_t is None:
                    self.first_pose_t = pose_t

                if self.last_pose is None:
                    self.last_pose = pose
                dpose = (pose - self.last_pose) / dt
                self.last_pose = pose
                self.last_pose_t = pose_t
                hat_x = np.vstack((pose, dpose))
                self._x_hat_list.append(np.concatenate(((t_abs,), hat_x.flatten())))

                K = np.array([[2/1.0, 2/1.0]])

                if pose_t - self.first_pose_t > 5.0:
                    x_t = np.array([[0.0, 0.0, -2.5*l_over_d_w],
                                    [0.0, 0.0, 0.0]])
                    u_t = np.array([0.0, 0.0, 0.0])
                    e = x_t - hat_x
                    if False: #np.abs(e[0, 2]) < l_over_d_w / 100.0 and np.abs(e[1, 2]) < l_over_d_w / 100.0:
                        # self._ep_gripper.close(power=50)
                        self.stop = True
                        self.t_stop = time.time()

                        u_e = np.array([0.0, 0.0, 0.0])
                    else:
                        u_e = (K @ e).flatten()

                    if self.stop:
                        if time.time() - self.t_stop > 2.0:
                            self._exit_everything.value = True

                    u = u_t + u_e
                else:
                    u_e = (K @ (x_t - hat_x)).flatten()
                    u = u_t + u_e

            self.int_u += dt*u
            self._u_list.append(np.concatenate(((t_abs,), u)))
            self._int_u_list.append(np.concatenate(((t_abs,), self.int_u)))
        except Exception as e:
            print(traceback.format_exc())

        self._x_t_list.append(np.concatenate(((t_abs,), x_t.flatten())))
        self._u_t_list.append(np.concatenate(((t_abs,), u_t)))

        try:
            if not self.stop:
                self._ep_chassis.drive_speed(x=self._b*self.int_u[2], y=self._b*self.int_u[0], timeout=0.5)
                a_efference = np.array([u[0], 0.0, u[2]])
            else:
                self._ep_chassis.drive_speed(x=0, y=0, timeout=0.5)
                a_efference = np.array([0, 0.0, 0])
        except Exception as e:
            print(traceback.format_exc())
        # print(sample[1:4], a_efference)

        # Overwrite IMU measurements with efference copy
        if self._efference:
            sample[1:4] = a_efference


        self._latest_timestamp.value = sample[0]
        self._backup.append(sample)

        if not self._earliest_timestamp_set:
            self._earliest_timestamp.value = sample[0]
            self._earliest_timestamp_set = True

        # Fill the multiprocess queue, TODO is this needed?
        while len(self._backup) > 0:
            try:
                self._queue.put_nowait(self._backup[0])
                self._backup.pop(0)
            except queue.Full:
                break

    def earliest_timestamp(self):
        return float(self._earliest_timestamp.value)

    def latest_timestamp(self):
        return float(self._latest_timestamp.value)

    def next_sample(self):
        try:
            sample = self._queue.get_nowait()
        except queue.Empty:
            sample = None
        return sample

    def save_data(self):
        file_name = os.path.join(self._record_dir, 'imu.pickle')

        with open(file_name, 'wb') as file:
            pickle.dump({'accel': np.array(self._accel_list),
                         'gyro':  np.array(self._gyro_list),
                         'u': np.array(self._u_list),
                         'u_t': np.array(self._u_t_list),
                         'int_u': np.array(self._int_u_list),
                         'x_t': np.array(self._x_t_list),
                         'x_hat': np.array(self._x_hat_list),
                        }, file, protocol=2)


def preprocess_image(frame, frame_gray):
    frame_gray_uint8 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tmp = frame_gray_uint8.astype(np.float32) * (1.0 / 255.0)
    frame_gray[...] = cv2.remap(tmp, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

class RoboMasterFrameSource(Process):
    def __init__(self, imu_source, record_dir=None):
        super(RoboMasterFrameSource, self).__init__()
        self._queue = Queue()
        self._stop_process = Value('b', False)
        self._earliest_timestamp = Value('d', -1.0)
        self._earliest_timestamp_set = False
        self._latest_timestamp = Value('d', -1.0)
        self._width = WIDTH
        self._height = HEIGHT
        self._fps = FPS
        self._num_frames = 0

        self._template_buffer = np.zeros((self._height, self._width), dtype=np.float32)

        self._shared_memories = []

        self._imu_source = imu_source
        self._record_dir = record_dir

        self._frame_times = []

        self._E_dt = 1/FPS
        self._ts_hat = None
        self._time_filter_alpha = 0.01

        # self._last_receive_ts = time.time()

        cv2.setNumThreads(1)

    def run(self):
        remove_shm_from_resource_tracker()

        ep_robot = robot.Robot()
        ep_robot.initialize(conn_type="sta")

        # ep_robot.gripper.open(power=50)
        ep_robot.robotic_arm.moveto(x=200, y=40).wait_for_completed()


        self._imu_source._ep_chassis = ep_robot.chassis
        self._imu_source._ep_gripper = ep_robot.gripper

        ep_robot.chassis.sub_imu(freq=50, callback=self._imu_source.add_sample)

        ep_camera = ep_robot.camera
        ep_camera.start_video_stream(display=False)

        try:
            while not self._stop_process.value:
                frame = ep_camera.read_cv2_image(strategy='pipeline')
                receive_ts = time.time()

                # delta = receive_ts - self._last_receive_ts
                # print(1.0 / delta)
                # self._last_receive_ts = receive_ts

                # if self._ts_hat is None:
                #     self._ts_hat = receive_ts
                # else:
                #     self._ts_hat += self._E_dt
                #     self._ts_hat = (1-self._time_filter_alpha) * self._ts_hat + self._time_filter_alpha * receive_ts
                self._ts_hat = receive_ts - 0.1 #+ (-0.0242) # TODO NO!

                self._process_frame(self._ts_hat, frame)
        except KeyboardInterrupt:
            print('RoboMasterFrameSource KeyboardInterruptcaught')
        finally:
            ep_robot.chassis.drive_speed(x=0, y=0, z=0, timeout=5)
            ep_camera.stop_video_stream()
            print('sleeping for robot to stop')
            time.sleep(2)
            ep_robot.close()

        if self._record_dir:
            print('RoboMasterFrameSource saving')
            file_name = os.path.join(self._record_dir, 'frame_metadata.pickle')
            with open(file_name, 'wb') as file:
                pickle.dump({'ts': np.array(self._frame_times)}, file, protocol=2)
            self._imu_source.save_data()

    def _process_frame(self, ts, frame):
        self._frame_times.append(ts)

        shm = shared_memory.SharedMemory(create=True, size=self._template_buffer.nbytes)
        frame_gray = np.ndarray(self._template_buffer.shape,
                                dtype=self._template_buffer.dtype,
                                buffer=shm.buf)
        preprocess_image(frame, frame_gray)

        self._queue.put((ts, shm))
        shm.close()

        if not self._earliest_timestamp_set:
            self._earliest_timestamp.value = ts
            self._earliest_timestamp_set = True
        self._latest_timestamp.value = ts

        if self._record_dir:
            image_dir = os.path.join(record_dir, 'images')
            frame_name = os.path.join(image_dir, 'frame_{:06d}.npy'.format(self._num_frames))
            with open(frame_name, 'wb') as f:
                np.save(f, frame)

        self._num_frames += 1

    def earliest_timestamp(self):
        return float(self._earliest_timestamp.value)

    def latest_timestamp(self):
        return float(self._latest_timestamp.value)

    def next_sample(self):
        try:
            (ts, shm) = self._queue.get_nowait()
            frame_gray = np.ndarray(self._template_buffer.shape,
                                    dtype=self._template_buffer.dtype,
                                    buffer=shm.buf)
            sample = (ts, frame_gray)
            self._shared_memories.append(shm)
            return sample
        except queue.Empty:
            sample = None
        return sample

    def free_sample(self):
        if len(self._shared_memories) > 0:
            shm = self._shared_memories.pop(0)
            shm.close()
            shm.unlink()

    def signal_stop(self):
        self._stop_process.value = True

class RobomasterTemplateSource(object):
    def __init__(self, patch_dim=50, resettable=True, record_dir=None):
        self._resettable = resettable
        self._record_dir = record_dir
        self._patch_params = {
            'patch_start_time': time.time() + 2.5,
            'patch_end_time': time.time() + 10000.0,
            'patch_coordinates': (int(5.0*WIDTH/8 - patch_dim),
                                  int(2.0*HEIGHT/4 - patch_dim),
                                  int(5.0*WIDTH/8 + patch_dim),
                                  int(2.0*HEIGHT/4 + patch_dim))
        }
        self._patches = []
        self._create_patch(self._patch_params['patch_start_time'])

        self._patch_index = -1

        self._exit = False

        self._force_new_patch = False

    def force_new_patch(self):
        self._force_new_patch = True

    def _create_patch(self, time):
        self._patches.append((time,
                              time + 100000.0,
                              self._patch_params['patch_coordinates']))

    def current_patch_valid(self, time):
        if True: #self._resettable:
            key = cv2.pollKey()
            #if key != -1:
            #    print('key {}'.format(key))
            if key == 114 or self._force_new_patch: # r key
                self._create_patch(time)
                self._force_new_patch = False
                return False
            elif key == 116: # t key
                return 'reset observer'
            elif key == 113: # q key
                self._exit = True

        if self._patch_index < 0:
            raise Exception('Current patch valid called before patch selected')
        return time < self._patches[self._patch_index][1]

    def get_new_patch(self, time):
        next_patch_index = self._patch_index + 1

        if next_patch_index >= len(self._patches):
            return None

        if time >= self._patches[next_patch_index][0]:
            self._patch_index = next_patch_index
            return self._patches[next_patch_index][2]

    def save_data(self):
        file_name = os.path.join(self._record_dir, 'templates_live.pickle')
        with open(file_name, 'wb') as file:
            pickle.dump({'patches': self._patches}, file, protocol=2)

def make_record_dir():
    previous_directories = sorted(glob.glob(os.path.join(BASE_RECORD_PATH, 'record_*')))
    if len(previous_directories) == 0:
        next_recording_number = 0
    else:
        next_recording_number = int(previous_directories[-1][-6:]) + 1

    record_dir = os.path.join(BASE_RECORD_PATH, 'record_{:06d}'.format(next_recording_number))
    images_dir = os.path.join(record_dir, 'images')

    os.mkdir(record_dir)
    os.mkdir(images_dir)
    return record_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize', dest='visualize', action='store_true', help='Visualize')
    parser.add_argument('--wait', dest='wait', action='store_true', help='Wait for key press when visualizing')
    parser.add_argument('--nogyro', dest='no_gyro', action='store_true', help='Disable gyro stabilization')
    parser.add_argument('--april', dest='april', action='store_true', help='Use apriltag for ground truth')
    parser.add_argument('--record', dest='record', action='store_true', help='Record data and events')
    parser.add_argument('--b', type=float, help='scalar gain b to apply to control effort')
    parser.add_argument('--efference', dest='efference', action='store_true', help='Report efference copy instead of IMU')

    args = parser.parse_args()

    if args.wait:
        wait_key = 0
    else:
        wait_key = 1

    WIDTH = 1280
    HEIGHT = 720
    FPS = 30
    patch_dim = 50
    strided_patch_size = 4000 # Track 4000 pixels
    max_flow_time = 0.5 * (1.0/FPS) # Works well enough

    if args.record:
        record_dir = make_record_dir()
    else:
        record_dir = None

    # From running calibrate_calculate.py
    K = np.array([[631.05345607,   0.,         646.8600196 ],
                  [  0.,         633.5803277,  357.86951071],
                  [  0.,           0.,           1.        ]])
    D = np.array([0.1726052, 0.43400192, -0.43320789, 0.04646433])

    resolution = (WIDTH, HEIGHT)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, resolution, cv2.CV_16SC2)

    if record_dir:
        file_name = os.path.join(record_dir, 'intrinsics.pickle')
        with open(file_name, 'wb') as file:
            pickle.dump({'K': K,
                         'D': D,
                         'resolution': resolution}, file, protocol=2)


    imu_source = RoboMasterIMUSource(no_gyro = args.no_gyro, record_dir=record_dir,
                                     b=args.b, efference=args.efference)
    phi_pose_subscriber = imu_source

    frame_source = RoboMasterFrameSource(imu_source, record_dir=record_dir)
    template_source = RobomasterTemplateSource(patch_dim=patch_dim,
                                               resettable = True,
                                               record_dir = record_dir)

    vector_pub = ZMQVectorPublisher()

    ttc_depth = TTCDepth(frame_source, imu_source, template_source, K,
                         visualize=args.visualize,
                         wait_key=wait_key,
                         max_flow_time=max_flow_time,
                         strided_patch_size=strided_patch_size,
                         april_ground_truth=args.april,
                         april_resize_to=None, #(270, 480),
                         vector_pub = vector_pub,
                         phi_pose_subscriber=phi_pose_subscriber,
                         phi_accel_power_thresh=0.001)

    cv2.setNumThreads(1)
    try:
        frame_source.start()

        start_collection_time = time.time()

        while not template_source._exit and not imu_source._exit_everything.value:
            #print(time.time() - start_time)
            imu_start_compute_time = ttc_depth._imu_time_computed_to
            flow_start_compute_time = ttc_depth._flow_time_computed_to
            observer_start_compute_time = ttc_depth._ttc_pose_observer_time_computed_to
            start_time = time.time()
            ttc_depth.update()
            end_time = time.time()
            imu_end_compute_time = ttc_depth._imu_time_computed_to
            flow_end_compute_time = ttc_depth._flow_time_computed_to
            observer_end_compute_time = ttc_depth._ttc_pose_observer_time_computed_to

            if imu_start_compute_time is not None:
                real_delta = end_time-start_time
                imu_delta = imu_end_compute_time - imu_start_compute_time
                flow_delta = flow_end_compute_time - flow_start_compute_time
                observer_delta = observer_end_compute_time - observer_start_compute_time

                if imu_delta > 0:
                    # TODO this compares system time with hardware timestamps, it doens't make sense
                    imu_lag = end_time - imu_end_compute_time
                    flow_lag = end_time - flow_end_compute_time
                    observer_lag = end_time - observer_end_compute_time

                    if (observer_lag > 0.25):
                        template_source.force_new_patch()

                    # print('real {:.03f} imu {:.03f} ratio {:.03f} imu lag {:.03f} flow lag {:.03f} obs lag {:.03f}'.format(
                    #       real_delta, imu_delta, imu_delta/real_delta, imu_lag, flow_lag, observer_lag))
                    #print('{:.02f}'.format(ttc_depth._ttc_list[-1][1]))
                    # print('{:.01f}'.format(ttc_depth._z_hat_list[-1][1]))

            if time.time() - start_collection_time > 10000.0:
                print('Ran for 10000 seconds, exiting')
                frame_source.signal_stop()

                print('Join frame source1')
                frame_source.join()
                break

            if imu_start_compute_time is None or observer_delta < 0.1:
                update_delta = end_time - start_time
                # Rate limit
                sleep_delta = (1.0/1000.0) - update_delta
                if sleep_delta > 0:
                     time.sleep(sleep_delta)

    except KeyboardInterrupt:
        print('Main process KeyboardInterrupt caught')

    finally:
        print('Attempting to clean up data sources')
        frame_source.signal_stop()
        frame_source.join()

    if args.record:
        template_source.save_data()

    print('Success, ignore the resource tracker')
