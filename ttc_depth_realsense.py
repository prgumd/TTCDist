###############################################################################
#
# File: ttc_depth_realsense.py
# Available under MIT license
#
# Estimate distance in realtime with Phi, TTC, and AprilTag using a Realsense D435i camera
# Supports saving the data to disk for later evaluation with ttc_depth_from_folder.py
# and ttc_depth_calc_error.py
#
# History:
# 07-22-21 - Levi Burner - Adapted file from ttc_depth_nx.py
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

# If pyrealsense2 is able to be installed with 
# pip install pyrealsense2
# then the first import metho works
# but if installed from source (as needed for Ubuntu 22.04 and on)
# the second import is needed
try:
    import pyrealsense2 as rs
    rs.__version__
except AttributeError as e:
    from pyrealsense2 import pyrealsense2 as rs

from ttc_depth import TTCDepth

from zmq_vector_publisher import ZMQVectorPublisher

BASE_RECORD_PATH = '../recordings'

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

class RealsenseIMUSource(object):
    def __init__(self, no_gyro, record_dir = None):
        self._no_gyro = no_gyro
        self._record_dir = record_dir

        self._queue = Queue()
        self._stop_process = Value('b', False)
        self._earliest_timestamp = Value('d', -1.0)
        self._earliest_timestamp_set = False
        self._latest_timestamp = Value('d', -1.0)
        self._backup = []

        self._last_accel_sample = np.array((0.0, 0.0, 0.0), dtype=np.float32)
        self._last_gyro_sample = np.array((0.0, 0.0, 0.0), dtype=np.float32)

        self._accel_list = []
        self._gyro_list = []

    def _add_sample(self, sample):
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

    def add_accel_sample(self, accel):
        ts = accel.timestamp / 1000.0

        data = accel.get_motion_data()
        
        # Transform the data from the realsense frame to the camera is identity
        accel_c = np.array((data.x, data.y, data.z), dtype=np.float32)

        self._last_accel_sample = accel_c
        sample = np.concatenate(((ts,), accel_c, self._last_gyro_sample))

        self._accel_list.append(np.concatenate(((ts,), accel_c)).tolist())

        self._add_sample(sample)

    def add_gyro_sample(self, gyro):
        if not self._no_gyro:
            ts = gyro.timestamp / 1000.0

            data = gyro.get_motion_data()
            gyro = np.array((data.x, data.y, data.z), dtype=np.float32)

            self._last_gyro_sample = gyro
            sample = np.concatenate(((ts,), self._last_accel_sample, gyro))

            self._gyro_list.append(np.concatenate(((ts,), gyro)).tolist())

            self._add_sample(sample)

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

    def signal_stop(self):
        self._stop_process.value = True

    def save_data(self):
        file_name = os.path.join(self._record_dir, 'imu.pickle')

        with open(file_name, 'wb') as file:
            pickle.dump({'accel': np.array(self._accel_list),
                         'gyro':  np.array(self._gyro_list)}, file, protocol=2)

def preprocess_image(frame, frame_gray):
    frame_gray[...] = frame.astype(np.float32) * (1.0 / 255.0)
    # = tmp # cv2.remap(tmp, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

class RealsenseFrameSource(Process):
    def __init__(self, realsense_imu_source, record_dir=None):
        super(RealsenseFrameSource, self).__init__()
        self._queue = Queue()
        self._stop_process = Value('b', False)
        self._earliest_timestamp = Value('d', -1.0)
        self._earliest_timestamp_set = False
        self._latest_timestamp = Value('d', -1.0)
        self._width = WIDTH
        self._height = HEIGHT
        self._fps = FPS
        self._imager = 1
        self._num_frames = 0

        self._accel_hz = 250
        self._gyro_hz = 400

        self._template_buffer = np.zeros((self._height, self._width), dtype=np.float32)

        self._shared_memories = []

        self._realsense_imu_source = realsense_imu_source
        self._record_dir = record_dir

        self._frame_times = []

        cv2.setNumThreads(1)

    def run(self):
        remove_shm_from_resource_tracker()

        pipeline = rs.pipeline()

        config = rs.config()
        config.enable_stream(rs.stream.infrared, self._imager, self._width, self._height, rs.format.y8, self._fps)
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, self._accel_hz)
        config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, self._gyro_hz)

        queue = rs.frame_queue(200)
        self._pipeline_profile = pipeline.start(config, queue)

        # https://community.intel.com/t5/Items-with-no-label/How-to-enable-disable-emitter-through-python-wrapper/td-p/547900
        device = self._pipeline_profile.get_device()
        depth_sensor = device.query_sensors()[0] # TODO NO!!!!
        depth_sensor.set_option(rs.option.emitter_enabled, 0)

        try:
            while not self._stop_process.value:
                frame = queue.wait_for_frame()
                if frame:
                    if frame.is_frameset():
                        self._process_frameset(frame.as_frameset())
                    elif frame.is_motion_frame():
                        profile = frame.get_profile()
                        if profile.stream_type() == rs.stream.accel:
                            self._process_accel(frame.as_motion_frame())
                        elif profile.stream_type() == rs.stream.gyro:
                            self._process_gyro(frame.as_motion_frame())
                else:
                    time.sleep(0.001)
        finally:
            pipeline.stop()

        if self._record_dir:
            self._realsense_imu_source.save_data()

            file_name = os.path.join(self._record_dir, 'frame_metadata.pickle')
            with open(file_name, 'wb') as file:
                pickle.dump({'ts': np.array(self._frame_times)}, file, protocol=2)

    def _process_frameset(self, frameset):
        ts = frameset.timestamp / 1000.0
        self._frame_times.append(ts)

        frame = np.asanyarray(frameset.get_infrared_frame().get_data())

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

    def _process_accel(self, motion_frame):
        self._realsense_imu_source.add_accel_sample(motion_frame)

    def _process_gyro(self, motion_frame):
        self._realsense_imu_source.add_gyro_sample(motion_frame)

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

class RealsenseTemplateSource(object):
    def __init__(self, patch_dim=50, resettable=True, record_dir=None):
        self._resettable = resettable
        self._record_dir = record_dir
        self._patch_params = {
            'patch_start_time': time.time() + 2.5,
            'patch_end_time': time.time() + 10000.0,
            'patch_coordinates': (int(WIDTH/2 - patch_dim), int(HEIGHT/2 - patch_dim), int(WIDTH/2 + patch_dim), int(HEIGHT/2 + patch_dim))
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

# Hacky way to get the realsense intrinsics from the camera
def get_realsense_intrinsics():
    pipeline = rs.pipeline()

    config = rs.config()
    config.enable_stream(rs.stream.infrared, 1, WIDTH, HEIGHT, rs.format.y8, FPS)
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 400)

    queue = rs.frame_queue(200)
    pipeline_profile = pipeline.start(config, queue)
    pipeline.stop()

    # TODO Shouldn't the imager be specified?
    profile = pipeline_profile.get_stream(rs.stream.infrared)
    intrinsics = profile.as_video_stream_profile().get_intrinsics()
    fx = intrinsics.fx
    fy = intrinsics.fy
    ppx = intrinsics.ppx
    ppy = intrinsics.ppy

    dist_coeffs = intrinsics.coeffs # TODO use these, they are all zero right now

    K = np.array(((fx,  0, ppx),
                  (0,  fy, ppy),
                  (0,  0,  1)),
                 dtype=np.float32)
    D = np.array((0.0, 0.0, 0.0, 0.0), dtype=np.float32)
    resolution = (WIDTH, HEIGHT)

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, resolution, cv2.CV_16SC2)

    accel_profile = pipeline_profile.get_stream(rs.stream.accel)
    gyro_profile  = pipeline_profile.get_stream(rs.stream.gyro)

    accel_extrinsics = accel_profile.get_extrinsics_to(profile)
    gyro_extrinsics  = gyro_profile.get_extrinsics_to(profile)

    print('accel extrinsics {}'.format(accel_extrinsics))
    print('gyro exstrinsics {}'.format(gyro_extrinsics))

    return K, D, map1, map2, resolution

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
    parser.add_argument('--high', dest='high', action='store_true', help='Highest fps possible with limited fov')
    parser.add_argument('--nogyro', dest='no_gyro', action='store_true', help='Disable gyro stabilization')
    parser.add_argument('--april', dest='april', action='store_true', help='Use apriltag for ground truth')
    parser.add_argument('--record', dest='record', action='store_true', help='Record data and events')

    args = parser.parse_args()

    if args.wait:
        wait_key = 0
    else:
        wait_key = 1

    if args.high:
        # Highest fps super narrow vertical fov
        WIDTH = 848
        HEIGHT = 100
        FPS = 300
        patch_dim = HEIGHT / 4
        strided_patch_size = patch_dim**2
        max_flow_time = 0.5 * (1.0/FPS) # Works well enough
    else:
        # Highest fps with standard fov
        WIDTH = 848
        HEIGHT = 480
        FPS = 90
        patch_dim = 50
        strided_patch_size = 4000 # Track 400 pixels
        max_flow_time = 0.5 * (1.0/FPS) # Works well enough


    if args.record:
        record_dir = make_record_dir()
    else:
        record_dir = None

    K, D, map1, map2, resolution = get_realsense_intrinsics()

    if record_dir:
        file_name = os.path.join(record_dir, 'intrinsics.pickle')
        with open(file_name, 'wb') as file:
            pickle.dump({'K': K,
                         'D': D,
                         'resolution': resolution}, file, protocol=2)

    imu_source = RealsenseIMUSource(no_gyro = args.no_gyro, record_dir=record_dir)
    frame_source = RealsenseFrameSource(imu_source, record_dir=record_dir)
    template_source = RealsenseTemplateSource(patch_dim=patch_dim,
                                              resettable = not args.high,
                                              record_dir = record_dir)

    vector_pub = ZMQVectorPublisher()

    ttc_depth = TTCDepth(frame_source, imu_source, template_source, K,
                         visualize=args.visualize,
                         wait_key=wait_key,
                         max_flow_time=max_flow_time,
                         strided_patch_size=strided_patch_size,
                         april_ground_truth=args.april,
                         april_resize_to=None, #(270, 480),
                         vector_pub = vector_pub)

    cv2.setNumThreads(1)
    try:
        frame_source.start()

        start_collection_time = time.time()

        while not template_source._exit:
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

                    print('real {:.03f} imu {:.03f} ratio {:.03f} imu lag {:.03f} flow lag {:.03f} obs lag {:.03f}'.format(
                          real_delta, imu_delta, imu_delta/real_delta, imu_lag, flow_lag, observer_lag))
                    #print('{:.02f}'.format(ttc_depth._ttc_list[-1][1]))
                    #print('{:.01f}'.format(ttc_depth._z_hat_list[-1][1]))

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

    except Exception as e:
        print('Exception caught')
        print(traceback.format_exc())

    finally:
        print('Attempting to clean up data sources')
        imu_source.signal_stop()
        frame_source.signal_stop()
        frame_source.join()

    if args.record:
        template_source.save_data()

    print('Success, ignore the resource tracker')
