###############################################################################
#
# File: ttc_depth_from_folder.py
# Available under MIT license
#
# Run Phi and TTC distance estimation using recordings made with ttc_depth_realsense.py
#
# History:
# 04-21-20 - Levi Burner - Adapted file from early prototype code
# 09-26-22 - Levi Burner - Open source release
#
###############################################################################

import argparse
import json
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' # Attempt to disable OpenBLAS multithreading, it makes the script slower
import glob
import pickle
import json
import time

import cv2
import numpy as np
from scipy import interpolate

from ttc_depth import TTCDepth
from utilities import (latest_recording_dir,
                       load_calibration_from_dir)
from zmq_vector_publisher import ZMQVectorPublisher, ZMQVectorPublisherSaver
from shutil import copyfile

class RecordedIMUSource(object):
    def __init__(self, record_dir, wy_bias = 0.0, wz_bias = 0.0):
        self._wy_bias = wy_bias
        self._wz_bias = wz_bias

        sensor_data = pickle.load(open(os.path.join(record_dir, 'imu.pickle'), 'rb'))
        self._accel_samples = sensor_data['accel']
        self._gyro_samples = sensor_data['gyro']

        self._gyro_samples[:, 2] += self._wy_bias * np.pi / 180.0
        self._gyro_samples[:, 3] += self._wz_bias * np.pi / 180.0

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
    frame_gray = frame.astype(np.float32) * (1.0 / 255.0)
    return frame_gray

class RecordedFrameSource(object):
    def __init__(self,
                 record_dir,
                 preload=False,
                 preprocess=False,
                 frame_skip=0):
        self._preload = preload
        self._preprocess = preprocess
        self._frame_skip = frame_skip

        self._frame_names_orig = sorted(list(glob.glob(os.path.join(record_dir, 'images/frame_*.npy'))))

        self._frame_names = self._frame_names_orig[::(frame_skip+1)]

        frame_metadata = pickle.load(open(os.path.join(record_dir, 'frame_metadata.pickle'), 'rb'))
        self._frame_ts = frame_metadata['ts'][::(frame_skip + 1)]

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

class RecordedTemplateSource(object):
    def __init__(self, record_dir):
        templates = pickle.load(open(os.path.join(record_dir, 'templates_live.pickle'), 'rb'))
        self._patches = templates['patches']

        # WIDTH = 848
        # HEIGHT = 480
        # patch_dim = 50

        # self._patch_params = {
        #     'patch_start_time': time.time() + 2.5,
        #     'patch_end_time': time.time() + 10000.0,
        #     'patch_coordinates': (int(WIDTH/2 - patch_dim), int(HEIGHT/2 - patch_dim), int(WIDTH/2 + patch_dim), int(HEIGHT/2 + patch_dim))
        # }

        self._patch_index = -1

    # def _create_patch(self, time):
    #     self._patches.append((time,
    #                           time + 100000.0,
    #                           self._patch_params['patch_coordinates']))

    def current_patch_valid(self, time):
        # TODO allow creating patches on the fly
        # if True: #self._resettable:
        #     key = cv2.pollKey()
        #     #if key != -1:
        #     #    print('key {}'.format(key))
        #     if key == 114: # r key
        #         self._create_patch(time)
        #         return False
        #     elif key == 116: # t key
        #         return 'reset observer'
        #     elif key == 113: # q key
        #         self._exit = True

        if self._patch_index < 0:
            raise Exception('Current patch valid called before patch selected')

        valid = True

        if time > self._patches[self._patch_index][1]:
            valid = False

        if self._patch_index + 1 < len(self._patches):
            if time > self._patches[self._patch_index + 1][0]:
                valid = False

        return valid

    def get_new_patch(self, time):
        next_patch_index = self._patch_index + 1

        if next_patch_index >= len(self._patches):
            return None

        if time >= self._patches[next_patch_index][0]:
            self._patch_index = next_patch_index
            return self._patches[next_patch_index][2]

def make_results_dir(recording_name, ttc_append='', vicon=False):
    if not os.path.isdir('results'):
        os.mkdir('results')

    results_dir = os.path.join('results', recording_name)
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    results_dir_ttc = os.path.join(results_dir, 'ttc'+ttc_append)
    if not os.path.isdir(results_dir_ttc):
        os.mkdir(results_dir_ttc)

    visualization_dir = os.path.join(results_dir, 'visualization')
    if not os.path.isdir(visualization_dir):
        os.mkdir(visualization_dir)

    if vicon:
        results_dir_vicon = os.path.join(results_dir, 'vicon')
        if not os.path.isdir(results_dir_vicon):
            os.mkdir(results_dir_vicon)


    return results_dir_ttc, visualization_dir

def vicon_file_name(record_dir, recording_name):
    record_number = int(recording_name[-6:])

    if record_number == 0:
        vicon_file_number_str = ''
    else:
        vicon_file_number_str = ' {}'.format(record_number)

    vicon_file_name = os.path.join(record_dir, '../vicon/d435i_ttc{}.csv'.format(vicon_file_number_str))
    return vicon_file_name

def run_on_directory(record_dir, args, max_visualize_rate, save_visualization, settings_name=''):
    K, D, map1, map2, resolution = load_calibration_from_dir(record_dir)

    params_file_name = os.path.join(record_dir, 'params.json')
    if os.path.exists(params_file_name):
        params = json.load(open(params_file_name))
    else:
        params = {'trim_end': 0.0}

    if args.save or save_visualization:
        recording_name = os.path.split(record_dir)[-1]

        vicon_file = vicon_file_name(record_dir, recording_name)
        if os.path.exists(vicon_file):
            results_dir, visualization_dir = make_results_dir(recording_name, ttc_append=settings_name, vicon=True)
            vicon_file_dest = os.path.join('results', recording_name, 'vicon', 'results.csv')
            copyfile(vicon_file, vicon_file_dest)
        else:
            results_dir, visualization_dir = make_results_dir(recording_name, vicon=False)

        vector_pub = ZMQVectorPublisherSaver()
    else:
        vector_pub = ZMQVectorPublisher()
        visualization_dir = None

    if args.nopublish:
        vector_pub = None

    frame_source = RecordedFrameSource(record_dir,
                                       preload=args.preload,
                                       preprocess=args.preprocess,
                                       frame_skip=args.frame_skip)
    imu_source = RecordedIMUSource(record_dir,
                                   wy_bias=args.wy_bias,
                                   wz_bias=args.wz_bias)
    template_source = RecordedTemplateSource(record_dir)

    strided_patch_size = 4000

    max_flow_time = 0.8 * (1.0/90.0) # Not important since real time isn't the problem here

    last_time = imu_source.latest_timestamp()

    ttc_depth = TTCDepth(frame_source, imu_source, template_source, K,
                         visualize=args.visualize,
                         wait_key=wait_key,
                         max_flow_time=max_flow_time,
                         max_visualize_rate=max_visualize_rate,
                         save_visualization=save_visualization,
                         save_visualization_dir=visualization_dir,
                         plot_start_t=args.plot_start,
                         plot_end_t=args.plot_end,
                         strided_patch_size=strided_patch_size,
                         april_ground_truth=args.april,
                         april_resize_to=april_resize_to,
                         max_april_rate=max_april_rate,
                         max_delta=max_delta,
                         vector_pub=vector_pub,
                         print_timing=print_timing,
                         ground_truth_source=ground_truth_source,
                         affine_skip=args.affine_skip)

    print('Beginning')
    start_time = time.time()

    track_start_time = None
    track_start_frame_count = None
    track_end_time = None
    track_end_frame_count = None
    while True:
        ttc_depth.update()
        cv2.waitKey(wait_key)

        if track_start_time is None:
            if ttc_depth._affine_tracker is not None:
                track_start_time = time.time()
                track_start_frame_count = ttc_depth._frames_processed

        if frame_source.latest_timestamp() is None or imu_source.latest_timestamp() is None:
            track_end_time = time.time()
            track_end_frame_count = ttc_depth._frames_processed
            print('Out of data')
            break

        if ttc_depth._ttc_pose_observer_time_computed_to > last_time - params['trim_end']:
            track_end_time = time.time()
            track_end_frame_count = ttc_depth._frames_processed
            print('Early end')
            break

    end_time = time.time()

    tracked_time = track_end_time - track_start_time
    tracked_frames = track_end_frame_count - track_start_frame_count

    num_samples = tracked_frames
    frames_start_time  = track_start_time
    frames_stop_time   = track_end_time
    print('======================================================')
    print('Took {:.3f} seconds'.format(track_end_time-track_start_time))
    print('Processed {} frames {:.2f} ms per frame'.format(num_samples, 1000*(track_end_time-track_start_time)/num_samples))
    print('{:.2f} Hz'.format(num_samples / (frames_stop_time-frames_start_time)))
    print('======================================================')

    #print('{:.2f}x realtime'.format((frames_stop_time-frames_start_time)/ (end_time-start_time)))

    # num_samples = frame_source._sample_index
    # frames_start_time  = frame_source._frame_ts.min()
    # frames_stop_time   = frame_source._frame_ts.max()
    # print('Took {:.3f} seconds'.format(end_time-start_time))
    # print('Processed {} frames {:.2f} ms per frame'.format(num_samples, 1000*(end_time-start_time)/num_samples))
    # print('{:.2f} Hz'.format(num_samples / (end_time-start_time)))
    # print('{:.2f}x realtime'.format((frames_stop_time-frames_start_time)/ (end_time-start_time)))

    if args.save:
        file_name = os.path.join(results_dir, 'results.pickle')
        results_dict = vector_pub.get_data()
        with open(file_name, 'wb') as file:
            pickle.dump({'params': None,
                         'results': results_dict}, file)

    return num_samples, (frames_stop_time - frames_start_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', dest='dir', type=str, help='Directory to load data from. If not specified the latest in the default location is used')
    parser.add_argument('--all', dest='all', action='store_true', help='Run through all recordings in folder')
    parser.add_argument('--visualize', dest='visualize', action='store_true', help='Visualize')
    parser.add_argument('--wait', dest='wait', action='store_true', help='Wait for key press when visualizing')
    parser.add_argument('--april', dest='april', action='store_true', help='Use apriltag for ground truth')
    parser.add_argument('--preload', dest='preload', action='store_true', help='Read all images before processing for benchmarking')
    parser.add_argument('--preprocess', dest='preprocess', action='store_true', help='Preprocess all images when preloading')
    parser.add_argument('--nopublish', dest='nopublish', action='store_true', help='Dont publish with ZMQ')
    parser.add_argument('--bench', dest='bench', action='store_true', help='Activate preload nopublish and some other things')
    parser.add_argument('--realtime', dest='realtime', action='store_true', help='Test ability to run in realtime with april feedback')
    parser.add_argument('--gt', dest='ground_truth_file', type=str, help='Load ground truth from file')
    parser.add_argument('--save', dest='save', action='store_true', help='Save results to results folder')
    parser.add_argument('--save_vis', dest='save_vis', action='store_true', help='Save visualization results at 30fps for video')
    parser.add_argument('--frame_skip', dest='frame_skip', type=int, default=0, help='camera frames to skip (decimation)')
    parser.add_argument('--affine_skip', dest='affine_skip', type=int, default=0, help='affine samples to skip (decimation)')
    parser.add_argument('--wy_bias', dest='wy_bias', type=float, default=0, help='Gyro bias y axis in deg per second')
    parser.add_argument('--wz_bias', dest='wz_bias', type=float, default=0, help='Gyro bias z axis in deg per second')
    parser.add_argument('--plot_start', dest='plot_start', type=float, default=0, help='Default visualization plot start time')
    parser.add_argument('--plot_end', dest='plot_end', type=float, default=60.0, help='Default visualization plot end time')

    args = parser.parse_args()

    if args.wait:
        wait_key = 0
    else:
        wait_key = 1

    if not args.dir:
        record_dir = latest_recording_dir()
    else:
        record_dir = args.dir

    if args.bench:
        args.preload = True
        #args.preprocess = True #Pre-converting to floating point is cheating
        args.nopublish = True
        num_runs = 1
        max_delta = 0.1
        print_timing = False # Can turn on for detailed info
    else:
        num_runs = 1
        max_delta = 0.01
        print_timing = False

    if args.realtime:
        args.preload = True
        args.april = True
        max_april_rate = 30.0
        max_delta = 0.1
        print_timing = False
        april_resize_to = None #(270,480)
    else:
        max_april_rate = 100.0
        april_resize_to=None

    if args.ground_truth_file:
        # TODO https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
        ground_truth_source = pickle.load(open(args.ground_truth_file, 'rb'), encoding='latin1')['poses']
    else:
        ground_truth_source = None

    if args.save_vis:
        max_visualize_rate = 30.0
        save_visualization = True
    else:
        max_visualize_rate = 60.0
        save_visualization = False

    # TODO how to disable threading
    cv2.setNumThreads(1)

    if not args.bench:
        SETTINGS = {
            '':               {'affine_skip': 0, 'wy_bias': 0, 'wz_bias': 0, 'april': True},
            #'_affine_skip_1': {'affine_skip': 1, 'wy_bias': 0, 'wz_bias': 0},
            # '_affine_skip_2': {'affine_skip': 2, 'wy_bias': 0, 'wz_bias': 0, 'april': False},
            # '_affine_skip_3': {'affine_skip': 3, 'wy_bias': 0, 'wz_bias': 0, 'april': False},
            #'_wy_bias_0_5':   {'affine_skip': 0, 'wy_bias': 0.5, 'wz_bias': 0},
            # '_wy_bias_1':     {'affine_skip': 0, 'wy_bias': 1, 'wz_bias': 0, 'april': False},
            # '_wy_bias_2':     {'affine_skip': 0, 'wy_bias': 2, 'wz_bias': 0, 'april': False},
            #'_wy_bias_3':     {'affine_skip': 0, 'wy_bias': 3, 'wz_bias': 0},
            #'_wy_bias_4':     {'affine_skip': 0, 'wy_bias': 4, 'wz_bias': 0},
            #'_wz_bias_0_5':     {'affine_skip': 0, 'wy_bias': 0, 'wz_bias': 0.5},
            # '_wz_bias_1':     {'affine_skip': 0, 'wy_bias': 0, 'wz_bias': 1, 'april': False},
            # '_wz_bias_2':     {'affine_skip': 0, 'wy_bias': 0, 'wz_bias': 2, 'april': False},
            #'_wz_bias_3':     {'affine_skip': 0, 'wy_bias': 0, 'wz_bias': 3},
            #'_wz_bias_4':     {'affine_skip': 0, 'wy_bias': 0, 'wz_bias': 4},
            #'_wz_bias_5':     {'affine_skip': 0, 'wy_bias': 0, 'wz_bias': 5},
        }
    else:
        SETTINGS = {
            '':               {'affine_skip': 0, 'wy_bias': 0, 'wz_bias': 0, 'april': False},
        }

    if not args.all:
        for i in range(num_runs):
            run_on_directory(record_dir, args, max_visualize_rate, save_visualization)
    else:
        recording_dirs = sorted(glob.glob(os.path.join(record_dir, 'record_*')))

        total_track_time = 0
        total_frames = 0

        for directory in recording_dirs:
            #if directory != '../recordings_21_11_12/record_000004':
            #    continue

            for key in SETTINGS.keys():
                print('Processing: {}'.format(directory))
                print('Settings', SETTINGS[key])
                args.affine_skip = SETTINGS[key]['affine_skip']
                args.wy_bias = SETTINGS[key]['wy_bias']
                args.wz_bias = SETTINGS[key]['wz_bias']
                args.april = SETTINGS[key]['april']

                for i in range(num_runs):
                    track_frames, track_time = run_on_directory(directory, args, max_visualize_rate, save_visualization,
                                                                settings_name=key)

                    total_track_time += track_time
                    total_frames += track_frames


        print('===================== Overall ========================')
        print('Took {:.3f} seconds'.format(total_track_time))
        print('Processed {} frames {:.2f} ms per frame'.format(total_frames, 1000*total_track_time/total_frames))
        print('{:.2f} Hz'.format(total_frames / total_track_time))
        print('======================================================')
