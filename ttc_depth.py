###############################################################################
#
# File: ttc_depth.py
# Available under MIT license
#
# Integrates all the algorithms in TTCDist
# Is used by ttc_depth_realesense.py, ttc_depth_robomaster.py, and ttc_depth_from_folder.py 
#
# History:
# 05-17-21 - Levi Burner - Created File
# 09-26-22 - Levi Burner - Open source release
#
###############################################################################

import argparse
import time
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from affine_flow import (AffineTrackRotInvariant,
                         draw_full_reverse_warp,
                         draw_derotated,
                         draw_warped_patch_location)
from affine_flow_time_to_contact import AffineFlowTimeToContactEstimator
from ttc_pose_observer import TTCPoseObserver
from phi_pose_observer import PhiPoseObserver

try:
    from apriltag_odometry import AprilPose
    APRIL_AVAILABLE = True
except ImportError:
    APRIL_AVAILABLE = False

from ahrs.common.quaternion import slerp # TODO remove

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def trim_signal(signal, min_start_time):
    if len(signal) == 0:
        return signal

    start_time = min(signal[-1][0], min_start_time)
    
    for i, x in enumerate(signal):
        if x[0] >= start_time:
            break

    start_i = max(0, i-1)

    return signal[start_i:]

def resample_signal(samples, t):
    index_list = [i for (i, x) in enumerate(samples) if x[0] < t]
    if len(index_list) == 0:
        return np.array(samples[-1][1:], dtype=np.float32)

    index = index_list[-1]

    if index == len(samples)-1:
        sample = np.array(samples[-1][1:], dtype=np.float32)
    else:
        sample_left  = np.array(samples[index][1:], dtype=np.float32)
        t_left       = samples[index][0]
        sample_right = np.array(samples[index+1][1:], dtype=np.float32)
        t_right      = samples[index+1][0]

        alpha = (t - t_left) / (t_right - t_left)

        sample = alpha*(sample_right - sample_left) + sample_left
    return sample

def resample_orientation(qs, t):
    index_list = [i for (i, q) in enumerate(qs) if q[0] < t]
    if len(index_list) == 0:
        return np.array(qs[0][1:5])

    index = index_list[-1]

    if index == len(qs)-1:
        return np.array(qs[-1][1:5])
    else:
        q_left  = np.array(qs[index][1:5])
        t_left  = qs[index][0]
        q_right = np.array(qs[index+1][1:5])
        t_right = qs[index+1][0]

        alpha = (t - t_left) / (t_right - t_left)

        q = slerp(q_left, q_right, np.array((alpha, )))[0]

    return np.array(q)

# Hamilton product
# https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
# Scalar first
def quat_mult(q, p):
    r = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    r[0] = q[0]*p[0] - q[1]*p[1] - q[2]*p[2] - q[3]*p[3]
    r[1] = q[0]*p[1] + q[1]*p[0] + q[2]*p[3] - q[3]*p[2]
    r[2] = q[0]*p[2] - q[1]*p[3] + q[2]*p[0] + q[3]*p[1]
    r[3] = q[0]*p[3] + q[1]*p[2] - q[2]*p[1] + q[3]*p[0]
    return r

# Scalar first
def quat_inv_no_norm(q):
    q_inv = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    q_inv[0] = q[0]
    q_inv[1:4] = -q[1:4]
    return q_inv

# Scalar first
# Forward eular, could do trapezoidal
def integrate_quaternion(q, gyr, dt):
    p = np.array([0.0, gyr[0], gyr[1], gyr[2]], dtype=np.float32)
    dot_q = 0.5 * quat_mult(q, p)
    q_unpacked = q + dt * dot_q
    return q_unpacked

class TTCDepth(object):
    def __init__(self, frame_source, imu_source, template_source, K,
                       visualize=False, wait_key=1, max_flow_time=None,
                       max_visualize_rate=30.0,
                       save_visualization=False,
                       save_visualization_dir=None,
                       plot_start_t=0.0,
                       plot_end_t=30.0,
                       strided_patch_size=100.0,
                       april_ground_truth=False, vector_pub = None,
                       max_april_rate=10.0,
                       april_resize_to = None,
                       max_delta=0.1,
                       print_timing=False,
                       ground_truth_source=None,
                       affine_skip=0,
                       phi_pose_subscriber=None,
                       phi_accel_power_thresh=2.0):
        self._frame_source = frame_source
        self._imu_source = imu_source
        self._K = K
        self._K_inv = np.linalg.inv(self._K)
        self._visualize = visualize
        self._visualize_derotation = False
        self._wait_key = wait_key
        self._max_flow_time = max_flow_time
        self._affine_skip = affine_skip

        self._max_visualize_rate = max_visualize_rate
        self._save_visualization = save_visualization
        self._save_visualization_dir = save_visualization_dir
        self._plot_start_t = plot_start_t
        self._plot_end_t = plot_end_t
        self._save_visualization_frame_id = 0

        self._max_april_rate = max_april_rate

        self._last_visualize_time = time.time()
        self._last_april_time = time.time()

        # Common choice
        #resize_to = (270, 480)
        self._april_resize_to = april_resize_to

        self._max_delta = max_delta

        self._orientation_estimator_last_t = None
        self._orientations_q = []
        self._accels_meas_c = []
        self._all_accels_meas_c = []


        self._april_ground_truth = april_ground_truth
        if self._april_ground_truth and not APRIL_AVAILABLE:
            raise Exception('AprilTag import failed and apriltag measurements were requested. Is the library installed?')

        self._vector_pub = vector_pub

        self._frame_q_w_c0 = np.array((1, 0, 0, 0), dtype=np.float32)

        self._template_source = template_source
        self._affine_tracker = None
        self._ttc_estimator = None
        self._ttc_pose_observer = None
        self._phi_pose_observer = None
        self._phi_accel_power_thresh = phi_accel_power_thresh
        self._ttc_pose_observer_creation_time = None
        self._phi_pose_observer_creation_time = None
        self._p_list = []
        self._ttc_list = []
        self._phi_list = []
        self._all_ttc_list = []
        self._z_hat_list = []
        self._phi_z_hat_list = []
        self._all_z_hat_list = []
        self._all_pose_hat_list = []
        self._ground_truth_pose_list = []
        self._R_c_c0_list = []

        self._imu_max_time_computed_to           = None
        self._imu_time_computed_to               = None
        self._flow_time_computed_to              = None
        self._ttc_pose_observer_time_computed_to = None
        self._phi_pose_observer_time_computed_to = None

        self._ttc_pose_observer_dt = 0.01
        self._phi_pose_observer_dt = 0.01
        self._ttc_pose_observer_seconds_to_keep = 2.0
        self._phi_pose_observer_seconds_to_keep = 2.0

        self._frame_unused_sample = None

        self._strided_patch_size = strided_patch_size

        self._april_pose = None

        self._print_timing = print_timing

        self._ground_truth_source = ground_truth_source

        self._last_p = None
        self._last_gt_p = None
        self._last_gt_p_t = None

        self._last_q_tw_tc_wxyz = np.array([1., 0., 0., 0.])

        self._frames_processed = 0

        self._phi_pose_subscriber = phi_pose_subscriber

        self._patch_coordinates = None

    def update(self):
        if not self._update_time_computed_to(): # Return early if no new data
            return

        if self._print_timing:
            start_time = time.time()
            start_imu_time_computed_to = self._imu_time_computed_to
        self._update_orientations()
        if self._print_timing:
            end_imu_time_computed_to = self._imu_time_computed_to
            end_time = time.time()
            delta_real = end_imu_time_computed_to - start_imu_time_computed_to
            delta = end_time - start_time
            print('imu      delta {:0.3f} data time {:0.3f} times realtime {:4.1f}'.format(delta, delta_real, delta_real/delta))

        if self._print_timing:
            start_flow_time_computed_to = self._flow_time_computed_to
            start_time = time.time()
        self._update_affine_flow()
        if self._print_timing:
            end_time = time.time()
            end_flow_time_computed_to = self._flow_time_computed_to
            delta_real = end_flow_time_computed_to - start_flow_time_computed_to
            delta = end_time - start_time
            print('flow     delta {:0.3f} data time {:0.3f} times realtime {:4.1f}'.format(delta, delta_real, delta_real/delta))

        if self._print_timing:
            start_time = time.time()
            start_observer_time_computed_to = self._ttc_pose_observer_time_computed_to
        self._update_ttc_pose_observer()
        if self._print_timing:
            end_observer_time_computed_to = self._ttc_pose_observer_time_computed_to
            end_time = time.time()
            delta_real = end_observer_time_computed_to - start_observer_time_computed_to
            delta = end_time - start_time
            print('ttc observer delta {:0.3f} data time {:0.3f} times realtime {:4.1f}'.format(delta, delta_real, delta_real/delta))

        if self._print_timing:
            start_time = time.time()
            start_observer_time_computed_to = self._phi_pose_observer_time_computed_to
        self._update_phi_pose_observer()
        if self._print_timing:
            end_observer_time_computed_to = self._phi_pose_observer_time_computed_to
            end_time = time.time()
            delta_real = end_observer_time_computed_to - start_observer_time_computed_to
            delta = end_time - start_time
            print('phi observer delta {:0.3f} data time {:0.3f} times realtime {:4.1f}'.format(delta, delta_real, delta_real/delta))

        if self._print_timing:
            start_time = time.time()
        trim_time = min(self._ttc_pose_observer_time_computed_to, self._phi_pose_observer_time_computed_to)
        self._orientations_q   = trim_signal(self._orientations_q, trim_time)
        self._accels_meas_c    = trim_signal(self._accels_meas_c, trim_time)
        self._R_c_c0_list      = trim_signal(self._R_c_c0_list, trim_time)
        self._p_list           = trim_signal(self._p_list, trim_time)
        self._ttc_list         = trim_signal(self._ttc_list, trim_time)
        self._phi_list         = trim_signal(self._phi_list, trim_time)
        self._z_hat_list       = trim_signal(self._z_hat_list, trim_time)
        self._phi_z_hat_list   = trim_signal(self._phi_z_hat_list, trim_time)
        if self._print_timing:
            end_time = time.time()
            delta_real = end_observer_time_computed_to - start_observer_time_computed_to
            delta = end_time - start_time
            print('trim     delta {:0.3f} data time {:0.3f} times realtime {:4.1f}'.format(delta, delta_real, delta_real/delta))
            print('=========================')

    def _update_time_computed_to(self):
        if self._imu_max_time_computed_to is None:
            imu_first_timestamp   = self._imu_source.earliest_timestamp()
            frame_first_timestamp = self._frame_source.earliest_timestamp()

            if imu_first_timestamp < 0 or frame_first_timestamp < 0:
                print('ttc depth sources do not have data yet')
                time.sleep(0.05) # TODO NO!
                return False

            self._imu_max_time_computed_to           = min(frame_first_timestamp, imu_first_timestamp)
            self._imu_time_computed_to               = self._imu_max_time_computed_to
            self._flow_time_computed_to              = self._imu_max_time_computed_to
            self._ttc_pose_observer_time_computed_to = frame_first_timestamp
            self._phi_pose_observer_time_computed_to = frame_first_timestamp

        # Compute up to some maximum timestep into the future
        max_update_time = self._imu_time_computed_to + self._max_delta

        # Find the latest IMU and frame timestamp less than or equal to that bound
        imu_max_update_time = self._imu_source.latest_timestamp()
        if imu_max_update_time is not None:
            imu_max_update_time = min(imu_max_update_time, max_update_time)

        frame_max_update_time = self._frame_source.latest_timestamp()
        if frame_max_update_time is not None:
            frame_max_update_time = min(frame_max_update_time, max_update_time)

        # If no new data is available then mention it
        # Detect if data seems to be coming in from one source but not the other
        if imu_max_update_time is None or frame_max_update_time is None:
            print('No new data')
            if frame_max_update_time is not None:
                if (frame_max_update_time - self._time_computed_to) > 1.0:
                    print('IMU data is lagging by {}'.format(frame_max_update_time - self._time_computed_to))
            if imu_max_update_time is not None:
                if (imu_max_update_time - self._time_computed_to) > 1.0:
                    print('Frame data is lagging by {}'.format(imu_max_update_time - self._time_computed_to))
            return False

        # Compute up to the satisfying IMU sample
        # Frames cannot be processed past the IMU sample since they depend on the orientation
        self._imu_max_time_computed_to = imu_max_update_time
        return True

    def _update_orientations(self):
        last_t = self._imu_time_computed_to
        while True:
            #start = time.time()
            imu_sample = self._imu_source.next_sample()
            #mid = time.time()
            if imu_sample is None:
                self._imu_time_computed_to = last_t
                break

            if self._orientation_estimator_last_t is None:
                t = imu_sample[0]
                self._orientation_estimator_last_t = t
                accel_meas_c = [0, -9.81, 0]
                gyro_c = [0, 0, 0]
                q_w_c_wxyz = [1., 0., 0., 0.]
                #mid = time.time()
            else:
                t     = imu_sample[0]
                accel_meas_c = imu_sample[1:4]
                gyro_c  = imu_sample[4:7]

                R_tc_c = np.array([[0, 0, 1],
                                   [-1, 0, 0],
                                   [0, -1, 0]])
                R_tw_w = R_tc_c
                R_w_tw = R_tw_w.transpose()

                accel_meas_tc = R_tc_c @ accel_meas_c
                gyro_tc  = R_tc_c @ gyro_c

                q_tw_tc_wxyz = integrate_quaternion(self._last_q_tw_tc_wxyz, gyro_tc, t - self._orientation_estimator_last_t)
                self._last_q_tw_tc_wxyz = q_tw_tc_wxyz

                # Scipy uses scalar last quaternion format
                R_tw_tc = R.from_quat([q_tw_tc_wxyz[1], q_tw_tc_wxyz[2], q_tw_tc_wxyz[3], q_tw_tc_wxyz[0]]).as_matrix().astype(np.float32)
                R_w_c = R_w_tw @ R_tw_tc @ R_tc_c

                q_w_c_xyzw = R.from_matrix(R_w_c).as_quat()
                # We use scalar first format
                q_w_c_wxyz = np.array([q_w_c_xyzw[3], q_w_c_xyzw[0], q_w_c_xyzw[1], q_w_c_xyzw[2]]).astype(np.float32)

                #mid = time.time()

            self._publish('gyro', t, np.array(gyro_c))
            self._orientations_q.append([t, q_w_c_wxyz[0], q_w_c_wxyz[1], q_w_c_wxyz[2], q_w_c_wxyz[3]])

            self._accels_meas_c.append([t, *accel_meas_c])
            self._all_accels_meas_c.append([t, *accel_meas_c])
            self._publish('accel_meas_c', t, np.array(accel_meas_c))

            self._orientation_estimator_last_t  = t
            last_t = t
            end = time.time()

            #delta = end - start
            #mid_delta = mid - start
            #print('delta {:03f} mid delta {:03f}'.format(delta, mid_delta))

            if t > self._imu_max_time_computed_to:
                self._imu_time_computed_to = t
                break


    def _update_patch(self, t, frame_gray, frame_q_w_c0):
        if self._affine_tracker is None:
            new_patch_coordinates = self._template_source.get_new_patch(t)
            if new_patch_coordinates is not None:

                patch_area = (new_patch_coordinates[2] - new_patch_coordinates[0]) * (new_patch_coordinates[3] - new_patch_coordinates[1])
                stride = np.sqrt(patch_area / self._strided_patch_size)

                self._affine_tracker = AffineTrackRotInvariant(patch_coordinates=new_patch_coordinates,
                                                               template_image=frame_gray,
                                                               template_q_c_to_fc=np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
                                                               K=self._K,
                                                               visualize=False, #self._visualize,
                                                               wait_key=self._wait_key,
                                                               stride=stride,
                                                               inverse=True,
                                                               max_update_time=self._max_flow_time)
                self._frame_q_w_c0 = frame_q_w_c0
                self._ttc_estimator = AffineFlowTimeToContactEstimator(new_patch_coordinates, self._K)
                self._ttc_pose_observer = TTCPoseObserver(dt=self._ttc_pose_observer_dt,
                                                          seconds_to_keep=self._ttc_pose_observer_seconds_to_keep)
                self._phi_pose_observer = PhiPoseObserver(dt=self._phi_pose_observer_dt,
                                                          seconds_to_keep=self._phi_pose_observer_seconds_to_keep,
                                                          accel_power_thresh=self._phi_accel_power_thresh)
                self._ttc_pose_observer_creation_time = t
                self._phi_pose_observer_creation_time = t
                self._patch_coordinates = new_patch_coordinates
                return True
        else:
            current_patch_valid_return = self._template_source.current_patch_valid(t)
            if current_patch_valid_return == 'reset observer':
                self._ttc_pose_observer.reset_ic()
                self._phi_pose_observer.reset_ic()
                return True
            elif not current_patch_valid_return:
                self._affine_tracker = None
                self._ttc_estimator  = None
                self._ttc_pose_observer = None
                self._phi_pose_observer = None
                self._ttc_pose_observer_creation_time = None
                self._phi_pose_observer_creation_time = None
                self._patch_coordinates = None
                return False
        return False

    def _update_affine_flow(self):
        while True:
            #start = time.time()
            if self._frame_unused_sample is None:
                sample = self._frame_source.next_sample()
            else:
                sample = self._frame_unused_sample

            #mid = time.time()

            if sample is None:
                break

            (t, frame_gray) = sample

            if t > self._imu_time_computed_to:
                self._frame_unused_sample = sample
                break
            else:
                self._frame_unused_sample = None

            process_april = True

            frame_q_w_c = resample_orientation(self._orientations_q, t)
            new_patch = self._update_patch(t, frame_gray, frame_q_w_c)

            # if new_patch:
                # while True:
                #     new_sample = self._frame_source.next_sample()
                #     if new_sample is None:
                #         break
                #     sample = new_sample

            frame_q_c0_c = quat_mult(quat_inv_no_norm(self._frame_q_w_c0), frame_q_w_c)

            if self._affine_tracker is not None:
                if self._last_p is None:
                #if len(self._p_list) == 0:
                    self._last_p = np.array((0, 0, 0, 0, 0, 0))
                #else:
                #    self._last_p = self._p_list[-1][1:]

                R_c_c0 = R.from_quat([frame_q_c0_c[1], frame_q_c0_c[2], frame_q_c0_c[3], frame_q_c0_c[0]]).as_matrix().astype(np.float32).transpose()
                
                if self._frames_processed % (self._affine_skip + 1) == 0:
                    self._R_c_c0_list.append((t, R_c_c0))
                    self._publish('R_fc_to_c', t, R_c_c0)

                #print(self._last_p)

                p = self._affine_tracker.update(self._last_p, frame_gray, R_c_c0)
                self._last_p = np.copy(p)

                #p = np.array((0, 0, 0, 0, 0, 0))
                if self._frames_processed % (self._affine_skip + 1) == 0:
                    self._p_list.append([t, ] + p.tolist())
                    self._publish('p', t, p)

                    res = self._ttc_estimator.estimate_ttc(p)
                    if res is not False:
                        x_dot_over_z, y_dot_over_z, z_dot_over_z, ttc_inv_xy = res

                        # x_dot_over_z and others are measured in the camera frame
                        # but we need them in the objects frame
                        # which means the sign is flipped
                        x_dot_over_z = -x_dot_over_z / (t - self._flow_time_computed_to)
                        y_dot_over_z = -y_dot_over_z / (t - self._flow_time_computed_to)
                        z_dot_over_z = -z_dot_over_z / (t - self._flow_time_computed_to)

                        self._ttc_list.append((t, x_dot_over_z, y_dot_over_z, z_dot_over_z))
                        self._all_ttc_list.append((t, x_dot_over_z, y_dot_over_z, z_dot_over_z))
                        self._publish('ttc_inv', t, np.array((x_dot_over_z, y_dot_over_z, z_dot_over_z), dtype=np.float32))
                    else:
                        print('failed to estimate ttc')

                    # Calculate phi and add to list
                    phi_x = p[4]
                    phi_y = p[5]
                    phi_z = 2 / (1 + p[0] + 1 + p[3])
                    self._phi_list.append((t, phi_x, phi_y, phi_z))
            else:
                self._last_p = np.array((0, 0, 0, 0, 0, 0))
                R_c_c0 = R.from_quat([0, 0, 0, 1]).as_matrix().astype(np.float32).transpose()
                if self._frames_processed % (self._affine_skip + 1) == 0:
                    self._p_list.append((t, 0, 0, 0, 0, 0, 0))
                    self._R_c_c0_list.append((t, R_c_c0))
                    self._ttc_list.append((t, 0, 0, 0))
                    self._phi_list.append((t, 0, 0, 0))
                    self._all_ttc_list.append((t, 0, 0, 0))

            if self._frames_processed % (self._affine_skip + 1) == 0:
                self._flow_time_computed_to = t

            if (self._visualize or self._visualize_derotation) and ((time.time() - self._last_visualize_time) > (1.0/self._max_visualize_rate)):
                if self._visualize:
                    if self._affine_tracker is None:
                        cv2.imshow('frame_gray', frame_gray)
                        self._save_visualization_func(frame_gray)
                    else:
                        frame_gray_copy = np.copy(frame_gray)
                        draw_warped_patch_location(frame_gray_copy, self._patch_coordinates, p, frame_q_c0_c, self._K)

                        frame_bgr_copy = cv2.cvtColor(frame_gray_copy, cv2.COLOR_GRAY2BGR)

                        if len(self._z_hat_list) > 0:
                            depth_str = '{:0.1f} ft'.format(-self._z_hat_list[-1][1] * 3.28084)
                            cv2.putText(frame_bgr_copy, depth_str, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), thickness=4)

                        if len(self._phi_z_hat_list) > 0:
                            depth_str = '{:0.1f} ft'.format(-self._phi_z_hat_list[-1][1] * 3.28084)
                            cv2.putText(frame_bgr_copy, depth_str, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), thickness=4)

                        cv2.imshow('frame_gray', frame_bgr_copy)
                        self._save_visualization_func(frame_gray_copy)

                if self._visualize_derotation:
                    frame_warped_back = draw_derotated(frame_gray, frame_q_c0_c, self._K)
                    cv2.imshow('affine flow', frame_warped_back)

                #cv2.waitKey(self._wait_key)
                self._last_visualize_time = time.time()

            if self._april_ground_truth and ((time.time() - self._last_april_time) > (1.0/self._max_april_rate)):
                self._update_april_ground_truth(t, frame_gray, R_c_c0)
                self._last_april_time = time.time()

            self._frame_source.free_sample()
            self._frames_processed += 1

            #end = time.time()

            #delta = end - start
            #mid_delta = mid - start
            #print('flow internal delta {:03f} mid delta {:03f}'.format(delta, mid_delta))

    def _update_ttc_pose_observer(self):
        while True:
            publish = False
            z_hat = 0.0
            accel_x_z_tf = 0.0
            accel_y_z_tf = 0.0
            accel_z_z_tf = 0.0

            next_t = self._ttc_pose_observer_time_computed_to + self._ttc_pose_observer_dt

            if next_t > self._flow_time_computed_to:
                break

            if self._ground_truth_source is not None:
                self._publish_ground_truth_source(next_t)

            self._ttc_pose_observer_time_computed_to += self._ttc_pose_observer_dt

            if self._ttc_pose_observer is not None:
                if next_t >= self._ttc_pose_observer_creation_time:
                    accel_meas_c = resample_signal(self._accels_meas_c, next_t)
                    frame_q_w_c = resample_orientation(self._orientations_q, next_t)
                    scaled_velocities = resample_signal(self._ttc_list, next_t)

                    # Calculate and plot linear acceleration
                    R_w_c = R.from_quat([frame_q_w_c[1], frame_q_w_c[2], frame_q_w_c[3], frame_q_w_c[0]]).as_matrix()
                    R_w_c0 = R.from_quat([self._frame_q_w_c0[1], self._frame_q_w_c0[2], self._frame_q_w_c0[3], self._frame_q_w_c0[0]]).as_matrix()
                    R_c0_c = R_w_c0.transpose() @ R_w_c
                    accel_meas_c0 = R_c0_c @ accel_meas_c

                    observer_ret = self._ttc_pose_observer.update(next_t, scaled_velocities, accel_meas_c0)

                    if observer_ret is not None:
                        z_hat, accel_x_z_tf, accel_y_z_tf, accel_z_z_tf = observer_ret
                        self._z_hat_list.append((next_t, z_hat))
                        self._all_z_hat_list.append((next_t, z_hat))
                        pose_hat = self._calc_pose_hat(next_t, z_hat)
                        self._all_pose_hat_list.append([next_t, *pose_hat])
                        publish = True
                        # print('ttc:', z_hat)
                else:
                    self._z_hat_list.append((next_t, 0))
                    self._all_z_hat_list.append((next_t, 0))
                    self._all_pose_hat_list.append([next_t, 0, 0, 0])
            else:
                self._z_hat_list.append((next_t, 0))
                self._all_z_hat_list.append((next_t, 0))
                self._all_pose_hat_list.append([next_t, 0, 0, 0])

            if publish:
                pose_hat = self._calc_pose_hat(next_t, z_hat)
                self._publish('accel_z_hat', next_t, np.array((accel_x_z_tf, accel_y_z_tf, accel_z_z_tf)))
                self._publish_pose_hat(next_t, pose_hat)

    def _update_phi_pose_observer(self):
        while True:
            publish = False
            z_hat = 0.0
            accel_x_z_tf = 0.0
            accel_y_z_tf = 0.0
            accel_z_z_tf = 0.0

            next_t = self._phi_pose_observer_time_computed_to + self._phi_pose_observer_dt

            if next_t > self._flow_time_computed_to:
                break

            # if self._ground_truth_source is not None:
            #     self._publish_ground_truth_source(next_t)

            self._phi_pose_observer_time_computed_to += self._phi_pose_observer_dt

            if self._phi_pose_observer is not None:
                if next_t >= self._phi_pose_observer_creation_time:
                    accel_meas_c = resample_signal(self._accels_meas_c, next_t)
                    frame_q_w_c = resample_orientation(self._orientations_q, next_t)
                    phi = resample_signal(self._phi_list, next_t)

                    # Calculate and plot linear acceleration
                    R_w_c = R.from_quat([frame_q_w_c[1], frame_q_w_c[2], frame_q_w_c[3], frame_q_w_c[0]]).as_matrix()
                    R_w_c0 = R.from_quat([self._frame_q_w_c0[1], self._frame_q_w_c0[2], self._frame_q_w_c0[3], self._frame_q_w_c0[0]]).as_matrix()
                    R_c0_c = R_w_c0.transpose() @ R_w_c
                    accel_meas_c0 = R_c0_c @ accel_meas_c

                    observer_ret = self._phi_pose_observer.update(next_t, phi, accel_meas_c0)

                    if observer_ret is not None:
                        z_hat, accel_x_z_tf, accel_y_z_tf, accel_z_z_tf = observer_ret
                        self._phi_z_hat_list.append((next_t, z_hat))
                        # self._all_z_hat_list.append((next_t, z_hat))
                        publish = True
                        # print('phi:', z_hat)
                else:
                    self._phi_z_hat_list.append((next_t, 0))
                    # self._all_z_hat_list.append((next_t, 0))
            else:
                self._phi_z_hat_list.append((next_t, 0))
                # self._all_z_hat_list.append((next_t, 0))

            if publish:
                pose_hat = self._calc_pose_hat(next_t, z_hat)
                if self._phi_pose_subscriber is not None:
                    self._phi_pose_subscriber.add_pose_hat(next_t, pose_hat)

                self._publish('phi_accel_z_hat', next_t, np.array((accel_x_z_tf, accel_y_z_tf, accel_z_z_tf)))
                self._publish_pose_hat(next_t, pose_hat, 'phi_pose_hat')

    def _calc_pose_hat(self, next_t, z_hat):
        p = resample_signal(self._p_list, next_t)

        A_p = np.array([[1+p[0],   p[2], p[4]],
                        [p[1],   1+p[3], p[5]],
                        [   0,        0,   1]], dtype=np.float32)

        patch_center_xy = np.array([(self._patch_coordinates[2] + self._patch_coordinates[0]) / 2.0,
                                    (self._patch_coordinates[3] + self._patch_coordinates[1]) / 2.0,
                                    1.0])
        point = patch_center_xy
        point = A_p @ point
        px = point[0]/point[2]
        py = point[1]/point[2]

        pose_hat = z_hat * self._K_inv @ np.array((px, py, 1.0), dtype=np.float32)
        return pose_hat

    def _publish_pose_hat(self, next_t, pose_hat, name='pose_hat'):
        if self._vector_pub:
            self._publish(name, next_t, pose_hat)

    def _update_april_ground_truth(self, t, frame_gray, R_c_c0):
        if self._april_resize_to is None:
            resize_to = frame_gray.shape
        else:
            resize_to = self._april_resize_to

        resize_ratio = frame_gray.shape[0] / resize_to[0]
        if self._april_pose is None:
            K = np.copy(self._K) # Hack!
            K = K / resize_ratio
            K[2, 2] = 1
            self._april_pose = AprilPose(K, family='tag36h11', marker_size_m=160/1000.0)


        frame_gray = cv2.resize(frame_gray, resize_to[::-1]) # Hack!

        #start = time.time()
        detections = self._april_pose.find_tags((frame_gray * 255).astype(np.uint8))
        #end = time.time()
        #print('Time {}'.format(end-start))

        # frame_gray_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        # self._april_pose.draw_detections(frame_gray_bgr, detections)
        # cv2.imshow('detections', frame_gray_bgr)

        if len(detections) > 1:
           raise Exception('Too many april tags')
        elif len(detections) == 1:
           detection = detections[0]
        else:
           detection = None

        # detection = self._april_pose.find_detection(detections, 19)

        if detection is not None:
            pose_c, rot = self._april_pose.find_pose_from_tag(detection)

            # The pose is measured in the camera frame but we need
            # it in the objects frame so take the negative
            pose_c0 = -R_c_c0.transpose() @ pose_c

            self._ground_truth_pose_list.append((t, pose_c0[0], pose_c0[1], pose_c0[2]))
            self._publish('ground_truth_pose', t, pose_c0)

            if self._last_gt_p is not None:
                dp = pose_c0 - self._last_gt_p
                v =  dp / (t - self._last_gt_p_t)
                F =  v / pose_c0[2]
                self._publish('ttc_inv_gt', t, F)
            else:
                v = None

            self._last_gt_p = pose_c0
            self._last_gt_p_t = t
            return v
        return None

        #end = time.time()
        #print('April {}'.format(end-start))


    def _publish(self, topic, t, data):
        #start = time.time()
        if self._vector_pub is not None:
            self._vector_pub.publish(topic, t, data)
        #end = time.time()
        #print('publish {}'.format(end-start))

    def _publish_ground_truth_source(self, t):
        position = resample_signal(self._ground_truth_source, t)
        self._publish('ground_truth_pose', t, position)

    def _visualize_plots(self):
        start_t = self._plot_start_t
        end_t = self._plot_end_t

        ttc_list = np.array(self._all_ttc_list)
        ttc_list[:, 0] = ttc_list[:, 0] - ttc_list[0, 0] - start_t

        accel_meas_c = np.array(self._all_accels_meas_c)
        accel_meas_c[:, 0] = accel_meas_c[:, 0] - accel_meas_c[0, 0] - start_t

        z_hat_list = np.array(self._all_z_hat_list)
        z_hat_list[:, 0] = z_hat_list[:, 0] - z_hat_list[0, 0] - start_t

        pose_hat_list = np.array(self._all_pose_hat_list)
        pose_hat_list[:, 0] = pose_hat_list[:, 0] - pose_hat_list[0, 0] - start_t

        if len(self._ground_truth_pose_list) > 0:
            ground_truth_pose = np.array(self._ground_truth_pose_list)
            ground_truth_pose[:, 0] = ground_truth_pose[:, 0] - ground_truth_pose[0, 0] - start_t
        else:
            ground_truth_pose = None

        if ttc_list[-1, 0] >= 0 and ttc_list[-1, 0] <= (end_t - start_t):
            fig = plt.figure()
            canvas = FigureCanvas(fig)

            plt.subplot(3,1,1)
            plt.plot(pose_hat_list[:, 0], pose_hat_list[:, 1])
            plt.title('Tau-constraint Position Estimation')
            plt.ylabel('X (m)')
            plt.xlim([0, end_t - start_t])
            plt.ylim([-1, 1])
            plt.grid()

            plt.subplot(3,1,2)
            plt.plot(pose_hat_list[:, 0], pose_hat_list[:, 2])
            plt.xlim([0, end_t - start_t])
            plt.ylim([-1, 1])
            plt.ylabel('Y (m)')
            plt.grid()

            plt.subplot(3,1,3)
            plt.plot(pose_hat_list[:, 0], pose_hat_list[:, 3], label='Z Tau-Constraint')

            if ground_truth_pose is not None:
                plt.plot(ground_truth_pose[:, 0], ground_truth_pose[:, 3], label='Z April Tag')

            plt.xlim([0, end_t - start_t])
            plt.ylim([-4, 0])
            # plt.legend()
            plt.ylabel('Z (m)')
            plt.xlabel('t (seconds)')
            plt.grid()

            plt.tight_layout()

            canvas.draw()
            width, height = fig.get_size_inches() * fig.get_dpi()
            data = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            plt.close()
            plot_bgr = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

            return plot_bgr
            #cv2.imshow('plot_bgr', plot_bgr)
        #plt.show()


    def _save_visualization_func(self, frame_gray):
        if self._save_visualization:
            plot_bgr = self._visualize_plots()

            frame_gray = np.clip(frame_gray, 0.0, 1.0)

            frame_gray_bgr_uint8 = (cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)*255).astype(np.uint8)

            #frame_gray_name = os.path.join(self._save_visualization_dir, 'frame_{:06d}.png'.format(self._save_visualization_frame_id))
            #cv2.imwrite(frame_gray_name, frame_gray_bgr_uint8)
            #self._save_visualization_frame_id += 1

            if plot_bgr is not None:
                (RES_Y, RES_X, _) = plot_bgr.shape
                camera_frame_res_x = int(frame_gray_bgr_uint8.shape[1] * (RES_Y / frame_gray_bgr_uint8.shape[0]))

                frame_gray_bgr_uint8_resized = cv2.resize(frame_gray_bgr_uint8, dsize=(camera_frame_res_x, RES_Y))

                video_bgr = np.hstack((frame_gray_bgr_uint8_resized, plot_bgr))
                cv2.imshow('Video frame', video_bgr)

                frame_name = os.path.join(self._save_visualization_dir, 'frame_{:06d}.png'.format(self._save_visualization_frame_id))
                cv2.imwrite(frame_name, video_bgr)
                self._save_visualization_frame_id += 1
