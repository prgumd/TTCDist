###############################################################################
#
# File: affine_flow.py
# Available under MIT license
#
# Estimate affine flow for a predetermined patch
#
# History:
# 04-23-20 - Levi Burner - Created file
# 09-26-22 - Levi Burner - Open source release
#
###############################################################################

import time

import cv2
import numpy as np
import numba as nb
from scipy.spatial.transform import Rotation as R

def derotate_image(frame, K, q_c_to_fc):
    R_c_to_fc = R.from_quat((q_c_to_fc[1], q_c_to_fc[2], q_c_to_fc[3], q_c_to_fc[0])).as_matrix()
    R_fc_to_c = R_c_to_fc.T

    # Derive this by considering p1 = (K R K_inv) (Z(X)/Z(RX)) p0
    top_two_rows = (K @ R_fc_to_c @ np.linalg.inv(K))[0:2, :]
    bottom_row = (R_fc_to_c @ np.linalg.inv(K))[2, :]

    map_pixel_c_to_fc = np.vstack((top_two_rows, bottom_row))
    map_pixel_c_to_fc_opencv = np.float32(map_pixel_c_to_fc.flatten().reshape(3,3))

    frame_derotated = cv2.warpPerspective(frame, map_pixel_c_to_fc_opencv, (frame.shape[1], frame.shape[0]), flags=cv2.WARP_INVERSE_MAP+cv2.INTER_LINEAR)
    return frame_derotated

def calculate_dW_dp_mults(rect, stride):
    region_size = (int((rect[2] - rect[0])/stride), int((rect[3] - rect[1])/stride))

    x_mult = stride*np.tile(np.arange(0, region_size[0]), region_size[1]) + rect[0]
    y_mult = stride*np.repeat(np.arange(0, region_size[1]), region_size[0]) + rect[1]
    return x_mult, y_mult

# From eq 35 of LK 20 years on
def invert_delta_p(last_p, T_considered, I_warped_back_considered, grad_T_dW, H_T_dW_inv):
    diff_T_I_shaped  = (T_considered - I_warped_back_considered).flatten()

    grad_J_to_p = diff_T_I_shaped @ grad_T_dW
    delta_p_inv = H_T_dW_inv @ grad_J_to_p

    p = -delta_p_inv

    p_inv_unscaled = np.array(
        (-p[0] - p[0]*p[3] + p[1] * p[2],
         -p[1],
         -p[2],
         -p[3] - p[0]*p[3] + p[1]*p[2],
         -p[4] - p[3]*p[4] + p[2]*p[5],
         -p[5] - p[0]*p[5] + p[1]*p[4]),
        dtype=np.float32)
    scale = 1.0 / ((1+p[0])*(1+p[3])-p[1]*p[2])
    delta_p = scale * p_inv_unscaled

    last_p = compose_warp(last_p, delta_p)

    return delta_p, last_p
invert_delta_p = nb.jit(nopython = True, cache = True, fastmath=True)(invert_delta_p)

# Calculate p_c s.t. W(p_c, x) = W(p, W_(dp, x))
# from eq 18 of LK 20 years on
def compose_warp(p, dp):
    p_c = np.array(
        (p[0] + dp[0] + p[0] * dp[0] + p[2] * dp[1],
         p[1] + dp[1] + p[1] * dp[0] + p[3] * dp[1],
         p[2] + dp[2] + p[0] * dp[2] + p[2] * dp[3],
         p[3] + dp[3] + p[1] * dp[2] + p[3] * dp[3],
         p[4] + dp[4] + p[0] * dp[4] + p[2] * dp[5],
         p[5] + dp[5] + p[1] * dp[4] + p[3] * dp[5]))
    return p_c
compose_warp = nb.jit(nopython = True, cache = True, fastmath=True)(compose_warp)

def make_rot_times_affine(stride, p, rect, R_fc_to_c, K, K_inv):
    # Affine matrix parameterized by p
    A_p = np.array([[stride*(1.0+p[0]),   stride*p[2], p[4] + (1.0+p[0])*rect[0] +     p[2]*rect[1]],
                    [stride*(p[1]),   stride*(1.0+p[3]), p[5] +     p[1]*rect[0] + (1.0+p[3])*rect[1]],
                    [   0.0,        0.0,                                         1.0]], dtype=np.float32)
    # Derived directly from perspective projection equations
    tmp = R_fc_to_c @ K_inv @ A_p

    K_cropped = np.ascontiguousarray(K[0:2, :])
    top = K_cropped @ tmp
    bot = tmp[2, :]

    rot_times_affine = np.vstack((top, np.atleast_2d(bot)))

    return rot_times_affine
make_rot_times_affine = nb.jit(nopython = True, cache = True, fastmath=True)(make_rot_times_affine)

def nb_norm(p, delta_p_stop):
    return np.linalg.norm(p) < delta_p_stop
nb_norm = nb.jit(nopython = True, cache = True, fastmath=True)(nb_norm)

def affineLKTracker(frame,
                    T_considered,
                    rect,
                    region_size,
                    p,
                    K, K_inv,
                    R_fc_to_c,
                    dW_dp_x_mult=None, dW_dp_y_mult=None,
                    stride=1,
                    inverse=True,
                    grad_T_dW=None,
                    H_T_dW_inv=None):
    #if dW_dp_x_mult is None or dW_dp_y_mult is None:
    #    dW_dp_x_mult, dW_dp_y_mult = calculate_dW_dp_mults(rect, region_size, stride)
    #start = time.time()
    rot_times_affine = make_rot_times_affine(stride, p, rect, R_fc_to_c, K, K_inv)
    I_warped_back_considered = cv2.warpPerspective(frame, rot_times_affine, region_size, flags=cv2.WARP_INVERSE_MAP+cv2.INTER_LINEAR)

    #mid = time.time()

    #bit_start = time.time()

    if not inverse:
        #bit_start = time.time()

        normalization_factor = 0.125 # 1/8 for sobel of size 3
        sobel_x = cv2.Sobel(I_warped_back_considered, cv2.CV_32F, 1, 0, ksize=3, scale=normalization_factor).flatten()
        sobel_y = cv2.Sobel(I_warped_back_considered, cv2.CV_32F, 0, 1, ksize=3, scale=normalization_factor).flatten()


        x_mult_sobel_x = dW_dp_x_mult * sobel_x
        x_mult_sobel_y = dW_dp_x_mult * sobel_y
        y_mult_sobel_x = dW_dp_y_mult * sobel_x
        y_mult_sobel_y = dW_dp_y_mult * sobel_y
        # vstack and transpose is faster than column stack
        grad_I_dW = np.vstack((x_mult_sobel_x, x_mult_sobel_y, y_mult_sobel_x, y_mult_sobel_y, sobel_x, sobel_y)).transpose()

        # TODO this may not be exploiting symmetry, is almost half the computation time
        H = np.einsum('ij,ik->jk', grad_I_dW, grad_I_dW, optimize=True)

        grad_J_to_p = diff_T_I_shaped @ grad_I_dW

        delta_p = np.linalg.solve(H, grad_J_to_p).reshape((6,))

        #bit_end = time.time()
    else:
        #bit_start = time.time()



        #delta_p_inv = np.sum(diff_T_I_shaped) * (H_T_dW_inv @ grad_T_dW)
        #print(delta_p_inv.shape)

        # Negative because we used T - I instead of I - T as in LK 20 years on
        #bit_end = time.time()

        delta_p, p = invert_delta_p(p, T_considered, I_warped_back_considered, grad_T_dW, H_T_dW_inv)
        #bit_end = time.time()

    #end = time.time()
    #print('end {:.2f} bit {:.2f} mid {:.2f}'.format(1000000*(end-bit_end), 1000000*(bit_end-mid), 1000000*(mid-start)))

    return delta_p, p, I_warped_back_considered


def draw_warped_patch_location(frame, rect, p, q_c_to_fc, K):
    points = np.array([
        (rect[0], rect[1], 1),
        (rect[2], rect[1], 1),
        (rect[2], rect[3], 1),
        (rect[0], rect[3], 1)
    ])

    A_p = np.array([[1+p[0],   p[2], p[4]],
                    [p[1],   1+p[3], p[5]],
                    [   0,        0,   1]], dtype=np.float32)

    # Derived directly from perspective projection equations
    R_fc_to_c = R.from_quat([q_c_to_fc[1], q_c_to_fc[2], q_c_to_fc[3], q_c_to_fc[0]]).as_matrix().astype(np.float32).transpose()
    K_inv = np.linalg.inv(K)

    tmp = R_fc_to_c @ K_inv @ A_p
    rot_times_affine = np.vstack((K[0:2, :] @ tmp, tmp[2, :]))

    points_warped = []
    for point in points:
        point = rot_times_affine @ point
        px = point[0]/point[2]
        py = point[1]/point[2]
        points_warped.append([px, py])
    points_warped = np.array(points_warped)

    points_warped = points_warped.reshape((-1, 1, 2)).astype(np.int32)

    cv2.polylines(frame, [points_warped], isClosed=True, color=255, thickness=2)

def draw_derotated(frame, q_c_to_fc, K):
    R_fc_to_c = R.from_quat([q_c_to_fc[1], q_c_to_fc[2], q_c_to_fc[3], q_c_to_fc[0]]).as_matrix().astype(np.float32).transpose()
    K_inv = np.linalg.inv(K)

    tmp = R_fc_to_c @ K_inv
    rot = np.vstack((K[0:2, :] @ tmp, tmp[2, :]))
    frame_derotated = cv2.warpPerspective(frame, rot, (frame.shape[1], frame.shape[0]), flags=cv2.WARP_INVERSE_MAP+cv2.INTER_LINEAR)
    return frame_derotated

def draw_full_reverse_warp(frame, rect, p, q_c_to_fc, K):
    R_fc_to_c = R.from_quat([q_c_to_fc[1], q_c_to_fc[2], q_c_to_fc[3], q_c_to_fc[0]]).as_matrix().astype(np.float32).transpose()
    K_inv = np.linalg.inv(K)

    # Affine matrix parameterized by p
    A_p = np.array([[1+p[0],   p[2], p[4]],
                    [p[1],   1+p[3], p[5]],
                    [   0,        0,   1]], dtype=np.float32)

    # Derived directly from perspective projection equations
    tmp = R_fc_to_c @ K_inv @ A_p
    rot_times_affine = np.vstack((K[0:2, :] @ tmp, tmp[2, :]))

    frame_warped_back = cv2.warpPerspective(frame, rot_times_affine, (frame.shape[1], frame.shape[0]), flags=cv2.WARP_INVERSE_MAP+cv2.INTER_LINEAR)

    points = np.array([
        (rect[0], rect[1]),
        (rect[2], rect[1]),
        (rect[2], rect[3]),
        (rect[0], rect[3])
    ])
    points = points.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(frame_warped_back, [points], isClosed=True, color=(0, 255, 0), thickness=1)

    return frame_warped_back

class AffineTrackRotInvariant:
    def __init__(self,
                 patch_coordinates,
                 template_image,
                 template_q_c_to_fc,
                 K,
                 delta_p_stop=0.1,
                 delta_p_mult=1.0,
                 visualize=False,
                 visualize_verbose=False,
                 wait_key=0,
                 stride=1.0,
                 inverse=True,
                 max_update_time=None):
        self._patch_coordinates = patch_coordinates
        self._K = K.astype(np.float32)
        self._K_inv = np.linalg.inv(K).astype(np.float32)
        self._delta_p_stop = delta_p_stop
        self._delta_p_mult = delta_p_mult
        self._stride = stride
        self._inverse = inverse
        self._max_update_time = max_update_time

        self._visualize = visualize
        self._visualize_verbose = visualize_verbose
        self._wait_key = wait_key

        if template_image is not None:
            self.set_template(patch_coordinates, template_image, template_q_c_to_fc)

    def set_template(self, patch_coordinates, template_image, template_q_c_to_fc):
        dW_dp_x_mult, dW_dp_y_mult = calculate_dW_dp_mults(patch_coordinates, self._stride)
        self._dW_dp_x_mult = dW_dp_x_mult
        self._dW_dp_y_mult = dW_dp_y_mult

        self._patch_coordinates = patch_coordinates
        self._template_image = template_image
        self._template_q_c_to_fc = template_q_c_to_fc

        # Get the image in the fixed orientation frame
        template_image_derotated = derotate_image(self._template_image, self._K, template_q_c_to_fc)

        self._template_image_derotated = template_image_derotated[self._patch_coordinates[1]:self._patch_coordinates[3], self._patch_coordinates[0]:self._patch_coordinates[2]]

        size_y = int((self._patch_coordinates[3] - self._patch_coordinates[1])/self._stride)
        size_x = int((self._patch_coordinates[2] - self._patch_coordinates[0])/self._stride)

        self._template_image_derotated = cv2.resize(self._template_image_derotated, (size_x, size_y))

        if self._inverse:
            normalization_factor = 0.125 # 1/8 for sobel of size 3
            sobel_x = cv2.Sobel(self._template_image_derotated, cv2.CV_32F, 1, 0, ksize=3, scale=normalization_factor).flatten()
            sobel_y = cv2.Sobel(self._template_image_derotated, cv2.CV_32F, 0, 1, ksize=3, scale=normalization_factor).flatten()

            x_mult_sobel_x = self._dW_dp_x_mult * sobel_x
            x_mult_sobel_y = self._dW_dp_x_mult * sobel_y
            y_mult_sobel_x = self._dW_dp_y_mult * sobel_x
            y_mult_sobel_y = self._dW_dp_y_mult * sobel_y
            # vstack and transpose is faster than column stack
            self._grad_T_dW = np.vstack((x_mult_sobel_x, x_mult_sobel_y, y_mult_sobel_x, y_mult_sobel_y, sobel_x, sobel_y)).transpose().astype(np.float32)

            # TODO this may not be exploiting symmetry, is almost half the computation time
            self._H_T_dW_inv = np.linalg.inv(np.einsum('ij,ik->jk', self._grad_T_dW, self._grad_T_dW, optimize=True)).astype(np.float32)
        else:
            self._grad_T_dW = None
            self._H_T_dW_inv = None

    def update(self, p, frame_gray, R_fc_to_c):
        steps = 0

        rect = self._patch_coordinates
        region_size = (int((rect[2] - rect[0])/self._stride), int((rect[3] - rect[1])/self._stride))

        p, I_warped_back_considered = affine_flow_loop(
             frame_gray,
             self._template_image_derotated,
             self._patch_coordinates,
             region_size,
             p,
             self._K,
             self._K_inv,
             R_fc_to_c,
             dW_dp_x_mult=self._dW_dp_x_mult,
             dW_dp_y_mult=self._dW_dp_y_mult,
             stride=self._stride,
             inverse=self._inverse,
             grad_T_dW=self._grad_T_dW,
             H_T_dW_inv=self._H_T_dW_inv,
             delta_p_stop = self._delta_p_stop,
             max_update_time = self._max_update_time)

        if self._visualize:
            cv2.imshow('affine flow progress', np.hstack((I_warped_back_considered, self._template_image_derotated)))
            #cv2.waitKey(self._wait_key)

        #print('Update time {} steps {} seconds'.format(steps, t_current - t_start))
        return p

def affine_flow_loop(frame,
                    T_considered,
                    rect,
                    region_size,
                    p,
                    K, K_inv,
                    R_fc_to_c,
                    dW_dp_x_mult=None, dW_dp_y_mult=None,
                    stride=1,
                    inverse=True,
                    grad_T_dW=None,
                    H_T_dW_inv=None,
                    delta_p_stop=None,
                    max_update_time=None):
    steps = 0
    t_start = time.time()
    old_p = p
    while True:
        #start = time.time()
        delta_p, p, I_warped_back_considered = affineLKTracker(
                         frame,
                         T_considered,
                         rect,
                         region_size,
                         p,
                         K,
                         K_inv,
                         R_fc_to_c,
                         dW_dp_x_mult,
                         dW_dp_y_mult,
                         stride,
                         inverse,
                         grad_T_dW,
                         H_T_dW_inv)

        #p = compose_warp(p, delta_p)

        #if self._visualize_verbose:
        #    cv2.imshow('affine flow progress verbose', np.hstack((I_warped_back_considered, self._template_image_derotated)))
            #cv2.waitKey(self._wait_key)

        #mid = time.time()
        steps += 1
        t_current = time.time()

        #if np.linalg.norm(delta_p) < delta_p_stop:
        #if nb_norm(delta_p) < delta_p_stop:
        if not nb_norm(p, 1e4): # sanity
            p = old_p
            break
        if nb_norm(delta_p, delta_p_stop):
            #print('Small delta_p {} steps {:.01f} hz'.format(steps, steps/(t_current - t_start)))
            break
        #end = time.time()
        #print(end-mid, mid-start)

        
        if max_update_time is not None:
            if t_current - t_start > max_update_time:
                print('Max update time {} steps {} seconds'.format(steps, t_current - t_start))
                break
    return p, I_warped_back_considered
