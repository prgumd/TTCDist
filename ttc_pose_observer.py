###############################################################################
#
# File: ttc_pose_observer.py
# Available under MIT license
#
# Position observer from phi and acceleration
#
# History:
# 04-26-20 - Levi Burner - Created file
# 09-26-22 - Levi Burner - Open source release
#
###############################################################################

import time
import numpy as np
from ttc_depth_integrals_accel_bias import accel_z_z_star_tf, accel_x_z_star_tf

def feedback_good(accel_power, accel_z, max_z):
    return (accel_z != np.NAN and accel_power > 2.0 and accel_z < max_z)

class TTCPoseObserver(object):
    def __init__(self,
                 z_0=None,
                 z_0_dot=0.0,
                 dt=0.005,
                 max_z=-0.1,
                 z_hat_gain=2.0,
                 seconds_to_keep = 1.0):
        self._dt = dt
        self._z_0 = None
        self._z_hat = self._z_0
        self._max_z = max_z
        self._min_initial_z = -7.5

        self._seconds_to_keep = seconds_to_keep
        self._num_samples_to_keep = int(self._seconds_to_keep / self._dt)

        self._t_list = []
        self._scaled_velocities_list = []
        self._a_fc_list = []

        self._A = np.array(((0, 1), (0, 0)))
        self._B = np.array((0, 1))
        self._L = np.array(((2, 0), (0, 20)))

    def reset_ic(self):
        self._z_hat = self._z_0

    def update(self, t, scaled_velocities, a_fc):
        self._t_list.append(t)
        self._scaled_velocities_list.append(scaled_velocities.tolist())
        self._a_fc_list.append(a_fc.tolist())

        accel_z_bias = None
        if len(self._t_list) == self._num_samples_to_keep:
            self._t_list = self._t_list[1:]
            self._scaled_velocities_list = self._scaled_velocities_list[1:]
            self._a_fc_list = self._a_fc_list[1:]

            # TODO NO!
            t = np.array(self._t_list)

            s_v = np.array(self._scaled_velocities_list)

            s_v_x = s_v[:, 0]
            s_v_y = s_v[:, 1]
            s_v_z = s_v[:, 2]

            a_fc = np.array(self._a_fc_list)
            a_x = a_fc[:, 0]
            a_y = a_fc[:, 1]
            a_z = a_fc[:, 2]

            (accel_x_z_tf, accel_x_bias) = accel_x_z_star_tf(t, s_v_x, s_v_z, a_x)
            if accel_x_z_tf != np.NAN:
                a_x_power = np.linalg.norm(a_x-accel_x_bias) / np.sqrt(self._num_samples_to_keep)
            else:
                a_x_power = 0.0
                print('x singular')

            (accel_y_z_tf, accel_y_bias) = accel_x_z_star_tf(t, s_v_y, s_v_z, a_y)
            if accel_y_z_tf != np.NAN:
                a_y_power = np.linalg.norm(a_y-accel_y_bias) / np.sqrt(self._num_samples_to_keep)
            else:
                a_y_power = 0.0
                print('y singular')

            (accel_z_z_tf, accel_z_bias) = accel_z_z_star_tf(t, s_v_z, a_z)
            if accel_z_z_tf != np.NAN:
                a_z_power = np.linalg.norm(a_z-accel_z_bias) / np.sqrt(self._num_samples_to_keep)
            else:
                a_z_power = 0.0
                print('z singular')
        else:
            return None

        accel_z_hat = 0.0
        accel_z_dot_hat = 0.0
        num_feedback = 0
        if feedback_good(a_x_power, accel_x_z_tf, self._max_z):
            accel_z_hat     += accel_x_z_tf
            accel_z_dot_hat += scaled_velocities[2] * accel_x_z_tf
            num_feedback    += 1
        else:
            accel_x_z_tf = 0.0

        if feedback_good(a_y_power, accel_y_z_tf, self._max_z):
            accel_z_hat     += accel_y_z_tf
            accel_z_dot_hat += scaled_velocities[2] * accel_y_z_tf
            num_feedback    += 1
        else:
            accel_y_z_tf = 0.0

        if feedback_good(a_z_power, accel_z_z_tf, self._max_z):
            accel_z_hat     += accel_z_z_tf
            accel_z_dot_hat += scaled_velocities[2] * accel_z_z_tf
            num_feedback    += 1
        else:
            accel_z_z_tf = 0.0

        if num_feedback > 0 and accel_z_bias is not None and self._z_hat is not None:
            accel_z_hat /= num_feedback
            accel_z_dot_hat /= num_feedback

            e = np.array((accel_z_hat, accel_z_dot_hat)) - self._z_hat

            z_hat_dot = self._A @ self._z_hat + self._B * (a_z[-1]-accel_z_bias) + self._L @ e

            self._z_hat = self._z_hat + z_hat_dot * self._dt
        elif num_feedback > 0 and self._z_hat is None:
            accel_z_hat /= num_feedback
            accel_z_dot_hat /= num_feedback

            # _min_initial_z is to filter an insane spike at the beginning of one sequence
            if accel_z_hat > self._min_initial_z:
                self._z_hat = np.array((accel_z_hat, accel_z_dot_hat))
            else:
                return None

        elif self._z_hat is not None:
            s_v_z_hat_dot = scaled_velocities[2] * self._z_hat[0]
            self._z_hat = np.array((self._z_hat[0] + s_v_z_hat_dot * self._dt,
                                    s_v_z_hat_dot))
        else:
            return None

        if self._z_hat[0] > self._max_z:
            self._z_hat[0] = self._max_z

        return self._z_hat[0], accel_x_z_tf, accel_y_z_tf, accel_z_z_tf
