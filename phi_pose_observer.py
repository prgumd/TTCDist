###############################################################################
#
# File: phi_pose_observer.py
# Available under MIT license
#
# Position observer from phi and acceleration
#
# History:
# 08-18-22 - Levi Burner - Created file
# 09-26-22 - Levi Burner - Open source release
#
###############################################################################

import numpy as np
from phi_depth_integrals_accel_bias import accel_x_phi_constraint_tf, accel_z_phi_constraint_tf

def feedback_good(accel_power, accel_z, max_z, accel_power_thresh):
    return (accel_z != np.NAN and accel_power > accel_power_thresh and accel_z < max_z)

class PhiPoseObserver(object):
    def __init__(self,
                 z_0=None,
                 z_0_dot=0.0,
                 dt=0.005,
                 max_z=-0.1,
                 z_hat_gain=2.0,
                 seconds_to_keep = 1.0,
                 accel_power_thresh = 2.0):
        self._dt = dt
        self._z_0 = None
        self._z_hat = self._z_0
        self._max_z = max_z
        self._min_initial_z = -7.5

        self._seconds_to_keep = seconds_to_keep
        self._num_samples_to_keep = int(self._seconds_to_keep / self._dt)

        self._t_list = []
        self._phi_list = []
        self._a_fc_list = []

        self._A = np.array(((0, 1), (0, 0)))
        self._B = np.array((0, 1))
        self._L = np.array(((2, 0), (0, 20)))

        self._accel_power_thresh = accel_power_thresh

    def reset_ic(self):
        self._z_hat = self._z_0

    def update(self, t, phi, a_fc):
        self._t_list.append(t)
        self._phi_list.append(phi.tolist())
        self._a_fc_list.append(a_fc.tolist())

        accel_z_bias = None
        if len(self._t_list) == self._num_samples_to_keep:
            self._t_list = self._t_list[1:]
            self._phi_list = self._phi_list[1:]
            self._a_fc_list = self._a_fc_list[1:]

            # TODO NO!
            t = np.array(self._t_list)

            phis = np.array(self._phi_list)

            phis_x = phis[:, 0]
            phis_y = phis[:, 1]
            phis_z = phis[:, 2]

            a_fc = np.array(self._a_fc_list)
            a_x = a_fc[:, 0]
            a_y = a_fc[:, 1]
            a_z = a_fc[:, 2]

            (accel_x_z_tf, accel_x_dz_tf, accel_x_bias), x_res = accel_x_phi_constraint_tf(t, phis_x, phis_z, a_x)
            if accel_x_z_tf != np.NAN:
                a_x_power = np.linalg.norm(a_x-accel_x_bias) / np.sqrt(self._num_samples_to_keep)
            else:
                a_x_power = 0.0
                print('x singular')

            (accel_y_z_tf, accel_y_dz_tf, accel_y_bias), y_res = accel_x_phi_constraint_tf(t, phis_y, phis_z, a_y)
            if accel_y_z_tf != np.NAN:
                a_y_power = np.linalg.norm(a_y-accel_y_bias) / np.sqrt(self._num_samples_to_keep)
            else:
                a_y_power = 0.0
                print('y singular')

            (accel_z_z_tf, accel_z_dz_tf, accel_z_bias), z_res = accel_z_phi_constraint_tf(t, phis_z, a_z)
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
        if feedback_good(a_x_power, accel_x_z_tf, self._max_z, self._accel_power_thresh):
            accel_z_hat     += accel_x_z_tf
            accel_z_dot_hat += accel_x_dz_tf
            num_feedback    += 1
        else:
            accel_x_z_tf = 0.0

        if feedback_good(a_y_power, accel_y_z_tf, self._max_z, self._accel_power_thresh):
            accel_z_hat     += accel_y_z_tf
            accel_z_dot_hat += accel_y_dz_tf
            num_feedback    += 1
        else:
            accel_y_z_tf = 0.0

        if feedback_good(a_z_power, accel_z_z_tf, self._max_z, self._accel_power_thresh):
            accel_z_hat     += accel_z_z_tf
            accel_z_dot_hat += accel_z_dz_tf
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
            new_z  = self._z_hat[0] * phi[2] / phis[-2, 2]
            new_dz = (new_z - self._z_hat[0]) / self._dt
            self._z_hat = np.array((new_z, new_dz))
        else:
            return None

        if self._z_hat[0] > self._max_z:
            self._z_hat[0] = self._max_z

        return self._z_hat[0], accel_x_z_tf, accel_y_z_tf, accel_z_z_tf

