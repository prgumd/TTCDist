###############################################################################
#
# File: affine_flow_time_to_contact.py
# Available under MIT license
#
# Calculate time to contact from affine flow parameters
#
# History:
# 04-23-20 - Levi Burner - Created file
# 09-26-22 - Levi Burner - Open source release
#
###############################################################################

import numpy as np

class AffineFlowTimeToContactEstimator(object):
    def __init__(self, patch_coordinates, K, stride=None):
        self._patch_coordinates = patch_coordinates
        self._K = K
        self._K_inv = np.linalg.inv(self._K)
        self._patch_center_xy = np.array([(self._patch_coordinates[2] + self._patch_coordinates[0]) / 2.0,
                                          (self._patch_coordinates[3] + self._patch_coordinates[1]) / 2.0])
        self._X_0_ttc = np.array([self._patch_center_xy[0], self._patch_center_xy[1], 1])
        self._last_p = None

    def estimate_ttc(self, p):
        if self._last_p is None:
            self._last_p = p
            return False

        A_0 = self._K_inv @ np.array([[1+self._last_p[0], self._last_p[2], self._last_p[4]],
                                      [self._last_p[1], 1+self._last_p[3], self._last_p[5]],
                                      [   0,      0,     1]]) @ self._K

        A_1 = self._K_inv @ np.array([[1+p[0], p[2], p[4]],
                                      [p[1], 1+p[3], p[5]],
                                      [   0,      0,     1]]) @ self._K
        self._last_p = p

        try:
            # dp = I - A_0 A_1^-1
            dp = np.linalg.solve(A_1.transpose(), (A_1 - A_0).transpose()).transpose()
        except np.linalg.LinAlgError:
            return False

        a = np.array([0, 0, dp[0, 0], dp[0, 1], dp[0, 2], dp[1, 0], dp[1, 1], dp[1, 2]])

        # Select a point to return the ttc for
        X_ttc = A_1 @ self._K_inv @ self._X_0_ttc

        EPS_XY = 1E-3
        x_dot_over_z, y_dot_over_z, _ = a_to_frequency_of_contact(a, X_ttc, EPS_XY)

        EPS_Z = 2E-3
        _, _, z_dot_over_z = a_to_frequency_of_contact(a, X_ttc, EPS_Z)

        return x_dot_over_z, y_dot_over_z, z_dot_over_z, X_ttc

def a_to_frequency_of_contact(a, X_ttc, EPS):
    if abs(a[4]) < EPS and abs(a[7]) < EPS: # Don't divide by a[4], a[7], assume U,V=0
        W_c =  (a[2] + a[6]) / 2
        U_a = -(a[2] - W_c)
        V_b = -(a[6] - W_c)

        W_a = 0
        W_b = 0

    elif abs(a[4]) < EPS: # Don't divide by a[4], assume U=0
        U_a = -a[5]*(a[4]/a[7])
        W_c =  a[2] + U_a
        V_b = -(a[6] - W_c)

        W_a = W_c * a[5] / a[7]
        W_b = 0

    elif abs(a[7]) < EPS: # Don't divide by a[7], assume V=0
        V_b = -a[3]*(a[7]/a[4])
        W_c =  a[6] + V_b
        U_a = -(a[2] - W_c)

        W_a = 0
        W_b = W_c * a[3] / a[4]

    else:
        U_a = -a[5]*(a[4]/a[7])
        V_b = -a[3]*(a[7]/a[4])
        W_c = ((a[2] + U_a) + (a[6] + V_b)) / 2

        W_a = W_c * a[5] / a[7]
        W_b = W_c * a[3] / a[4]

    x_dot_over_z =   U_a * X_ttc[0] - a[3] * X_ttc[1] - a[4]
    y_dot_over_z = -a[5] * X_ttc[0] +  V_b * X_ttc[1] - a[7]
    z_dot_over_z =  W_a * X_ttc[0] + W_b * X_ttc[1] + W_c

    return x_dot_over_z, y_dot_over_z, z_dot_over_z
