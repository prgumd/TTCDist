###############################################################################
#
# File: phi_depth_integrals_accel_bias.py
# Available under MIT license
#
# Estimate distance using the Phi constraint in the presence of
# a constant acceleration bias
#
# History:
# 09-26-22 - Levi Burner - Open source release
#
###############################################################################

import numpy as np
import numba as nb

def cumulative_int(dt, x):
    return dt * np.cumsum(x)

def accel_z_phi_constraint(times, z_over_z0, accel_z):
    z_over_z0 = z_over_z0 / z_over_z0[0]

    # TODO all times are needed but they are assumed to be evenly spaced
    dt = times[1] - times[0]
    int_accel  = cumulative_int(dt, accel_z)
    iint_accel = cumulative_int(dt, int_accel)

    E = z_over_z0 - 1
    R = -(times - times[0])
    D1 = 0.5 * np.square(R)
    Da = iint_accel

    A = np.stack((E, R, D1), axis=1)
    b = Da.reshape((A.shape[0], 1))

    z, res, rank, s = np.linalg.lstsq(A, b, rcond=-1)
    return z.reshape((3,)), res

def accel_z_phi_constraint_tf(times, z_over_z0, accel_z):
    z_star, res = accel_z_phi_constraint(np.flip(times), np.flip(z_over_z0), np.flip(accel_z))
    return z_star, res

def accel_x_phi_constraint(times, x_over_z, z_over_z0, accel_x):
    dx_over_z0 = x_over_z * (z_over_z0 / z_over_z0[0])
    dx_over_z0 -= dx_over_z0[0]

    # TODO all times are needed but they are assumed to be evenly spaced
    dt = times[1] - times[0]
    int_accel  = cumulative_int(dt, accel_x)
    iint_accel = cumulative_int(dt, int_accel)

    P = dx_over_z0
    R = -(times - times[0])
    D1 = 0.5 * np.square(R)
    Da = iint_accel

    A = np.stack((P, R, D1), axis=1)
    b = Da.reshape((A.shape[0], 1))

    x, res, rank, s = np.linalg.lstsq(A, b, rcond=-1)

    return x.reshape((3,)), res

def accel_x_phi_constraint_tf(times, x_over_z, z_over_z0, accel_x):
    x_star, res = accel_x_phi_constraint(np.flip(times), np.flip(x_over_z), np.flip(z_over_z0), np.flip(accel_x))
    return x_star, res

# Setup numba
# TODO No fast math because lots of the values are super tiny and might be sub-normal (not sure)
cumulative_int = nb.jit(nopython = True, cache = True, fastmath=False)(cumulative_int)
accel_z_phi_constraint = nb.jit(nopython = True, cache = True, fastmath=False)(accel_z_phi_constraint)
accel_z_phi_constraint_tf = nb.jit(nopython = True, cache = True, fastmath=False)(accel_z_phi_constraint_tf)
accel_x_phi_constraint = nb.jit(nopython = True, cache = True, fastmath=False)(accel_x_phi_constraint)
accel_x_phi_constraint_tf = nb.jit(nopython = True, cache = True, fastmath=False)(accel_x_phi_constraint_tf)
