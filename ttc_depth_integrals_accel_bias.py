###############################################################################
#
# File: ttc_depth_integrals_accel_bias.py
# Available under MIT license
#
# Estimate distance using the Tau constraint in the presence of
# a constant acceleration bias
#
# History:
# 09-26-22 - Levi Burner - Open source release
#
###############################################################################

import numpy as np
import numba as nb
import time

def cumulative_int(dt, x):
    return dt * np.cumsum(x)

# Calculate Phi
def Phi_t_t_0(dt, dot_z_over_z):
    int_dot_z_over_z = cumulative_int(dt, dot_z_over_z)
    phi = np.exp(int_dot_z_over_z)
    return phi

def accel_z_z_star_t0_no_inv(times, dot_z_over_z, accel_z):
    # TODO all times are needed but they are assumed to be evenly spaced
    #mid0 = time.time()
    dt = times[1] - times[0]

    int_accel  = cumulative_int(dt, accel_z)
    iint_accel = cumulative_int(dt, int_accel)

    phi = Phi_t_t_0(dt, dot_z_over_z)

    F_action = phi - (1 + ((times - times[0]) * dot_z_over_z[0]))

    times_squared = np.square(times-times[0])
    #total_time = (times[-1] - times[0])
    #int_times_squared_squared = (1./5.)*(total_time * total_time * total_time * total_time * total_time)

    a =           dt * np.sum(np.square(F_action))
    b =           dt * np.sum(times_squared*F_action)
    #c = (1./4.) * int_times_squared_squared
    c = (1./4.) * dt * np.sum(np.square(times_squared))

    d = -2.0    * dt * np.sum(iint_accel*F_action)
    e = -         dt * np.sum(iint_accel*times_squared)

    P = np.array(((a, b/2),
                  (b/2, c)))
    c = np.array((d, e))

    return P, c

def accel_z_z_star_t0(times, dot_z_over_z, accel_z):
    P, c = accel_z_z_star_t0_no_inv(times, dot_z_over_z, accel_z)

    try:
        z_star = np.linalg.solve(P, -0.5*c)
    except np.linalg.LinAlgError:
        z_star = (np.NAN, np.NAN)

    return z_star

def accel_z_z_star_tf(times, dot_z_over_z, accel_z):
    #start = time.time()
    z_star = accel_z_z_star_t0(np.flip(times), np.flip(dot_z_over_z), np.flip(accel_z))
    #end = time.time()
    #print(end-start)
    return z_star

def accel_x_z_star_t0_no_inv(times, dot_x_over_z, dot_z_over_z, accel_x):
    # TODO all times are needed but they are assumed to be evenly spaced
    #mid0 = time.time()
    dt = times[1] - times[0]

    int_accel  = cumulative_int(dt, accel_x)
    iint_accel = cumulative_int(dt, int_accel)

    phi = Phi_t_t_0(dt, dot_z_over_z)

    F_action = cumulative_int(dt, dot_x_over_z*phi) - ((times-times[0])*dot_x_over_z[0])

    times_squared = np.square(times-times[0])
    # TODO using the closed form makes the results worse...
    #int_times_squared_squared = (1./5.)*((times[-1] - times[0])**5)

    a =           dt * np.sum(np.square(F_action))
    b =           dt * np.sum(times_squared*F_action)
    #c = (1./4.) * int_times_squared_squared
    c = (1./4.) * dt * np.sum(np.square(times_squared))

    d = -2.0    * dt * np.sum(iint_accel*F_action)
    e = -         dt * np.sum(iint_accel*times_squared)

    P = np.array(((a, b/2),
                  (b/2, c)))
    c = np.array((d, e))

    return P, c

def accel_x_z_star_t0(times, dot_x_over_z, dot_z_over_z, accel_x):
    P, c = accel_x_z_star_t0_no_inv(times, dot_x_over_z, dot_z_over_z, accel_x)

    try:
        z_star = np.linalg.solve(P, -0.5*c)
    except np.linalg.LinAlgError:
        z_star = (np.NAN, np.NAN)

    return z_star

def accel_x_z_star_tf(times, dot_x_over_z, dot_z_over_z, accel_x):
    #start = time.time()
    z_star = accel_x_z_star_t0(np.flip(times), np.flip(dot_x_over_z), np.flip(dot_z_over_z), np.flip(accel_x))
    #end = time.time()
    #print(end-start)
    return z_star

# Setup numba
# No fast math because lots of the values are super tiny and might be sub-normal (not sure)
cumulative_int = nb.jit(nopython = True, cache = True, fastmath=False)(cumulative_int)
Phi_t_t_0 = nb.jit(nopython = True, cache = True, fastmath=False)(Phi_t_t_0)
accel_z_z_star_t0_no_inv = nb.jit(nopython = True, cache = True, fastmath=False)(accel_z_z_star_t0_no_inv)
# accel_z_z_star_tf = nb.jit(nopython = True, cache = True, fastmath=False)(accel_z_z_star_tf)
accel_x_z_star_t0_no_inv = nb.jit(nopython = True, cache = True, fastmath=False)(accel_x_z_star_t0_no_inv)
# accel_x_z_star_tf = nb.jit(nopython = True, cache = True, fastmath=False)(accel_x_z_star_tf)
