###############################################################################
#
# File: robomaster_pickle_mat.py
# Available under MIT license
#
# Export recordings made by ttc_depth_robomaster.py to Matlab .mat files
# for plotting
#
# History:
# 09-26-22 - Levi Burner - Open source release
#
###############################################################################

import pickle
from scipy.io import savemat
import os
import glob
import numpy as np

BASE_RESULTS_PATH = '../robomaster_recordings'

if __name__ == '__main__':
    recordings = sorted(glob.glob(os.path.join(BASE_RESULTS_PATH, 'record_*')))

    for recording in recordings:
        print('Processing: {}'.format(recording))
        with open(os.path.join(recording, 'imu.pickle'), 'rb') as f:
            p = pickle.load(f)

            dictionary = {}
            for key in p:
                dictionary[key] = np.array(p[key])

            savemat(os.path.join(recording, 'imu.mat'), dictionary)