###############################################################################
#
# File: utilities.py
# Available under MIT license
#
# Utility functions used by multiple files
#
# History:
# 04-23-20 - Levi Burner - Created file
# 09-26-22 - Levi Burner - Open source release
#
###############################################################################

import glob
import os
import pickle

import cv2
import numpy as np

def latest_recording_dir():
    datasets = sorted(glob.glob('../recordings/record_*/'))
    if len(datasets) == 0:
        raise Exception('No data directories in default location and no directory specified')
    else:
        dataset_dir = datasets[-1]
    return dataset_dir

def load_calibration_from_dir(record_dir):
    calibration = pickle.load(open(os.path.join(record_dir, 'intrinsics.pickle'), 'rb'))
    resolution = calibration['resolution']
    K = calibration['K']
    D = calibration['D']

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, resolution, cv2.CV_16SC2)

    return K, D, map1, map2, resolution
