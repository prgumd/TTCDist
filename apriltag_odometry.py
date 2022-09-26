###############################################################################
#
# File: apriltag_odometry.py
# Available under MIT license
#
# Find an apriltag in an image and estimate pose relative to it
#
# History:
# 04-23-20 - Levi Burner - Created file
# 09-26-22 - Levi Burner - Open source release
#
###############################################################################

import cv2
import numpy as np

from apriltag import apriltag

class AprilPose(object):
    def __init__(self, K, family='tagStandard41h12', marker_size_m=(4.5 * 2.54 / 100)):
        # Hard code a bunch of defaults for now, they are not important
        family = family
        threads = 8
        max_hamming = 0
        decimate = 1
        blur = 0.8
        refine_edges = False
        debug = False

        # Camera intrinsics
        self._K = K

        self._detector = apriltag(family, threads, max_hamming, decimate, blur, refine_edges, debug)

        m_half_size = marker_size_m / 2

        marker_center = np.array((0, 0, 0))
        marker_points = []
        marker_points.append(marker_center + (-m_half_size, m_half_size, 0))
        marker_points.append(marker_center + ( m_half_size, m_half_size, 0))
        marker_points.append(marker_center + ( m_half_size, -m_half_size, 0))
        marker_points.append(marker_center + (-m_half_size, -m_half_size, 0))
        self._marker_points = np.array(marker_points)

    def find_tags(self, frame_gray):
        detections = self._detector.detect(frame_gray)
        return detections

    def find_detection(self, detections, id):
        for (i, detection) in enumerate(detections):
            if detection['id'] == id:
                return detection
        return None

    def find_pose_from_tag(self, detection):
        object_points = self._marker_points
        image_points = detection['lb-rb-rt-lt']

        pnp_ret = cv2.solvePnP(object_points, image_points, self._K, distCoeffs=None,flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if pnp_ret[0] == False:
            raise Exception('Error solving PnP')

        r = pnp_ret[1]
        p = pnp_ret[2]

        return p.reshape((3,)), r.reshape((3,))

    def draw_detections(self, frame, detections):
        for detection in detections:
            pts = detection['lb-rb-rt-lt'].reshape((-1, 1, 2)).astype(np.int32)
            frame = cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=5)
            cv2.circle(frame, tuple(detection['center'].astype(np.int32)), 5, (0, 0, 255), -1)
