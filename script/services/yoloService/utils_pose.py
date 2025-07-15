#用来处理姿势

import numpy as np
import cv2
from .constants import POSE_PAIRS

def draw_pose(frame, kpts, color=(0, 255, 0)):
    for point in kpts:
        x, y = int(point[0]), int(point[1])
        cv2.circle(frame, (x, y), 3, color, -1)
    for i, j in POSE_PAIRS:
        if i < len(kpts) and j < len(kpts):
            pt1, pt2 = tuple(kpts[i]), tuple(kpts[j])
            cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, 2)

def get_center_point(kpts):
    x = np.mean(kpts[:, 0])
    y = np.mean(kpts[:, 1])
    return (int(x), int(y))

def angle_between_points(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
