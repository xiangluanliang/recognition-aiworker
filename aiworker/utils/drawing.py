# aiworker/utils/drawing.py
import cv2
import numpy as np
from ..config import POSE_PAIRS


def draw_pose(frame, kpts, color=(0, 255, 0)):
    """在帧上绘制单人的姿态关键点和骨骼。"""
    for i, j in POSE_PAIRS:
        if i < len(kpts) and j < len(kpts):
            pt1 = (int(kpts[i][0]), int(kpts[i][1]))
            pt2 = (int(kpts[j][0]), int(kpts[j][1]))
            cv2.line(frame, pt1, pt2, color, 2)
    for point in kpts:
        x, y = int(point[0]), int(point[1])
        cv2.circle(frame, (x, y), 3, color, -1)


def draw_abnormal_zone(frame, zone_points_list):
    """在帧上绘制多个异常区域的多边形边框。"""
    for points in zone_points_list:
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

