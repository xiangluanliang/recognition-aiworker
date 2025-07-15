# aiworker/yolo/yolo_detector.py
import numpy as np
from ultralytics import YOLO
import os
import logging

from ..config import MODEL_DIR, YOLO_POSE_MODEL_FILENAME

class YoloDetector:
    """
    一个封装了YOLOv8姿态估计模型的检测器。
    这是一个重量级对象，建议在服务启动时只初始化一次。
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        model_path = os.path.join(MODEL_DIR, YOLO_POSE_MODEL_FILENAME)
        try:
            self.pose_model = YOLO(model_path)
            self.logger.info(f"YOLO pose model loaded successfully from {model_path}")
        except Exception as e:
            self.logger.critical(f"Failed to load YOLO pose model: {e}")
            self.pose_model = None

    def _get_center_point(self, kpts: np.ndarray) -> tuple[int, int]:
        """计算一组关键点的几何中心。"""
        x = np.mean(kpts[:, 0])
        y = np.mean(kpts[:, 1])
        return (int(x), int(y))

    def detect_people(self, frame: np.ndarray) -> tuple[list, list, list]:
        """
        在给定的帧上检测所有人。

        Returns:
            A tuple containing:
            - kpts_list (list): 每个检测到的人的17个关键点 [17, 2]。
            - centers (list): 每个人的中心点坐标。
            - confidences (list): 每个人的检测置信度。
        """
        if self.pose_model is None:
            self.logger.warning("YOLO model not loaded, cannot perform detection.")
            return [], [], []

        results = self.pose_model(frame, verbose=False)  # verbose=False to suppress console output
        kpts_list, centers, confidences = [], [], []

        for r in results:
            if r.keypoints is None or r.boxes is None:
                continue

            keypoints_xy = r.keypoints.xy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            for i in range(len(keypoints_xy)):
                pts = keypoints_xy[i]
                conf = float(confs[i])
                kpts_list.append(pts)
                centers.append(self._get_center_point(pts))
                confidences.append(conf)

        return kpts_list, centers, confidences
