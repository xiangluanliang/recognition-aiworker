# aiworker/yolo/yolo_detector.py
import numpy as np
from ultralytics import YOLO
import os
import logging

from ..config import MODEL_DIR


class YoloDetector:
    """
    一个封装了YOLOv8模型的检测器。
    初始化时传入模型文件名。
    """

    def __init__(self, weights_filename: str):
        self.logger = logging.getLogger(__name__)
        model_path = os.path.join(MODEL_DIR, weights_filename)
        try:
            self.model = YOLO(model_path)
            self.logger.info(f"YOLO model loaded successfully from {model_path}")
        except Exception as e:
            self.logger.critical(f"Failed to load YOLO model: {e}")
            self.model = None

    def _get_center_point(self, kpts: np.ndarray) -> tuple[int, int]:
        x = np.mean(kpts[:, 0])
        y = np.mean(kpts[:, 1])
        return int(x), int(y)

    def detect_people(self, frame: np.ndarray) -> tuple[list, list, list]:
        if self.model is None:
            self.logger.warning("YOLO model not loaded, cannot perform detection.")
            return [], [], []

        results = self.model(frame, verbose=False)
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


