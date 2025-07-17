# aiworker/yolo/behavior_processor.py
import cv2
import os
import numpy as np
from collections import defaultdict, deque
import time
import logging

# --- 导入所有解耦后的独立模块 ---
from .yolo_detector import YoloDetector
from .logic_tracker import match_person_id
from .event_checkers import check_fall, check_intrusion, detect_fight
from ..services.api_client import fetch_warning_zones_for_camera, log_event
from ..utils.drawing import draw_pose, draw_abnormal_zone
from ..utils.file_saver import save_clip, save_event_image
from ..config import *


class AbnormalBehaviorProcessor:
    """
    一个有状态的处理器，用于检测单个视频流中的异常行为。
    这个类的每个实例对应一个独立的视频流，并管理其所有状态。
    """
    def __init__(self, camera_id: int, yolo_detector: YoloDetector, fps: int, enabled_detectors: list[str]):
        self.logger = logging.getLogger(__name__)
        self.camera_id = camera_id
        self.yolo_detector = yolo_detector
        self.fps = fps
        self.active_detectors = set(enabled_detectors)
        self.logger.info(f"处理器为摄像头 {camera_id} 初始化，启用的功能: {self.active_detectors}")
        self.frame_idx = 0
        self.prev_centers = {}
        self.video_buffer = deque(maxlen=int(self.fps * CLIP_DURATION_SECONDS))
        self.person_history = defaultdict(list)
        self.person_fall_status = defaultdict(
            lambda: {'fall_frame_count': 0, 'is_falling': False, 'cooldown_counter': 0})
        self.zone_status_cache = defaultdict(dict)
        self.recorded_intrusions = set()
        self.recorded_conflicts = set()
        self.fight_kpts_history = defaultdict(lambda: deque(maxlen=5))  # 打架检测需要最近5帧的姿态
        self.prev_centers_history = defaultdict(lambda: deque(maxlen=5))  # 历史中心点，计算速度加速度等

        self.warning_zones = []

        if 'intrusion_detection' in self.active_detectors:
            raw_zones_data = fetch_warning_zones_for_camera(self.camera_id)

            # 优化：解析每个区域的独立配置
            for zone_data in raw_zones_data:
                polygon = [[pt['x'], pt['y']] for pt in zone_data.get('zone_points', [])]
                if not polygon:
                    continue

                # 为每个区域单独计算和存储其配置
                stay_seconds = zone_data.get('safe_time', DEFAULT_STAY_SECONDS)
                self.warning_zones.append({
                    'id': zone_data.get('id'),
                    'name': zone_data.get('name', '未命名区域'),
                    'polygon': polygon,
                    'stay_frames': int(self.fps * stay_seconds),
                    'safe_dist': zone_data.get('safe_distance', DEFAULT_SAFE_DISTANCE)
                })

            self.logger.info(f"摄像头 {self.camera_id} 加载了 {len(self.warning_zones)} 个处理后的警戒区域。")

        self.logger.info(
            f"Processor for camera {camera_id} initialized with {len(self.warning_zones[self.camera_id])} zones.")

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        处理单帧图像，执行完整的检测、追踪、并根据配置动态判断、绘制流程。
        """
        self.frame_idx += 1
        processed_frame = frame.copy()

        kpts_list, centers, confidences = self.yolo_detector.detect_people(processed_frame)
        if not kpts_list:
            return processed_frame, {}

        ids = match_person_id(centers, self.prev_centers, PERSON_MATCHING_THRESHOLD)
        self.prev_centers = {pid: center for pid, center in zip(ids, centers)}

        all_event_pids = set()

        if 'fight_detection' in self.active_detectors:
            for i, kpts in enumerate(kpts_list):
                self.fight_kpts_history[ids[i]].append(kpts.copy())

            conflict_pairs = detect_fight(
                ids, centers, self.fight_kpts_history,
                FIGHT_DISTANCE_THRESHOLD, FIGHT_MOTION_THRESHOLD, FIGHT_ORIENTATION_SIMILARITY_THRESHOLD
            )
            for pid1, pid2 in conflict_pairs:
                for pid in [pid1, pid2]:
                    all_event_pids.add(pid)
                    if (pid, int(time.time()) // 10) not in self.recorded_conflicts:
                        self.recorded_conflicts.add((pid, int(time.time()) // 10))
                        self._log_event('conflict', pid, 0.99, frame)

        for i, (pid, kpts) in enumerate(zip(ids, kpts_list)):
            is_fall_event = False
            is_intruding_event = False

            x1, y1 = np.min(kpts[:, 0]), np.min(kpts[:, 1])
            x2, y2 = np.max(kpts[:, 0]), np.max(kpts[:, 1])
            bbox = (int(x1), int(y1), int(x2), int(y2))

            if 'fall_detection' in self.active_detectors:
                is_fall_event, is_new_fall, score = check_fall(
                    pid, kpts, bbox, self.person_fall_status,
                    FALL_ANGLE_THRESHOLD, FALL_WINDOW_SIZE, FALL_COOLDOWN_FRAMES
                )
                if is_new_fall:
                    all_event_pids.add(pid)
                    self._log_event('person_fall', pid, score, frame)

            if 'intrusion_detection' in self.active_detectors:
                is_intruding, new_intrusion_zones_info = check_intrusion(
                    pid, bbox, centers[i],
                    self.warning_zones,
                    self.recorded_intrusions, self.zone_status_cache, self.frame_idx
                )
                for zone_info in new_intrusion_zones_info:
                    all_event_pids.add(pid)
                    self._log_event('intrusion', pid, confidences[i], frame,
                                    details={'zone_id': zone_info['id'], 'zone_name': zone_info['name']})

            is_in_any_event = pid in all_event_pids or is_fall_event or is_intruding_event
            color = (0, 0, 255) if is_in_any_event else (0, 255, 0)

            draw_pose(processed_frame, kpts, color)
            cv2.rectangle(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"ID:{pid}"
            cv2.putText(processed_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        draw_abnormal_zone(processed_frame, self.warning_zones.get(self.camera_id, []))

        return processed_frame, {}

    def _log_event(self, event_type: str, pid: int, confidence: float, frame: np.ndarray, details: dict = None):
        """
        内部辅助函数，用于统一处理事件的文件保存和API上报。
        """
        print(
            f"[事件] Camera {self.camera_id} 检测到 {event_type}，Person ID={pid}，置信度={confidence:.2f}，详情={details}")
        self.logger.info(f"Camera {self.camera_id}: Logging event '{event_type}' for person ID {pid}.")

        # 1. 保存证据文件 (调用FileSaver模块)
        clip_path = save_clip(
            pid, self.frame_idx, self.video_buffer, self.fps, f'{event_type}_clips', event_type
        )
        image_path = save_event_image(
            frame, pid, self.frame_idx, f'{event_type}_images', event_type
        )

        # 2. 准备上报数据
        event_data = {
            'camera_id': self.camera_id,
            'event_type': event_type,
            'confidence': confidence,
            'image_path': image_path,
            'video_clip_path': clip_path,
            'detection_details': details or {}
        }

        # 3. 上报事件 (调用ApiClient模块)
        log_event(event_data)
