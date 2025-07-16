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
from .event_checkers import check_fall, check_intrusion
from ..services.api_client import fetch_warning_zones, log_event
from ..utils.drawing import draw_pose, draw_abnormal_zone
from ..utils.file_saver import save_clip, save_event_image
from ..config import *


class AbnormalBehaviorProcessor:
    """
    一个有状态的处理器，用于检测单个视频流中的异常行为。
    这个类的每个实例对应一个独立的视频流，并管理其所有状态。
    """

    def __init__(self, camera_id: int, pose_detector: YoloDetector, fight_detector: YoloDetector, fps: int):
        self.logger = logging.getLogger(__name__)
        self.camera_id = camera_id
        self.yolo_pose_detector = pose_detector  # 姿态检测模型
        self.yolo_fight_detector = fight_detector  # 打架检测模型
        self.fps = fps

        # --- 初始化此视频流的所有状态变量 ---
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

        # --- 初始化时从API获取警戒区域配置 ---
        zone_data = fetch_warning_zones(self.camera_id)

        if zone_data and isinstance(zone_data, list) and len(zone_data) > 0:
            first_zone_info = zone_data[0]
            raw_zones = [first_zone_info]  # 或者你后端设计是只发一个区域，直接包成列表
            zones = []
            for zone in raw_zones:
                points = zone.get('zone_points', [])
                converted = [[pt['x'], pt['y']] for pt in points]
                zones.append(converted)

            self.warning_zones = {self.camera_id: zones}
            self.stay_frames_required = int(self.fps * first_zone_info.get('safe_time', DEFAULT_STAY_SECONDS))
            self.safe_distance = first_zone_info.get('safe_distance', DEFAULT_SAFE_DISTANCE)
        else:
            self.warning_zones = {self.camera_id: []}
            self.stay_frames_required = int(self.fps * DEFAULT_STAY_SECONDS)
            self.safe_distance = DEFAULT_SAFE_DISTANCE

        self.logger.info(
            f"Processor for camera {camera_id} initialized with {len(self.warning_zones[self.camera_id])} zones.")

    # 你后续 process_frame 和 _log_event 里 调用检测时记得传对应模型，比如
    # detect_fight(..., self.yolo_fight_detector)
    # 姿态检测时用 self.yolo_pose_detector.detect_people(...)

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        处理单帧图像，执行完整的检测、追踪、判断、绘制流程。
        """
        self.frame_idx += 1
        processed_frame = frame.copy()
        self.video_buffer.append(frame)  # 持续缓冲原始视频帧，用于保存切片

        # 1. 目标检测 (调用YoloDetector模块)
        kpts_list, centers, confidences = self.yolo_pose_detector.detect_people(processed_frame)

        zones = self.warning_zones.get(self.camera_id, [])
        print(f"Drawing {len(zones)} zones: {zones}")
        draw_abnormal_zone(processed_frame, zones)

        if not kpts_list:
            return processed_frame, {}

        # 2. 目标追踪 (调用LogicTracker模块)
        if not centers or not self.prev_centers:
            # 如果任意一方为空，直接给 ids 一个简单序号列表，避免空数组传入match_person_id
            ids = list(range(len(centers)))
        else:
            ids = match_person_id(centers, self.prev_centers, PERSON_MATCHING_THRESHOLD)
        self.prev_centers = {pid: center for pid, center in zip(ids, centers)}

        all_event_pids = set()

        # 打架检测：使用模型输出，且加冷却机制，避免重复报警
        fight_ids = self.yolo_fight_detector.detect_fight(processed_frame)  # 返回打架人员ID列表
        current_cooldown_bucket = int(time.time()) // 10
        for pid in fight_ids:
            if (pid, current_cooldown_bucket) not in self.recorded_conflicts:
                self.recorded_conflicts.add((pid, current_cooldown_bucket))
                print(f"[事件] Camera {self.camera_id} 检测到 conflict，Person ID={pid}，置信度=0.99")
                all_event_pids.add(pid)
                self._log_event('conflict', pid, 0.99, frame)

        # 遍历每个人，检测摔倒和入侵
        for i, kpts in enumerate(kpts_list):
            pid = ids[i]

            # 检测摔倒
            is_fall, is_new_fall = check_fall(
                pid, kpts, self.person_history, self.person_fall_status,
                FALL_ANGLE_THRESHOLD, FALL_WINDOW_SIZE, FALL_COOLDOWN_FRAMES
            )

            # 检测入侵
            x1, y1 = np.min(kpts[:, 0]), np.min(kpts[:, 1])
            x2, y2 = np.max(kpts[:, 0]), np.max(kpts[:, 1])
            bbox = (int(x1), int(y1), int(x2), int(y2))
            is_intruding, new_intrusion_zones = check_intrusion(
                pid, bbox, centers[i], self.camera_id, self.warning_zones,
                self.recorded_intrusions, self.zone_status_cache, self.frame_idx,
                self.stay_frames_required, self.safe_distance
            )

            # 如果是新发生的事件，则记录并上报
            if is_new_fall:
                print(f"[事件] Camera {self.camera_id} 检测到 person_fall，Person ID={pid}，置信度={confidences[i]:.2f}")
                all_event_pids.add(pid)
                self._log_event('person_fall', pid, confidences[i], frame)

            for zone_index in new_intrusion_zones:
                print(f"[事件] Camera {self.camera_id} 检测到 intrusion，Person ID={pid}，区域={zone_index}")
                all_event_pids.add(pid)
                self._log_event('intrusion', pid, confidences[i], frame, details={'zone_index': zone_index})

            # 5. 绘制结果 (调用Drawing模块)
            is_in_event = pid in all_event_pids or is_fall or is_intruding
            color = (0, 0, 255) if is_in_event else (0, 255, 0)

            draw_pose(processed_frame, kpts, color)
            cv2.rectangle(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            label = f"ID:{pid}"
            cv2.putText(processed_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 绘制警戒区
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
