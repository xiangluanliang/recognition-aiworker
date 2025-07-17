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
from ..services.api_client import fetch_warning_zones, log_event
from ..utils.drawing import draw_pose, draw_abnormal_zone
from ..utils.file_saver import save_clip, save_event_image
from ..config import *


class AbnormalBehaviorProcessor:
    """
    一个有状态的处理器，用于检测单个视频流中的异常行为。
    这个类的每个实例对应一个独立的视频流，并管理其所有状态。
    """

    def __init__(self, camera_id: int, yolo_detector: YoloDetector, fps: int):
        self.logger = logging.getLogger(__name__)
        self.camera_id = camera_id
        self.yolo_detector = yolo_detector  # 接收全局唯一的YOLO检测器实例
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
        self.prev_centers_history = defaultdict(lambda: deque(maxlen=5))  # 历史中心点，计算速度加速度等

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

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        处理单帧图像，执行完整的检测、追踪、判断、绘制流程。
        """
        self.frame_idx += 1
        processed_frame = frame.copy()
        self.video_buffer.append(frame)  # 持续缓冲原始视频帧，用于保存切片

        # 1. 目标检测 (调用YoloDetector模块)
        kpts_list, centers, confidences = self.yolo_detector.detect_people(processed_frame)

        zones = self.warning_zones.get(self.camera_id, [])
        # print(f"Drawing {len(zones)} zones: {zones}")
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

        # 3. 更新用于打架检测的历史姿态数据
        for i, kpts in enumerate(kpts_list):
            self.fight_kpts_history[ids[i]].append(kpts.copy())

        for pid, center in zip(ids, centers):
            self.prev_centers_history[pid].append(center)

        # 4. 事件检测 (调用EventCheckers模块)
        conflict_pairs_with_scores = detect_fight(
            ids, centers, self.fight_kpts_history,self.prev_centers_history,
            FIGHT_DISTANCE_THRESHOLD, FIGHT_MOTION_THRESHOLD, FIGHT_ORIENTATION_SIMILARITY_THRESHOLD,
            FIGHT_SPEED_THRESHOLD,FIGHT_ACCELERATION_THRESHOLD,FIGHT_KPTS_CHANGE_THRESHOLD
        )

        all_event_pids = set()  # 记录本帧所有参与事件的人员ID，用于高亮绘制
        # 处理打架事件
        for pid1, pid2, conflict_scores in conflict_pairs_with_scores:
            for pid in [pid1, pid2]:
                all_event_pids.add(pid)
                # 10秒内对同一个人只上报一次打架事件，避免事件风暴
                if (pid, int(time.time()) // 10) in self.recorded_conflicts: continue
                self.recorded_conflicts.add((pid, int(time.time()) // 10))
                self._log_event('conflict', pid, conflict_scores, frame)

        # 遍历每个人，检测摔倒和入侵
        for i, kpts in enumerate(kpts_list):
            pid = ids[i]

            # 先算bbox
            x1, y1 = np.min(kpts[:, 0]), np.min(kpts[:, 1])
            x2, y2 = np.max(kpts[:, 0]), np.max(kpts[:, 1])
            bbox = (int(x1), int(y1), int(x2), int(y2))

            # 检测摔倒
            is_fall, is_new_fall, score = check_fall(
                pid, kpts, bbox, self.person_fall_status,
                FALL_ANGLE_THRESHOLD, FALL_WINDOW_SIZE, FALL_COOLDOWN_FRAMES
            )
            # 检测入侵
            is_intruding, new_intrusion_zones = check_intrusion(
                pid, bbox, centers[i], self.camera_id, self.warning_zones,
                self.recorded_intrusions, self.zone_status_cache, self.frame_idx,
                self.stay_frames_required, self.safe_distance
            )

            # 如果是新发生的事件，则记录并上报
            if is_new_fall:
                all_event_pids.add(pid)
                self._log_event('person_fall', pid, score, frame)

            for zone_index in new_intrusion_zones:
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
