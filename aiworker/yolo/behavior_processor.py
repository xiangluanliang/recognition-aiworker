# aiworker/yolo/behavior_processor.py
from datetime import datetime,timezone
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
        self.logger.info(f"处理器为摄像头 {self.camera_id} 初始化，启用的功能: {self.active_detectors}")

        self.frame_idx = 0
        self.prev_centers = {}
        self.video_buffer = deque(maxlen=int(self.fps * CLIP_DURATION_SECONDS))
        self.person_fall_status = defaultdict(
            lambda: {'fall_frame_count': 0, 'is_falling': False, 'cooldown_counter': 0})
        self.zone_status_cache = defaultdict(dict)
        self.recorded_intrusions = set()
        self.recorded_conflicts = set()
        self.fight_kpts_history = defaultdict(lambda: deque(maxlen=5))
        self.center_histories = defaultdict(lambda: deque(maxlen=5))
        self.person_last_seen = {}
        self.recent_falls = {}
        self.person_last_seen = {}
        self.recent_falls = {}

        self.warning_zones = []
        if 'intrusion_detection' in self.active_detectors:
            raw_zones_data = fetch_warning_zones_for_camera(self.camera_id)
            self.logger.info(f"从API获取到摄像头 {self.camera_id} 的原始区域数据: {raw_zones_data}")

            # 计算缩放比例
            x_scale = FRAME_WIDTH / CANONICAL_ZONE_WIDTH
            y_scale = FRAME_HEIGHT / CANONICAL_ZONE_HEIGHT

            for zone_data in raw_zones_data:
                scaled_polygon = [
                    [int(pt['x'] * x_scale), int(pt['y'] * y_scale)]
                    for pt in zone_data.get('zone_points', [])
                ]

                if not scaled_polygon:
                    continue

                self.warning_zones.append({
                    'id': zone_data.get('id'),
                    'name': zone_data.get('name', '未命名区域'),
                    'polygon': scaled_polygon,
                    'stay_frames': int(self.fps * zone_data.get('safe_time', DEFAULT_STAY_SECONDS)),
                    'safe_dist': zone_data.get('safe_distance', DEFAULT_SAFE_DISTANCE)
                })

            self.logger.info(f"摄像头 {self.camera_id} 加载了 {len(self.warning_zones)} 个处理后的警戒区域。")

    def _cleanup_stale_ids(self, current_pids: set, stale_threshold_frames: int = 300):
        """
        清理超过N帧（默认为10秒，假设30fps）未出现的陈旧ID，以释放内存。
        """
        stale_ids = [pid for pid, frame in self.person_last_seen.items() if
                     self.frame_idx - frame > stale_threshold_frames]

        if stale_ids:
            self.logger.info(f"清理陈旧ID: {stale_ids}")
            for pid in stale_ids:
                self.person_last_seen.pop(pid, None)
                self.prev_centers.pop(pid, None)
                self.person_fall_status.pop(pid, None)
                self.fight_kpts_history.pop(pid, None)

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        处理单帧图像，执行完整的检测、追踪、并根据配置动态判断、绘制流程。
        """
        self.frame_idx += 1
        processed_frame = frame.copy()

        polygons_to_draw = [zone['polygon'] for zone in self.warning_zones]
        draw_abnormal_zone(processed_frame, polygons_to_draw)

        kpts_list, centers, confidences = self.yolo_detector.detect_people(processed_frame)
        if not kpts_list:
            if self.frame_idx % 100 == 0:
                self._cleanup_stale_ids(set())
            return processed_frame, {}

        ids = match_person_id(centers, self.prev_centers, PERSON_MATCHING_THRESHOLD)
        self.prev_centers = {pid: center for pid, center in zip(ids, centers)}
        current_pids = set(ids)

        for pid in current_pids:
            self.person_last_seen[pid] = self.frame_idx

        for i, pid in enumerate(ids):
            self.center_histories[pid].append(centers[i])

        all_event_pids = set()

        if 'fight_detection' in self.active_detectors:
            for i, pid in enumerate(ids):
                self.fight_kpts_history[pid].append(kpts_list[i].copy())

            conflict_pairs = detect_fight(
                ids, centers, self.fight_kpts_history, self.center_histories,
                dist_thresh=FIGHT_DISTANCE_THRESHOLD,
                motion_thresh=FIGHT_MOTION_THRESHOLD,
                orient_thresh=FIGHT_ORIENTATION_SIMILARITY_THRESHOLD,
                speed_thresh=FIGHT_SPEED_THRESHOLD,
                accel_thresh=FIGHT_ACCELERATION_THRESHOLD,
                kpts_change_thresh=FIGHT_KPTS_CHANGE_THRESHOLD
            )
            for pid1, pid2, fight_score in conflict_pairs:
                for pid in [pid1, pid2]:
                    all_event_pids.add(pid)
                    if (pid, int(time.time()) // 10) not in self.recorded_conflicts:
                        self.logger.error(f"检测到打架行为，评分为: {fight_score}")
                        self.recorded_conflicts.add((pid, int(time.time()) // 10))
                        self._log_event('conflict', pid, fight_score, frame)

        for i, (pid, kpts) in enumerate(zip(ids, kpts_list)):
            x1, y1 = np.min(kpts[:, 0]), np.min(kpts[:, 1])
            x2, y2 = np.max(kpts[:, 0]), np.max(kpts[:, 1])
            bbox = (int(x1), int(y1), int(x2), int(y2))

            is_new_fall_event = False
            if 'fall_detection' in self.active_detectors:
                is_falling, is_new_fall, score = check_fall(
                    pid, kpts, bbox, self.person_fall_status,
                    FALL_ANGLE_THRESHOLD, FALL_WINDOW_SIZE, FALL_COOLDOWN_FRAMES
                )
                if is_new_fall:
                    is_new_fall_event = True
                    all_event_pids.add(pid)
                    self.logger.error(f"检测到摔倒 (分数: {score:.2f})。")
                    self._log_event('person_fall', pid, score, frame)

            # 调用入侵检测逻辑
            is_intruding, new_intrusion_zones_info = check_intrusion(
                pid, bbox, centers[i],
                self.warning_zones,
                self.recorded_intrusions, self.zone_status_cache, self.frame_idx
            )

            # 如果检测到新的入侵事件，上报
            if new_intrusion_zones_info:
                self.logger.error(f"检测到区域入侵行为。")
                for zone_info in new_intrusion_zones_info:
                    self._log_event('intrusion', pid, confidences[i], frame,
                                    details={'zone_id': zone_info['id'], 'zone_name': zone_info['name']})

            # --- 可视化调试信息 ---
            debug_texts = []
            is_in_event = is_new_fall_event or is_intruding
            display_color = (0, 0, 255) if is_in_event else (0, 255, 0)

            if is_intruding:
                display_color = (0, 165, 255)  # 橙色表示正在侵入中

                # 查找此人对应的计时器信息
                for zone_info in self.warning_zones:
                    zone_id = zone_info['id']
                    cache_key = f"{pid}_{zone_id}"

                    if cache_key in self.zone_status_cache:
                        status = self.zone_status_cache[cache_key]
                        stay_duration = self.frame_idx - status['start_frame']

                        # 将帧数转换为秒
                        stay_seconds = stay_duration / self.fps
                        required_seconds = zone_info['stay_frames'] / self.fps

                        debug_texts.append(f"Zone {zone_id}: INSIDE")
                        debug_texts.append(f"Timer: {stay_seconds:.1f}s / {required_seconds:.1f}s")

                        # 如果已触发警报，颜色变为红色
                        if (pid, zone_id) in self.recorded_intrusions:
                            display_color = (0, 0, 255)
                            debug_texts.append("ALARM TRIGGERED!")
                        break  # 只显示第一个正在侵入的区域信息

            # 绘制姿态和边界框
            draw_pose(processed_frame, kpts, display_color)
            cv2.rectangle(processed_frame, bbox[:2], bbox[2:], display_color, 2)

            # 在头顶绘制ID和调试文本
            label = f"ID:{pid}"
            cv2.putText(processed_frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color,
                        2)

            for idx, text in enumerate(debug_texts):
                y_pos = bbox[1] - 35 - (idx * 20)
                cv2.putText(processed_frame, text, (bbox[0], y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, display_color, 2)

        if self.frame_idx % 100 == 0:
            self._cleanup_stale_ids(current_pids)

        return processed_frame, {}

    def _log_event(self, event_type: str, pid: int, confidence: float, frame: np.ndarray, details: dict = None):
        """
        内部辅助函数，用于统一处理事件的文件保存和API上报。
        增加了异常处理，确保文件保存失败不会导致整个线程崩溃。
        """
        print(
            f"[事件] Camera {self.camera_id} 检测到 {event_type}，Person ID={pid}，置信度={confidence:.2f}，详情={details}")
        self.logger.info(f"Camera {self.camera_id}: Logging event '{event_type}' for person ID {pid}.")

        clip_path = None
        image_path = None

        try:
            if frame is not None:
                image_path = save_event_image(
                    frame, pid, self.frame_idx, f'{event_type}_images', event_type
                )

            if self.video_buffer:
                clip_path = save_clip(
                    pid, self.frame_idx, self.video_buffer, self.fps, f'{event_type}_clips', event_type
                )
        except Exception as e:
            self.logger.error(f"保存事件 '{event_type}' 的证据文件时发生异常: {e}", exc_info=True)

        event_data = {
            'time': datetime.now(timezone.utc).isoformat(),
            'camera': self.camera_id,
            'event_type': event_type,
            'confidence': confidence,
            'image_path': image_path,
            'video_clip_path': clip_path,
            'detection_details': details or {}
        }

        try:
            log_event(event_data)
        except Exception as e:
            self.logger.error(f"上报事件 '{event_type}' 到Django API时失败: {e}", exc_info=True)
