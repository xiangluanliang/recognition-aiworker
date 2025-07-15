#主要函数，调用检测处理视频

import cv2
import os

from collections import defaultdict, deque

import django
import numpy as np

from config import settings
from .yolo_utils import detect_people
from .logic_tracker import match_person_id
from .event_handlers import check_fall,  check_abnormal_overlap, check_intrusion, get_warning_zones_by_camera, \
    min_distance_to_polygon,detect_fight
from .utils_pose import draw_pose
from .constants import CLIP_DURATION_SECONDS
from api.models import EventLog, Camera
from django.utils import timezone

# 设置 Django 的配置模块路径（指定 base.py）
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.base')
django.setup()

# 然后再导入 Django 的配置模块
from django.conf import settings

#在图像上绘制多个异常区域的多边形边框。
def draw_abnormal_zone(frame, zone_points_list):
    for points in zone_points_list:
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

#保存视频切片内容
def save_clip(pid, frame_idx, clip_buffer, fps, subfolder, event_type):
    base_dir = os.path.join(settings.MEDIA_ROOT, 'subject_images', subfolder)
    os.makedirs(base_dir, exist_ok=True)

    clip_path = os.path.join(base_dir, f"{event_type}_{pid}_{frame_idx}.mp4")

    if not clip_buffer:
        return None
    height, width, _ = clip_buffer[0].shape
    writer = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for f in clip_buffer:
        writer.write(f)
    writer.release()

    # 返回相对路径用于数据库
    rel_path =  os.path.relpath(clip_path, str(settings.MEDIA_ROOT))
    return rel_path
#保存图片
def save_event_image(frame, pid, frame_idx, subfolder, event_type):
    base_dir = os.path.join(settings.MEDIA_ROOT, 'subject_images', subfolder)
    os.makedirs(base_dir, exist_ok=True)

    filename = f"{event_type}_{pid}_{frame_idx}.jpg"
    full_path = os.path.join(base_dir, filename)
    cv2.imwrite(full_path, frame)

    rel_path = os.path.relpath(full_path, str(settings.MEDIA_ROOT))
    return rel_path

#主要函数，调用进行判断各种异常（有时间的话可以拆分一下）
def detect_abnormal_behavior(video_path, output_path, camera_id, stay_seconds, safe_distance):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "❌ 无法打开视频", None

    camera = Camera.objects.filter(id=camera_id).first()
    if not camera:
        camera = Camera.objects.create(id=camera_id, name=f"Camera {camera_id}", is_active=True)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    stay_frames_required = int(fps * stay_seconds)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    if not out.isOpened():
        return False, "❌ 无法创建输出视频，请检查路径和编码器", None

    frame_idx = 0
    abnormal_count = 0
    prev_centers = {}
    fall_clip_buffer = defaultdict(lambda: deque(maxlen=int(fps * CLIP_DURATION_SECONDS)))
    person_history = defaultdict(list)
    person_fall_status = defaultdict(lambda: {'fall_frame_count': 0, 'is_falling': False})
    zone_status_cache = defaultdict(dict)
    recorded_intrusions = set()
    recorded_conflicts = set()
    fight_kpts_history = defaultdict(lambda: deque(maxlen=5))

    warning_zone_map = get_warning_zones_by_camera(camera_id)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        draw_abnormal_zone(frame, warning_zone_map[camera_id])  # ✅ 恢复绘制区域

        kpts_list, centers, confidences = detect_people(frame)
        ids = match_person_id(centers, prev_centers)

        for i, center in enumerate(centers):
            pid = ids[i]
            fall_clip_buffer[pid].append(frame.copy())


        for i, kpts in enumerate(kpts_list):
            fight_kpts_history[ids[i]].append(kpts.copy())

        # 检测打架
        conflict_pairs = detect_fight(ids, centers, kpts_list, frame_idx, fight_kpts_history)
        conflict_detected = False
        conflict_persons = set()

        for pid1, pid2 in conflict_pairs:
            conflict_detected = True
            for pid in [pid1, pid2]:
                conflict_persons.add(pid)
                if (pid, frame_idx // fps) in recorded_conflicts:
                    continue
                recorded_conflicts.add((pid, frame_idx // fps))
                clip_path = save_clip(pid, frame_idx, fall_clip_buffer[pid], fps, 'conflict_clips', 'conflict')
                image_path = save_event_image(frame, pid, frame_idx, 'conflict_clips', 'conflict')

                EventLog.objects.create(
                    event_type='conflict',
                    camera=camera,
                    time=timezone.now(),
                    confidence=0.99,
                    image_path=os.path.join('subject_images', image_path),
                    video_clip_path=os.path.join('subject_images', clip_path),
                    person=None
                )
                abnormal_count += 1

        intrusion_msgs = []
        for i, kpts in enumerate(kpts_list):
            pid = ids[i]
            center = centers[i]
            conf = confidences[i]
            is_fall, is_new_fall = check_fall(pid, kpts, center, frame_idx, person_history, person_fall_status)

            x1, y1 = int(kpts[:, 0].min()), int(kpts[:, 1].min())
            x2, y2 = int(kpts[:, 0].max()), int(kpts[:, 1].max())
            bbox = (x1, y1, x2, y2)

            abnormal_zones, intrusion_texts, in_danger_now = check_intrusion(
                bbox=bbox,
                center=center,
                camera_id=camera_id,
                frame_idx=frame_idx,
                fps=fps,
                stay_frames_required=stay_frames_required,
                safe_distance=safe_distance,
                warning_zones=warning_zone_map,
                status_cache=zone_status_cache
            )
            intrusion_msgs.extend(intrusion_texts)
            is_intruding = in_danger_now

            #颜色显示
            if pid in conflict_persons:
                color = (0, 165, 255)  # 橙色
            elif is_fall or is_intruding:
                color = (0, 0, 255)  # 红色
            else:
                color = (0, 255, 0)  # 绿色

            #摔倒记录
            if is_fall:
                if is_new_fall:
                    abnormal_count += 1
                    clip_path = save_clip(pid, frame_idx, fall_clip_buffer[pid], fps, 'fall_clips', 'fall')
                    image_path = save_event_image(frame, pid, frame_idx, 'fall_clips', 'fall')

                    EventLog.objects.create(
                        event_type='person_fall',
                        camera=camera,
                        time=timezone.now(),
                        confidence=conf,
                        image_path=os.path.join('subject_images', image_path),
                        video_clip_path=os.path.join('subject_images', clip_path),
                        person=None
                    )

            #区域入侵记录
            if abnormal_zones:
                for zone_index, polygon in abnormal_zones:
                    if (pid, zone_index) in recorded_intrusions:
                        continue
                    recorded_intrusions.add((pid, zone_index))

                    abnormal_count += 1
                    clip_path = save_clip(pid, frame_idx, fall_clip_buffer[pid], fps, 'intrusion_clips', 'intrusion')
                    image_path = save_event_image(frame, pid, frame_idx, 'intrusion_clips', 'intrusion')

                    EventLog.objects.create(

                        event_type='intrusion',
                        camera=camera,
                        time=timezone.now(),
                        confidence=conf,
                        image_path=os.path.join('subject_images', image_path),
                        video_clip_path=os.path.join('subject_images', clip_path),
                        person=None
                    )

            draw_pose(frame, kpts, color)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 显示各个区域的最小距离
            for zone_index, polygon in enumerate(warning_zone_map[camera_id]):
                dist = min_distance_to_polygon(center, polygon)
                text = f"Z{zone_index}: {dist:.1f}px"
                cv2.putText(frame, text, (x1, y1 + 20 + 20 * zone_index), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

            # 同时显示 fall/unfall 和 intrusion/unintrusion 状态
            status_text = []
            status_text.append("Fall" if is_fall else "Unfall")
            status_text.append("Intrusion" if is_intruding else "Unintrusion")
            if pid in conflict_persons:
                status_text.append("Conflict")
            label = f"{' | '.join(status_text)}  ID:{pid}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if conflict_detected:
            cv2.putText(frame, "They are FIGNTING!!!!!", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 3)

        # 显示状态栏文字（入侵提示）
        for idx, msg in enumerate(intrusion_msgs):
            cv2.putText(frame, msg, (10, height - 20 - 25 * idx), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    return True, f"共检测到异常事件: {abnormal_count}", output_path
