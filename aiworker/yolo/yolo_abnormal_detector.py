# 之前的主函数，调用各种其他写好的内容，对传入的视频的帧进行判断

from collections import defaultdict, deque
import cv2
import os
import datetime

import numpy as np

from aiworker.yolo.constants import POSE_PAIRS, MEDIA_ROOT
from aiworker.yolo.event_handlers import detect_people, match_person_id, detect_fight, check_fall, check_intrusion


# 绘制
def draw_pose(frame, kpts, color=(0, 255, 0)):
    for point in kpts:
        x, y = int(point[0]), int(point[1])
        cv2.circle(frame, (x, y), 3, color, -1)
    for i, j in POSE_PAIRS:
        if i < len(kpts) and j < len(kpts):
            pt1, pt2 = tuple(kpts[i]), tuple(kpts[j])
            cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, 2)
#保存视频切片内容
def save_clip(pid, frame_idx, clip_buffer, fps, subfolder, event_type):
    base_dir = os.path.join(MEDIA_ROOT, 'subject_images', subfolder)
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
    rel_path =  os.path.relpath(clip_path, str(MEDIA_ROOT))
    return rel_path
#保存图片
def save_event_image(frame, pid, frame_idx, subfolder, event_type):
    base_dir = os.path.join(MEDIA_ROOT, 'subject_images', subfolder)
    os.makedirs(base_dir, exist_ok=True)

    filename = f"{event_type}_{pid}_{frame_idx}.jpg"
    full_path = os.path.join(base_dir, filename)
    cv2.imwrite(full_path, frame)

    rel_path = os.path.relpath(full_path, str(MEDIA_ROOT))
    return rel_path

# 在图像上绘制多个异常区域的多边形边框。
def draw_abnormal_zone(frame, zone_points_list):
    for points in zone_points_list:
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

# 主函数，传入帧，调用检测各种情况
def process_abnormal_single_frame(
    frame, frame_idx, fps, camera_id,
    stay_seconds, safe_distance,
    prev_centers, fall_clip_buffer, person_history,
    person_fall_status, zone_status_cache,
    recorded_intrusions, recorded_conflicts,
    fight_kpts_history, warning_zone_map, camera,
    log_event_to_django  # 记得传这个函数进来
):
    abnormal_count = 0
    intrusion_msgs = []

    draw_abnormal_zone(frame, warning_zone_map[camera_id])

    kpts_list, centers, confidences = detect_people(frame)
    ids = match_person_id(centers, prev_centers)
    prev_centers.clear()
    prev_centers.update({pid: center for pid, center in zip(ids, centers)})

    for i, center in enumerate(centers):
        pid = ids[i]
        fall_clip_buffer[pid].append(frame.copy())

    for i, kpts in enumerate(kpts_list):
        fight_kpts_history[ids[i]].append(kpts.copy())

    # 打架检测
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

            event = {
                'event_type': 'conflict',
                'camera': camera.id,
                'time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'confidence': 0.99,
                'image_path': os.path.join('subject_images', image_path),
                'video_clip_path': os.path.join('subject_images', clip_path),
                'person': None
            }
            log_event_to_django(event)
            abnormal_count += 1

    # 摔倒与入侵检测
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
            stay_frames_required=int(fps * stay_seconds),
            safe_distance=safe_distance,
            warning_zones=warning_zone_map,
            status_cache=zone_status_cache
        )
        intrusion_msgs.extend(intrusion_texts)

        if is_fall and is_new_fall:
            abnormal_count += 1
            clip_path = save_clip(pid, frame_idx, fall_clip_buffer[pid], fps, 'fall_clips', 'fall')
            image_path = save_event_image(frame, pid, frame_idx, 'fall_clips', 'fall')

            event = {
                'event_type': 'person_fall',
                'camera': camera.id,
                'time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'confidence': conf,
                'image_path': os.path.join('subject_images', image_path),
                'video_clip_path': os.path.join('subject_images', clip_path),
                'person': None
            }
            log_event_to_django(event)

        for zone_index, _ in abnormal_zones:
            if (pid, zone_index) in recorded_intrusions:
                continue
            recorded_intrusions.add((pid, zone_index))
            abnormal_count += 1
            clip_path = save_clip(pid, frame_idx, fall_clip_buffer[pid], fps, 'intrusion_clips', 'intrusion')
            image_path = save_event_image(frame, pid, frame_idx, 'intrusion_clips', 'intrusion')

            event = {
                'event_type': 'intrusion',
                'camera': camera.id,
                'time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'confidence': conf,
                'image_path': os.path.join('subject_images', image_path),
                'video_clip_path': os.path.join('subject_images', clip_path),
                'person': None
            }
            log_event_to_django(event)

        # 绘制标签
        color = (0, 0, 255) if is_fall or in_danger_now else (0, 165, 255) if pid in conflict_persons else (0, 255, 0)
        draw_pose(frame, kpts, color)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label_parts = []
        label_parts.append("Fall" if is_fall else "Unfall")
        label_parts.append("Intrusion" if in_danger_now else "Unintrusion")
        if pid in conflict_persons:
            label_parts.append("Conflict")
        label = " | ".join(label_parts) + f" ID:{pid}"

        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 顶部提示
    if conflict_detected:
        cv2.putText(frame, "They are FIGHTING!!!!!", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 3)

    for idx, msg in enumerate(intrusion_msgs):
        cv2.putText(frame, msg, (10, frame.shape[0] - 20 - 25 * idx),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return frame, abnormal_count
