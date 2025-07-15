# 异常事件检测函数
import os
import cv2
import numpy as np
from collections import deque,defaultdict
from django.utils import timezone
from .utils_pose import angle_between_points
from api.models import EventLog,WarningZone
from shapely.geometry import Point, Polygon


# ========== 异常事件检测 ==========

#----------危险区域入侵检测----------

def point_in_polygon(point, polygon):
    from matplotlib.path import Path
    return Path(polygon).contains_point(point)
# 从数据库中根据摄像头信息获取异常区域
def get_warning_zones_by_camera(camera_id):
    """
    返回某摄像头下所有激活的多边形异常区域，格式为：
    { camera_id: [ [point1, point2, ...], [polygon2], ... ] }
    """
    zones = WarningZone.objects.filter(camera_id=camera_id, is_active=True)
    polygon_list = []

    for zone in zones:
        # zone.zone_points 是 [{"x":123, "y":456}, ...]
        polygon = [(pt["x"], pt["y"]) for pt in zone.zone_points]
        if polygon:
            polygon_list.append(polygon)

    return {camera_id: polygon_list}
# 最小距离判断（中心点距离异常区域，目前不用）
def min_distance_to_polygon(point, polygon):
    min_dist = float('inf')
    px, py = point
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        line_vec = np.array([x2 - x1, y2 - y1])
        point_vec = np.array([px - x1, py - y1])
        line_len = np.dot(line_vec, line_vec)
        if line_len == 0:
            dist = np.linalg.norm(point_vec)
        else:
            t = max(0, min(1, np.dot(point_vec, line_vec) / line_len))
            projection = np.array([x1, y1]) + t * line_vec
            dist = np.linalg.norm(projection - np.array([px, py]))
        min_dist = min(min_dist, dist)
    return min_dist
# 最小距离判断: 人物框边上去点的最小距离检测
def min_distance_bbox_to_polygon(bbox, polygon, num_samples_per_edge=5):
    x1, y1, x2, y2 = bbox
    sample_points = []

    # 四个边均匀采样点（包括角点）
    for i in range(num_samples_per_edge + 1):
        ratio = i / num_samples_per_edge
        # 上边
        sample_points.append((int(x1 + (x2 - x1) * ratio), y1))
        # 下边
        sample_points.append((int(x1 + (x2 - x1) * ratio), y2))
        # 左边
        sample_points.append((x1, int(y1 + (y2 - y1) * ratio)))
        # 右边
        sample_points.append((x2, int(y1 + (y2 - y1) * ratio)))

    # 取这些点中，距离 polygon 最近的那个距离
    return min(min_distance_to_polygon(pt, polygon) for pt in sample_points)
# 判断是否触发异常
def check_intrusion(bbox,center,camera_id,frame_idx,fps,stay_frames_required,safe_distance,warning_zones,status_cache):
    abnormal_events = []
    abnormal_msgs = []
    in_danger_now = False  # 只要在危险区域就设为 True

    for zone_index, polygon_points in enumerate(warning_zones.get(camera_id, [])):
        min_dist = min_distance_bbox_to_polygon(bbox, polygon_points)

        pid = f"{camera_id}_{zone_index}"
        if point_in_polygon(center, polygon_points) or min_dist < safe_distance:
            in_danger_now = True
            if pid not in status_cache:
                status_cache[pid] = {'start': frame_idx, 'recorded': False}
            else:
                if not status_cache[pid]['recorded']:
                    stay_duration = frame_idx - status_cache[pid]['start']
                    if stay_duration >= stay_frames_required:
                        abnormal_events.append((zone_index, polygon_points))
                        abnormal_msgs.append(f"区域 {zone_index} 入侵中")
                        status_cache[pid]['recorded'] = True
        else:
            status_cache.pop(pid, None)

    return abnormal_events, abnormal_msgs,in_danger_now
#异常距离检测(目前不用了)
def check_abnormal_overlap(bbox, zone_coords):
    """
    判断人物框（bbox）是否与异常区域（zone_coords）重叠。
    bbox: (x1, y1, x2, y2)
    zone_coords: (zx1, zy1, zx2, zy2)
    """
    x1, y1, x2, y2 = bbox
    zx1, zy1, zx2, zy2 = zone_coords

    # 判断是否有交集
    inter_x1 = max(x1, zx1)
    inter_y1 = max(y1, zy1)
    inter_x2 = min(x2, zx2)
    inter_y2 = min(y2, zy2)

    # 有重叠区域
    return inter_x1 < inter_x2 and inter_y1 < inter_y2

# ----------  摔倒检测  ----------

#摔倒情况检测
def check_fall(pid, kpts, center, frame_idx, person_history, person_fall_status, fall_window_size=3, cooldown_threshold=150):
    mid_shoulder = (kpts[5] + kpts[6]) / 2
    mid_hip = (kpts[11] + kpts[12]) / 2
    mid_knee = (kpts[13] + kpts[14]) / 2
    angle = angle_between_points(mid_shoulder, mid_hip, mid_knee)
    nose_y = kpts[0][1]
    knee_y = min(kpts[13][1], kpts[14][1])
    height = np.linalg.norm(mid_shoulder - mid_knee)

    person_history[pid].append({
        'angle': angle,
        'center': center,
        'nose_y': nose_y,
        'knee_y': knee_y,
        'height': height,
        'frame': frame_idx
    })

    status = person_fall_status[pid]

    # 冷却中，不再记录新的摔倒事件（但仍返回 is_fall=True 以供 UI 红框绘制）
    if status.get('cooldown_counter', 0) > 0:
        status['cooldown_counter'] -= 1
        return True, False

    # 判断摔倒趋势
    trend_fall = False
    if len(person_history[pid]) >= 2:
        a1 = person_history[pid][-2]['angle']
        a2 = angle
        angle_diff = abs(a2 - a1)
        c1 = np.array(person_history[pid][-2]['center'])
        c2 = np.array(center)
        move_dist = np.linalg.norm(c2 - c1)
        trend_fall = (angle_diff > 30 and move_dist > 50)

    is_fall = (angle < 85 and nose_y > knee_y - 20) or trend_fall

    if is_fall:
        status['fall_frame_count'] += 1
        if status['fall_frame_count'] >= fall_window_size:
            if not status.get('is_falling', False):
                # 新的摔倒状态开始
                status['is_falling'] = True
                status['cooldown_counter'] = cooldown_threshold  # 冷却帧数
                return True, True  # ✅ 是摔倒 + 是新事件
            return True, False  # 是摔倒，但已记录
        return True, False  # 判断为摔倒，但未达连续帧数
    else:
        status['fall_frame_count'] = 0
        status['is_falling'] = False
        return False, False

# ----------  打架检测  ----------

upper_kpts_indices = [5, 6, 7, 8, 9, 10]

def upper_body_motion_std(kpts_deque):
    kpts_array = np.array(kpts_deque)  # shape: (n_frames, 17, 2)
    upper_body = kpts_array[:, upper_kpts_indices, :]
    std = np.std(upper_body, axis=0).mean()
    return std


def estimate_orientation(kpts):
    kpts = np.array(kpts)

    if kpts.shape[1] == 2:
        # 没有置信度分量，默认可信
        shoulder_mid = (kpts[5, :2] + kpts[6, :2]) / 2
        nose = kpts[0, :2]
    elif kpts.shape[1] == 3:
        # 有置信度则判断置信度
        if kpts[5, 2] > 0.5 and kpts[6, 2] > 0.5:
            shoulder_mid = (kpts[5, :2] + kpts[6, :2]) / 2
            nose = kpts[0, :2] if kpts[0, 2] > 0.5 else shoulder_mid
        else:
            return np.array([0.0, 0.0])
    else:
        return np.array([0.0, 0.0])

    direction = nose - shoulder_mid
    return direction / (np.linalg.norm(direction) + 1e-5)

def orientation_similarity(vec1, vec2):
    cos_theta = np.dot(vec1, vec2)
    return abs(cos_theta)  # 越接近1则角度越小，越不像对打

fight_history = defaultdict(lambda: deque(maxlen=5))
#检测打架情况
def detect_fight(ids, centers, kpts_list, frame_idx, fight_kpts_history):
    conflicts = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            pid1, pid2 = ids[i], ids[j]
            c1, c2 = np.array(centers[i]), np.array(centers[j])
            dist = np.linalg.norm(c1 - c2)

            if dist < 100:  # 靠得近
                # 判断动作幅度（使用关键点历史）
                if len(fight_kpts_history[pid1]) == 5 and len(fight_kpts_history[pid2]) == 5:
                    motion1 = upper_body_motion_std(fight_kpts_history[pid1])
                    motion2 = upper_body_motion_std(fight_kpts_history[pid2])
                    if motion1 > 6 and motion2 > 6:
                        # 判断朝向
                        vec1 = estimate_orientation(kpts_list[i])
                        vec2 = estimate_orientation(kpts_list[j])
                        if orientation_similarity(vec1, vec2) < 0.3:  # 接近面对面
                            conflicts.append((pid1, pid2))
    return conflicts
