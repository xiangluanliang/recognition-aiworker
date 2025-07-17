# aiworker/yolo/event_checkers.py
import numpy as np
from matplotlib.path import Path


# --- Fall Detection Logic ---
def _angle_between_points(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

# 肩高比判断是否水平
def is_likely_horizontal(bbox, threshold=1.2):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    if height == 0:
        return False
    return (width / height) > threshold

def check_fall(pid, kpts, bbox, person_fall_status, base_angle_thresh, window_size, cooldown_frames):

    # -----  条件1：关键角度

    # 计算关键角度
    mid_shoulder = (kpts[5] + kpts[6]) / 2
    mid_hip = (kpts[11] + kpts[12]) / 2

    angle = _angle_between_points(kpts[0], mid_shoulder, mid_hip)

    # -----  条件2：肩宽比

    # 宽高比计算
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    wh_ratio = width / (height + 1e-5)

    # -----  条件3：姿势水平情况

    # 判断姿势是否“水平”
    is_horizontal = is_likely_horizontal(bbox)
    angle_thresh = base_angle_thresh + 15 if is_horizontal else base_angle_thresh

    # -----  条件4：高度变化检测
    status = person_fall_status[pid]
    if 'prev_height' not in status:
        status['prev_height'] = height
    height_change = abs(height - status['prev_height']) / (status['prev_height'] + 1e-5)
    status['prev_height'] = height

    # -----  分数计算

    # 角度得分（0-30）
    angle_score = 30 if angle < 10 else np.clip((angle_thresh - angle) / angle_thresh * 30, 0, 30)
    # 宽高比得分（0-50）
    wh_score = np.clip((wh_ratio - 1.2) / 1.8 * 40, 0, 40)
    # 高度变化分数
    motion_score = np.clip(height_change / 0.5 * 10, 0, 10)
    # 动作持续得分（0-20）
    status = person_fall_status[pid]
    duration_score = np.clip(status['fall_frame_count'] / window_size * 20, 0, 20)

    total_score = (angle_score + wh_score + duration_score + motion_score )/100

    status = person_fall_status[pid]

    if status['cooldown_counter'] > 0:
        status['cooldown_counter'] -= 1
        return True, False ,total_score # 冷却中，仍认为是摔倒状态，但不是新事件

    score_thresh = 0.6  # 总分超过60
    extreme_wh_ratio = 1.2  # 宽高比超过1.2
    is_fall = angle < angle_thresh or total_score >= score_thresh or wh_ratio > extreme_wh_ratio or (height_change > 0.3)

    if is_fall:
        status['fall_frame_count'] += 1
        if status['fall_frame_count'] >= window_size:
            if not status['is_falling']:
                status['is_falling'] = True
                status['cooldown_counter'] = cooldown_frames
                return True, True ,total_score # 是摔倒 + 是新事件
            return True, False ,total_score # 是摔倒，但已记录
    else:
        status['fall_frame_count'] = 0
        status['is_falling'] = False
    return is_fall, False,total_score

# --- Intrusion Detection Logic ---
def _point_in_polygon(point, polygon):
    return Path(polygon).contains_point(point)


# --- Helper function: Point to Polygon Distance ---
def _min_distance_point_to_polygon(point, polygon):
    """计算一个点到多邊形所有边的最短距离。"""
    min_dist = float('inf')
    px, py = point
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]

        # 计算点到线段的距离
        line_vec = np.array([x2 - x1, y2 - y1])
        point_vec = np.array([px - x1, py - y1])
        line_len_sq = np.dot(line_vec, line_vec)

        if line_len_sq == 0:  # 线段是一个点
            dist = np.linalg.norm(point_vec)
        else:
            # t 是点在线段上的投影位置，范围在 [0, 1]
            t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
            projection = np.array([x1, y1]) + t * line_vec
            dist = np.linalg.norm(projection - np.array([px, py]))

        min_dist = min(min_dist, dist)
    return min_dist


# --- Helper function: Bounding Box to Polygon Distance ---
def _min_distance_bbox_to_polygon(bbox, polygon, num_samples_per_edge=3):
    """
    通过在边界框上采样点，估算边界框到多边形的最短距离。
    """
    x1, y1, x2, y2 = bbox
    sample_points = []

    # 在四个边上均匀采样点（包括角点）
    for i in range(num_samples_per_edge + 1):
        ratio = i / num_samples_per_edge
        sample_points.append((x1 + (x2 - x1) * ratio, y1))  # Top
        sample_points.append((x1 + (x2 - x1) * ratio, y2))  # Bottom
        sample_points.append((x1, y1 + (y2 - y1) * ratio))  # Left
        sample_points.append((x2, y1 + (y2 - y1) * ratio))  # Right

    # 返回所有采样点中，距离多边形最近的那个距离
    return min(_min_distance_point_to_polygon(pt, polygon) for pt in sample_points)


def check_intrusion(pid, bbox, center, camera_id, warning_zones, recorded_intrusions, status_cache, frame_idx,
                    stay_frames, safe_dist):
    newly_detected_zones = []
    is_currently_intruding = False

    for zone_index, polygon in enumerate(warning_zones.get(camera_id, [])):
        min_dist = _min_distance_bbox_to_polygon(bbox, polygon)
        cache_key = f"{pid}_{zone_index}"

        if _point_in_polygon(center, polygon) or min_dist < safe_dist:
            is_currently_intruding = True
            if cache_key not in status_cache:
                status_cache[cache_key] = frame_idx  # 记录进入的帧号

            stay_duration = frame_idx - status_cache[cache_key]
            if stay_duration >= stay_frames:
                if (pid, zone_index) not in recorded_intrusions:
                    recorded_intrusions.add((pid, zone_index))
                    newly_detected_zones.append(zone_index)
        else:
            status_cache.pop(cache_key, None)  # 离开区域，清除缓存

    return is_currently_intruding, newly_detected_zones


# --- Fight Detection Logic ---
# 速度计算
def calc_center_velocity(center_history):
    if len(center_history) < 2:
        return 0
    vec_a = np.array(center_history[-1])
    vec_b = np.array(center_history[-2])

    return np.linalg.norm(vec_a - vec_b)
# 加速度计算
def calc_center_acceleration(center_history):
    if len(center_history) < 3:
        return 0
    p1 = np.array(center_history[-1])
    p2 = np.array(center_history[-2])
    p3 = np.array(center_history[-3])
    v1 = p1 - p2
    v2 = p2 - p3
    return np.linalg.norm(v1 - v2)
# 关键点变化测试
def calc_kpts_change(kpts_deque):
    if len(kpts_deque) < 2:
        return 0
    arr = np.array(kpts_deque)
    diffs = np.linalg.norm(arr[1:] - arr[:-1], axis=2)  # 计算相邻帧关键点位移，shape=(len-1, num_keypoints)
    mean_change = diffs.mean()
    return mean_change

# 计算上半身关键点变化幅度
def _upper_body_motion_std(kpts_deque):
    upper_kpts_indices = [5, 6, 7, 8, 9, 10]
    upper_body = np.array(kpts_deque)[:, upper_kpts_indices, :]
    return np.std(upper_body, axis=0).mean()
# 估计人物朝向的单位向量
def _estimate_orientation(kpts):
    shoulder_mid = (kpts[5] + kpts[6]) / 2
    direction = kpts[0] - shoulder_mid
    return direction / (np.linalg.norm(direction) + 1e-5)
# 检测视频中人物是否打架可能
def detect_fight(ids, centers, fight_kpts_history, center_histories,
                 dist_thresh, motion_thresh, orient_thresh,
                 speed_thresh, accel_thresh, kpts_change_thresh):
    conflicts = []

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            pid1, pid2 = ids[i], ids[j]

            # 1. 距离判断
            dist = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
            if dist >= dist_thresh:
                continue
            dist_score = max(0.0, 1.0 - dist / dist_thresh)

            # 2. 判断关键点历史是否足够
            if len(fight_kpts_history[pid1]) < 5 or len(fight_kpts_history[pid2]) < 5:
                continue

            # 3. 速度和加速度
            speed1 = calc_center_velocity(center_histories[pid1])
            speed2 = calc_center_velocity(center_histories[pid2])
            accel1 = calc_center_acceleration(center_histories[pid1])
            accel2 = calc_center_acceleration(center_histories[pid2])
            if speed1 < speed_thresh and speed2 < speed_thresh:
                continue
            if accel1 < accel_thresh and accel2 < accel_thresh:
                continue
            speed_score = min((speed1 + speed2) / 2 / speed_thresh, 1.0)
            accel_score = min((accel1 + accel2) / 2 / accel_thresh, 1.0)

            # 4. 关键点变化
            kpts_change1 = calc_kpts_change(fight_kpts_history[pid1])
            kpts_change2 = calc_kpts_change(fight_kpts_history[pid2])
            if kpts_change1 < kpts_change_thresh and kpts_change2 < kpts_change_thresh:
                continue
            kpts_change_score = min((kpts_change1 + kpts_change2) / 2 / kpts_change_thresh, 1.0)


            # 5. 上半身运动判断
            motion1 = _upper_body_motion_std(fight_kpts_history[pid1])
            motion2 = _upper_body_motion_std(fight_kpts_history[pid2])
            if motion1 < motion_thresh or motion2 < motion_thresh:
                continue
            motion_score = min((motion1 + motion2) / 2 / motion_thresh, 1.0)

            # 6. 朝向判断
            vec1 = _estimate_orientation(list(fight_kpts_history[pid1])[-1])
            vec2 = _estimate_orientation(list(fight_kpts_history[pid2])[-1])
            dot = np.dot(vec1, vec2)
            if dot >= -orient_thresh:
                continue
            face_score = min(1.0, (abs(dot) - orient_thresh) / (1.0 - orient_thresh))

            # 5. 综合评分
            fight_score = round(
                0.25 * motion_score +
                0.20 * dist_score +
                0.20 * face_score +
                0.15 * speed_score +
                0.10 * accel_score +
                0.10 * kpts_change_score,
                3
            )

            # 6. 最终加入结果
            conflicts.append((pid1, pid2, fight_score))

    return conflicts


