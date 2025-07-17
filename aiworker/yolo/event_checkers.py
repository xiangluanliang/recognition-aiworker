# aiworker/yolo/event_checkers.py
import numpy as np
from matplotlib.path import Path

from aiworker.config import FRAME_SKIP_RATE, INTRUSION_GRACE_PERIOD_FRAMES


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

    fps = 25
    frame_interval = FRAME_SKIP_RATE

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

    if 'fall_time' not in status:
        status['fall_time'] = 0.0

    # -----  分数计算

    # 角度得分（0-30）
    angle_score = 30 if angle < 10 else np.clip((angle_thresh - angle) / angle_thresh * 30, 0, 30)
    # 宽高比得分（0-50）
    wh_score = np.clip((wh_ratio - 1) / 1.2 * 40, 0, 40)
    # 高度变化分数
    motion_score = np.clip(height_change / 0.3 * 10, 0, 10)
    # 动作持续得分（0-20）
    # status = person_fall_status[pid]
    # duration_score = np.clip(status['fall_frame_count'] / window_size * 20, 0, 20)
    duration_score = np.clip(status['fall_time'] / 0.3 * 20, 0, 20)

    total_score = (1.5 * angle_score + 1.2 * wh_score + 1.0 * motion_score + 1.8 * duration_score) / 100

    status = person_fall_status[pid]

    if status['cooldown_counter'] > 0:
        status['cooldown_counter'] -= 1
        return True, False, 0.9

    is_posture_fallen = angle < angle_thresh
    is_shape_or_motion_abnormal = (wh_ratio > 1.2) or (height_change > 0.3)

    is_fall = is_posture_fallen and is_shape_or_motion_abnormal

    if is_fall:
        status['fall_time'] += (frame_interval / fps)
        if status['fall_time'] >= 0.2:
            if not status.get('is_falling', False):
                status['is_falling'] = True
                status['cooldown_counter'] = cooldown_frames
                return True, True, total_score
            return True, False, total_score
    else:
        status['fall_frame_count'] = 0
        status['is_falling'] = False

    if status['fall_frame_count'] >= window_size:
        if not status['is_falling']:
            status['is_falling'] = True
            status['cooldown_counter'] = cooldown_frames
            score = 0.95
            return True, True, score
        return True, False, 0.8

    return False, False, 0.1

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


def check_intrusion(pid, bbox, center, zone_list, recorded_intrusions, status_cache, frame_idx):
    """
    重构后的入侵检测逻辑，引入了“宽容期”以应对检测波动。
    """
    newly_detected_zones = []
    is_currently_intruding_any_zone = False

    for zone_info in zone_list:
        zone_id = zone_info['id']
        polygon = zone_info['polygon']
        safe_dist = zone_info['safe_dist']
        stay_frames = zone_info['stay_frames']
        cache_key = f"{pid}_{zone_id}"

        is_inside_zone = _point_in_polygon(center, polygon) or _min_distance_bbox_to_polygon(bbox, polygon) < safe_dist

        if is_inside_zone:
            is_currently_intruding_any_zone = True

            # 如果这个人是第一次进入这个区域
            if cache_key not in status_cache:
                # 初始化计时器和状态
                status_cache[cache_key] = {
                    'start_frame': frame_idx,
                    'last_seen_frame': frame_idx
                }
            else:
                # 如果不是第一次，只更新“最后见到”的帧
                status_cache[cache_key]['last_seen_frame'] = frame_idx

            # 计算总的滞留时长
            stay_duration = frame_idx - status_cache[cache_key]['start_frame']

            # 判断是否达到报警条件
            if stay_duration >= stay_frames:
                # 如果这个事件还没有被记录过
                if (pid, zone_id) not in recorded_intrusions:
                    recorded_intrusions.add((pid, zone_id))
                    newly_detected_zones.append(zone_info)
        else:
            # 如果人不在区域内，检查是否超过了“宽容期”
            if cache_key in status_cache:
                frames_since_last_seen = frame_idx - status_cache[cache_key]['last_seen_frame']

                # 只有当人离开区域超过宽容期后，才真正重置计时器
                if frames_since_last_seen > INTRUSION_GRACE_PERIOD_FRAMES:
                    status_cache.pop(cache_key, None)

    return is_currently_intruding_any_zone, newly_detected_zones


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
            center1, center2 = np.array(centers[i]), np.array(centers[j])

            # 1. 距离判断
            dist = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
            dist_score = max(0.0, 1.0 - dist / dist_thresh)

            # 跳过太远的人
            if dist_score <= 0:
                continue

            # 2. 判断关键点历史是否足够
            if len(fight_kpts_history[pid1]) < 5 or len(fight_kpts_history[pid2]) < 5:
                continue

            # 3. 速度和加速度
            speed1 = calc_center_velocity(center_histories[pid1])
            speed2 = calc_center_velocity(center_histories[pid2])
            speed_score = min((speed1 + speed2) / 2 / speed_thresh, 1.0)

            accel1 = calc_center_acceleration(center_histories[pid1])
            accel2 = calc_center_acceleration(center_histories[pid2])
            accel_score = min((accel1 + accel2) / 2 / accel_thresh, 1.0)

            # 4. 关键点变化
            kpts_change1 = calc_kpts_change(fight_kpts_history[pid1])
            kpts_change2 = calc_kpts_change(fight_kpts_history[pid2])
            kpts_change_score = min((kpts_change1 + kpts_change2) / 2 / kpts_change_thresh, 1.0)

            # 5. 上半身运动判断
            motion1 = _upper_body_motion_std(fight_kpts_history[pid1])
            motion2 = _upper_body_motion_std(fight_kpts_history[pid2])
            motion_score = min((motion1 + motion2) / 2 / motion_thresh, 1.0)

            # 6. 朝向判断
            vec1 = _estimate_orientation(list(fight_kpts_history[pid1])[-1])
            vec2 = _estimate_orientation(list(fight_kpts_history[pid2])[-1])
            orientation_dot = np.dot(vec1, vec2)
            face_score = min(1.0, (abs(orientation_dot) - orient_thresh) / (1.0 - orient_thresh)) if orientation_dot < -orient_thresh else 0.0

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

            # 统计满足的强打架特征个数
            strong_signs = 0
            if motion_score > 0.5:  # 上半身剧烈运动
                strong_signs += 1
            if face_score > 0.5:  # 面对面朝向
                strong_signs += 1
            if speed_score > 0.5:  # 移动速度较快
                strong_signs += 1
            if kpts_change_score > 0.5:  # 关键点剧烈变化
                strong_signs += 1

            # 6. 最终加入结果
            if fight_score >= 0.5 and strong_signs >= 2:
                conflicts.append((pid1, pid2, fight_score))

    return conflicts


