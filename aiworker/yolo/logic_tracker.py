# aiworker/yolo/logic_tracker.py
import numpy as np
from typing import Dict, List, Tuple, Set


def match_person_id(
        current_centers: List[Tuple[int, int]],
        prev_centers: Dict[int, Tuple[int, int]],
        threshold: int
) -> List[int]:
    """
    通过最小距离法，为当前帧检测到的人匹配之前帧的ID。
    """
    # ✅ 核心修正：添加处理边界情况的保护逻辑 (Guard Clause)
    if not current_centers:
        return []

    # 如果没有上一帧的记录可供匹配（例如，这是视频的第一帧），
    # 那么当前帧检测到的所有人都是“新人物”，直接为他们分配从0开始的新ID。
    if not prev_centers:
        return list(range(len(current_centers)))

    # --- 只有在 current_centers 和 prev_centers 都有数据时，才执行以下复杂的匹配逻辑 ---

    ids = [-1] * len(current_centers)
    used_prev_ids: Set[int] = set()

    # 计算所有距离
    dist_matrix = np.array(
        [[np.linalg.norm(np.array(c) - np.array(pc)) for pc in prev_centers.values()] for c in current_centers]
    )
    prev_pids = list(prev_centers.keys())

    # 贪心匹配
    # 增加一个检查，确保dist_matrix不是空的，以防万一
    if dist_matrix.size == 0:
        # 如果矩阵为空，说明无法匹配，直接分配新ID
        next_id = max(prev_centers.keys(), default=-1) + 1
        return list(range(next_id, next_id + len(current_centers)))

    while np.min(dist_matrix) < threshold:
        min_val = np.min(dist_matrix)
        if min_val >= threshold:
            break

        curr_idx, prev_idx_in_list = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        pid = prev_pids[prev_idx_in_list]

        if ids[curr_idx] == -1 and pid not in used_prev_ids:
            ids[curr_idx] = pid
            used_prev_ids.add(pid)

        dist_matrix[curr_idx, :] = float('inf')
        dist_matrix[:, prev_idx_in_list] = float('inf')

    # 为未匹配到的人分配新ID
    next_id = max(prev_centers.keys(), default=-1) + 1
    for i in range(len(ids)):
        if ids[i] == -1:
            ids[i] = next_id
            next_id += 1

    return ids
