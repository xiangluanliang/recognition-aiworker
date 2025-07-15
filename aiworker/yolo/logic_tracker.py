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

    Args:
        current_centers: 当前帧检测到的所有人的中心点列表。
        prev_centers: 上一帧的ID与中心点的映射字典。
        threshold: 匹配的最大像素距离。

    Returns:
        一个与 current_centers 对应的ID列表。
    """
    if not current_centers:
        return []

    ids = [-1] * len(current_centers)
    used_prev_ids: Set[int] = set()

    # 计算所有距离
    dist_matrix = np.array(
        [[np.linalg.norm(np.array(c) - np.array(pc)) for pc in prev_centers.values()] for c in current_centers]
    )
    prev_pids = list(prev_centers.keys())

    # 贪心匹配
    while np.min(dist_matrix) < threshold:
        min_val = np.min(dist_matrix)
        if min_val >= threshold:
            break

        # 找到距离最近的一对
        curr_idx, prev_idx_in_list = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)

        pid = prev_pids[prev_idx_in_list]

        # 如果双方都未被匹配，则成功匹配
        if ids[curr_idx] == -1 and pid not in used_prev_ids:
            ids[curr_idx] = pid
            used_prev_ids.add(pid)

        # 将这对的距离设为无穷大，不再参与后续匹配
        dist_matrix[curr_idx, :] = float('inf')
        dist_matrix[:, prev_idx_in_list] = float('inf')

    # 为未匹配到的人分配新ID
    next_id = max(prev_centers.keys(), default=0) + 1
    for i in range(len(ids)):
        if ids[i] == -1:
            ids[i] = next_id
            next_id += 1

    return ids
