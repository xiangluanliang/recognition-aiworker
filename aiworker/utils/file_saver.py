
# aiworker/utils/file_saver.py
import cv2
import os
from ..config import MEDIA_ROOT


def save_clip(pid, frame_idx, clip_buffer, fps, subfolder, event_type):
    """保存视频切片。"""
    base_dir = os.path.join(MEDIA_ROOT, 'subject_images', subfolder)
    os.makedirs(base_dir, exist_ok=True)
    clip_path = os.path.join(base_dir, f"{event_type}_{pid}_{frame_idx}.mp4")

    if not clip_buffer: return None

    height, width, _ = clip_buffer[0].shape
    writer = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for f in clip_buffer:
        writer.write(f)
    writer.release()

    return os.path.relpath(clip_path, str(MEDIA_ROOT))


def save_event_image(frame, pid, frame_idx, subfolder, event_type):
    """保存事件快照图片。"""
    base_dir = os.path.join(MEDIA_ROOT, 'subject_images', subfolder)
    os.makedirs(base_dir, exist_ok=True)
    filename = f"{event_type}_{pid}_{frame_idx}.jpg"
    full_path = os.path.join(base_dir, filename)
    cv2.imwrite(full_path, frame)

    return os.path.relpath(full_path, str(MEDIA_ROOT))