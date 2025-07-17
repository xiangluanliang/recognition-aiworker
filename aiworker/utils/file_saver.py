# aiworker/utils/file_saver.py
import os
import cv2
import datetime
import logging
import subprocess
import shutil

from ..config import MEDIA_ROOT

logger = logging.getLogger(__name__)


def save_clip(person_id, frame_idx, video_buffer, fps, sub_dir='clips', event_type='event'):
    """
    保存视频切片，并自动转码为Web兼容格式(H.264/AAC)。
    """
    if not video_buffer:
        logger.warning("视频缓冲区为空，无法保存切片。")
        return None

    # ✅ 步骤 1: 检查服务器上是否存在 ffmpeg 命令
    if not shutil.which('ffmpeg'):
        logger.error("未找到 ffmpeg 命令，无法进行视频转码。请在服务器上安装 ffmpeg。")
        return None

    # --- 路径和文件名设置 ---
    base_dir = os.path.join(MEDIA_ROOT, sub_dir)
    os.makedirs(base_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{event_type}_pid{person_id}_frame{frame_idx}_{timestamp}"

    # 定义临时文件和最终文件的路径
    temp_path = os.path.join(base_dir, f"{base_filename}_temp.mp4")
    final_path = os.path.join(base_dir, f"{base_filename}.mp4")

    # --- 步骤 2: 使用 OpenCV 快速保存一个临时的、未优化的视频文件 ---
    try:
        frame_height, frame_width, _ = video_buffer[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用一个基础的编码器
        writer = cv2.VideoWriter(temp_path, fourcc, fps, (frame_width, frame_height))

        for frame in video_buffer:
            if frame is not None:
                writer.write(frame)
        writer.release()
        logger.info(f"临时视频文件已保存: {temp_path}")

    except Exception as e:
        logger.error(f"使用OpenCV写入临时视频文件时失败: {e}", exc_info=True)
        return None

    # --- 步骤 3: 调用 ffmpeg 对临时文件进行转码，生成最终的Web兼容文件 ---
    try:
        logger.info(f"开始转码视频文件到: {final_path}")
        # 这就是您提供的、工作正常的ffmpeg命令
        ffmpeg_command = [
            'ffmpeg',
            '-y',  # 覆盖输出文件
            '-i', temp_path,
            '-c:v', 'libx264',  # H.264视频编码
            '-profile:v', 'high',
            '-level', '4.0',
            '-preset', 'fast',
            '-c:a', 'aac',  # AAC音频编码
            '-b:a', '128k',
            final_path
        ]

        # 执行命令，并隐藏控制台输出
        result = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        logger.info("视频转码成功。")
        return final_path

    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg转码失败。返回码: {e.returncode}")
        logger.error(f"FFmpeg stdout: {e.stdout}")
        logger.error(f"FFmpeg stderr: {e.stderr}")
        return None  # 转码失败，返回None

    finally:
        # --- 步骤 4: 无论成功与否，都删除临时文件 ---
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"已删除临时视频文件: {temp_path}")

def save_event_image(frame, person_id, frame_idx, sub_dir='images', event_type='event'):
    """保存事件的单帧截图。"""
    try:
        base_dir = os.path.join(MEDIA_ROOT, sub_dir)
        os.makedirs(base_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{event_type}_pid{person_id}_frame{frame_idx}_{timestamp}.jpg"
        full_path = os.path.join(base_dir, filename)

        cv2.imwrite(full_path, frame)
        logger.info(f"成功保存事件图片: {full_path}")
        return full_path
    except Exception as e:
        logger.error(f"保存事件图片失败: {e}", exc_info=True)
        return None