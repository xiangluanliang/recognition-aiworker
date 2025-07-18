# aiworker/utils/file_saver.py
import os
import cv2
import datetime
import logging
import subprocess
import shutil

from ..config import MEDIA_ROOT

logger = logging.getLogger(__name__)


def save_clip(person_id, frame_idx, video_buffer, fps, sub_dir='clips', event_type='event', audio_path=None):
    """
    保存视频切片 (新版：统一ffmpeg转码流程，支持音频合并)。
    """
    if not video_buffer:
        logger.warning("视频缓冲区为空，无法保存切片。")
        return None

    if not shutil.which('ffmpeg'):
        logger.error("未找到 ffmpeg 命令，无法进行视频转码或合并。请在服务器上安装 ffmpeg。")
        return None

    # --- 路径和文件名设置 ---
    base_dir = os.path.join(MEDIA_ROOT, sub_dir)
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{event_type}_pid{person_id}_frame{frame_idx}_{timestamp}"
    
    temp_video_path = os.path.join(base_dir, f"{base_filename}_temp_video.mp4")
    final_path = os.path.join(base_dir, f"{base_filename}.mp4")

    # --- 步骤 1: 将视频帧写入临时的无声视频文件 ---
    try:
        frame_height, frame_width, _ = video_buffer[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_video_path, fourcc, float(fps), (frame_width, frame_height))
        for frame in video_buffer:
            if frame is not None:
                writer.write(frame)
        writer.release()
    except Exception as e:
        logger.error(f"使用OpenCV写入临时视频文件时失败: {e}", exc_info=True)
        return None

    # --- 步骤 2: 动态构建并执行 ffmpeg 命令 ---
    try:
        ffmpeg_command = ['ffmpeg', '-y', '-i', temp_video_path]
        
        # 如果有音频，则添加音频输入和相关编码参数
        if audio_path and os.path.exists(audio_path):
            logger.info(f"检测到音频文件 '{audio_path}'，将进行音视频合并。")
            ffmpeg_command.extend(['-i', audio_path])
            ffmpeg_command.extend(['-c:v', 'libx264', '-preset', 'fast', '-c:a', 'aac', '-shortest'])
        else:
            # 如果没有音频，则仅进行视频转码，并明确指定无音频流
            logger.info(f"未提供有效音频，仅进行视频转码。")
            ffmpeg_command.extend(['-c:v', 'libx264', '-preset', 'fast', '-an']) # -an 标志表示 "audio none"
            
        ffmpeg_command.append(final_path)

        # 执行命令
        result = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        logger.info(f"ffmpeg 处理完成，最终文件保存至: {final_path}")
        return final_path

    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg 处理失败。返回码: {e.returncode}, Stderr: {e.stderr}")
        # 如果合并/转码失败，作为备用方案，尝试保留原始的临时文件
        if not os.path.exists(final_path):
            shutil.move(temp_video_path, final_path)
        return final_path
    finally:
        # --- 步骤 3: 清理临时的视频文件 ---
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            
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