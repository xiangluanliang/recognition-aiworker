import os
import subprocess
from script.services.audioService.event_handlers import handle_audio_file
FFMPEG_PATH = r"D:\DevTools\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"

def extract_audio_from_video(video_path, output_audio_path=None):
    print(f"[extract_audio_from_video] video_path: {video_path}")

    if not output_audio_path:
        base, _ = os.path.splitext(video_path)
        output_audio_path = base + ".wav"

    command = [
        FFMPEG_PATH,
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_audio_path
    ]

    print(f"[extract_audio_from_video] Running command: {' '.join(command)}")

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"[extract_audio_from_video] 提取完成: {output_audio_path}")
        return output_audio_path
    except subprocess.CalledProcessError as e:
        print(f"[extract_audio_from_video] 提取失败: {e.stderr.decode()}")
        return None


def analyze_video_audio(video_path):
    print(f"[analyze_video_audio] 开始分析视频: {video_path}")
    audio_path = extract_audio_from_video(video_path)

    if not audio_path or not os.path.exists(audio_path):
        print(f"[analyze_video_audio] 提取音频失败或文件不存在: {audio_path}")
        return {"error": "音频提取失败"}

    print(f"[analyze_video_audio] 音频路径: {audio_path}")
    result = handle_audio_file(audio_path)
    print(f"[analyze_video_audio] 模型返回结果: {result}")
    return result
