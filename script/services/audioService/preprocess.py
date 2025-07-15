# preprocess.py
import librosa
import os
import subprocess
import tempfile

def extract_audio_from_video(video_path):
    # 创建临时音频文件
    temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_audio_path = temp_audio.name
    temp_audio.close()

    # ffmpeg命令：提取音频，转为wav格式，单声道，16k采样率（你可以根据模型需求调整）
    command = [
        'ffmpeg', '-y', '-i', video_path,
        '-vn',  # 不要视频流
        '-ac', '1',  # 单声道
        '-ar', '16000',  # 采样率16k
        '-f', 'wav',
        temp_audio_path
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        os.unlink(temp_audio_path)
        raise RuntimeError(f"ffmpeg extract audio failed: {result.stderr.decode()}")

    return temp_audio_path

def load_audio(filepath, sr=16000):
    waveform, sample_rate = librosa.load(filepath, sr=sr, mono=True)
    return waveform
