# aiworker/audio/event_handlers.py

from .audio_detect import AudioEventDetector
from .preprocess import load_audio

try:
    audio_detector = AudioEventDetector()
except Exception as e:
    print(f"致命错误：无法初始化 AudioEventDetector: {e}")
    audio_detector = None

INTERESTING_CLASSES = {
    "Explosion": 1.0,
    "Gunshot": 2.0,
    "Screaming": 1.5,
    "Alarm": 1.0,
    "Firecracker": 0.8,
    "Shout": 1.2,
    "Yell": 1.2,
    "Crying, sobbing": 0.7
}


def is_abnormal(label, score):
    if label not in INTERESTING_CLASSES:
        return False
    # 权重越高，阈值越低，越容易触发告警
    threshold = 0.25 / INTERESTING_CLASSES[label]
    return score > threshold


def trigger_alarm(event, confidence, processor):
    """
    当音频事件触发时，调用 AbnormalBehaviorProcessor 实例来记录事件和保存视频切片。
    """
    if processor:
        print(f"🚨 音频事件触发视频保存：{event} (置信度 {confidence:.2f})")
        # 复用已有的 _log_event 方法，实现音视频联合证据保全
        event_type_key = f'audio_{event.lower().replace(", ", "_").replace(" ", "_")}'
        processor._log_event(
            event_type=event_type_key,
            pid=0,  # 0 代表环境事件
            confidence=confidence,
            frame=processor.video_buffer[-1] if processor.video_buffer else None,
            details={'trigger': 'audio'}
        )
    else:
        print(f"🚨 触发异常声学告警：{event} (置信度 {confidence:.2f})")


def handle_audio_file(path: str, processor):
    """
    处理单个音频文件的完整流程。
    """
    if not audio_detector:
        print("错误：音频检测器未成功初始化，跳过处理。")
        return []

    # 1. 加载音频文件，确保采样率为32k
    try:
        waveform = load_audio(path, sr=32000)
    except Exception as e:
        print(f"错误：加载音频文件 {path} 失败: {e}")
        return []

    results = audio_detector.detect(waveform)

    abnormal_events = []
    for result in results:
        label = result['label']
        score = result['score']
        if label in INTERESTING_CLASSES:
            if is_abnormal(label, score):
                trigger_alarm(label, score, processor)
            abnormal_events.append(result)

    return abnormal_events
