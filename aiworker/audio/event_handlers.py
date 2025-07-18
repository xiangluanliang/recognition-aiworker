# aiworker/audio/event_handlers.py
import time
import logging
from .audio_detect import AudioEventDetector
from .preprocess import load_audio
logger = logging.getLogger(__name__)
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
    threshold = 0.05 / INTERESTING_CLASSES[label]
    return score > threshold


def trigger_alarm(event, confidence, processor):
    """
    当音频事件触发时：
    1. 将事件信息暂存到处理器，用于后续的音视频融合。
    2. 继续调用 _log_event，直接记录独立的音频告警事件。
    """
    if processor:
        logger.info(f"======> trigger_alarm 已被成功调用！声音: '{event}', 置信度: {confidence:.2f} <======")

        event_info = {
            'label': event,
            'score': confidence,
            'timestamp': time.time()
        }
        processor.last_audio_event_for_fusion = event_info

        event_type_key = f'audio_{event.lower().replace(", ", "_").replace(" ", "_")}'
        processor._log_event(
            event_type=event_type_key,
            pid=0,
            confidence=confidence,
            frame=processor.video_buffer[-1] if processor.video_buffer else None,
            details={'trigger': 'audio_only'}
        )


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
