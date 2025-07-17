# event_handlers.py
from .audio_detect import detect_audio_events
from .preprocess import load_audio

INTERESTING_CLASSES = {
    "Screaming": 1.0,
    "Shout": 1.0,
    "Fight": 1.0,
    "Gunshot, gunfire": 1.2,
    "Fusillade": 1.2,
    "Explosion": 1.5,
    "Crying, sobbing": 0.8,
    "Child screaming": 1.0,
    "Yell": 1.0,
    "Groan": 0.7,
    "Whimper": 0.7,
    "Wail, moan": 0.7,
}

def is_abnormal(label, score):
    if label not in INTERESTING_CLASSES:
        return False
    threshold = 0.25 / INTERESTING_CLASSES[label]  # 0.25是你现在的阈值
    return score > threshold

def handle_audio_file(path, processor):
    waveform = load_audio(path)
    results = detect_audio_events(waveform)
    filtered_results = []
    for label, score in results:
        if label in INTERESTING_CLASSES:
            if is_abnormal(label, score):
                # --- 修改点 2：调用新的 trigger_alarm，并把 processor 传下去 ---
                trigger_alarm(label, score, processor)
            filtered_results.append({"label": label, "score": score})
    return filtered_results


def trigger_alarm(event, confidence, processor):
    """
    当音频事件触发时，调用 AbnormalBehaviorProcessor 实例来记录事件和保存视频切片。
    """
    if processor:
        print(f"🚨 音频事件触发视频保存：{event} (置信度 {confidence:.2f})")
        # 复用已有的 _log_event 方法，实现音视频联合证据保全
        # 我们虚构一个 person_id=0 来代表这是由环境（音频）触发的事件
        processor._log_event(
            event_type=f'audio_{event.lower().replace(" ", "_")}',
            pid=0,
            confidence=confidence,
            frame=processor.video_buffer[-1] if processor.video_buffer else None,
            details={'trigger': 'audio'}
        )
    else:
        print(f"🚨 触发异常声学告警：{event} (置信度 {confidence:.2f})")