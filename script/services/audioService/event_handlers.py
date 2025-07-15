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

def handle_audio_file(path):
    waveform = load_audio(path)
    results = detect_audio_events(waveform)
    filtered_results = []
    for label, score in results:
        if label in INTERESTING_CLASSES:
            if is_abnormal(label, score):
                trigger_alarm(label, score)
            filtered_results.append({"label": label, "score": score})
    return filtered_results

def trigger_alarm(event, confidence):
    # 这里写告警的具体逻辑，比如：
    # 1) 写数据库
    # 2) 发送消息通知
    # 3) 触发摄像头录像/联动
    print(f"🚨 触发异常声学告警：{event} (置信度 {confidence:.2f})")
