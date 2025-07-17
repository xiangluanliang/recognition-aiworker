# run_audio_detect.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './models_code')))

import torch
import librosa
import numpy as np
import pandas as pd

from models_code.Cnn14 import Cnn14
from models_code.pytorch_utils import do_mixup, interpolate, pad_framewise_output

# 设置关注的类别及其权重（权重越高，阈值越低 => 越容易告警）
INTERESTING_CLASSES = {
    "Explosion": 1.0,
    "Gunshot": 2.0,
    "Scream": 1.5,
    "Alarm": 1.0,
    "Firecracker": 0.8
}

def is_abnormal(label, score):
    if label not in INTERESTING_CLASSES:
        return False
    threshold = 0.25 / INTERESTING_CLASSES[label]
    return score > threshold

def trigger_alarm(event, confidence):
    # 模拟告警行为，可以改成写数据库、发消息等
    print(f"🚨 异常声音告警：{event} (置信度 {confidence:.2f})")

# -------------------- 加载模型 --------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320,
              mel_bins=64, fmin=50, fmax=14000, classes_num=527)
checkpoint = torch.load('models/Cnn14_mAP=0.431.pth', map_location=device)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# -------------------- 加载标签 --------------------
labels = pd.read_csv('metadata/class_labels_indices.csv')
idx_to_label = dict(zip(labels['index'], labels['display_name']))

# -------------------- 读取音频 --------------------
audio_path = 'audio/explosion.WAV'
waveform, sr = librosa.load(audio_path, sr=32000, mono=True)
if len(waveform) < 32000:
    waveform = np.pad(waveform, (0, 32000 - len(waveform)))
else:
    waveform = waveform[:32000]
waveform_tensor = torch.tensor(waveform).float().unsqueeze(0).to(device)

# -------------------- 推理 --------------------
with torch.no_grad():
    output = model(waveform_tensor)
    clipwise_output = output['clipwise_output'].cpu().numpy()[0]

# -------------------- 处理结果 --------------------
top_indices = np.argsort(clipwise_output)[::-1][:10]
filtered_results = []

print("\n🎧 Top Predicted Sounds:")
for i in top_indices:
    label = idx_to_label[i]
    score = clipwise_output[i]
    if label in INTERESTING_CLASSES:
        if is_abnormal(label, score):
            trigger_alarm(label, score)
        filtered_results.append({"label": label, "score": float(score)})
        print(f"{label}: {score:.3f} {'✅ 告警' if is_abnormal(label, score) else ''}")

# 返回结构化结果（如你嵌入服务端可用）
print("\n📦 Structured Results:")
print(filtered_results)
