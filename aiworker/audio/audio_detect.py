# aiworker/audio/audio_detect.py

import torch
import numpy as np
import pandas as pd
import os
import logging

from .models_code.Cnn14 import Cnn14
from ..config import AUDIO_MODEL_DIR, METADATA_DIR, AUDIO_MODEL_CHECKPOINT_FILENAME, AUDIO_MODEL_LABELS_FILENAME

logger = logging.getLogger(__name__)


class AudioEventDetector:
    """
    封装了Cnn14音频事件检测模型的服务类。
    负责加载模型、标签和执行推理。
    """

    def __init__(self):
        logger.info("正在初始化音频事件检测器 (Cnn14)...")
        self.device = 'cpu'  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"音频模型将运行在设备: {self.device}")

        model_path = os.path.join(AUDIO_MODEL_DIR, AUDIO_MODEL_CHECKPOINT_FILENAME)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"音频模型文件未找到: {model_path}")

        self.model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320,
                           mel_bins=64, fmin=50, fmax=14000, classes_num=527)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
        logger.info("Cnn14模型加载成功。")

        # 3. 加载类别标签
        labels_path = os.path.join(METADATA_DIR, AUDIO_MODEL_LABELS_FILENAME)
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"音频标签文件未找到: {labels_path}")

        labels_df = pd.read_csv(labels_path)
        self.idx_to_label = dict(zip(labels_df['index'], labels_df['display_name']))
        logger.info("音频类别标签加载成功。")

    def detect(self, waveform: np.ndarray) -> list[dict]:
        """
        对输入的音频波形进行事件检测。
        """
        # 1. 预处理：确保输入长度符合模型要求（1秒，32000个采样点）
        if len(waveform) < 32000:
            waveform = np.pad(waveform, (0, 32000 - len(waveform)))
        else:
            waveform = waveform[:32000]

        # 2. 转换为Tensor并移动到指定设备
        waveform_tensor = torch.tensor(waveform).float().unsqueeze(0).to(self.device)

        # 3. 执行模型推理
        with torch.no_grad():
            output = self.model(waveform_tensor)
            # 获取片段级别的输出概率
            clipwise_output = output['clipwise_output'].cpu().numpy()[0]

        # 4. 处理并返回结果
        # 找到分数最高的10个类别
        top_indices = np.argsort(clipwise_output)[::-1][:10]

        results = []
        for i in top_indices:
            label = self.idx_to_label.get(i, "未知类别")
            score = clipwise_output[i]
            results.append({"label": label, "score": float(score)})

        return results