# audio_detect.py
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

# 加载模型和标签表
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
class_map_path = tf.keras.utils.get_file(
    'yamnet_class_map.csv',
    'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
)
class_names = pd.read_csv(class_map_path)['display_name']

def detect_audio_events(waveform):
    scores, embeddings, spectrogram = yamnet_model(waveform)
    mean_scores = tf.reduce_mean(scores, axis=0)
    top5 = tf.argsort(mean_scores, direction='DESCENDING')[:5]
    return [(class_names[int(i)], float(mean_scores[i])) for i in top5]

