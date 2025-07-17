# aiworker/config.py
import os

# --- General Service Configuration ---
# 从环境变量获取，如果未设置则使用默认值
DJANGO_API_TOKEN = os.environ.get('DJANGO_API_TOKEN', '3d814802906b91d7947518f5d0191a42795cace7')
DJANGO_API_BASE_URL = os.environ.get('DJANGO_API_URL', 'https://8.152.101.217/api/test/api/')
RTMP_SERVER_URL = os.environ.get('RTMP_SERVER_URL', 'rtmp://localhost:9090/live')
CACHE_REFRESH_INTERVAL = 300  # 5 minutes
# MEDIA_ROOT = os.environ.get('MEDIA_ROOT', '/media') # 媒体文件保存根目录
MEDIA_ROOT = '/srv/http/recognition_media'

# --- Stream Processing Parameters ---
FRAME_WIDTH = 854
FRAME_HEIGHT = 480
FRAME_SKIP_RATE = 3  # 每5帧进行一次昂贵的AI计算
JPEG_QUALITY = 70

# --- Model Paths and Filenames ---
# 基础模型目录
MODEL_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)),'..', 'dnn_models')

# Face Detector (OpenCV DNN)
FACE_DETECTOR_PROTOTXT_FILENAME = 'opencv_face_detector.pbtxt'
FACE_DETECTOR_WEIGHTS_FILENAME = 'opencv_face_detector_uint8.pb'

# Face Recognition (ONNX)
FACE_RECOGNITION_MODEL_FILENAME = 'InceptionResnetV1_vggface2.onnx'

# Liveness Detection (ONNX)
OULU_LIVENESS_MODEL_FILENAME = 'OULU_Protocol_2_model_0_0.onnx'

# Landmark Predictor (Dlib)
DLIB_LANDMARK_PREDICTOR_FILENAME = 'shape_predictor_68_face_landmarks.dat'


# --- AI Model Tuning Parameters ---

# Face Detector
FACE_DETECTOR_CONFIDENCE_THRESHOLD = 0.5

# Face Recognition
FACE_RECOGNITION_THRESHOLD = 1.00  # 欧氏距离阈值，越小越相似

# Liveness - OULU Model
OULU_LIVENESS_INPUT_SIZE = (224, 224)
OULU_LIVENESS_THRESHOLD = 0.5  # 分数高于此值为真人
OULU_LIVENESS_HARD_THRESHOLD = 0.2 # 分数低于此值立刻判定为欺诈
MIN_EFFECTIVE_LIVENESS_ROI_SIZE = 100 # 人脸框小于此像素尺寸，则不进行OULU活体检测

# Liveness - Blink Detection
EYE_AR_THRESH = 0.23  # 眼睛纵横比阈值
# 连续闭眼帧数 一般需要至少为2，但我们的服务器很难得到一个连续帧
# 为了优化用户体验，我们将改变眨眼检测的算法
# 不再要求“连续闭眼”，而是检测一个更符合生理特征的“睁眼 -> 闭眼 -> 睁眼”的完整序列。
# 哪怕“闭眼”状态只被我们捕捉到一帧，只要它发生在一个“睁眼”状态之后，
# 我们就认为用户有眨眼的意图，并开始寻找下一个“睁眼”状态来确认整个动作。
EYE_AR_CONSEC_FRAMES = 1
BLINK_TIMEOUT_FRAMES = 150  # 眨眼检测超时帧数

# --- UI and Guidance Parameters ---
RECOMMENDED_FACE_RECT_RATIO = 0.5  # 推荐人脸区域占画面的比例
RECOMMENDED_FACE_MIN_PIXELS = 150  # 推荐人脸的最小像素尺寸

# --- Abnormal Behavior Detection Parameters ---
# Models
YOLO_POSE_MODEL_FILENAME = "yolov8l-pose.pt"

# Tracking
PERSON_MATCHING_THRESHOLD = 60 # 追踪时匹配ID的像素距离阈值

#这个是切片的时长，但是我们的视频短短的
CLIP_DURATION_SECONDS = 2

# Fall Detection
FALL_ANGLE_THRESHOLD = 35.0 # 身体倾斜角度阈值
FALL_WINDOW_SIZE = 3 # 连续N帧满足条件才判断为摔倒
FALL_COOLDOWN_FRAMES = 150 # 摔倒事件上报后的冷却时间

# Intrusion Detection (默认值，实际会从API获取)
DEFAULT_STAY_SECONDS = 5
DEFAULT_SAFE_DISTANCE = 50.0

# Fight Detection
FIGHT_DISTANCE_THRESHOLD = 180           # 人体之间距离阈值（像素）
FIGHT_MOTION_THRESHOLD = 3.5             # 运动标准差阈值
FIGHT_ORIENTATION_SIMILARITY_THRESHOLD = -0.1  # 朝向点积阈值，负数代表面对面
FIGHT_SPEED_THRESHOLD = 10.0             # 速度阈值（像素/帧）
FIGHT_ACCELERATION_THRESHOLD = 5.0       # 加速度阈值（像素/帧²）
FIGHT_KPTS_CHANGE_THRESHOLD = 0.1        # 关键点变化阈值

# Drawing
POSE_PAIRS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16)
]

# audio

AUDIO_MODEL_DIR = os.path.join(MODEL_DIR, 'audio_models')
METADATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'metadata')

AUDIO_MODEL_CHECKPOINT_FILENAME = "Cnn14_mAP=0.431.pth"
AUDIO_MODEL_LABELS_FILENAME = "class_labels_indices.csv"

# deepseek key
DEEPSEEK_API_KEY = 'sk-981044236a6a4402a991d5b83c8b61e2'
