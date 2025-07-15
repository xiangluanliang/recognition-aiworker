# aiworker/config.py
import os

# --- General Service Configuration ---
# 从环境变量获取，如果未设置则使用默认值
DJANGO_API_TOKEN = os.environ.get('DJANGO_API_TOKEN', '3d814802906b91d7947518f5d0191a42795cace7')
DJANGO_API_BASE_URL = os.environ.get('DJANGO_API_URL', 'http://127.0.0.1:8000/api/test/')
RTMP_SERVER_URL = os.environ.get('RTMP_SERVER_URL', 'rtmp://localhost:9090/live')
CACHE_REFRESH_INTERVAL = 300  # 5 minutes

# --- Stream Processing Parameters ---
FRAME_WIDTH = 854
FRAME_HEIGHT = 480
FRAME_SKIP_RATE = 5  # 每5帧进行一次昂贵的AI计算
JPEG_QUALITY = 70

# --- Model Paths and Filenames ---
# 基础模型目录
MODEL_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'dnn_models')

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
FACE_DETECTOR_CONFIDENCE_THRESHOLD = 0.4

# Face Recognition
FACE_RECOGNITION_THRESHOLD = 0.8  # 欧氏距离阈值，越小越相似

# Liveness - OULU Model
OULU_LIVENESS_INPUT_SIZE = (224, 224)
OULU_LIVENESS_THRESHOLD = 0.015  # 分数高于此值为真人
MIN_EFFECTIVE_LIVENESS_ROI_SIZE = 100 # 人脸框小于此像素尺寸，则不进行OULU活体检测

# Liveness - Blink Detection
EYE_AR_THRESH = 0.25  # 眼睛纵横比阈值
EYE_AR_CONSEC_FRAMES = 2  # 连续闭眼帧数
BLINK_TIMEOUT_FRAMES = 150  # 眨眼检测超时帧数

# --- UI and Guidance Parameters ---
RECOMMENDED_FACE_RECT_RATIO = 0.5  # 推荐人脸区域占画面的比例
RECOMMENDED_FACE_MIN_PIXELS = 150  # 推荐人脸的最小像素尺寸
