import cv2
import os
import numpy as np
import logging
from scipy.spatial.distance import euclidean

#from external.liveness_detector.predictor import LivenessDetector

# 配置日志。将级别设置为 DEBUG 以获取最详细的输出。
# 修正了格式字符串，移除了可能导致错误的 %(funcName)s (在某些Python版本或特定上下文中)
# 使用 %(name)s 来表示logger的名称（这里是FaceRecognizer）， %(funcName)s 可以单独在消息中打印如果需要
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')


class VisionServiceWorker:
    """
    一个纯粹的AI计算类，封装了所有核心视觉逻辑。
    【重要修改】增加了图像预处理（CLAHE）和更精细的参数调整。
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(VisionServiceWorker, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            current_script_dir = os.path.abspath(os.path.dirname(__file__))
            self.MODEL_DIR = os.path.join(current_script_dir, '../../', 'dnn_models')
            self.FACE_DETECTOR_PROTOTXT_PATH = os.path.join(self.MODEL_DIR, 'opencv_face_detector.pbtxt')
            self.FACE_DETECTOR_WEIGHTS_PATH = os.path.join(self.MODEL_DIR, 'opencv_face_detector_uint8.pb')
            self.FACE_RECOGNITION_MODEL_PATH = os.path.join(self.MODEL_DIR, 'nn4.small2.v1.t7')
            self.FACE_DETECTOR_CONFIDENCE_THRESHOLD = 0.4
            self.FACE_RECOGNITION_THRESHOLD = 0.8
            self.LIVENESS_BLUR_THRESHOLD = 100.0
            self.FACE_DETECTOR_NET = None
            self.FACE_RECOGNITION_NET = None
            self._load_all_models()
            self.logger = logging.getLogger(__name__)  # 初始化一个logger实例
#            self.liveness_model_path = os.path.join(current_script_dir, '..', 'resource', 'anti_spoof_models',
#                                                    '2.7_80x80_MiniFASNetV2.pth')
#            self.liveness_detector = LivenessDetector(self.liveness_model_path)

    def _load_all_models(self):
        # 获取当前 logger 实例，这样在日志中会显示 (FaceRecognizer)
        logger = logging.getLogger(__name__) 
        logger.info("Loading face models...")
        try:
            # 确保模型文件存在
            if not os.path.exists(self.FACE_DETECTOR_PROTOTXT_PATH):
                raise FileNotFoundError(f"Face detector prototxt not found: {self.FACE_DETECTOR_PROTOTXT_PATH}")
            if not os.path.exists(self.FACE_DETECTOR_WEIGHTS_PATH):
                raise FileNotFoundError(f"Face detector weights not found: {self.FACE_DETECTOR_WEIGHTS_PATH}")
            if not os.path.exists(self.FACE_RECOGNITION_MODEL_PATH):
                raise FileNotFoundError(f"Face recognition model not found: {self.FACE_RECOGNITION_MODEL_PATH}")

            self.FACE_DETECTOR_NET = cv2.dnn.readNet(self.FACE_DETECTOR_PROTOTXT_PATH, self.FACE_DETECTOR_WEIGHTS_PATH)
            self.FACE_RECOGNITION_NET = cv2.dnn.readNetFromTorch(self.FACE_RECOGNITION_MODEL_PATH)
            logger.info("Face models loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load models: {e}. Please check model paths: {self.MODEL_DIR}")
            raise RuntimeError("Model loading failed.")

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        logger = logging.getLogger(__name__)
        if frame is None or frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            logger.warning("Received empty or invalid frame for preprocessing. Returning original frame.")
            return frame

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        equalized_gray = clahe.apply(gray_frame)
        processed_frame = cv2.cvtColor(equalized_gray, cv2.COLOR_GRAY2BGR)
        
        logger.debug("Applied CLAHE preprocessing to frame.")
        return processed_frame

    def detect_faces(self, frame: np.ndarray) -> list:
        logger = logging.getLogger(__name__)
        if self.FACE_DETECTOR_NET is None:
            logger.warning("Face detector model not loaded. Cannot detect faces.")
            return []
        if frame is None or frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            logger.warning("Received empty or invalid frame for face detection.")
            return []

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
        self.FACE_DETECTOR_NET.setInput(blob)
        detections = self.FACE_DETECTOR_NET.forward()

        detected_faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.FACE_DETECTOR_CONFIDENCE_THRESHOLD:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                startX, startY, endX, endY = box.astype("int")
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w, endX), min(h, endY)

                if (endX - startX) > 0 and (endY - startY) > 0:
                    detected_faces.append({'box_coords': [startX, startY, endX, endY], 'confidence': float(confidence)})
        logger.debug(f"Detected {len(detected_faces)} faces with confidence > {self.FACE_DETECTOR_CONFIDENCE_THRESHOLD}")
        return detected_faces

    def extract_face_features(self, face_image: np.ndarray) -> np.ndarray:
        logger = logging.getLogger(__name__)
        if self.FACE_RECOGNITION_NET is None:
            logger.warning("Face recognition model not loaded. Cannot extract features.")
            return np.array([])
        if face_image is None or face_image.size == 0 or face_image.shape[0] == 0 or face_image.shape[1] == 0:
            logger.warning("Received empty or invalid face_image for feature extraction.")
            return np.array([])

        face_blob = cv2.dnn.blobFromImage(cv2.resize(face_image, (96, 96)), 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        self.FACE_RECOGNITION_NET.setInput(face_blob)
        features = self.FACE_RECOGNITION_NET.forward()
        logger.debug(f"Extracted face features of shape: {features.shape}")
        return features.flatten()

    def recognize_face_identity(self, frame: np.ndarray, known_faces_data: list) -> list:
        self.logger.debug("Starting face recognition process.")
        results = []
        preprocessed_frame = self.preprocess_frame(frame)
        detected_faces = self.detect_faces(preprocessed_frame)

        for face_info in detected_faces:
            x1, y1, x2, y2 = face_info['box_coords']
            cropped_face = frame[y1:y2, x1:x2]

            if cropped_face.size == 0:
                self.logger.warning("Cropped face region is empty.")
                continue

            face_embedding = self.extract_face_features(cropped_face)
            if face_embedding.size == 0:
                self.logger.warning("Failed to extract features from face region.")
                continue

            min_distance = float('inf')
            identity = 'Stranger'
            person_id = None
            person_state = None

            for known_face in known_faces_data:
                if 'face_embedding' not in known_face or known_face['face_embedding'] is None:
                    continue

                known_embedding = np.array(known_face['face_embedding'], dtype=np.float32)
                distance = euclidean(face_embedding, known_embedding)

                if distance < min_distance:
                    min_distance = distance
                    identity = known_face['name']
                    person_id = known_face['id']
                    person_state = known_face['state']

            final_identity = identity if min_distance < self.FACE_RECOGNITION_THRESHOLD else 'Stranger'
            final_person_id = person_id if final_identity != 'Stranger' else None
            final_person_state = person_state if final_identity != 'Stranger' else None

            final_distance = float(min_distance) if min_distance != float('inf') else None
            # 【 KERNKORREKTUR 】 Alle numerischen Werte in Python-Standardtypen umwandeln, bevor sie dem Wörterbuch hinzugefügt werden.
            results.append({
                'box_coords': [int(c) for c in face_info['box_coords']],  # Ensures all coordinates are standard int
                'confidence': float(face_info['confidence']),
                'identity': final_identity,
                'distance': final_distance,
                'person_id': int(final_person_id) if final_person_id is not None else None,
                'person_state': int(final_person_state) if final_person_state is not None else None,
            })

        self.logger.debug(f"Recognition finished. Found {len(results)} results.")
        return results

    def perform_liveness_check(self, frame: np.ndarray, detected_faces: list) -> bool:
        logger = logging.getLogger(__name__)
        if not detected_faces:
            logger.debug("No faces detected for liveness check. Passing by default.")
            return True
        if frame is None or frame.size == 0:
            logger.warning("Invalid frame for liveness check.")
            return True

        for face_info in detected_faces:
            x1, y1, x2, y2 = face_info['box_coords']
            face_roi = frame[y1:y2, x1:x2]

            if face_roi.size == 0 or face_roi.shape[0] < 20 or face_roi.shape[1] < 20:
                logger.warning("Face ROI too small or invalid. Skipping.")
                continue

            try:
                prob_real = self.liveness_detector.predict(face_roi)
                logger.info(f"Liveness probability: {prob_real:.4f}")
                if prob_real < 0.8:  # 阈值可调
                    logger.warning("Liveness check failed. Possible spoof.")
                    return False
            except Exception as e:
                logger.error(f"Liveness check exception: {e}")
                return False

        return True


# 实例化 VisionServiceWorker。这会在模块导入时发生，确保模型只加载一次。
try:
    vision_worker_instance = VisionServiceWorker()
except RuntimeError as e:
    logging.critical(f"CRITICAL: Failed to initialize VisionServiceWorker. AI functions will not work. Error: {e}")
    vision_worker_instance = None


def process_frame_face_recognition(frame: np.ndarray, known_faces_data: list, camera_id: int = 0) -> (np.ndarray, dict):
    """
    一个用于外部调用的标准接口函数，接收视频帧和已知人脸数据。
    它返回处理后的帧和包含事件信息的结构化字典。
    【已修正】此版本简化了逻辑，消除了重复的AI处理。
    """
    # 检查AI worker实例是否成功初始化
    if vision_worker_instance is None:
        processed_frame = frame.copy()
        cv2.putText(processed_frame, "AI WORKER OFFLINE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return processed_frame, {'status': 'error', 'message': 'AI Worker not initialized', 'events_to_log': []}

    # 复制帧用于绘制，以避免修改原始输入帧
    processed_frame_for_display = frame.copy()
    # 初始化返回数据的结构
    detection_data = {
        'status': 'success',
        'liveness_passed': True,
        'persons': [],
        'events_to_log': []
    }

    # 1. 调用核心识别函数，获取所有识别结果。
    #    这个函数内部已经包含了预处理、人脸检测和特征比对的完整流程。
    recognition_results = vision_worker_instance.recognize_face_identity(frame, known_faces_data)
    detection_data['persons'] = recognition_results

    # 2. 从识别结果中提取人脸框，用于后续的活体检测。
    #    活体检测应在原始图像上进行，以获得最真实的纹理信息。
    detected_faces_for_liveness = [{'box_coords': res['box_coords']} for res in recognition_results]
    is_live = vision_worker_instance.perform_liveness_check(frame, detected_faces_for_liveness)

    # 3. 如果活体检测失败，则标记并直接返回
    if not is_live:
        logging.warning("Liveness fraud detected! Skipping face recognition.")
        detection_data['liveness_passed'] = False
        detection_data['events_to_log'].append({
            'event_type': 'LIVENESS_FRAUD_DETECTED', 'confidence': 1.0, 'person_id': None, 'identity': 'Liveness_Fraud'
        })
        # 在画面上绘制醒目的欺骗攻击警告
        cv2.putText(processed_frame_for_display, "【欺骗攻击】检测到非活体！", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return processed_frame_for_display, detection_data

    # 4. 如果活体检测通过，则遍历识别结果，准备事件日志并绘制结果
    has_dangerous_person_this_frame = False
    has_stranger_this_frame = False
    known_names_on_screen = set()

    for res in recognition_results:
        x1, y1, x2, y2 = res['box_coords']
        identity = res['identity']
        person_state = res.get('person_state')
        confidence = res['confidence']
        distance = res['distance']

        # 根据识别结果确定颜色和事件类型
        color = (0, 255, 0)  # 默认绿色 (已知人员)
        event_type = 'FACE_KNOWN_PERSON_DETECTED'
        person_id = res['person_id']

        if identity == 'Stranger':
            color = (0, 165, 255)  # 橙色 (陌生人)
            event_type = 'FACE_STRANGER_DETECTED'
            has_stranger_this_frame = True
            person_id = None
        elif person_id is not None and person_state == 1:
            color = (0, 0, 255)  # 红色 (危险人物)
            event_type = 'FACE_DANGEROUS_PERSON_DETECTED'
            has_dangerous_person_this_frame = True
        elif identity in ['unknown_feature_error', 'unknown_cropped_error']:
            color = (255, 0, 0)  # 蓝色 (处理错误)
            event_type = 'FACE_PROCESSING_ERROR'
            person_id = None
        else:
            known_names_on_screen.add(identity)

        # 准备事件数据，由上层服务(local_ai_service.py)决定是否发送
        detection_data['events_to_log'].append({
            'event_type': event_type,
            'confidence': confidence,
            'person_id': person_id,
            'identity': identity
        })

        # 在人脸周围绘制矩形框和文本标签
        label = f"{identity}" if distance is None else f"{identity} D:{distance:.2f}"
        cv2.rectangle(processed_frame_for_display, (x1, y1), (x2, y2), color, 2)
        cv2.putText(processed_frame_for_display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    # 5. 在帧顶部添加一个整体的概览消息
    overall_message = ""
    if has_dangerous_person_this_frame:
        overall_message = "【紧急警报】发现危险人员！"
    elif has_stranger_this_frame:
        overall_message = "【警告】发现陌生人！"
    elif known_names_on_screen:
        overall_message = f"【正常】已识别: {', '.join(known_names_on_screen)}"
    elif len(detected_faces_for_liveness) > 0:
        overall_message = "正在识别..."
    else:
        overall_message = "未检测到人脸"

    cv2.putText(processed_frame_for_display, overall_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # 6. 返回最终处理好的图像帧和包含所有信息的字典
    return processed_frame_for_display, detection_data



