# aiworker/face/vision_service.py
import cv2
import os
import numpy as np
import logging
import onnxruntime

# --- 导入我们拆分出去的子模块和配置 ---
from .liveness_detector import LivenessDetector
from aiworker.config import (
    MODEL_DIR,
    OULU_LIVENESS_MODEL_FILENAME,
    DLIB_LANDMARK_PREDICTOR_FILENAME,
    FACE_DETECTOR_PROTOTXT_FILENAME,
    FACE_DETECTOR_WEIGHTS_FILENAME,
    FACE_RECOGNITION_MODEL_FILENAME,
    FACE_DETECTOR_CONFIDENCE_THRESHOLD,
    FACE_RECOGNITION_THRESHOLD
)
from scipy.spatial.distance import euclidean


class VisionServiceWorker:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(VisionServiceWorker, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return

        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing VisionServiceWorker...")

        oulu_model_path = os.path.join(MODEL_DIR, OULU_LIVENESS_MODEL_FILENAME)
        dlib_predictor_path = os.path.join(MODEL_DIR, DLIB_LANDMARK_PREDICTOR_FILENAME)
        face_detector_prototxt = os.path.join(MODEL_DIR, FACE_DETECTOR_PROTOTXT_FILENAME)
        face_detector_weights = os.path.join(MODEL_DIR, FACE_DETECTOR_WEIGHTS_FILENAME)
        face_rec_model_path = os.path.join(MODEL_DIR, FACE_RECOGNITION_MODEL_FILENAME)

        try:
            # ✅ **核心修正：在此处创建并持有 LivenessDetector 实例**
            self.liveness_detector = LivenessDetector(oulu_model_path, dlib_predictor_path)

            # 加载人脸检测器 (OpenCV DNN)
            self.face_detector_net = cv2.dnn.readNet(face_detector_prototxt, face_detector_weights)

            # 加载人脸识别模型 (ONNX)
            self.face_recognition_net = onnxruntime.InferenceSession(face_rec_model_path,
                                                                     providers=['CPUExecutionProvider'])
            self.face_rec_input_name = self.face_recognition_net.get_inputs()[0].name
            self.face_rec_output_name = self.face_recognition_net.get_outputs()[0].name

            self.logger.info("All vision models loaded successfully.")

        except Exception as e:
            self.logger.critical(f"CRITICAL: Failed to load vision models: {e}")
            raise RuntimeError("One or more vision models failed to load.")

        self._initialized = True

    def detect_faces(self, frame: np.ndarray) -> list:
        if self.FACE_DETECTOR_NET is None:
            self.logger.warning("Face detector model not loaded. Cannot detect faces.")
            return []
        if frame is None or frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            self.logger.warning("Received empty or invalid frame for face detection.")
            return []
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0),
                                     swapRB=False, crop=False)
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
        self.logger.debug(
            f"Detected {len(detected_faces)} faces with confidence > {self.FACE_DETECTOR_CONFIDENCE_THRESHOLD}")
        return detected_faces

    def _extract_face_features(self, face_image: np.ndarray) -> np.ndarray:
        if self.FACE_RECOGNITION_NET is None:
            self.logger.warning("Face recognition model not loaded. Cannot extract features.")
            return np.array([])
        if face_image is None or face_image.size == 0 or face_image.shape[0] == 0 or face_image.shape[1] == 0:
            self.logger.warning("Received empty or invalid face_image for feature extraction.")
            return np.array([])
        processed_image = cv2.resize(face_image, (160, 160))
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        processed_image = processed_image.astype(np.float32) / 255.0
        processed_image = (processed_image - 0.5) * 2.0
        input_tensor = np.transpose(processed_image, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        features = self.FACE_RECOGNITION_NET.run([self.face_rec_output_name], {self.face_rec_input_name: input_tensor})[
            0]
        self.logger.debug(f"Extracted face features of shape: {features.shape}")
        return features.flatten()

    def recognize_faces(self, frame: np.ndarray, detected_faces: list, known_faces_data: list) -> list:
        """
        将检测到的人脸与已知人脸库进行比对，返回每个人的身份信息。
        """
        results = []
        for face_info in detected_faces:
            x1, y1, x2, y2 = face_info['box_coords']
            cropped_face = frame[y1:y2, x1:x2]

            if cropped_face.size == 0:
                self.logger.warning("Cropped face region is empty, skipping recognition.")
                continue

            # 提取当前人脸的特征
            face_embedding = self._extract_face_features(cropped_face)
            if face_embedding.size == 0:
                self.logger.warning("Failed to extract features from face, skipping recognition.")
                continue

            # 初始化最佳匹配结果
            min_distance = float('inf')
            best_match_identity = 'Stranger'
            best_match_person_id = None
            best_match_person_state = None

            # 遍历已知人脸库进行比对
            for known_face in known_faces_data:
                if 'face_embedding' not in known_face or known_face['face_embedding'] is None:
                    continue

                known_embedding = np.array(known_face['face_embedding'], dtype=np.float32)

                # 计算欧氏距离
                distance = euclidean(face_embedding, known_embedding)

                # 如果找到了更近的距离，则更新最佳匹配
                if distance < min_distance:
                    min_distance = distance
                    best_match_identity = known_face.get('name', 'Unknown')
                    best_match_person_id = known_face.get('id')
                    best_match_person_state = known_face.get('state')

            # 根据阈值决定最终身份
            final_identity = best_match_identity if min_distance < FACE_RECOGNITION_THRESHOLD else 'Stranger'
            final_person_id = best_match_person_id if final_identity != 'Stranger' else None
            final_person_state = best_match_person_state if final_identity != 'Stranger' else None
            final_distance = float(min_distance) if min_distance != float('inf') else None

            # 将完整结果添加到列表
            results.append({
                'box_coords': [int(c) for c in face_info['box_coords']],
                'confidence': float(face_info['confidence']),
                'identity': final_identity,
                'distance': final_distance,
                'person_id': int(final_person_id) if final_person_id is not None else None,
                'person_state': int(final_person_state) if final_person_state is not None else None,
            })

        return results