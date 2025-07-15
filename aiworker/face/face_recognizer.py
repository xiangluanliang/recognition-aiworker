# face_recognizer.py

import cv2
import os
import numpy as np
import logging
from scipy.spatial.distance import euclidean


class VisionServiceWorker:
    """
    一个纯粹的AI计算类，封装了所有核心视觉逻辑。
    【修改】所有数据库相关的代码已被移除。
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
            self.MODEL_DIR = os.path.join(current_script_dir, '..', 'dnn_models')
            self.FACE_DETECTOR_PROTOTXT_PATH = os.path.join(self.MODEL_DIR, 'opencv_face_detector.pbtxt')
            self.FACE_DETECTOR_WEIGHTS_PATH = os.path.join(self.MODEL_DIR, 'opencv_face_detector_uint8.pb')
            self.FACE_DETECTOR_CONFIDENCE_THRESHOLD = 0.4
            self.FACE_RECOGNITION_MODEL_PATH = os.path.join(self.MODEL_DIR, 'nn4.small2.v1.t7')
            self.FACE_RECOGNITION_THRESHOLD = 0.8
            self.FACE_DETECTOR_NET = None
            self.FACE_RECOGNITION_NET = None
            self._load_all_models()

    def _load_all_models(self):
        logging.info("Loading face models...")
        try:
            self.FACE_DETECTOR_NET = cv2.dnn.readNet(self.FACE_DETECTOR_PROTOTXT_PATH, self.FACE_DETECTOR_WEIGHTS_PATH)
            self.FACE_RECOGNITION_NET = cv2.dnn.readNetFromTorch(self.FACE_RECOGNITION_MODEL_PATH)
            logging.info("Face models loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load models: {e}")
            raise RuntimeError("Model loading failed.")

    def detect_faces(self, frame: np.ndarray) -> list:
        if self.FACE_DETECTOR_NET is None: return []
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.FACE_DETECTOR_NET.setInput(blob)
        detections = self.FACE_DETECTOR_NET.forward()
        detected_faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.FACE_DETECTOR_CONFIDENCE_THRESHOLD:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                detected_faces.append({'box_coords': box.astype("int").tolist(), 'confidence': float(confidence)})
        return detected_faces

    def extract_face_features(self, face_image: np.ndarray) -> np.ndarray:
        if self.FACE_RECOGNITION_NET is None: return np.array([])
        face_blob = cv2.dnn.blobFromImage(cv2.resize(face_image, (96, 96)), 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True)
        self.FACE_RECOGNITION_NET.setInput(face_blob)
        return self.FACE_RECOGNITION_NET.forward().flatten()

    def recognize_face_identity(self, frame: np.ndarray, known_faces_data: list) -> list:
        results = []
        detected_faces = self.detect_faces(frame)
        for face_info in detected_faces:
            x1, y1, x2, y2 = face_info['box_coords']
            cropped_face = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            if cropped_face.size == 0: continue

            face_embedding = self.extract_face_features(cropped_face)
            if face_embedding.size == 0: continue

            min_distance, identity, person_id, person_state = float('inf'), 'Stranger', None, None

            for known_face in known_faces_data:
                if 'face_embedding' not in known_face or known_face['face_embedding'] is None:
                    continue

                known_embedding = np.array(known_face['face_embedding'], dtype=np.float32)
                distance = euclidean(face_embedding, known_embedding)
                if distance < min_distance:
                    min_distance, identity, person_id, person_state = distance, known_face['name'], known_face['id'], \
                    known_face['state']

            final_identity = identity if min_distance < self.FACE_RECOGNITION_THRESHOLD else 'Stranger'

            results.append({
                'box_coords': face_info['box_coords'], 'confidence': face_info['confidence'],
                'identity': final_identity, 'distance': float(min_distance),
                'person_id': person_id if final_identity != 'Stranger' else None,
                'person_state': person_state if final_identity != 'Stranger' else None
            })
        return results

    def perform_liveness_check(self, frame: np.ndarray, detected_faces: list) -> bool:
        if not detected_faces: return True
        LIVENESS_BLUR_THRESHOLD = 100.0
        for face_info in detected_faces:
            x1, y1, x2, y2 = face_info['box_coords']
            face_roi = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            if face_roi.size < (20 * 20): continue
            gray_face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            laplacian_variance = cv2.Laplacian(gray_face_roi, cv2.CV_64F).var()
            if laplacian_variance < LIVENESS_BLUR_THRESHOLD:
                logging.warning(f"Liveness check failed, variance: {laplacian_variance}")
                return False
        return True


try:
    vision_worker_instance = VisionServiceWorker()
except RuntimeError as e:
    logging.critical(f"CRITICAL: Failed to initialize VisionServiceWorker. AI functions will not work. Error: {e}")
    vision_worker_instance = None


def process_frame_face_recognition(frame: np.ndarray, known_faces_data: list, camera_id: int = 0) -> (np.ndarray, dict):
    """
    【修改】标准接口函数现在接收 known_faces_data 参数。
    它返回处理后的帧和包含事件信息的结构化字典。
    """
    if vision_worker_instance is None:
        processed_frame = frame.copy()
        cv2.putText(processed_frame, "AI WORKER OFFLINE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return processed_frame, {'status': 'error', 'message': 'AI Worker not initialized', 'events_to_log': []}

    processed_frame = frame.copy()
    detection_data = {
        'status': 'success',
        'liveness_passed': True,
        'persons': [],
        'events_to_log': []
    }

    detected_faces = vision_worker_instance.detect_faces(processed_frame)
    is_live = vision_worker_instance.perform_liveness_check(processed_frame, detected_faces)

    if not is_live:
        detection_data['liveness_passed'] = False
        detection_data['events_to_log'].append({
            'event_type': 'LIVENESS_FRAUD_DETECTED', 'confidence': 1.0, 'person_id': None
        })
        cv2.putText(processed_frame, "LIVENESS FRAUD DETECTED!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return processed_frame, detection_data

    recognition_results = vision_worker_instance.recognize_face_identity(processed_frame, known_faces_data)
    detection_data['persons'] = recognition_results

    for res in recognition_results:
        event_to_log = None
        if res['identity'] == 'Stranger':
            event_to_log = {'event_type': 'FACE_STRANGER_DETECTED', 'person_id': None, 'confidence': res['confidence']}
        elif res.get('person_state') == 1:
            event_to_log = {'event_type': 'FACE_DANGEROUS_PERSON_DETECTED', 'person_id': res['person_id'],
                            'confidence': res['confidence']}

        if event_to_log:
            detection_data['events_to_log'].append(event_to_log)

        x1, y1, x2, y2 = res['box_coords']
        identity = res['identity']
        color = (0, 255, 0)
        if identity == 'Stranger':
            color = (0, 165, 255)
        elif res.get('person_state') == 1:
            color = (0, 0, 255)
        label = f"{identity} D:{res['distance']:.2f}"
        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(processed_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return processed_frame, detection_data