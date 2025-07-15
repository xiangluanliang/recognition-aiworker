import cv2
import os
import numpy as np
import logging
from scipy.spatial.distance import euclidean
import onnxruntime
import dlib
from imutils import face_utils

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# --- 活体检测模型参数 ---
OULU_LIVENESS_INPUT_SIZE = (224, 224)
OULU_LIVENESS_THRESHOLD = 0.015 # 调整：更宽松的阈值

# --- 人脸识别模型路径 ---
FACE_RECOGNITION_MODEL_FILENAME = 'InceptionResnetV1_vggface2.onnx'

# --- Dlib 关键点预测器模型路径 ---
DLIB_LANDMARK_PREDICTOR_FILENAME = 'shape_predictor_68_face_landmarks.dat'

# --- 眨眼检测参数 ---
EYE_AR_THRESH = 0.25 # 调整：更低的EAR阈值更容易检测眨眼
EYE_AR_CONSEC_FRAMES = 2 # 调整：连续帧数减少，更快检测眨眼
BLINK_TIMEOUT_FRAMES = 150 # 调整：给予更多帧数（时间）来完成眨眼

# --- 新增：人脸区域要求参数 (用于绘制指导框和活体推理) ---
RECOMMENDED_FACE_RECT_RATIO = 0.5 # 推荐人脸区域占画面宽度/高度的比例
RECOMMENDED_FACE_MIN_PIXELS = 150 # 推荐人脸的最小尺寸（以较短边为准），用于绘制指导框
# MIN_EFFECTIVE_LIVENESS_ROI_SIZE = OULU_LIVENESS_INPUT_SIZE[0] // 2 # 此参数已存在


class VisionServiceWorker:
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

            self.FACE_RECOGNITION_MODEL_PATH = os.path.join(self.MODEL_DIR, FACE_RECOGNITION_MODEL_FILENAME)

            self.OULU_LIVENESS_MODEL_PATH = os.path.join(self.MODEL_DIR, 'OULU_Protocol_2_model_0_0.onnx')
            self.DLIB_LANDMARK_PREDICTOR_PATH = os.path.join(self.MODEL_DIR, DLIB_LANDMARK_PREDICTOR_FILENAME)

            self.FACE_DETECTOR_CONFIDENCE_THRESHOLD = 0.4
            self.FACE_RECOGNITION_THRESHOLD = 0.8

            self.DLIB_FACE_DETECTOR = dlib.get_frontal_face_detector()
            self.DLIB_LANDMARK_PREDICTOR = None
            self.lStart, self.lEnd = None, None
            self.rStart, self.rEnd = None, None

            self.blink_counter = 0
            self.total_blinks = 0
            self.blink_detection_active = False
            self.frames_since_last_blink = 0

            self.FACE_DETECTOR_NET = None
            self.FACE_RECOGNITION_NET = None
            self.OULU_LIVENESS_SESSION = None

            # FIX: Initialize self.logger BEFORE calling _load_all_models()
            self.logger = logging.getLogger(__name__)
            self._load_all_models()

    def _load_all_models(self):
        self.logger.info("Loading AI models...")
        try:
            if not os.path.exists(self.FACE_DETECTOR_PROTOTXT_PATH):
                raise FileNotFoundError(f"Face detector prototxt not found: {self.FACE_DETECTOR_PROTOTXT_PATH}")
            if not os.path.exists(self.FACE_DETECTOR_WEIGHTS_PATH):
                raise FileNotFoundError(f"Face detector weights not found: {self.FACE_DETECTOR_WEIGHTS_PATH}")
            self.FACE_DETECTOR_NET = cv2.dnn.readNet(self.FACE_DETECTOR_PROTOTXT_PATH, self.FACE_DETECTOR_WEIGHTS_PATH)

            if not os.path.exists(self.FACE_RECOGNITION_MODEL_PATH):
                raise FileNotFoundError(f"Face recognition model not found: {self.FACE_RECOGNITION_MODEL_PATH}")
            self.FACE_RECOGNITION_NET = onnxruntime.InferenceSession(self.FACE_RECOGNITION_MODEL_PATH, providers=['CPUExecutionProvider'])
            self.face_rec_input_name = self.FACE_RECOGNITION_NET.get_inputs()[0].name
            self.face_rec_output_name = self.FACE_RECOGNITION_NET.get_outputs()[0].name

            if not os.path.exists(self.OULU_LIVENESS_MODEL_PATH):
                raise FileNotFoundError(f"OULU_Protocol_2_model_0_0.onnx not found: {self.OULU_LIVENESS_MODEL_PATH}")
            self.OULU_LIVENESS_SESSION = onnxruntime.InferenceSession(
                self.OULU_LIVENESS_MODEL_PATH,
                providers=['CPUExecutionProvider']
            )
            self.oulu_liveness_input_name = self.OULU_LIVENESS_SESSION.get_inputs()[0].name
            self.oulu_liveness_output_name = self.OULU_LIVENESS_SESSION.get_outputs()[0].name

            if not os.path.exists(self.DLIB_LANDMARK_PREDICTOR_PATH):
                raise FileNotFoundError(f"Dlib landmark predictor not found: {self.DLIB_LANDMARK_PREDICTOR_PATH}")
            self.DLIB_LANDMARK_PREDICTOR = dlib.shape_predictor(self.DLIB_LANDMARK_PREDICTOR_PATH)
            (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

            self.logger.info("All AI models loaded successfully.")
        except Exception as e:
            self.logger.critical(f"CRITICAL: Failed to load AI models: {e}. AI functions may not work correctly.")
            self.FACE_DETECTOR_NET = None
            self.FACE_RECOGNITION_NET = None
            self.OULU_LIVENESS_SESSION = None
            self.DLIB_LANDMARK_PREDICTOR = None
            raise RuntimeError("One or more AI models failed to load.")

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            self.logger.warning("Received empty or invalid frame for preprocessing. Returning original frame.")
            return frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        equalized_gray = clahe.apply(gray_frame)
        processed_frame = cv2.cvtColor(equalized_gray, cv2.COLOR_GRAY2BGR)
        self.logger.debug("Applied CLAHE preprocessing to frame.")
        return processed_frame

    def detect_faces(self, frame: np.ndarray) -> list:
        if self.FACE_DETECTOR_NET is None:
            self.logger.warning("Face detector model not loaded. Cannot detect faces.")
            return []
        if frame is None or frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            self.logger.warning("Received empty or invalid frame for face detection.")
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
        self.logger.debug(f"Detected {len(detected_faces)} faces with confidence > {self.FACE_DETECTOR_CONFIDENCE_THRESHOLD}")
        return detected_faces

    def extract_face_features(self, face_image: np.ndarray) -> np.ndarray:
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
        features = self.FACE_RECOGNITION_NET.run([self.face_rec_output_name], {self.face_rec_input_name: input_tensor})[0]
        self.logger.debug(f"Extracted face features of shape: {features.shape}")
        return features.flatten()

    def recognize_face_identity(self, frame: np.ndarray, known_faces_data: list) -> list:
        self.logger.debug("Starting face recognition process.")
        results = []
        # Note: This method is designed to perform its own face detection if called directly.
        # In local_ai_service.py's current flow, `detect_faces` is called first, then `perform_liveness_check`,
        # then `recognize_face_identity`. For efficiency, `recognize_face_identity` could ideally take `detected_faces_info`
        # as an argument. However, to minimize structural changes, we'll keep its current signature,
        # acknowledging that `local_ai_service.py` will handle the overall flow and filtering.
        preprocessed_frame = self.preprocess_frame(frame)
        detected_faces = self.detect_faces(preprocessed_frame)

        for face_info in detected_faces:
            x1, y1, x2, y2 = face_info['box_coords']
            cropped_face = frame[y1:y2, x1:x2]

            if cropped_face.size == 0:
                self.logger.warning("Cropped face region is empty for recognition.")
                continue

            face_embedding = self.extract_face_features(cropped_face)
            if face_embedding.size == 0:
                self.logger.warning("Failed to extract features from face region for recognition.")
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
            results.append({
                'box_coords': [int(c) for c in face_info['box_coords']],
                'confidence': float(face_info['confidence']),
                'identity': final_identity,
                'distance': final_distance,
                'person_id': int(final_person_id) if final_person_id is not None else None,
                'person_state': int(final_person_state) if final_person_state is not None else None,
            })

        self.logger.debug(f"Recognition finished. Found {len(results)} results.")
        return results

    def _oulu_liveness_preprocess_and_infer(self, cropped_face: np.ndarray) -> (float, str):
        if self.OULU_LIVENESS_SESSION is None:
            raise RuntimeError("OULU_Protocol_2_model_0_0.onnx model not loaded.")
        processed_image = cv2.resize(cropped_face, OULU_LIVENESS_INPUT_SIZE)
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        processed_image = processed_image.astype(np.float32)
        processed_image = (processed_image - 127.5) / 127.5
        input_tensor = np.transpose(processed_image, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        oulu_outputs = self.OULU_LIVENESS_SESSION.run([self.oulu_liveness_output_name], {self.oulu_liveness_input_name: input_tensor})[0]
        flattened_output = oulu_outputs.flatten()
        if flattened_output.size > 0:
            liveness_score = flattened_output[0].item()
            predicted_class_name = "LIVE" if liveness_score >= OULU_LIVENESS_THRESHOLD else "SPOOF"
        else:
            self.logger.error(f"OULU Liveness model output is empty: {oulu_outputs.shape}")
            liveness_score = 0.0
            predicted_class_name = "ERROR_EMPTY_OUTPUT"
        return liveness_score, predicted_class_name

    def _eye_aspect_ratio(self, eye_points):
        A = euclidean(eye_points[1], eye_points[5])
        B = euclidean(eye_points[2], eye_points[4])
        C = euclidean(eye_points[0], eye_points[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def perform_liveness_check(self, frame: np.ndarray, detected_faces_info: list, require_blink: bool = False) -> (bool, list):
        """
        使用 OULU_Protocol_2_model_0_0.onnx 模型结合眨眼检测进行活体判断。
        返回 (is_overall_live, liveness_details_list)
        liveness_details_list 包含每个脸的 {'box_coords', 'oulu_score', 'oulu_result', 'blink_status'}
        """
        overall_live_status = True
        liveness_results_per_face = []

        if not detected_faces_info:
            self.logger.debug("No faces detected for liveness check. Passing.")
            if self.blink_detection_active:
                self.logger.debug("No faces, resetting blink detection state.")
                self.blink_detection_active = False
                self.frames_since_last_blink = 0
            return True, []

        if require_blink and not self.blink_detection_active:
            self.blink_detection_active = True
            self.frames_since_last_blink = 0
            self.blink_counter = 0
            self.logger.info("Blink detection activated.")

        for face_info in detected_faces_info:
            x1, y1, x2, y2 = face_info['box_coords']
            cropped_face = frame[y1:y2, x1:x2]

            face_width = x2 - x1
            face_height = y2 - y1

            oulu_score = 0.0
            oulu_result_str = "N/A"
            blink_status = "N/A"

            if self.OULU_LIVENESS_SESSION is None or \
               (cropped_face.shape[0] < MIN_EFFECTIVE_LIVENESS_ROI_SIZE or cropped_face.shape[1] < MIN_EFFECTIVE_LIVENESS_ROI_SIZE):
                self.logger.warning(f"Face (ROI: {face_width}x{face_height}): Too small for reliable OULU Liveness. Score defaulted to 0.0.")
                oulu_score = 0.0
                oulu_result_str = "SPOOF (Size)"
            else:
                try:
                    score, predicted_name = self._oulu_liveness_preprocess_and_infer(cropped_face)
                    oulu_score = score
                    oulu_result_str = predicted_name

                    if score < OULU_LIVENESS_THRESHOLD:
                        self.logger.warning(f"Face (ROI: {face_width}x{face_height}): OULU liveness detected SPOOF! Score: {oulu_score:.4f} (Predicted: {oulu_result_str})")
                    else:
                        self.logger.info(f"Face (ROI: {face_width}x{face_height}): OULU liveness detected LIVE. Score: {oulu_score:.4f} (Predicted: {oulu_result_str})")

                except Exception as e:
                    self.logger.error(f"Error during OULU Liveness inference for Face (ROI: {face_width}x{face_height}): {e}. Result defaulted to SPOOF.")
                    oulu_result_str = "ERROR (OULU Liveness)"
                    oulu_score = 0.0

            if require_blink and self.DLIB_LANDMARK_PREDICTOR is not None:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                dlib_rect = dlib.rectangle(x1, y1, x2, y2)
                try:
                    shape = self.DLIB_LANDMARK_PREDICTOR(gray_frame, dlib_rect)
                    shape = face_utils.shape_to_np(shape)
                    leftEye = shape[self.lStart:self.lEnd]
                    rightEye = shape[self.rStart:self.rEnd]
                    leftEAR = self._eye_aspect_ratio(leftEye)
                    rightEAR = self._eye_aspect_ratio(rightEye)
                    ear = (leftEAR + rightEAR) / 2.0
                    self.logger.debug(f"Face EAR: {ear:.2f}, Blink counter: {self.blink_counter}, Total blinks: {self.total_blinks}")

                    if ear < EYE_AR_THRESH:
                        self.blink_counter += 1
                        blink_status = f"EYES_CLOSED (EAR: {ear:.2f})"
                    else:
                        if self.blink_counter >= EYE_AR_CONSEC_FRAMES:
                            self.total_blinks += 1
                            self.logger.info(f"Blink detected! Total blinks: {self.total_blinks}")
                            self.frames_since_last_blink = 0
                            blink_status = "BLINK_DETECTED"
                        else:
                            blink_status = f"EYES_OPEN (EAR: {ear:.2f})"
                        self.blink_counter = 0

                    if self.blink_detection_active:
                        self.frames_since_last_blink += 1
                        if self.frames_since_last_blink >= BLINK_TIMEOUT_FRAMES:
                            self.logger.warning(f"Blink timeout: No natural blink detected in {BLINK_TIMEOUT_FRAMES} frames.")
                            blink_status = "BLINK_TIMEOUT"
                except Exception as e:
                    self.logger.error(f"Error during Dlib landmark/blink detection for face {x1,y1,x2,y2}: {e}")
                    blink_status = "DLIB_ERROR"
            else:
                blink_status = "NOT_REQUIRED/DLIB_UNAVAILABLE"

            # --- 综合判断逻辑 (保持不变，这是关键的活体判断逻辑) ---
            final_live_status_for_face = True
            oulu_is_live_by_threshold = (oulu_score >= OULU_LIVENESS_THRESHOLD)
            blink_is_detected = ("BLINK_DETECTED" == blink_status)

            if require_blink:
                if not oulu_is_live_by_threshold:
                    final_live_status_for_face = False
                elif oulu_is_live_by_threshold:
                    if blink_is_detected:
                        final_live_status_for_face = True
                    elif self.frames_since_last_blink < BLINK_TIMEOUT_FRAMES:
                        final_live_status_for_face = True
                    else: # OULU is LIVE, but blink timed out
                        final_live_status_for_face = False
            else: # If blink is not required, rely solely on OULU
                final_live_status_for_face = oulu_is_live_by_threshold

            if not final_live_status_for_face:
                overall_live_status = False

            liveness_results_per_face.append({
                'box_coords': [int(c) for c in face_info['box_coords']],
                'oulu_score': oulu_score,
                'oulu_result': oulu_result_str,
                'blink_status': blink_status,
                'combined_live_status': final_live_status_for_face
            })
        return overall_live_status, liveness_results_per_face