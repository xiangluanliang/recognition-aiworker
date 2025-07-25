import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial.distance import euclidean
import onnxruntime
import logging

from aiworker.config import (
    OULU_LIVENESS_INPUT_SIZE, OULU_LIVENESS_THRESHOLD, MIN_EFFECTIVE_LIVENESS_ROI_SIZE,
    EYE_AR_THRESH, BLINK_TIMEOUT_FRAMES
)


class LivenessDetector:
    def __init__(self, oulu_model_path: str, dlib_predictor_path: str = None):
        self.logger = logging.getLogger(__name__)
        self.session = onnxruntime.InferenceSession(oulu_model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # static_image_mode=False 表示处理视频流
        # max_num_faces=1 只处理画面中最大的一张脸，提高性能
        # min_detection_confidence=0.5 设置检测阈值
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.LEFT_EYE_EAR_INDICES = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_EAR_INDICES = [33, 160, 158, 133, 153, 144]

        self.reset_blink_state()

    def reset_blink_state(self):
        """重置眨眼检测的状态，用于新的会话。"""
        self.has_seen_eyes_open = False
        self.has_seen_eyes_closed = False
        self.blink_completed = False

    def _eye_aspect_ratio(self, eye_points):
        A = euclidean(eye_points[1], eye_points[5])
        B = euclidean(eye_points[2], eye_points[4])
        C = euclidean(eye_points[0], eye_points[3])
        return (A + B) / (2.0 * C)

    def _check_blinks(self, frame: np.ndarray):
        """
        使用 Mediapipe 检测眨眼。
        不再需要 face_rect 和 gray_frame。
        """
        if self.blink_completed:
            return "BLINK_COMPLETED"

        try:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = self.mp_face_mesh.process(image_rgb)
            image_rgb.flags.writeable = True

            if not results.multi_face_landmarks:
                return "NO_FACE_DETECTED"

            face_landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape

            left_eye_points = np.array(
                [[face_landmarks[i].x * w, face_landmarks[i].y * h] for i in self.LEFT_EYE_EAR_INDICES])
            right_eye_points = np.array(
                [[face_landmarks[i].x * w, face_landmarks[i].y * h] for i in self.RIGHT_EYE_EAR_INDICES])

            leftEAR = self._eye_aspect_ratio(left_eye_points)
            rightEAR = self._eye_aspect_ratio(right_eye_points)
            ear = (leftEAR + rightEAR) / 2.0

            # 根据当前帧的眼睛开合状态，更新我们的“目击”记录
            if ear < EYE_AR_THRESH:
                self.has_seen_eyes_closed = True
                current_status_msg = "EYES_CLOSED"
            else:
                self.has_seen_eyes_open = True
                current_status_msg = "EYES_OPEN"

            # 检查是否两种状态都已集齐
            if self.has_seen_eyes_open and self.has_seen_eyes_closed:
                self.blink_completed = True  # 锁定完成状态
                return "BLINK_COMPLETED"

            # 如果还未集齐，返回当前状态
            return f"{current_status_msg} (EAR: {ear:.2f})"

        except Exception as e:
            self.logger.error(f"Error during Mediapipe blink detection: {e}")
            return "MEDIAPIPE_ERROR"

    def _check_oulu_liveness(self, cropped_face):
        if not self.session: return 0.0, "OULU_MODEL_UNAVAILABLE"
        if cropped_face.shape[0] < MIN_EFFECTIVE_LIVENESS_ROI_SIZE or cropped_face.shape[
            1] < MIN_EFFECTIVE_LIVENESS_ROI_SIZE:
            return 0.0, "SPOOF (Size)"
        try:
            image = cv2.resize(cropped_face, OULU_LIVENESS_INPUT_SIZE)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = (image.astype(np.float32) - 127.5) / 127.5
            tensor = np.transpose(image, (2, 0, 1))
            tensor = np.expand_dims(tensor, axis=0)
            outputs = self.session.run([self.output_name], {self.input_name: tensor})[0]
            score = outputs.flatten()[0].item()
            result_str = "LIVE" if score >= OULU_LIVENESS_THRESHOLD else "SPOOF"
            return score, result_str
        except Exception as e:
            self.logger.error(f"Error during OULU inference: {e}")
            return 0.0, "ERROR (OULU)"

    def perform_liveness_check(self, frame, detected_faces_info, require_blink=False):
        liveness_results = []
        blink_status = "NOT_REQUIRED"
        if require_blink:
            blink_status = self._check_blinks(frame)

        for face_info in detected_faces_info:
            x1, y1, x2, y2 = face_info['box_coords']
            cropped_face = frame[y1:y2, x1:x2]
            oulu_score, oulu_result = self._check_oulu_liveness(cropped_face)

            is_live = (oulu_result == "LIVE")
            if require_blink:
                is_live = is_live and (blink_status == "BLINK_COMPLETED")

            liveness_results.append({
                'box_coords': [int(c) for c in face_info['box_coords']],
                'oulu_score': oulu_score,
                'oulu_result': oulu_result,
                'blink_status': blink_status,
                'combined_live_status': is_live
            })

        # if self.blink_completed:
        #     self.reset_blink_state()

        return liveness_results
