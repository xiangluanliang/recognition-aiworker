# aiworker/face/liveness_detector.py
import cv2
import numpy as np
import dlib
from imutils import face_utils
from scipy.spatial.distance import euclidean
import onnxruntime
import logging

from aiworker.config import (
    OULU_LIVENESS_INPUT_SIZE, OULU_LIVENESS_THRESHOLD, MIN_EFFECTIVE_LIVENESS_ROI_SIZE,
    EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES, BLINK_TIMEOUT_FRAMES
)


class LivenessDetector:
    def __init__(self, oulu_model_path: str, dlib_predictor_path: str):
        self.logger = logging.getLogger(__name__)
        try:
            self.session = onnxruntime.InferenceSession(oulu_model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.logger.info("OULU Liveness model loaded successfully.")
        except Exception as e:
            self.logger.critical(f"Failed to load OULU Liveness model: {e}")
            self.session = None

        try:
            self.landmark_predictor = dlib.shape_predictor(dlib_predictor_path)
            (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
            self.logger.info("Dlib landmark predictor loaded successfully.")
        except Exception as e:
            self.logger.critical(f"Failed to load Dlib landmark predictor: {e}")
            self.landmark_predictor = None

        # Blink detection state variables
        self.blink_counter = 0
        self.total_blinks = 0
        self.blink_detection_active = False
        self.frames_since_last_blink = 0

    def _eye_aspect_ratio(self, eye_points):
        A = euclidean(eye_points[1], eye_points[5])
        B = euclidean(eye_points[2], eye_points[4])
        C = euclidean(eye_points[0], eye_points[3])
        return (A + B) / (2.0 * C)

    def _check_blinks(self, frame_gray, face_rect):
        if not self.landmark_predictor:
            return "DLIB_UNAVAILABLE"

        try:
            shape = self.landmark_predictor(frame_gray, face_rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            leftEAR = self._eye_aspect_ratio(leftEye)
            rightEAR = self._eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            if ear < EYE_AR_THRESH:
                self.blink_counter += 1
                return f"EYES_CLOSED (EAR: {ear:.2f})"
            else:
                if self.blink_counter >= EYE_AR_CONSEC_FRAMES:
                    self.total_blinks += 1
                    self.frames_since_last_blink = 0
                    return "BLINK_DETECTED"
                self.blink_counter = 0
                return f"EYES_OPEN (EAR: {ear:.2f})"
        except Exception as e:
            self.logger.error(f"Error during blink detection: {e}")
            return "DLIB_ERROR"

    def _check_oulu_liveness(self, cropped_face):
        if not self.session:
            return 0.0, "OULU_MODEL_UNAVAILABLE"

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
        overall_live_status = True
        liveness_results = []
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if require_blink and not self.blink_detection_active:
            self.blink_detection_active = True
            self.frames_since_last_blink = 0
            self.total_blinks = 0
            self.blink_counter = 0

        for face_info in detected_faces_info:
            x1, y1, x2, y2 = face_info['box_coords']
            cropped_face = frame[y1:y2, x1:x2]

            oulu_score, oulu_result = self._check_oulu_liveness(cropped_face)

            blink_status = "NOT_REQUIRED"
            if require_blink:
                dlib_rect = dlib.rectangle(x1, y1, x2, y2)
                blink_status = self._check_blinks(gray_frame, dlib_rect)
                if self.blink_detection_active:
                    self.frames_since_last_blink += 1
                    if self.frames_since_last_blink >= BLINK_TIMEOUT_FRAMES and self.total_blinks == 0:
                        blink_status = "BLINK_TIMEOUT"

            # Combine results
            is_live = oulu_result == "LIVE"
            if require_blink:
                is_live = is_live and (
                            blink_status == "BLINK_DETECTED" or self.frames_since_last_blink < BLINK_TIMEOUT_FRAMES)

            if not is_live:
                overall_live_status = False

            liveness_results.append({
                'box_coords': [int(c) for c in face_info['box_coords']],
                'oulu_score': oulu_score,
                'oulu_result': oulu_result,
                'blink_status': blink_status,
                'combined_live_status': is_live
            })

        return overall_live_status, liveness_results
