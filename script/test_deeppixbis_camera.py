import cv2
import numpy as np
import os
import sys
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入 face_recognizer2 模块的 VisionServiceWorker 实例和相关常量
try:
    # BLINK_TIMEOUT_FRAMES is imported here, so we should use it directly
    from face_recognizer2 import VisionServiceWorker, OULU_LIVENESS_THRESHOLD, MIN_EFFECTIVE_LIVENESS_ROI_SIZE, BLINK_TIMEOUT_FRAMES
except ImportError as e:
    logger.critical(f"Error importing VisionServiceWorker from face_recognizer2.py: {e}")
    logger.critical("Please ensure face_recognizer2.py is in the same directory and all its dependencies are met (e.g., dlib, imutils, onnxruntime).")
    sys.exit(1)

# 全局实例化 VisionServiceWorker，它会加载所有模型
try:
    liveness_tester_instance = VisionServiceWorker()
except RuntimeError:
    logger.critical("Failed to initialize VisionServiceWorker. Exiting.")
    sys.exit(1)

# --- 人脸区域要求参数 (用于绘制指导框) ---
RECOMMENDED_FACE_RECT_RATIO = 0.5 # 推荐人脸区域占画面宽度/高度的比例
RECOMMENDED_FACE_MIN_PIXELS = 150 # 推荐人脸的最小尺寸（以较短边为准），用于绘制指导框

def run_camera_liveness_test(camera_id=0):
    """
    打开摄像头，实时进行活体检测。
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.critical(f"Error: Could not open camera {camera_id}. Please check if it's connected and not in use.")
        return

    logger.info(f"Camera {camera_id} opened successfully. Displaying live feed. Press 'q' to quit.")

    frame_skip_detection = 2 # 每隔2帧进行一次人脸检测
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to grab frame from camera. Exiting...")
            break

        display_frame = frame.copy()
        h, w, _ = display_frame.shape

        # --- 绘制推荐人脸区域指导框 ---
        rect_w = int(w * RECOMMENDED_FACE_RECT_RATIO)
        rect_h = int(h * RECOMMENDED_FACE_RECT_RATIO)
        rect_side = min(rect_w, rect_h)
        if rect_side < RECOMMENDED_FACE_MIN_PIXELS:
            rect_side = RECOMMENDED_FACE_MIN_PIXELS

        center_x, center_y = w // 2, h // 2
        start_x_guide = center_x - rect_side // 2
        start_y_guide = center_y - rect_side // 2
        end_x_guide = center_x + rect_side // 2
        end_y_guide = center_y + rect_side // 2

        guide_color = (255, 255, 255) # Default white
        guide_thickness = 2
        overall_status_message = "Please center your face."
        overall_status_color = (100, 100, 100) # Gray

        detected_faces = []
        if frame_count % frame_skip_detection == 0:
            detected_faces = liveness_tester_instance.detect_faces(frame)

        face_in_guide_box = False
        face_size_ok = False

        if detected_faces:
            # 仅显示第一个检测到的人脸的活体详情
            main_face_info = detected_faces[0]
            x1, y1, x2, y2 = main_face_info['box_coords']

            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            face_width = x2 - x1
            face_height = y2 - y1

            if (x1 >= start_x_guide and y1 >= start_y_guide and x2 <= end_x_guide and y2 <= end_y_guide):
                face_in_guide_box = True

            if min(face_width, face_height) >= RECOMMENDED_FACE_MIN_PIXELS:
                face_size_ok = True

            # 执行活体检测 (包含OULU Liveness和眨眼)
            is_overall_live_result, liveness_details = liveness_tester_instance.perform_liveness_check(
                frame, detected_faces, require_blink=True # 明确要求眨眼检测
            )

            if liveness_details:
                main_face_liveness = liveness_details[0]
                oulu_score = main_face_liveness['oulu_score']
                oulu_result_str = main_face_liveness['oulu_result']
                blink_status_str = main_face_liveness['blink_status']
                combined_live_status = main_face_liveness['combined_live_status'] # 综合判断结果

                if not combined_live_status:
                    overall_status_message = "SPOOFING DETECTED!"
                    overall_status_color = (0, 0, 255) # Red
                    box_color = (0, 0, 255) # Red
                else: # combined_live_status is True
                    overall_status_message = "LIVE Person detected."
                    overall_status_color = (0, 255, 0) # Green
                    box_color = (0, 255, 0) # Green

                # This line generates the label for the bounding box
                liveness_status_text = f"OULU: {oulu_result_str} ({oulu_score:.2f}) | Blink: {blink_status_str}"

            else: # 如果 liveness_details 为空，可能人脸检测没成功，或 VisionServiceWorker 实例异常
                overall_status_message = "No faces/AI issue."
                overall_status_color = (0, 165, 255) # Orange
                liveness_status_text = "Processing..."
                box_color = (255, 255, 0) # Yellow (processing)


            # Draw bounding box for the main face
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)

            # Draw text label for the main face
            cv2.putText(display_frame, liveness_status_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            # Display blink counts and status
            cv2.putText(display_frame, f"Total Blinks: {liveness_tester_instance.total_blinks}", (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if liveness_tester_instance.blink_detection_active and liveness_tester_instance.total_blinks == 0:
                # Use the imported BLINK_TIMEOUT_FRAMES directly
                cv2.putText(display_frame, f"Please Blink! ({liveness_tester_instance.frames_since_last_blink}/{BLINK_TIMEOUT_FRAMES})", (w - 300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


        # Update guide box color and overall message based on detected faces & quality
        # This logic is complex and relies on the `overall_status_message` which is set by the liveness check
        # It needs to be carefully ordered to reflect the most critical status.
        if "SPOOFING DETECTED!" in overall_status_message:
            guide_color = (0, 0, 255) # Red for spoof
        elif "LIVE Person detected." in overall_status_message:
            guide_color = (0, 255, 0) # Green for live
        elif detected_faces and face_in_guide_box and face_size_ok: # Before liveness check results are final
            guide_color = (0, 255, 0) # Green: Good position for liveness check
            overall_status_message = "Face OK. Performing Liveness Check."
            overall_status_color = (0, 255, 0)
        elif detected_faces and (not face_in_guide_box or not face_size_ok):
            guide_color = (0, 165, 255) # Orange: Adjust position/size
            if not face_in_guide_box:
                overall_status_message = "Please center your face."
            elif not face_size_ok:
                overall_status_message = "Move closer/further."
            overall_status_color = (0, 165, 255)
        elif not detected_faces:
            guide_color = (255, 255, 255) # White: No face detected
            overall_status_message = "No face detected."
            overall_status_color = (100, 100, 100)
        # If overall status message was already set to "No faces/AI issue.", keep it.


        # Draw the guide box
        cv2.rectangle(display_frame, (start_x_guide, start_y_guide), (end_x_guide, end_y_guide), guide_color, guide_thickness)

        # Display overall status on top of the frame
        cv2.putText(display_frame, overall_status_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, overall_status_color, 2, cv2.LINE_AA)

        cv2.imshow('OULU Liveness + Blink Liveness Test (Press Q to Quit)', display_frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("User pressed 'q'. Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Application exited gracefully.")


if __name__ == "__main__":
    run_camera_liveness_test(camera_id=0) # 0 for default camera, try 1, 2 etc. if you have multiple.