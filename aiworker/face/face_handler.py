# aiworker/face/face_handler.py
import cv2
import base64
import numpy as np
from .vision_service import VisionServiceWorker
from aiworker.config import (
    RECOMMENDED_FACE_RECT_RATIO, RECOMMENDED_FACE_MIN_PIXELS, BLINK_TIMEOUT_FRAMES
)


def process_frame_for_stream(vision_worker: VisionServiceWorker, frame: np.ndarray, known_faces_data: list,
                             camera_id: int, perform_heavy_ai: bool):
    """
    为实时视频流处理单帧，包含UI绘制和可选的AI计算。
    """
    detection_data = {
        'persons': [],
        'liveness_passed': True,
        'liveness_details_per_face': [],
        'events_to_log': []
    }
    processed_frame = frame.copy()
    h, w, _ = processed_frame.shape

    # 1. 人脸检测 (每帧都做，相对轻量)
    detected_faces = vision_worker.detect_faces(frame)
    liveness_details = []

    # 2. 昂贵的AI计算 (跳帧执行)
    if perform_heavy_ai and detected_faces:
        liveness_status, liveness_details = vision_worker.liveness_detector.perform_liveness_check(
            frame, detected_faces, require_blink=False
        )
        detection_data['liveness_passed'] = liveness_status
        detection_data['liveness_details_per_face'] = liveness_details

        recognized_identities = vision_worker.recognize_faces(frame, detected_faces, known_faces_data)

        # 合并结果并记录事件
        for person in recognized_identities:
            matched_liveness = next((ld for ld in liveness_details if ld['box_coords'] == person['box_coords']), None)
            if matched_liveness:
                person['liveness_info'] = matched_liveness
                if not matched_liveness['combined_live_status']:
                    # 填充完整的事件数据
                    event_to_log = {
                        'event_type': 'LIVENESS_FRAUD_DETECTED',
                        # 即使是欺诈，也尝试记录识别出的身份信息
                        'person_id': person.get('person_id'),
                        'person_name': person.get('identity', 'Unknown'),  # 可能是 'Stranger' 或识别出的名字
                        # 'detection_details' 包含了所有技术细节，非常重要
                        'detection_details': {
                            'box_coords': person['box_coords'],
                            'message': 'Liveness check failed.',
                            'liveness_info': {
                                'oulu_score': matched_liveness.get('oulu_score'),
                                'oulu_result': matched_liveness.get('oulu_result'),
                                'blink_status': matched_liveness.get('blink_status'),
                            }
                        }
                    }
                    detection_data['events_to_log'].append(event_to_log)
            detection_data['persons'].append(person)

    # 3. UI绘制 (每帧都做，保证流畅)
    _draw_guidance_and_results(processed_frame, detected_faces, detection_data['persons'], vision_worker)

    return processed_frame, detection_data


def process_frame_for_api(vision_worker: VisionServiceWorker, frame: np.ndarray, known_faces_data: list):
    """
    为单帧API请求进行处理，进行严格的活体检测，并返回完整JSON。
    """
    detected_faces = vision_worker.detect_faces(frame)
    if not detected_faces:
        response_json = {
            'status': 'error',
            'message': 'No face detected',
            'liveness_passed': False,
            'persons': [],
            'processed_image': None
        }
        return response_json, frame  # 第二个返回值是原始帧，用于可能的内部调试

    # 对于单帧API，我们强制要求眨眼以达到最高的防伪级别
    liveness_details = vision_worker.liveness_detector.perform_liveness_check(
        frame, detected_faces, require_blink=True
    )
    liveness_status = bool(liveness_details) and all(
        item.get('combined_live_status', False) for item in liveness_details
    )

    recognized_identities = vision_worker.recognize_faces(frame, detected_faces, known_faces_data)

    persons_data = []
    for person in recognized_identities:
        matched_liveness = next((ld for ld in liveness_details if ld['box_coords'] == person['box_coords']), None)
        if matched_liveness:
            person['liveness_info'] = matched_liveness
        persons_data.append(person)

    processed_image = frame.copy()
    _draw_guidance_and_results(processed_image, detected_faces, persons_data, vision_worker)

    _, buffer = cv2.imencode('.jpg', processed_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    response_json = {
        'status': 'success',
        'persons': persons_data,
        'liveness_passed': liveness_status,
        'processed_image': f"data:image/jpeg;base64,{img_base64}"
    }
    return response_json, processed_image


def _draw_guidance_and_results(frame, detected_faces, persons_data, vision_worker):
    """
    一个独立的绘制函数，负责将所有指导信息和分析结果绘制到帧上。

    Args:
        frame (np.ndarray): 要绘制的图像帧。
        detected_faces (list): 原始人脸检测结果列表，用于判断是否有脸。
        persons_data (list): 包含识别和活体信息的详细人员列表。
        vision_worker (VisionServiceWorker): VisionServiceWorker 实例，用于获取眨眼状态。
    """
    h, w, _ = frame.shape

    # --- 1. 计算指导框位置 ---
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

    # --- 2. 根据检测结果，确定指导文字和颜色 ---
    guide_color = (255, 255, 255)  # 默认白色
    status_message = "No face detected."
    status_color = (100, 100, 100)  # 默认灰色

    if detected_faces:
        # 以第一个检测到的人脸为主要判断依据
        main_face_box = detected_faces[0]['box_coords']
        mx1, my1, mx2, my2 = main_face_box
        main_face_width = mx2 - mx1

        face_in_guide_box = (
                    mx1 >= start_x_guide and my1 >= start_y_guide and mx2 <= end_x_guide and my2 <= end_y_guide)
        face_size_ok = main_face_width >= RECOMMENDED_FACE_MIN_PIXELS

        # 检查是否已有最终的活体/欺诈判断结果
        is_spoof = any(p.get('liveness_info', {}).get('combined_live_status') is False for p in persons_data)
        is_live = all(p.get('liveness_info', {}).get('combined_live_status', True) for p in
                      persons_data) if persons_data else False

        if is_spoof:
            status_message = "SPOOFING DETECTED!"
            status_color = (0, 0, 255)  # 红色
        elif is_live:
            status_message = "LIVE Person detected."
            status_color = (0, 255, 0)  # 绿色
        elif face_in_guide_box and face_size_ok:
            status_message = "Face OK. Please wait..."
            status_color = (0, 255, 0)  # 绿色
        else:
            status_color = (0, 165, 255)  # 橙色
            if not face_in_guide_box:
                status_message = "Please center your face."
            elif not face_size_ok:
                status_message = "Please move closer."

    # 根据状态文字最终决定指导框颜色
    if "SPOOFING" in status_message:
        guide_color = (0, 0, 255)
    elif "LIVE" in status_message or "Face OK" in status_message:
        guide_color = (0, 255, 0)
    elif "center" in status_message or "closer" in status_message:
        guide_color = (0, 165, 255)

    # --- 3. 开始绘制 ---
    # 绘制指导框
    cv2.rectangle(frame, (start_x_guide, start_y_guide), (end_x_guide, end_y_guide), guide_color, 2)
    # 绘制顶部状态信息
    cv2.putText(frame, status_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)

    # 绘制眨眼提示 (如果需要)
    # 注意: vision_worker.liveness_detector 才是真正持有眨眼状态的对象
    if vision_worker.liveness_detector.blink_detection_active:
        total_blinks = vision_worker.liveness_detector.total_blinks
        if total_blinks == 0:
            blink_frames = vision_worker.liveness_detector.frames_since_last_blink
            cv2.putText(frame, f"Please Blink! ({blink_frames}/{BLINK_TIMEOUT_FRAMES})",
                        (w - 300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Total Blinks: {total_blinks}",
                    (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # 绘制每个识别出的人脸信息
    for person in persons_data:
        x1, y1, x2, y2 = person['box_coords']
        identity = person.get('identity', 'Unknown')
        liveness_info = person.get('liveness_info', {})
        is_live = liveness_info.get('combined_live_status', False)

        # 根据活体和身份确定框的颜色和标签
        box_color = (0, 255, 0)  # 默认绿色
        display_label = identity

        if not is_live:
            box_color = (0, 0, 255)  # 红色 (欺诈)
            display_label = f"SPOOF ({liveness_info.get('oulu_result', 'N/A')})"
        elif identity == 'Stranger':
            box_color = (0, 255, 255)  # 黄色 (陌生人)
        elif person.get('person_state') == 2:
            box_color = (0, 165, 255)  # 橙色 (危险人员)
            display_label = f"DANGER ({identity})"

        # 添加距离到标签
        distance = person.get('distance')
        if distance is not None:
            display_label += f" ({distance:.2f})"

        # 绘制人脸框和身份标签
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, display_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2, cv2.LINE_AA)

        # 绘制详细的活体检测结果
        oulu_text = f"Liveness: {liveness_info.get('oulu_result', 'N/A')} ({liveness_info.get('oulu_score', 0.0):.2f})"
        blink_text = f"Blink: {liveness_info.get('blink_status', 'N/A')}"
        cv2.putText(frame, oulu_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA)
        cv2.putText(frame, blink_text, (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA)

    return frame
