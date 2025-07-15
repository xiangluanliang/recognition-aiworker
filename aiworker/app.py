# local_ai_service.py
import base64
import os
import cv2
import datetime

import numpy as np
import requests
import threading
import logging
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from yolo_detector import process_single_frame as person_detector
# 导入 VisionServiceWorker 类以及新的指南框常量
from face_recognizer2 import VisionServiceWorker, RECOMMENDED_FACE_RECT_RATIO, RECOMMENDED_FACE_MIN_PIXELS, BLINK_TIMEOUT_FRAMES
from report_generator import process_report_generation

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - (AI-Worker) - %(message)s')

vision_worker_instance = VisionServiceWorker()

def process_frame_face_recognition(frame, known_faces_data, camera_id):
    """
    负责集成 VisionServiceWorker 的人脸识别和活体检测功能。
    这个函数作为 local_ai_service.py 中 'face_recognition' AI 函数的包装器。
    它确保人脸识别只在活体检测通过后进行，并绘制指南框。
    """
    detection_data = {
        'persons': [],
        'liveness_passed': True, # 整体活体状态，若有任何非活体，则设为False
        'liveness_details_per_face': [],
        'events_to_log': []
    }
    processed_frame = frame.copy() # 用于绘制的帧
    h, w, _ = processed_frame.shape

    # --- 新增：绘制推荐人脸区域指导框的逻辑 ---
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

    guide_color = (255, 255, 255) # 默认白色
    guide_thickness = 2
    overall_status_message_display = "Please center your face." # 顶部状态消息
    overall_status_color_display = (100, 100, 100) # 灰色

    # 1. 预处理帧并检测人脸
    preprocessed_frame_for_detection = vision_worker_instance.preprocess_frame(frame)
    detected_faces = vision_worker_instance.detect_faces(preprocessed_frame_for_detection)

    # 2. 对所有检测到的人脸进行活体检测
    # 对于实时视频流，通常不强制眨眼 (require_blink=False)
    # API 端点 (`recognize_frame_api`) 会明确设置为 True
    overall_liveness_status, liveness_details = vision_worker_instance.perform_liveness_check(
        frame=frame,
        detected_faces_info=detected_faces,
        require_blink=False # 对于实时流，通常不强制眨眼。
    )
    detection_data['liveness_passed'] = overall_liveness_status
    detection_data['liveness_details_per_face'] = liveness_details

    # --- 更新指南框颜色和顶部状态消息 (遵循 test_deeppixbis_camera.py 的逻辑) ---
    face_in_guide_box = False
    face_size_ok = False

    if detected_faces:
        # 考虑第一个检测到的面部作为主要参考
        main_face_box_coords = detected_faces[0]['box_coords']
        mx1, my1, mx2, my2 = main_face_box_coords
        main_face_width = mx2 - mx1
        main_face_height = my2 - my1

        if (mx1 >= start_x_guide and my1 >= start_y_guide and mx2 <= end_x_guide and my2 <= end_y_guide):
            face_in_guide_box = True

        if min(main_face_width, main_face_height) >= RECOMMENDED_FACE_MIN_PIXELS:
            face_size_ok = True

        # 根据活体检测的最终结果设置最高优先级状态
        if overall_liveness_status: # 如果所有脸都活体通过
            overall_status_message_display = "LIVE Person detected."
            overall_status_color_display = (0, 255, 0) # 绿色
        else: # 如果有任何脸未通过活体（欺骗或太小）
            overall_status_message_display = "SPOOFING DETECTED!"
            overall_status_color_display = (0, 0, 255) # 红色

        # 如果最终状态不是欺骗或活体，则根据位置更新
        if overall_status_message_display not in ["SPOOFING DETECTED!", "LIVE Person detected."]:
            if face_in_guide_box and face_size_ok:
                overall_status_message_display = "Face OK. Performing Liveness Check."
                overall_status_color_display = (0, 255, 0) # 绿色：位置良好
            else:
                overall_status_color_display = (0, 165, 255) # 橙色：调整位置/大小
                if not face_in_guide_box:
                    overall_status_message_display = "Please center your face."
                elif not face_size_ok:
                    overall_status_message_display = "Move closer/further."
    else: # 没有检测到人脸
        overall_status_message_display = "No face detected."
        overall_status_color_display = (100, 100, 100) # 灰色

    # 最后根据整体状态确定指南框颜色
    if "SPOOFING DETECTED!" in overall_status_message_display:
        guide_color = (0, 0, 255) # 红色
    elif "LIVE Person detected." in overall_status_message_display:
        guide_color = (0, 255, 0) # 绿色
    elif overall_status_message_display == "Face OK. Performing Liveness Check.":
        guide_color = (0, 255, 0) # 绿色
    elif overall_status_message_display in ["Please center your face.", "Move closer/further."]:
        guide_color = (0, 165, 255) # 橙色
    else: # 默认无脸状态
        guide_color = (255, 255, 255) # 白色


    # 绘制指南框
    cv2.rectangle(processed_frame, (start_x_guide, start_y_guide), (end_x_guide, end_y_guide), guide_color, guide_thickness)

    # 在顶部显示整体状态信息
    cv2.putText(processed_frame, overall_status_message_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, overall_status_color_display, 2, cv2.LINE_AA)

    # 显示眨眼提示 (如果需要且活跃)
    if vision_worker_instance.blink_detection_active and vision_worker_instance.total_blinks == 0:
        cv2.putText(processed_frame, f"Please Blink! ({vision_worker_instance.frames_since_last_blink}/{BLINK_TIMEOUT_FRAMES})",
                    (w - 300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    # 也显示总眨眼次数
    cv2.putText(processed_frame, f"Total Blinks: {vision_worker_instance.total_blinks}",
                (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


    # 3. 合并活体详情和识别结果，并根据活体状态确定最终输出和报警
    all_recognized_faces = vision_worker_instance.recognize_face_identity(frame, known_faces_data)

    detection_data['persons'] = [] # 清空，重新填充
    for face_rec_candidate in all_recognized_faces:
        matched_liveness_info = next((ld for ld in liveness_details if ld['box_coords'] == face_rec_candidate['box_coords']), None)

        current_face_liveness_info = {
            'oulu_score': 0.0, 'oulu_result': 'N/A', 'blink_status': 'N/A', 'combined_live_status': False # 默认为非活体以确保安全
        }
        if matched_liveness_info:
            current_face_liveness_info = matched_liveness_info
        
        face_rec_candidate['liveness_info'] = current_face_liveness_info
        
        # 活体检测判断
        is_live_combined = current_face_liveness_info.get('combined_live_status', False)

        color = (0, 255, 0) # 默认绿色 (合法用户，活体)
        display_label = face_rec_candidate['identity']

        # 如果活体检测失败 (非活人)，则标记为欺骗攻击，并触发报警
        if not is_live_combined:
            color = (0, 0, 255) # 红色 (欺骗攻击)
            display_label = f"SPOOF ({current_face_liveness_info['oulu_result']})"
            detection_data['liveness_passed'] = False # 标记整体活体失败

            detection_data['events_to_log'].append({
                'event_type': 'LIVENESS_FRAUD_DETECTED',
                'person_id': face_rec_candidate.get('person_id'),
                'person_name': face_rec_candidate.get('identity', 'Unknown'),
                'detection_details': {
                    'box_coords': face_rec_candidate['box_coords'],
                    'oulu_score': current_face_liveness_info['oulu_score'],
                    'oulu_result': current_face_liveness_info['oulu_result'],
                    'blink_status': current_face_liveness_info['blink_status']
                }
            })
            app.logger.warning(f"Liveness fraud detected for {display_label} in camera {camera_id}!")

        elif face_rec_candidate['identity'] == 'Stranger':
            color = (0, 255, 255) # 黄色 (陌生人，活体)
        elif face_rec_candidate['identity'] != 'Stranger' and face_rec_candidate.get('person_state') == 2: # 假设2为危险人员状态
            color = (0, 165, 255) # 橙色 (危险人员，活体)
            display_label = f"DANGER ({face_rec_candidate['identity']})"
            # 记录危险人员事件
            detection_data['events_to_log'].append({
                'event_type': 'DANGER_PERSON_DETECTED',
                'person_id': face_rec_candidate['person_id'],
                'person_name': face_rec_candidate['identity'],
                'detection_details': {
                    'box_coords': face_rec_candidate['box_coords'],
                    'liveness_info': current_face_liveness_info
                }
            })
            app.logger.warning(f"Danger person detected: {face_rec_candidate['identity']} in camera {camera_id}!")
        
        # 添加距离到标签（如果可用）
        if face_rec_candidate['distance'] is not None:
            display_label += f" ({face_rec_candidate['distance']:.2f})"

        # 绘制边界框
        x1, y1, x2, y2 = face_rec_candidate['box_coords']
        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(processed_frame, display_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # 绘制活体信息 (下方)
        cv2.putText(processed_frame, f"Liveness: {current_face_liveness_info['oulu_result']} ({current_face_liveness_info['oulu_score']:.2f})",
                            (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        cv2.putText(processed_frame, f"Blink: {current_face_liveness_info['blink_status']}",
                            (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

        # 填充 detection_data['persons']
        detection_data['persons'].append(face_rec_candidate) # 包含liveness_info


        # 只有活体且识别成功的用户才记录“人员识别”事件
        if face_rec_candidate['identity'] != 'Stranger' and is_live_combined:
            detection_data['events_to_log'].append({
                'event_type': 'PERSON_RECOGNIZED',
                'person_id': face_rec_candidate['person_id'],
                'person_name': face_rec_candidate['identity'],
                'detection_details': {
                    'box_coords': face_rec_candidate['box_coords'],
                    'liveness_info': current_face_liveness_info
                }
            })
            app.logger.info(f"Person recognized: {face_rec_candidate['identity']} in camera {camera_id} (LIVE)!")


    return processed_frame, detection_data


AI_FUNCTIONS = {
    'person_detection': person_detector,
    'face_recognition': process_frame_face_recognition,
}

DJANGO_API_TOKEN = os.environ.get('DJANGO_API_TOKEN', '3d814142795cace7')
DJANGO_API_BASE_URL = os.environ.get('DJANGO_API_URL', 'http://172.21.167.220:8000/api/test/')
RTMP_SERVER_URL = os.environ.get('RTMP_SERVER_URL', 'rtmp://localhost:9090/live')
KNOWN_FACES_CACHE = []
CACHE_REFRESH_INTERVAL = 300


def fetch_and_cache_known_faces():
    global KNOWN_FACES_CACHE
    try:
        url = os.path.join(DJANGO_API_BASE_URL, 'known-faces/')
        response = requests.get(url, timeout=10, headers={"Authorization": f"Token {DJANGO_API_TOKEN}"})
        response.raise_for_status()
        KNOWN_FACES_CACHE = response.json()
        app.logger.info(f"Successfully refreshed known faces cache. Loaded {len(KNOWN_FACES_CACHE)} faces.")
    except Exception as e:
        app.logger.error(f"Failed to fetch known faces from Django: {e}")


def schedule_face_cache_refresh():
    fetch_and_cache_known_faces()
    threading.Timer(CACHE_REFRESH_INTERVAL, schedule_face_cache_refresh).start()


def log_event_to_django(event_data: dict):
    def _send_request():
        try:
            url = os.path.join(DJANGO_API_BASE_URL, 'log-event/')
            requests.post(url, json=event_data, timeout=5, headers={"Authorization": f"Token {DJANGO_API_TOKEN}"})
        except Exception as e:
            app.logger.error(f"Failed to log event to Django: {e}")

    threading.Thread(target=_send_request).start()


def capture_and_process_thread(stream_id: str, ai_function_name: str, camera_id: str):
# def capture_and_process_thread(stream_id: str, ai_function_name: str):
#     """
#     这个函数在后台线程中运行，负责：
#     1. 从RTMP服务器拉流。
#     2. 根据ai_function_name对帧进行处理（或不处理）。
#     3. 将处理后的帧编码为JPEG并存入全局缓存。
#     """
#     cache_key = (stream_id, ai_function_name)
#     stream_lock = video_streams_cache[cache_key]['lock']
#
#     rtmp_url = f'{RTMP_SERVER_URL}{stream_id}'
#     cap = cv2.VideoCapture(rtmp_url)
#     if not cap.isOpened():
#         app.logger.error(f"Cannot open stream: {rtmp_url}")
#         # 可以考虑在缓存中设置一个错误状态
#         return
#
#     app.logger.info(f"Thread started for stream '{stream_id}' with AI function '{ai_function_name}'")
#
#     frame_skip = 5
#     frame_count = 0
#     fail_count = 0
#
#     process_function = AI_FUNCTIONS.get(ai_function_name)
#
#     while True:
#
#         success, frame = cap.read()
#         if not success:
#             fail_count += 1
#             if fail_count > 100:
#                 app.logger.error(f"Stream '{stream_id}' disconnected. Stopping thread.")
#                 break
#             time.sleep(0.1)  # 稍作等待
#             continue
#         fail_count = 0
#
#         frame = cv2.resize(frame, (854, 480))
#
#         processed_frame = frame
#
#         if process_function and frame_count % frame_skip == 0:
#             if ai_function_name == 'face_recognition':
#                 processed_frame, detection_data = process_function(
#                     frame.copy(),
#                     known_faces_data=KNOWN_FACES_CACHE,
#                     camera_id=int(stream_id)
#                 )
#             else:
#                 processed_frame, detection_data = process_function(
#                     frame.copy(),
#                     camera_id=int(stream_id)
#                 )
#
#             if detection_data and 'events_to_log' in detection_data:
#                 for event in detection_data['events_to_log']:
#                     event['camera'] = int(stream_id)
#                     event['time'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
#                     log_event_to_django(event)
#
#         ret, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
#         if ret:
#             with stream_lock:
#                 video_streams_cache[cache_key]['frame_bytes'] = buffer.tobytes()
#
#         frame_count += 1
#
#     cap.release()
#     with video_streams_cache[cache_key]['lock']:
#         del video_streams_cache[cache_key]
#     app.logger.info(f"Thread stopped and cache cleaned for {cache_key}")

# def capture_and_process_thread(stream_id: str, ai_function_name: str):
    """
    这个函数在后台线程中运行，负责：
    1.从RTMP服务器拉流。
    2.根据ai_function_name对帧进行处理（或不处理）。
    3.将处理后的帧编码为JPEG并存入全局缓存。
    """
    cache_key = (stream_id, ai_function_name, camera_id)
    stream_lock = video_streams_cache[cache_key]['lock']

    camera_id_int = int(camera_id)

    rtmp_url = f'{RTMP_SERVER_URL}{stream_id}'
    cap = cv2.VideoCapture(rtmp_url)
    if not cap.isOpened():
        app.logger.error(f"Cannot open stream: {rtmp_url}")
        return

    app.logger.info(f"Thread started for stream '{stream_id}' with AI function '{ai_function_name}', camera '{camera_id}'")

    frame_skip = 5
    frame_count = 0
    fail_count = 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    camera_id = int(stream_id)

    # ✅ 初始化异常检测缓存
    from collections import defaultdict, deque
    prev_centers = {}
    fall_clip_buffer = defaultdict(lambda: deque(maxlen=int(fps * 5)))
    person_history = defaultdict(list)
    person_fall_status = defaultdict(lambda: {'fall_frame_count': 0, 'is_falling': False})
    zone_status_cache = defaultdict(dict)
    recorded_intrusions = set()
    recorded_conflicts = set()
    fight_kpts_history = defaultdict(lambda: deque(maxlen=5))

    # ✅ 拉取摄像头异常区域
    def fetch_warning_zones(camera_id):
        try:
            response = requests.get(
                f"{DJANGO_API_BASE_URL}get-warning-zones/{camera_id}/",
                headers={"Authorization": f"Token {DJANGO_API_TOKEN}"},
                verify=False
            )
            response.raise_for_status()
            return response.json()  # 格式: {camera_id: [[(x, y), (x, y), ...], ...]}
        except Exception as e:
            app.logger.error(f"Failed to fetch warning zones: {e}")
            return {camera_id: []}

    warning_zone_map = fetch_warning_zones(camera_id)

    # ✅ 伪造 camera 对象（简化处理）
    camera = {
        'id': camera_id,
        'name': f"Camera {camera_id}"
    }

    process_function = AI_FUNCTIONS.get(ai_function_name)

    while True:
        success, frame = cap.read()
        if not success:
            fail_count += 1
            if fail_count > 100:
                app.logger.error(f"Stream '{stream_id}' disconnected. Stopping thread.")
                break
            time.sleep(0.1)
            continue
        fail_count = 0

        frame = cv2.resize(frame, (854, 480))
        processed_frame = frame

        # 每隔 frame_skip 帧处理一次
        if process_function and frame_count % frame_skip == 0:
            if ai_function_name == 'face_recognition':
                processed_frame, detection_data = process_function(
                    frame,
                    known_faces_data=KNOWN_FACES_CACHE,
                    camera_id=camera_id_int
                )

            elif ai_function_name == 'abnormal_detection':
                from aiworker.yolo.yolo_abnormal_detector import process_abnormal_single_frame

                processed_frame = process_abnormal_single_frame(
                    frame=frame.copy(),
                    frame_idx=frame_count,
                    fps=fps,
                    camera_id=camera_id,
                    stay_seconds=5,  # 可动态传入
                    safe_distance=50.0,  # 可动态传入
                    prev_centers=prev_centers,
                    fall_clip_buffer=fall_clip_buffer,
                    person_history=person_history,
                    person_fall_status=person_fall_status,
                    zone_status_cache=zone_status_cache,
                    recorded_intrusions=recorded_intrusions,
                    recorded_conflicts=recorded_conflicts,
                    fight_kpts_history=fight_kpts_history,
                    warning_zone_map=warning_zone_map,
                    camera=camera,
                    log_event_to_django=log_event_to_django
                )

            else:
                # 其他功能（人流量统计等）
                processed_frame, detection_data = process_function(
                    frame.copy(),
                    camera_id=camera_id_int
                )

            # --- Stream Processor: 统一处理报警事件日志发送 ---
            if detection_data and 'events_to_log' in detection_data:
                for event in detection_data['events_to_log']:
                    event['camera'] = camera_id
                    event['time'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    log_event_to_django(event)
        else:
            # 如果不处理这一帧，只使用原始帧进行显示，但为了确保指南框和顶部状态消息持续更新，
            # 即使不执行完整的AI处理，也应该至少调用 process_frame_face_recognition 来获取这些视觉信息。
            # 这是一个权衡：要么每帧都运行大部分逻辑来更新UI，要么只在处理帧时更新。
            # 为了指南框和状态实时性，建议即使跳帧也调用。
            # 然而，为了保持原有的 frame_skip 逻辑，我们这里保留只在处理帧时才调用AI函数的做法。
            # 这意味着在跳过的帧上，指南框和状态不会更新。
            # 如果需要每帧都更新UI元素，需要修改此处的逻辑，让 process_frame_face_recognition
            # 即使在不进行完整AI推理时也能提供绘制结果。
            # For simplicity and minimal invasiveness, we stick to the original frame skipping for AI processing.
            processed_frame = frame.copy() # Copy to ensure we can draw on it later if needed (though not done here)

            # 即使是跳过的帧，也可以选择绘制指南框和顶部状态消息，但这需要
            # process_frame_face_recognition 能够返回一个 'empty' 或 'passthrough' 模式的帧。
            # 为了不改变 process_frame_face_recognition 的复杂性，我们让它只在实际处理时绘制。
            # 因此，在跳过的帧上，不会出现指南框或状态更新。

        # 这里的 processed_frame 可能是来自 AI 函数的结果，也可能是原始帧 (如果跳过处理)
        if processed_frame is None: processed_frame = frame.copy() # 确保不为 None
        ret, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ret:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        frame_count += 1
    cap.release()
    with stream_lock:
        video_streams_cache.pop(cache_key, None)
    app.logger.info(f"Thread stopped and cache cleaned for {cache_key}")


def stream_generator(stream_id: str, ai_function_name: str, camera_id: str):
    """
    从缓存中读取对应 camera_id 的帧并持续推送给前端。
    """
    cache_key = (stream_id, ai_function_name, camera_id)

    # 等待后台线程生成第一帧
    while cache_key not in video_streams_cache or 'frame_bytes' not in video_streams_cache[cache_key]:
        app.logger.debug(f"Waiting for first frame of {cache_key}...")
        time.sleep(0.5)
        if cache_key not in video_streams_cache:
            app.logger.warning(f"Stream {cache_key} seems to have failed to start.")
            return

    app.logger.info(f"Client connected to stream generator for {cache_key}")

    while cache_key in video_streams_cache:
        try:
            with video_streams_cache[cache_key]['lock']:
                frame_bytes = video_streams_cache[cache_key].get('frame_bytes')

            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(1 / 20)  # 控制帧率
        except KeyError:
            app.logger.warning(f"Cache key {cache_key} was removed. Closing generator.")
            break

@app.route('/<ai_function_name>/<stream_id>/<camera_id>')
def video_feed(ai_function_name: str, stream_id: str, camera_id: str):
    """
    Flask 路由：处理视频流请求，按 (stream_id, ai_function_name, camera_id) 缓存并启动线程。
    """
    cache_key = (stream_id, ai_function_name, camera_id)

    if cache_key not in video_streams_cache:
        app.logger.info(f"Cache miss for {cache_key}. Creating new processing thread.")
        lock = threading.Lock()
        thread = threading.Thread(
            target=capture_and_process_thread,
            args=(stream_id, ai_function_name, camera_id),  # 传入 camera_id
            daemon=True
        )
        video_streams_cache[cache_key] = {'thread': thread, 'lock': lock, 'frame_bytes': None}
        thread.start()

    return Response(
        stream_generator(stream_id, ai_function_name, camera_id),  # 传入 camera_id
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/ai/recognize-frame', methods=['POST'])
def recognize_frame_api():
    data = request.json
    image_base64 = data.get('image_data')

    if not image_base64:
        return jsonify({'status': 'error', 'message': '未提供 image_data'}), 400

    try:
        if ',' in image_base64:
            header, encoded = image_base64.split(',', 1)
        else:
            encoded = image_base64

        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("图像解码失败")
        else:
            app.logger.info("API: 图像解码成功")

    except Exception as e:
        app.logger.error(f"API: Base64 image decoding failed: {e}")
        return jsonify({'status': 'error', 'message': '无效的图像数据'}), 400

    # --- API endpoint calls VisionServiceWorker methods directly for more control ---
    # For API calls, we don't draw the guide box here as it's a single frame
    # The processed_image_for_api_response will have the detection results drawn on it.

    preprocessed_frame_api = vision_worker_instance.preprocess_frame(frame)
    detected_faces_api = vision_worker_instance.detect_faces(preprocessed_frame_api)

    # 对于 API 调用，明确强制眨眼检测以进行严格的活体验证
    overall_liveness_api_status, liveness_details_api = vision_worker_instance.perform_liveness_check(
        frame=frame,
        detected_faces_info=detected_faces_api,
        require_blink=True # 强制眨眼检测
    )

    api_response_persons = []
    processed_image_for_api_response = frame.copy() # 用于API响应的绘制帧

    # 遍历所有检测到的脸，根据活体结果决定是否进行识别和绘制
    for face_liveness_info in liveness_details_api:
        box_coords = face_liveness_info['box_coords']
        is_live_combined = face_liveness_info.get('combined_live_status', False)

        face_rec_info = {
            'box_coords': box_coords,
            'confidence': 0.0,
            'identity': 'Unknown',
            'distance': None,
            'person_id': None,
            'person_state': None,
            'liveness_info': face_liveness_info
        }
        
        original_detected_face = next((d for d in detected_faces_api if d['box_coords'] == box_coords), None)
        if original_detected_face:
            face_rec_info['confidence'] = original_detected_face['confidence']


        if is_live_combined:
            temp_rec_results = vision_worker_instance.recognize_face_identity(frame, KNOWN_FACES_CACHE)
            matched_rec = next((r for r in temp_rec_results if r['box_coords'] == box_coords), None)
            
            if matched_rec:
                face_rec_info.update({
                    'identity': matched_rec['identity'],
                    'distance': matched_rec['distance'],
                    'person_id': matched_rec['person_id'],
                    'person_state': matched_rec['person_state'],
                })
        else:
            face_rec_info['identity'] = 'SPOOF'
            face_rec_info['person_id'] = None
            face_rec_info['person_state'] = None
            app.logger.warning(f"API: Liveness check failed for face at {box_coords}. Marked as SPOOF.")


        color = (0, 255, 0)
        display_label = face_rec_info['identity']

        if not is_live_combined:
            color = (0, 0, 255)
            display_label = f"SPOOF ({face_liveness_info['oulu_result']})"
            log_event_to_django({
                'event_type': 'LIVENESS_FRAUD_DETECTED',
                'camera': 0,
                'time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'person_id': face_rec_info.get('person_id'),
                'person_name': face_rec_info.get('identity', 'Unknown/Spoof'),
                'detection_details': {
                    'box_coords': face_rec_info['box_coords'],
                    'oulu_score': face_liveness_info['oulu_score'],
                    'oulu_result': face_liveness_info['oulu_result'],
                    'blink_status': face_liveness_info['blink_status']
                }
            })
            app.logger.warning(f"API Call: Liveness fraud detected for {display_label}!")

        elif face_rec_info['identity'] == 'Stranger':
            color = (0, 255, 255)
        elif face_rec_info['identity'] != 'Stranger' and face_rec_info.get('person_state') == 2:
            color = (0, 165, 255)
            display_label = f"DANGER ({face_rec_info['identity']})"
            log_event_to_django({
                'event_type': 'DANGER_PERSON_DETECTED',
                'camera': 0,
                'time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'person_id': face_rec_info['person_id'],
                'person_name': face_rec_info['identity'],
                'detection_details': {
                    'box_coords': face_rec_info['box_coords'],
                    'liveness_info': face_liveness_info
                }
            })
            app.logger.warning(f"API Call: Danger person detected: {face_rec_info['identity']}!")

        if face_rec_info['distance'] is not None:
            display_label += f" ({face_rec_info['distance']:.2f})"
        
        x1, y1, x2, y2 = face_rec_info['box_coords']
        cv2.rectangle(processed_image_for_api_response, (x1, y1), (x2, y2), color, 2)
        cv2.putText(processed_image_for_api_response, display_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        cv2.putText(processed_image_for_api_response, f"Liveness: {face_liveness_info['oulu_result']} ({face_liveness_info['oulu_score']:.2f})",
                            (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        cv2.putText(processed_image_for_api_response, f"Blink: {face_liveness_info['blink_status']}",
                            (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

        api_response_persons.append(face_rec_info)


        if face_rec_info['identity'] != 'Stranger' and is_live_combined:
            log_event_to_django({
                'event_type': 'PERSON_RECOGNIZED',
                'camera': 0,
                'time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'person_id': face_rec_info['person_id'],
                'person_name': face_rec_info['identity'],
                'detection_details': {
                    'box_coords': face_rec_info['box_coords'],
                    'liveness_info': face_liveness_info
                }
            })
            app.logger.info(f"API Call: Person recognized: {face_rec_info['identity']} (LIVE)!")


    _, buffer = cv2.imencode('.jpg', processed_image_for_api_response)
    processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'status': 'success',
        'persons': api_response_persons,
        'liveness_passed': overall_liveness_api_status,
        'liveness_info': liveness_details_api,
        'processed_image': f"data:image/jpeg;base64,{processed_image_base64}"
    })


def fetch_summary_for_report():
    """从Django后端获取日报所需的数据摘要。"""
    api_url = f"{DJANGO_API_BASE_URL}daily-report/today"
    headers = {"Authorization": f"Token {DJANGO_API_TOKEN}"}
    app.logger.info(f"ReportGen: Fetching data from {api_url}")
    response = requests.get(api_url, headers=headers, timeout=20, verify=False)
    response.raise_for_status()
    return response.json()


@app.route('/ai/generate-report', methods=['POST'])
def generate_report_endpoint():
    """
    API端点，调用独立的AI模块来生成日报。
    """
    app.logger.info("Received request to generate daily report.")
    try:
        summary_data = fetch_summary_for_report()
        report_content = process_report_generation(summary_data)

        return jsonify({
            "status": "success",
            "message": "Daily report generated and submitted successfully.",
            "content": report_content
        }), 200

    except Exception as e:
        app.logger.error(f"Failed to execute report generation task: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.logger.info(">>> Unified AI Worker Service Starting <<<")
    app.logger.info(f">>> Registered real-time AI functions: {list(AI_FUNCTIONS.keys())}")
    if not DJANGO_API_TOKEN:
        app.logger.warning(
            "CRITICAL: DJANGO_API_TOKEN environment variable is not set. Service may not function correctly.")

    app.logger.info(f">>> Django API URL set to: {DJANGO_API_BASE_URL}")
    app.logger.info(f">>> RTMP Stream Source URL set to: {RTMP_SERVER_URL}")
    app.logger.info(">>> Report generation module is also loaded.")
    if 'face_recognition' in AI_FUNCTIONS:
        threading.Thread(target=schedule_face_cache_refresh, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)