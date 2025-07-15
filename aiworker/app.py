# app.py
import base64
import cv2
import numpy as np
import threading
import logging
import time
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

# --- 导入重构后的模块 ---
from aiworker.config import *
from aiworker.services.api_client import fetch_known_faces, log_event
from aiworker.face.vision_service import VisionServiceWorker
from aiworker.face.face_handler import process_frame_for_api, process_frame_for_stream as process_face_stream
from aiworker.yolo.behavior_processor import AbnormalBehaviorProcessor
from aiworker.yolo.yolo_detector import YoloDetector

# from aiworker.report.report_generator import process_report_generation

# --- Flask App 初始化 ---
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - (AI-Worker) - %(message)s')

# --- 全局实例和缓存 ---
# 全局AI模型实例 (重量级对象，只加载一次)
vision_worker = VisionServiceWorker()
yolo_detector = YoloDetector()

# 全局缓存
video_streams_cache = {}
known_faces_cache = []
AI_FUNCTIONS = ['face_recognition', 'abnormal_detection']

# --- 后台任务：定时刷新已知人脸缓存 ---
def schedule_face_cache_refresh():
    global known_faces_cache
    known_faces_cache = fetch_known_faces()
    threading.Timer(CACHE_REFRESH_INTERVAL, schedule_face_cache_refresh).start()


def capture_and_process_thread(stream_id: str, ai_function_name: str, camera_id: str):
    cache_key = (stream_id, ai_function_name, camera_id)
    stream_lock = video_streams_cache[cache_key]['lock']
    camera_id_int = int(camera_id)

    rtmp_url = f'{RTMP_SERVER_URL}{stream_id}'
    cap = cv2.VideoCapture(rtmp_url)
    if not cap.isOpened():
        app.logger.error(f"Cannot open stream: {rtmp_url}")
        return

    app.logger.info(f"Thread started for stream '{stream_id}' with AI '{ai_function_name}' for camera '{camera_id}'")

    # --- 为需要状态的AI功能初始化对应的处理器实例 ---
    processor_instance = None
    if ai_function_name == 'abnormal_detection':
        fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
        processor_instance = AbnormalBehaviorProcessor(camera_id_int, yolo_detector, fps)

    frame_count = 0
    while cache_key in video_streams_cache:
        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # --- 跳帧逻辑 ---
        if frame_count % FRAME_SKIP_RATE != 0:
            frame_count += 1
            continue  # 跳过此帧的AI处理

        processed_frame = frame
        if ai_function_name == 'face_recognition':
            # 人脸识别的跳帧逻辑在其内部处理，以保证UI流畅
            processed_frame, _ = process_face_stream(
                vision_worker, frame, known_faces_cache, camera_id_int, perform_heavy_ai=True
            )
        elif ai_function_name == 'abnormal_detection' and processor_instance:
            processed_frame, _ = processor_instance.process_frame(frame)

        # 将处理后的帧放入缓存
        ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ret:
            with stream_lock:
                video_streams_cache[cache_key]['frame_bytes'] = buffer.tobytes()

        frame_count += 1

    cap.release()
    video_streams_cache.pop(cache_key, None)
    app.logger.info(f"Thread stopped for {cache_key}")

@app.route('/<ai_function_name>/<stream_id>/<camera_id>')
def video_feed(ai_function_name: str, stream_id: str, camera_id: str):
    if ai_function_name not in AI_FUNCTIONS:
        return Response(f"Error: AI function '{ai_function_name}' not found.", status=404)

    cache_key = (stream_id, ai_function_name, camera_id)
    if cache_key not in video_streams_cache:
        app.logger.info(f"Cache miss for {cache_key}. Creating new processing thread.")
        video_streams_cache[cache_key] = {'lock': threading.Lock(), 'frame_bytes': None}
        thread = threading.Thread(
            target=capture_and_process_thread,
            args=(stream_id, ai_function_name, camera_id, video_streams_cache, vision_worker, known_faces_cache),
            daemon=True
        )
        thread.start()

    return Response(
        stream_generator(cache_key),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def stream_generator(cache_key):
    """从缓存中读取帧并推送给前端。"""
    app.logger.info(f"Client connected to stream generator for {cache_key}")
    while cache_key in video_streams_cache:
        with video_streams_cache[cache_key]['lock']:
            frame_bytes = video_streams_cache[cache_key].get('frame_bytes')
        if frame_bytes:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(1 / 25)  # 控制推流帧率


# --- 单帧识别API端点 ---
@app.route('/ai/recognize-frame', methods=['POST'])
def recognize_frame_api():
    image_base64 = request.json.get('image_data')
    if not image_base64:
        return jsonify({'status': 'error', 'message': '未提供 image_data'}), 400

    try:
        img_bytes = base64.b64decode(image_base64.split(',', 1)[-1])
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None: raise ValueError("图像解码失败")
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'无效的图像数据: {e}'}), 400

    # 调用独立的API处理函数
    response_json, _ = process_frame_for_api(vision_worker, frame, known_faces_cache)

    # 记录事件 (API调用通常不上报普通识别事件，只上报欺诈或危险人员)
    if response_json.get('persons'):
        for person in response_json['persons']:
            if person.get('identity') == 'SPOOF' or person.get('person_state') == 2:
                log_event({'event_type': 'DANGER_OR_SPOOF_API', 'details': person})

    return jsonify(response_json)


# --- 服务启动 ---
if __name__ == '__main__':
    app.logger.info(">>> Unified AI Worker Service Starting (Refactored) <<<")
    threading.Thread(target=schedule_face_cache_refresh, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
