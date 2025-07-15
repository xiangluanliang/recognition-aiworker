# local_ai_service.py
import base64
import os
import time
import cv2
import datetime

import numpy as np
import requests
import threading
import logging
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from aiworker.yolo.yolo_detector import process_single_frame as person_detector
from aiworker.face.face_recognizer2 import process_frame_face_recognition
# from aiworker.utils.report_generator import process_report_generation

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - (AI-Worker) - %(message)s')
video_streams_cache = {}
AI_FUNCTIONS = {
    'person_detection': person_detector,
    'face_recognition': process_frame_face_recognition,
    # 未来有新功能，在这里继续添加...
}

DJANGO_API_TOKEN = os.environ.get('DJANGO_API_TOKEN', '3d814802906b91d7947518f5d0191a42795cace7')
DJANGO_API_BASE_URL = os.environ.get('DJANGO_API_URL', 'https://8.152.101.217/api/test/api/')
RTMP_SERVER_URL = os.environ.get('RTMP_SERVER_URL', 'rtmp://localhost:9090/live/')
KNOWN_FACES_CACHE = []
CACHE_REFRESH_INTERVAL = 300


def fetch_and_cache_known_faces():
    global KNOWN_FACES_CACHE
    try:
        url = os.path.join(DJANGO_API_BASE_URL, 'known-faces/')
        response = requests.get(url, timeout=10, headers={"Authorization": f"Token {DJANGO_API_TOKEN}"},verify=False)
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
            requests.post(url, json=event_data, timeout=5, headers={"Authorization": f"Token {DJANGO_API_TOKEN}"},verify=False)
        except Exception as e:
            app.logger.error(f"Failed to log event to Django: {e}")

    threading.Thread(target=_send_request).start()


def capture_and_process_thread(stream_id: str, ai_function_name: str, camera_id: str):
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

        if process_function and frame_count % frame_skip == 0:
            if ai_function_name == 'face_recognition':
                processed_frame, detection_data = process_function(
                    frame.copy(),
                    known_faces_data=KNOWN_FACES_CACHE,
                    camera_id=camera_id_int
                )
            else:
                processed_frame, detection_data = process_function(
                    frame.copy(),
                    camera_id=camera_id_int
                )

            if detection_data and 'events_to_log' in detection_data:
                for event in detection_data['events_to_log']:
                    event['camera'] = camera_id
                    event['time'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    log_event_to_django(event)

        ret, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ret:
            with stream_lock:
                video_streams_cache[cache_key]['frame_bytes'] = buffer.tobytes()

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
        # 解码Base64图片
        # 前端发送的 toDataURL() 结果会包含一个头部 "data:image/jpeg;base64,"，需要去掉
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
            app.logger.info("图像解码成功")

    except Exception as e:
        app.logger.error(f"Base64 image decoding failed: {e}")
        return jsonify({'status': 'error', 'message': '无效的图像数据'}), 400
    app.logger.info("已知人脸信息")
    app.logger.info(KNOWN_FACES_CACHE)
    # --- 调用我们标准的人脸识别功能 ---
    # 注意：这里我们硬编码了使用 'face_recognition' 功能
    # 第二个参数是缓存的已知人脸数据，由后台定时任务更新
    processed_frame, detection_data = process_frame_face_recognition(
        frame,
        known_faces_data=KNOWN_FACES_CACHE,
        camera_id=0  # 身份认证场景通常用0或一个虚拟ID
    )

    # 我们只需要将结构化的检测数据返回给前端，不需要返回处理后的图片
    return jsonify(detection_data)


def fetch_summary_for_report():
    """从Django后端获取日报所需的数据摘要。"""
    # 这个函数现在只负责调用API
    api_url = f"{DJANGO_API_BASE_URL}daily-report/"
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
        # 1. 从Django获取数据
        # summary_data = fetch_summary_for_report()

        # 2. 【核心修改】调用外部模块的函数进行AI处理
        # report_content = process_report_generation(summary_data)


        return jsonify({
            "status": "success",
            "message": "Daily report generated and submitted successfully.",
            "content": "功能正在维护中" #report_content
        }), 200

    except Exception as e:
        app.logger.error(f"Failed to execute report generation task: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


app.logger.info(">>> Unified AI Worker Service Starting <<<")
app.logger.info(f">>> Registered real-time AI functions: {list(AI_FUNCTIONS.keys())}")
if not DJANGO_API_TOKEN:
    app.logger.warning(
        "CRITICAL: DJANGO_API_TOKEN environment variable is not set. Service may not function correctly.")

app.logger.info(f">>> Django API URL set to: {DJANGO_API_BASE_URL}")
app.logger.info(f">>> RTMP Stream Source URL set to: {RTMP_SERVER_URL}")
app.logger.info(">>> Report generation module is also loaded.")

# 启动时先执行一次，然后开启定时刷新
if 'face_recognition' in AI_FUNCTIONS:
    app.logger.info(">>> 正在初始化已知人脸数据获取任务...")
    # 直接调用一次函数以暴露启动错误
    fetch_and_cache_known_faces()
    # 再启动后台定时刷新任务
    threading.Thread(target=schedule_face_cache_refresh, daemon=True).start()

