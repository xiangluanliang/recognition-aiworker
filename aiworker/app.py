# app.py
import base64
import json

import cv2
import numpy as np
import threading
import logging
import subprocess
import tempfile
import time

import requests
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from flask_sock import Sock

# --- 导入重构后的模块 ---
from aiworker.config import *
from aiworker.services.api_client import fetch_known_faces, log_event
from aiworker.face.vision_service import VisionServiceWorker
from aiworker.face.face_handler import process_frame_for_api
from aiworker.yolo.behavior_processor import AbnormalBehaviorProcessor
from aiworker.yolo.yolo_detector import YoloDetector


# 导入音频模块
# from aiworker.audio.event_handlers import handle_audio_file
# from aiworker.report.report_generator import process_report_generation

# --- Flask App 初始化 ---
app = Flask(__name__)
CORS(app)
sock = Sock(app)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - (AI-Worker) - %(message)s')

# --- 全局实例和缓存 ---
# 全局AI模型实例 (重量级对象，只加载一次)
vision_worker = VisionServiceWorker()
yolo_detector = YoloDetector(YOLO_POSE_MODEL_FILENAME)

# 全局缓存
video_streams_cache = {}
AI_FUNCTIONS = ['abnormal_detection']

# def audio_detect_thread(rtmp_url_inner, camera_id_inner, processor):
#     while True:
#         audio_path = None  # 初始化
#         try:
#             with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
#                 audio_path = tmp_audio.name
#
#             command = [
#                 "ffmpeg", "-y", "-i", rtmp_url_inner,
#                 "-t", "5",
#                 "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
#                 audio_path
#             ]
#             subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#
#             # 用 handle_audio_file 时，传入 processor 实例 ---
#             handle_audio_file(audio_path, processor)
#
#         except Exception as e:
#             app.logger.error(f"[AudioThread-{camera_id_inner}] 音频处理失败: {e}")
#
#         finally:
#             if audio_path and os.path.exists(audio_path):
#                 os.remove(audio_path)
#
#         time.sleep(10)


def capture_and_process_thread(stream_id: str, ai_function_name: str, camera_id: str):
    cache_key = (stream_id, ai_function_name, camera_id)
    stream_lock = video_streams_cache[cache_key]['lock']
    camera_id_int = int(camera_id)

    rtmp_url = f'{RTMP_SERVER_URL}{stream_id}'
    cap = cv2.VideoCapture(rtmp_url)

    processor_instance = None
    if ai_function_name == 'abnormal_detection':
        fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
        processor_instance = AbnormalBehaviorProcessor(camera_id_int, yolo_detector, fps)

    if not cap.isOpened():
        app.logger.error(f"Cannot open stream: {rtmp_url}")
        return 
        # 即使视频流打开失败，我们依然可以尝试启动音频检测
        # if processor_instance:
        #     audio_thread = threading.Thread(
        #         target=audio_detect_thread,
        #         args=(rtmp_url, camera_id, processor_instance),
        #         daemon=True
        #     )
        #     audio_thread.start()
        # if not cap.isOpened():
        #     return

    app.logger.info(f"Thread started for stream '{stream_id}' with AI '{ai_function_name}' for camera '{camera_id}'")

    frame_count = 0
    while cache_key in video_streams_cache:
        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            cap.release()
            cap = cv2.VideoCapture(rtmp_url)
            app.logger.warning(f"Stream {stream_id} disconnected. Attempting to reconnect...")
            continue

        frame_count += 1

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        if frame_count % FRAME_SKIP_RATE != 0:
            continue

        # --- 帧处理逻辑 ---
        processed_frame = frame
        if ai_function_name == 'abnormal_detection' and processor_instance:
            if hasattr(processor_instance, 'video_buffer'):
                processor_instance.video_buffer.append(frame.copy())
            processed_frame, _ = processor_instance.process_frame(frame)

        ret, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ret:
            with stream_lock:
                video_streams_cache[cache_key]['frame_bytes'] = buffer.tobytes()

    cap.release()
    video_streams_cache.pop(cache_key, None)
    app.logger.info(f"Thread stopped for {cache_key}")
@app.route('/<ai_function_name>/<stream_id>/<camera_id>')
def video_feed(ai_function_name: str, stream_id: str, camera_id: str):
    if ai_function_name not in AI_FUNCTIONS:
        return Response(f"Error: AI stream function '{ai_function_name}' not found.", status=404)

    cache_key = (stream_id, ai_function_name, camera_id)
    if cache_key not in video_streams_cache:
        app.logger.info(f"Cache miss for {cache_key}. Creating new processing thread.")
        video_streams_cache[cache_key] = {'lock': threading.Lock(), 'frame_bytes': None}
        thread = threading.Thread(
            target=capture_and_process_thread,
            args=(stream_id, ai_function_name, camera_id),
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
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        time.sleep(1 / 25)  # 控制推流帧率


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

    known_faces_data = fetch_known_faces()
    response_json, _ = process_frame_for_api(vision_worker, frame, known_faces_data)

    if not response_json.get('liveness_passed', True):
        log_event({'event_type': 'LIVENESS_FRAUD_API', 'details': response_json})

    return jsonify(response_json)


@sock.route('/ws/liveness_check')
def liveness_check_websocket(ws):
    """
    处理活体检测的WebSocket连接。
    加入了跳帧逻辑以优化性能。
    """
    app.logger.info("WebSocket client connected for liveness check.")

    session_vision_worker = VisionServiceWorker()
    known_faces_data = fetch_known_faces()

    frame_counter = 0
    frame_skip_rate = 3

    try:
        while True:
            image_data_base64 = ws.receive(timeout=10)
            if image_data_base64 is None: break
            frame_counter += 1

            try:
                img_bytes = base64.b64decode(image_data_base64.split(',', 1)[-1])
                frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                if frame is None: continue
            except Exception:
                continue

            if frame_counter % frame_skip_rate == 0:
                response_json, _ = process_frame_for_api(
                    session_vision_worker, frame, known_faces_data
                )

                is_final_result = False
                final_status = 'processing'

                if frame_counter > BLINK_TIMEOUT_FRAMES:
                    is_final_result = True
                    final_status = 'timeout'
                    response_json['message'] = '活体检测超时'
                    response_json['liveness_passed'] = False
                else:
                    persons = response_json.get('persons', [])
                    if persons:  # 只有在检测到人脸时才进行后续判断
                        first_person = persons[0]
                        liveness_info = first_person.get('liveness_info', {})

                        # 1. 快速失败：OULU分数过低
                        if liveness_info.get('oulu_score', 1.0) < OULU_LIVENESS_HARD_THRESHOLD:
                            is_final_result = True
                            final_status = 'fraud_detected'
                            response_json['message'] = '检测到欺诈攻击'
                            response_json['liveness_passed'] = False

                        # 2. 最终成功：眨眼完成
                        elif liveness_info.get('blink_status') == 'BLINK_COMPLETED':
                            is_final_result = True
                            final_status = 'success'
                            response_json['message'] = '活体检测通过'
                            response_json['liveness_passed'] = True

                if is_final_result:
                    app.logger.info(f"Final result determined: {final_status}")
                    response_json['status'] = 'final'
                    ws.send(json.dumps(response_json))
                    break
                else:
                    response_json['status'] = 'processing'
                    ws.send(json.dumps(response_json))

    except Exception as e:
        app.logger.error(f"Error in liveness WebSocket: {e}", exc_info=True)
    finally:
        app.logger.info("WebSocket client disconnected.")


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
        summary_data = fetch_summary_for_report()

        report_content = "模块升级中"
        # report_content = process_report_generation(summary_data)
        return jsonify({
            "status": "success",
            "message": "Daily report generated and submitted successfully.",
            "content": report_content
        }), 200
    except Exception as e:
        app.logger.error(f"Failed to execute report generation task: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# --- 服务启动 ---
if __name__ == '__main__':
    app.logger.info(">>> Unified AI Worker Service Starting (Refactored) <<<")
    app.run(host='0.0.0.0', port=5000, threaded=True)
