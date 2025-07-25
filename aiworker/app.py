# app.py
import base64
import json
import queue

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

from aiworker.audio.event_handlers import audio_detector, INTERESTING_CLASSES, is_abnormal, trigger_alarm
from aiworker.audio.preprocess import load_audio
# --- 导入重构后的模块 ---
from aiworker.config import *
from aiworker.services.api_client import *
from aiworker.face.vision_service import VisionServiceWorker
from aiworker.face.face_handler import process_frame_for_api
from aiworker.yolo.behavior_processor import AbnormalBehaviorProcessor
from aiworker.yolo.yolo_detector import YoloDetector
# from aiworker.audio.event_handlers import handle_audio_file
from aiworker.utils.report_generator import process_report_generation

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

def audio_detect_thread(rtmp_url, camera_id, processor):
    """ 音频检测线程 """
    logger = app.logger
    logger.info(f"[AudioThread-{camera_id}] 音频检测线程已启动。")
    while True:
        audio_path = None
        try:
            # 创建一个不会自动删除的临时文件
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir='/tmp') as tmp_audio:
                audio_path = tmp_audio.name

            command = [
                "ffmpeg", "-y", "-i", rtmp_url, "-t", "5",
                "-vn", "-acodec", "pcm_s16le", "-ar", "32000", "-ac", "1",
                audio_path
            ]
            logger.info(f"[AudioThread-{camera_id}] 准备执行ffmpeg命令，截取音频到: {audio_path}")
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info(f"[AudioThread-{camera_id}] ffmpeg命令执行成功。")
            processor.latest_audio_clip_path = audio_path

            waveform = load_audio(audio_path, sr=32000)
            results = audio_detector.detect(waveform)

            if results:
                top_result = results[0]
                logger.info(
                    f"[AudioThread-{camera_id}] 音频AI检测最高分结果: {top_result['label']} (分数: {top_result['score']:.2f})")

            for result in results:
                label = result['label']
                score = result['score']
                if label in INTERESTING_CLASSES and is_abnormal(label, score):
                    trigger_alarm(label, score, processor)


        except subprocess.CalledProcessError as e:
            logger.error(f"[AudioThread-{camera_id}] ffmpeg命令执行失败! Stderr: {e.stderr}")
        except Exception as e:
            logger.error(f"[AudioThread-{camera_id}] 音频处理线程发生未知错误: {e}", exc_info=True)

        time.sleep(5)
        
def frame_reader_thread(cap: cv2.VideoCapture, frame_queue: queue.Queue):
    """一个专门负责从VideoCapture对象中读取帧并放入队列的线程。"""
    app.logger.info("读帧线程已启动。")
    while True:
        if not cap.isOpened():
            break

        success, frame = cap.read()
        if not success:
            break

        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            with frame_queue.mutex:
                frame_queue.queue.clear()
            frame_queue.put(frame)

    app.logger.info("读帧线程已退出。")
    frame_queue.put(None)


def capture_and_process_thread(ai_function_name: str, camera_id: str):
    camera_id_int = int(camera_id)
    cache_key = (ai_function_name, camera_id)
    stream_lock = video_streams_cache[cache_key]['lock']

    camera_details = get_camera_details(camera_id_int)

    if not camera_details or not camera_details.get('password'):
        logger.error(f"无法为摄像头 {camera_id} 获取有效配置或推流码，线程退出。")
        video_streams_cache.pop(cache_key, None)
        return

    stream_key = camera_details.get('password')
    active_detectors = camera_details.get('active_detectors', [])
    logger.info(f"成功为摄像头 {camera_id} 获取到配置: {active_detectors}, 推流码: {stream_key}")

    rtmp_url = f'{RTMP_SERVER_URL}{stream_key}'
    cap = cv2.VideoCapture(rtmp_url)
    if not cap.isOpened():
        app.logger.error(f"VideoCapture无法打开流 {rtmp_url}，线程退出。")
        video_streams_cache.pop(cache_key, None)
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
    processor_instance = AbnormalBehaviorProcessor(
        camera_id_int, yolo_detector, fps, enabled_detectors=active_detectors
    )

    # 启动音频检测线程
    audio_thread = threading.Thread(
        target=audio_detect_thread,
        args=(rtmp_url, camera_id_int, processor_instance),
        daemon=True
    )
    audio_thread.start()

    frame_queue = queue.Queue(maxsize=50)
    reader = threading.Thread(target=frame_reader_thread, args=(cap, frame_queue), daemon=True)
    reader.start()
    app.logger.info(f"处理线程已为 {cache_key} 启动，开始进入主循环。")

    frame_count = 0
    while cache_key in video_streams_cache:
        try:
            frame = frame_queue.get(timeout=10)

            if frame is None:
                app.logger.info("收到读帧线程的退出信号，处理线程即将退出。")
                break

            frame_count += 1
            if processor_instance and hasattr(processor_instance, 'video_buffer'):
                processor_instance.video_buffer.append(frame.copy())

            if frame_count % FRAME_SKIP_RATE != 0:
                continue

            resized_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            processed_frame, _ = processor_instance.process_frame(resized_frame)

            ret, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ret:
                with stream_lock:
                    video_streams_cache[cache_key]['frame_bytes'] = buffer.tobytes()

        except queue.Empty:
            app.logger.warning(f"处理线程等待新帧超时（10秒），判定流 {camera_id} 已中断。")

    cap.release()
    video_streams_cache.pop(cache_key, None)
    app.logger.info(f"主处理线程已为 {cache_key} 停止。")


@app.route('/cameras/<int:camera_id>')
def video_feed(camera_id: str):
    ai_function_name = 'abnormal_detection'

    cache_key = (ai_function_name, str(camera_id))

    if cache_key not in video_streams_cache:
        app.logger.info(f"Cache miss for {cache_key}. Creating new processing thread.")
        video_streams_cache[cache_key] = {'lock': threading.Lock(), 'frame_bytes': None}
        thread = threading.Thread(
            target=capture_and_process_thread,
            args=(ai_function_name, str(camera_id)),
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
    session_vision_worker.liveness_detector.reset_blink_state()

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
                    response_json['message'] = '活体检测超时'
                    response_json['liveness_passed'] = False

                # 检查是否眨眼已完成 (这是会话结束的标志之一)
                elif response_json.get('persons') and \
                     response_json['persons'][0].get('liveness_info', {}).get('blink_status') == 'BLINK_COMPLETED':
                    is_final_result = True
                    if response_json.get('liveness_passed'):
                        response_json['message'] = '活体检测通过'
                    else:
                        response_json['message'] = '活体检测失败 (欺诈攻击)'
                
                # 检查是否被OULU模型硬性拒绝
                elif response_json.get('persons') and \
                     response_json['persons'][0].get('liveness_info', {}).get('oulu_score', 1.0) < OULU_LIVENESS_HARD_THRESHOLD:
                    is_final_result = True
                    response_json['message'] = '检测到欺诈攻击'
                    response_json['liveness_passed'] = False

                if is_final_result:
                    app.logger.info(f"Final result determined: {final_status}")
                    response_json['status'] = 'final'
                    ws.send(json.dumps(response_json))
                    break
                else:
                    response_json['status'] = 'processing'
                    response_json['liveness_passed'] = None
                    ws.send(json.dumps(response_json))

    except Exception as e:
        app.logger.error(f"Error in liveness WebSocket: {e}", exc_info=True)
    finally:
        app.logger.info("WebSocket client disconnected.")


@app.route('/ai/generate-report', methods=['POST'])
def generate_report_endpoint():
    """
    API端点，调用独立的AI模块来生成日报。
    """
    app.logger.info("Received request to generate daily report.")
    try:
        # 1. 从Django获取数据
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


# --- 服务启动 ---
if __name__ == '__main__':
    app.logger.info(">>> Unified AI Worker Service Starting (Refactored) <<<")
    app.run(host='0.0.0.0', port=5000, threaded=True)
