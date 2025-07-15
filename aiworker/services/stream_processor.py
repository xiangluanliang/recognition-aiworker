# aiworker/services/stream_processor.py
import cv2
import time
import datetime
import logging
import threading

from aiworker.config import RTMP_SERVER_URL, FRAME_WIDTH, FRAME_HEIGHT, FRAME_SKIP_RATE, JPEG_QUALITY
from aiworker.services.api_client import log_event
from aiworker.face.face_handler import process_frame_for_stream

# from aiworker.yolo.yolo_handler import process_frame_for_yolo # 示例

logger = logging.getLogger(__name__)


def capture_and_process_thread(stream_id: str, ai_function_name: str, camera_id: str, video_streams_cache: dict,
                               vision_worker, known_faces_data):
    """后台线程，负责拉流、处理和缓存帧。"""
    cache_key = (stream_id, ai_function_name, camera_id)
    stream_lock = video_streams_cache[cache_key]['lock']
    camera_id_int = int(camera_id)

    rtmp_url = f'{RTMP_SERVER_URL}{stream_id}'
    cap = cv2.VideoCapture(rtmp_url)
    if not cap.isOpened():
        logger.error(f"Cannot open stream: {rtmp_url}")
        return

    logger.info(f"Thread started for stream '{stream_id}' with AI '{ai_function_name}' for camera '{camera_id}'")

    frame_count = 0
    while cache_key in video_streams_cache:
        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        perform_heavy_ai = (frame_count % FRAME_SKIP_RATE == 0)

        processed_frame, detection_data = frame, None

        if ai_function_name == 'face_recognition':
            processed_frame, detection_data = process_frame_for_stream(
                vision_worker, frame, known_faces_data, camera_id_int, perform_heavy_ai
            )
        # elif ai_function_name == 'person_detection':
        #     processed_frame, detection_data = process_frame_for_yolo(...)

        if detection_data and detection_data.get('events_to_log'):
            for event in detection_data['events_to_log']:
                event['camera'] = camera_id_int
                event['time'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                log_event(event)

        ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ret:
            with stream_lock:
                video_streams_cache[cache_key]['frame_bytes'] = buffer.tobytes()

        frame_count += 1

    cap.release()
    video_streams_cache.pop(cache_key, None)
    logger.info(f"Thread stopped for {cache_key}")
