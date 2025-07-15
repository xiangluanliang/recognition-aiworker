# aiworker/services/api_client.py
import requests
import logging
from aiworker.config import DJANGO_API_BASE_URL, DJANGO_API_TOKEN

logger = logging.getLogger(__name__)

def fetch_known_faces():
    """从Django后端获取已知人脸数据。"""
    try:
        url = f"{DJANGO_API_BASE_URL}known-faces/"
        headers = {"Authorization": f"Token {DJANGO_API_TOKEN}"}
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        logger.info(f"Successfully fetched {len(response.json())} known faces.")
        return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch known faces: {e}")
        return []

def log_event(event_data: dict):
    """向Django后端上报一个事件。"""
    try:
        url = f"{DJANGO_API_BASE_URL}log-event/"
        headers = {"Authorization": f"Token {DJANGO_API_TOKEN}"}
        requests.post(url, json=event_data, timeout=5, headers=headers)
    except Exception as e:
        logger.error(f"Failed to log event to Django: {e}")

def fetch_warning_zones(camera_id: int) -> dict:
    """从Django后端获取指定摄像头的警戒区配置。"""
    default_zones = {'zones': [], 'safe_time': 5, 'safe_distance': 50.0}
    try:
        # 注意：URL应该从config.py中获取
        url = f"{DJANGO_API_BASE_URL}warning-zones/by-camera/{camera_id}/"
        headers = {"Authorization": f"Token {DJANGO_API_TOKEN}"}
        response = requests.get(url, headers=headers, verify=False, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"获取摄像头 {camera_id} 的警戒区失败: {e}")
        return default_zones
