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
        response = requests.get(url, timeout=10, headers=headers, verify=False)
        response.raise_for_status()
        logger.info(f"Successfully fetched {len(response.json())} known faces.")
        return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch known faces: {e}")
        return []

def log_event(event_data: dict):
    url = f"{DJANGO_API_BASE_URL}log-event/"
    headers = {"Authorization": f"Token {DJANGO_API_TOKEN}"}
    try:
        response = requests.post(url, json=event_data, headers=headers, timeout=5, verify=False)
        if response.status_code >= 400:
            logger.error(f"上报事件失败，状态码: {response.status_code}, 响应: {response.text}")
        else:
            logger.info(f"成功上报事件: {event_data.get('event_type')}")
    except requests.exceptions.RequestException as e:
        logger.error(f"获取摄像头 {camera_id} 的警戒区失败: {e}")
        return default_zones


def fetch_safety_config(camera_id: int) -> dict:
    """
    从 Django 后端获取某个摄像头的安全配置（safe_distance 和 safe_time）。
    返回格式: {'safe_distance': float, 'safe_time': int}，失败时返回默认值。
    """
    default_config = {'safe_distance': 50.0, 'safe_time': 5}
    try:
        url = f"{DJANGO_API_BASE_URL}safety-config/{camera_id}/"
        headers = {"Authorization": f"Token {DJANGO_API_TOKEN}"}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()

        # 后端返回的是一个列表，我们默认取第一个配置
        if isinstance(data, list) and len(data) > 0:
            return {
                'safe_distance': float(data[0].get('safe_distance', 50.0)),
                'safe_time': int(data[0].get('safe_time', 5))
            }

    except requests.exceptions.RequestException as e:
        logger.error(f"获取摄像头 {camera_id} 的安全配置失败: {e}")

    return default_config

