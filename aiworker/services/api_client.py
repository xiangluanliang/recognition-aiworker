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
    """向Django后端上报一个事件。"""
    try:
        url = f"{DJANGO_API_BASE_URL}log-event/"
        headers = {"Authorization": f"Token {DJANGO_API_TOKEN}"}
        requests.post(url, json=event_data, timeout=5, headers=headers, verify=False)
    except Exception as e:
        logger.error(f"Failed to log event to Django: {e}")


def fetch_warning_zones_for_camera(camera_id: int) -> list:
    """
    根据摄像头ID，从Django API获取其所有关联的警戒区域。
    """
    if not DJANGO_API_TOKEN:
        logger.error("Django API Token未配置，无法获取警戒区域。")
        return []

    url = f"{DJANGO_API_BASE_URL}warning_zones/by-camera/{camera_id}/"
    headers = {"Authorization": f"Token {DJANGO_API_TOKEN}"}

    try:
        response = requests.get(url, headers=headers, timeout=5, verify=False)
        response.raise_for_status()
        zones = response.json()
        logger.info(f"成功为摄像头 {camera_id} 获取到 {len(zones)} 个警戒区域。")
        return zones

    except requests.exceptions.RequestException as e:
        logger.error(f"请求摄像头 {camera_id} 的警戒区域时失败: {e}")
        return []


def get_camera_details(camera_id: int) -> dict | None:
    """
    根据摄像头ID从Django API获取其详细信息，包括启用的AI功能。
    """
    if not DJANGO_API_TOKEN:
        logger.error("Django API Token未配置，无法获取摄像头详情。")
        return None

    url = f"{DJANGO_API_BASE_URL}cameras/{camera_id}/"
    headers = {
        "Authorization": f"Token {DJANGO_API_TOKEN}"
    }

    try:
        response = requests.get(url, headers=headers, timeout=5, verify=False)
        response.raise_for_status()

        camera_data = response.json()
        logger.info(f"成功从API获取到摄像头 {camera_id} 的详情。")
        return camera_data

    except requests.exceptions.RequestException as e:
        logger.error(f"请求摄像头 {camera_id} 详情时发生网络错误: {e}")
        return None
