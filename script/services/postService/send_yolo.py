# api/services/postService/send_yolo.py

from setup_django import *

import requests
import os
from django.core.files.base import ContentFile
from django.db import connection
from api.models import VideoAnalysisTask


# 这是一个辅助函数，专门用于从AI Worker下载文件
def download_file_from_worker(download_url: str) -> ContentFile | None:
    """从本地AI Worker下载文件，并将其作为Django的ContentFile对象返回"""
    try:
        print(f"准备从AI Worker下载结果视频: {download_url}")
        # 注意：这里的请求也是通过隧道发出的
        response = requests.get(download_url, timeout=300)
        response.raise_for_status()
        print("结果视频下载成功。")
        # ContentFile允许我们从内存中的字节数据创建一个Django可用的文件对象
        return ContentFile(response.content)
    except requests.exceptions.RequestException as e:
        print(f"从AI Worker下载处理结果视频失败: {e}")
        return None

def my_yolo(task_id: int, public_video_url: str):
    """
    完整的后台处理任务：派发任务给AI Worker，获取结果，下载结果视频，并更新数据库。
    """
    print(f"后台任务启动：处理任务ID {task_id}")
    connection.close()

    AI_WORKER_URL = os.getenv('AI_WORKER_BASE_URL')
    if not AI_WORKER_URL:
        print(f"任务 {task_id} 失败: 未在.env文件中配置AI_WORKER_BASE_URL")
        return  # 提前退出
    try:
        task = VideoAnalysisTask.objects.get(id=task_id)
        task.status = 1  # 1. 更新状态为“正在处理”
        task.save()

        # 2. 发送处理指令到AI Worker
        tunnel_endpoint = f"{AI_WORKER_URL}/process-video"
        response = requests.post(tunnel_endpoint, json={"video_url": public_video_url}, timeout=300)
        response.raise_for_status()
        ai_results = response.json()
        print(f"从AI Worker收到响应: {ai_results}")

        if ai_results.get('status') == 'success':
            # 3. 如果成功，获取结果视频的下载路径
            processed_video_url_path = ai_results.get('processed_video_download_url')
            if processed_video_url_path:
                # 构造完整的下载URL
                full_download_url = f"{AI_WORKER_URL}{processed_video_url_path}"

                # 下载处理后的视频内容
                processed_video_content = download_file_from_worker(full_download_url)

                if processed_video_content:
                    # 4. 【关键】将下载的视频内容保存到模型的FileField中
                    original_filename = os.path.basename(public_video_url)
                    # Django的FileField.save()方法会处理文件的物理保存
                    task.processed_video.save(f"processed_{original_filename}", processed_video_content, save=False)

            # 5. 更新任务记录的其他字段
            task.status = 2  # 2代表“处理成功”
            task.analysis_result = ai_results.get('detection_data')
            task.save()  # 一次性保存所有更改
            print(f"任务 {task_id} 已成功完成并保存所有数据。")
        else:
            raise Exception(ai_results.get('message', '未知的AI处理错误'))

    except Exception as e:
        print(f"任务 {task_id} 处理失败: {e}")
        connection.close()
        task = VideoAnalysisTask.objects.get(id=task_id)
        task.status = -1
        task.analysis_result = {'error': str(e)}
        task.save()
# 假设您已经定义了异常区域坐标和安全距离
# abnormal_zone_coords = [300, 300, 200, 200]
# safe_distance = 50  # 示例安全距离
# video_path = "D:\\Desktop\\abnormal_5.mp4"
# response = requests.post("http://localhost:5000/detect-abnormal", json={"video_path": video_path})


payload = {
    "video_path": "D:\\Desktop\\abnormal_4.mp4",
    "camera_id": 1,
    "stay_seconds": 0.1,
    "safe_distance": 100
}
response = requests.post("http://localhost:5000/detect-abnormal", json=payload)

if response.ok:
    data = response.json()
    print("✅ 检测完成")
    print("检测信息：", data["message"])
    print("处理后视频地址：", data["video_url"])
else:
    print("请求失败:", response.text)

