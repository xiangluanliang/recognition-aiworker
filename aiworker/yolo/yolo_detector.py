# yolo_detector.py

from ultralytics import YOLO
import cv2
import subprocess
import shutil
import os

# 模型应该在脚本加载时初始化一次，避免重复加载
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "dnn_models", "yolov8n.pt")

def process_single_frame(frame):
    """
    【新增的核心函数 - 用于实时流处理】
    接收一个从视频流中捕获的独立图像帧（NumPy数组），进行YOLOv8检测，
    然后返回两个结果：
    1. processed_frame: 已经绘制了检测框和标签的图像帧。
    2. detection_data: 一个包含该帧内所有检测信息的结构化字典。
    """
    # 1. 在当前帧上运行YOLOv8模型
    results = model(frame)

    # 2. 准备一个帧的副本用于绘制，以避免修改原始帧
    processed_frame = frame.copy()

    # 3. 初始化一个字典来存储当前帧的检测数据
    detection_data = {'person_count': 0, 'detections': []}

    # 4. 遍历检测结果
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # 我们只关心标签为'person'的检测结果
            if label == 'person':
                # 将人员计数加一
                detection_data['person_count'] += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # 将详细信息存入字典
                detection_data['detections'].append({
                    'label': label,
                    'confidence': round(conf, 2),
                    'box': [x1, y1, x2, y2]
                })

                # 5. 在图像副本上绘制边界框和标签
                # 绘制绿色矩形框
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 在框上方添加文字标签
                cv2.putText(processed_frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 6. 返回处理完成的帧和结构化的检测数据
    return processed_frame, detection_data



def fix_video_for_streaming(video_path):
    """
    【新增的辅助函数】
    使用ffmpeg将视频重新编码为H.264格式，并将moov atom移动到文件开头，
    使其适合网络流媒体播放。
    """
    print(f"开始优化视频文件以便于网络播放: {video_path}")
    # 创建一个临时文件名，用于存放优化后的视频
    temp_path = video_path + ".temp_fixed.mp4"

    # 构建ffmpeg命令
    command = [
        'ffmpeg', '-y',  # -y: 如果输出文件已存在则直接覆盖
        '-i', video_path,  # -i: 指定输入文件
        '-c:v', 'libx264',  # -c:v: 指定视频编码器为H.264
        '-pix_fmt', 'yuv420p',  # -pix_fmt: 指定像素格式以保证最大兼容性
        '-preset', 'veryfast',  # -preset: 编码速度与质量的平衡，veryfast速度很快
        '-c:a', 'aac',  # -c:a: 指定音频编码器为AAC
        '-movflags', '+faststart',  # -movflags +faststart: 将索引移动到文件开头的关键参数
        temp_path
    ]

    try:
        # 执行命令，使用PIPE隐藏ffmpeg在控制台的大量输出
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 用优化后的文件覆盖原始的输出文件
        shutil.move(temp_path, video_path)
        print(f"成功优化视频文件: {video_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg 优化失败: {e.stderr.decode()}")
        # 如果优化失败，确保删除可能已创建的临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False
    except FileNotFoundError:
        print("错误: ffmpeg 命令未找到。请确保 ffmpeg 已安装并处于系统的PATH中。")
        return False
def detect_humans_with_yolov8(video_path, output_video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # 返回一个包含错误信息的字典
        return {'success': False, 'message': '无法打开视频文件', 'detections': []}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 确保输出视频的编码器是广泛支持的 'mp4v'
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    all_detections = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        frame_detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                # 我们只处理'person'
                if label == 'person':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    # 记录检测结果
                    frame_detections.append({
                        'label': label,
                        'confidence': round(conf, 2),
                        'box': [x1, y1, x2, y2]
                    })

                    # 在帧上绘制矩形框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 将处理过的帧写入新视频
        out.write(frame)
        if frame_detections:
            all_detections.append({'frame': frame_index, 'detections': frame_detections})

        frame_index += 1

    cap.release()
    out.release()
    print(f"OpenCV 已完成视频写入: {output_video_path}")

    fix_video_for_streaming(output_video_path)

    # 返回一个包含所有信息的字典
    return {
        'success': True,
        'message': f'检测完成，共处理 {frame_index} 帧。',
        'detection_data': all_detections,
        'output_path': output_video_path
    }
