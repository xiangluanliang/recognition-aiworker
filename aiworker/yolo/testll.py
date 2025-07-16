# aiworker/yolo/testll.py
import cv2
import time

from aiworker.yolo.yolo_detector import YoloDetector
from aiworker.yolo.behavior_processor import AbnormalBehaviorProcessor
from aiworker.config import YOLO_POSE_MODEL_FILENAME, YOLO_FIGHT_MODEL_FILENAME, MODEL_DIR


def main():
    # 相机 ID 和视频流地址
    camera_id = 1
    video_path = "rtmp://127.0.0.1/live/test"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("视频流读取失败！")
        return
    print("拉流成功，开始处理...")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25  # 默认值 25
    fps = int(fps)

    # 初始化两个 YOLO 模型
    pose_detector = YoloDetector(weights_filename=YOLO_POSE_MODEL_FILENAME)
    fight_detector = YoloDetector(weights_filename=YOLO_FIGHT_MODEL_FILENAME)

    # 初始化行为处理器
    processor = AbnormalBehaviorProcessor(camera_id, pose_detector, fight_detector, fps)

    frame_skip = 6  # 跳过4帧，处理第5帧
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频读取结束或失败")
            break

        if frame_idx % (frame_skip + 1) == 0:
            processed_frame, _ = processor.process_frame(frame)
        else:
            processed_frame = frame  # 跳帧时直接显示原帧

        cv2.imshow('行为检测', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
