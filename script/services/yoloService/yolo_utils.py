# 调用yolo检测人物
from ultralytics import YOLO
from .utils_pose import get_center_point
pose_model = YOLO("yolov8l-pose.pt")

def detect_people(frame):
    results = pose_model(frame)
    kpts_list, centers, confidences = [], [], []

    for r in results:
        if r.keypoints is None or r.boxes is None:
            continue

        keypoints_xy = r.keypoints.xy.cpu().numpy()  # [num_people, 17, 2]
        confs = r.boxes.conf.cpu().numpy()           # [num_people]

        for i in range(len(keypoints_xy)):
            pts = keypoints_xy[i]
            conf = float(confs[i])
            kpts_list.append(pts)
            centers.append(get_center_point(pts))
            confidences.append(conf)

    return kpts_list, centers, confidences
