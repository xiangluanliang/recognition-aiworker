from ultralytics import YOLO
import cv2

# 加载暴力检测模型
fight_model = YOLO(r'D:\project\recognition-api\Yolo_nano_weights.pt')  # 替换为你的模型路径

# 打架检测函数
def detect_fight_box(frame):
    results = fight_model.predict(source=[frame], imgsz=640, verbose=False)
    for r in results:
        for cls, box in zip(r.boxes.cls, r.boxes.xyxy):
            if int(cls) == 1:  # 类别 1 表示暴力行为
                x1, y1, x2, y2 = map(int, box)
                return True, (x1, y1, x2, y2)
    return False, None

# 视频输入路径
video_path = 'D:\\Desktop\\conflict_2.mp4'  # 替换为你的输入视频
output_path = 'output_fight.mp4'

# 打开视频
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ 无法打开视频")
    exit()

# 获取视频参数
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

# 创建视频写入器
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# 遍历帧进行检测
while True:
    ret, frame = cap.read()
    if not ret:
        break

    is_fight, bbox = detect_fight_box(frame)
    if is_fight:
        cv2.rectangle(frame, bbox[:2], bbox[2:], (0, 165, 255), 2)
        cv2.putText(frame, "Fighting detected", (bbox[0], bbox[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

    out.write(frame)  # 写入输出视频
    cv2.imshow('Fight Detection', frame)
    if cv2.waitKey(1) == 27:  # ESC键退出
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ 视频处理完成，保存为：", output_path)
