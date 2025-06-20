import cv2
import time

# 載入 Haar Cascade 模型
cascade_path = "opencv_project/cascade.xml"
fire_extinguisher_cascade = cv2.CascadeClassifier(cascade_path)

def resize_to_max(img, max_width=960, max_height=720):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1)
    return cv2.resize(img, (int(w * scale), int(h * scale)))

# 打開攝影機（0 是內建鏡頭，如果 iVCam 是第二台鏡頭請試試 1、2）
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ 無法開啟攝影機")
    exit()

# 取得畫面尺寸資訊
ret, frame = cap.read()
if not ret:
    print("❌ 無法讀取攝影機畫面")
    exit()

frame = resize_to_max(frame)
frame_height, frame_width = frame.shape[:2]

# 初始化影片寫入器（儲存影片）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output2.mp4', fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 讀取攝影機畫面失敗")
        break

    frame = resize_to_max(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    fire_extinguishers = fire_extinguisher_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=60,
        minSize=(80, 160)
    )

    for (x, y, w, h) in fire_extinguishers:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    out.write(frame)  # 每一幀寫進影片
    cv2.imshow("滅火器即時偵測", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 偵測結束，影片已儲存為 output.mp4")
        break

cap.release()
out.release()
cv2.destroyAllWindows()
