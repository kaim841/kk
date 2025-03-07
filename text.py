import cv2
import numpy as np

# 加载 Haar 分类器
face_cascade = cv2.CascadeClassifier(r"E:\haarcascade_eye.xml")
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

# 初始化美颜参数
slim_ratio = 1.0
eye_ratio = 1.0
mouth_ratio = 1.0

def adjust_face(frame, face, landmarks):
    """
    根据用户输入的滑动条值调整瘦脸、大眼、嘴巴比例。
    """
    global slim_ratio, eye_ratio, mouth_ratio

    # 瘦脸处理
    (x, y, w, h) = face
    if slim_ratio != 1.0:
        center = (x + w // 2, y + h // 2)
        M = cv2.getRotationMatrix2D(center, 0, slim_ratio)
        frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

    # 大眼、嘴巴大小可以扩展处理 landmarks
    return frame

def apply_makeup(frame, eyes, nose, mouth):
    """
    添加实时妆容效果：眼线、唇彩、腮红。
    """
    # 眼线（绘制眼睛轮廓）
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

    # 唇彩（嘴巴区域填充）
    for (mx, my, mw, mh) in mouth:
        cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (0, 0, 255), -1)

    # 腮红（左右两侧脸颊）
    if len(nose) > 0:
        for (nx, ny, nw, nh) in nose:
            cv2.circle(frame, (nx + nw // 2 - 40, ny + nh), 30, (255, 192, 203), -1)
            cv2.circle(frame, (nx + nw // 2 + 40, ny + nh), 30, (255, 192, 203), -1)

    return frame

def nothing(x):
    pass

# 创建滑动条窗口
cv2.namedWindow('Adjustments')
cv2.createTrackbar('Slim Face', 'Adjustments', 10, 20, nothing)
cv2.createTrackbar('Big Eyes', 'Adjustments', 10, 20, nothing)
cv2.createTrackbar('Mouth Size', 'Adjustments', 10, 20, nothing)

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 转为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # 从滑动条获取参数
    slim_ratio = cv2.getTrackbarPos('Slim Face', 'Adjustments') / 10
    eye_ratio = cv2.getTrackbarPos('Big Eyes', 'Adjustments') / 10
    mouth_ratio = cv2.getTrackbarPos('Mouth Size', 'Adjustments') / 10

    for (x, y, w, h) in faces:
        # 绘制人脸框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 提取人脸区域
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # 检测眼睛
        eyes = eye_cascade.detectMultiScale(roi_gray)
        # 检测鼻子
        nose = nose_cascade.detectMultiScale(roi_gray)
        # 检测嘴巴
        mouth = mouth_cascade.detectMultiScale(roi_gray)

        # 调整面部
        frame = adjust_face(frame, (x, y, w, h), [])

        # 添加妆容
        frame = apply_makeup(frame, eyes, nose, mouth)

    # 显示视频
    cv2.imshow('Face Beauty', frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()