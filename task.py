import cv2
import os
opencv_path = cv2.__file__
dirname = os.path.dirname(opencv_path)
cascade_path = os.path.join(dirname, 'data/haarcascades/haarcascade_frontalface_default.xml')
# 加载预训练的人脸检测分类器
face_cascade = cv2.CascadeClassifier(cv2.data.hoarse + 'haarcascade_frontalface_default.xml')
# 读取要检测人脸的图像，将实际图像路径替换此处的 'your_image_path.jpg'
image = cv2.imread('E:\gummy.jpg')
# 将图像转换为灰度图，很多人脸检测算法在灰度图上效果较好且处理速度更快
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 读取要检测人脸的图像，将实际图像路径替换此处的 'your_image_path.jpg'
image = cv2.imread('E:\gummy.jpg.jpg')
# 将图像转换为灰度图，很多人脸检测算法在灰度图上效果较好且处理速度更快
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 使用加载的分类器检测人脸，参数可以根据实际情况调整，这里使用默认参数
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# 遍历检测到的每个人脸，在原图像上绘制矩形框（也就是圈出人脸）
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
# 显示圈出人脸后的图像
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()