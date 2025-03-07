import cv2
import numpy as np


# 读取包含苹果的图像（这里假设图像已存在，名为 'apples.jpg'）
image = cv2.imread('E:\\apple.jpg')
# 转换到HSV颜色空间，因为在HSV空间中对颜色的分割更方便
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义红苹果的HSV颜色范围（大致范围，可根据实际情况调整）
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
# 创建红苹果的掩膜
mask_red = cv2.inRange(hsv, lower_red, upper_red)

# 定义青苹果的HSV颜色范围（大致范围，可根据实际情况调整）
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])
# 创建青苹果的掩膜
mask_green = cv2.inRange(hsv, lower_green, upper_green)

# 计算红苹果的反向投影
roi_hist_red = cv2.calcHist([hsv], [0], mask_red, [180], [0, 180])
cv2.normalize(roi_hist_red, roi_hist_red, 0, 255, cv2.NORM_MINMAX)
back_project_red = cv2.calcBackProject([hsv], [0], roi_hist_red, [0, 180], 1)

# 计算青苹果的反向投影
roi_hist_green = cv2.calcHist([hsv], [0], mask_green, [180], [0, 180])
cv2.normalize(roi_hist_green, roi_hist_green, 0, 255, cv2.NORM_MINMAX)
back_project_green = cv2.calcBackProject([hsv], [0], roi_hist_green, [0, 180], 1)

# 显示结果（可以根据实际情况进一步处理，比如进行阈值处理、轮廓检测等使标记更清晰准确）
cv2.imshow('Original Image', image)
cv2.imshow('Red Apple Back Projection', back_project_red)
cv2.imshow('Green Apple Back Projection', back_project_green)
cv2.waitKey(0)
cv2.destroyAllWindows()