import cv2
import numpy as np
import matplotlib.pyplot as plt


# 读取风景图
img = cv2.imread('E:\wok1.jpg')
# 转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 计算灰度直方图
hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])

# 绘制灰度直方图
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("Number of Pixels")
plt.plot(hist_gray)
plt.xlim([0, 256])

# 读取风景图
img = cv2.imread('E:\wok1.jpg')

# RGB空间直方图
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color = col)
plt.title("RGB Histogram")
plt.xlabel("Bins")
plt.ylabel("Number of Pixels")
plt.xlim([0, 256])
plt.show()

# 转换为HSV空间
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue = hsv[:, :, 0]
sat = hsv[:, :, 1]
val = hsv[:, :, 2]

# HSV空间直方图
hist_hue = cv2.calcHist([hue], [0], None, [180], [0, 180])
hist_sat = cv2.calcHist([sat], [0], None, [256], [0, 256])
hist_val = cv2.calcHist([val], [0], None, [256], [0, 256])

plt.figure()
plt.subplot(311)
plt.title("HSV Histogram - Hue")
plt.xlabel("Bins")
plt.ylabel("Number of Pixels")
plt.plot(hist_hue)
plt.xlim([0, 180])

plt.subplot(312)
plt.title("HSV Histogram - Saturation")
plt.xlabel("Bins")
plt.ylabel("Number of Pixels")
plt.plot(hist_sat)
plt.xlim([0, 256])

plt.subplot(313)
plt.title("HSV Histogram - Value")
plt.xlabel("Bins")
plt.ylabel("Number of Pixels")
plt.plot(hist_val)
plt.xlim([0, 256])
plt.show()
# 读取风景图
img = cv2.imread('E:\wok1.jpg')

# 转换为灰度图用于直方图均衡
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 直方图均衡
equ = cv2.equalizeHist(gray)
plt.show()
cv2.imshow("gray",gray)
cv2.imshow("Original", gray)
cv2.imshow("Equalized", equ)
cv2.waitKey(0)
cv2.destroyAllWindows()