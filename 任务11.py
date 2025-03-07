import cv2
import numpy as np

# 读取图像
img = cv2.imread("E:\ww (4).jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

# 使用cv2.HoughLines()检测直线
lines1 = cv2.HoughLines(edges, 1, np.pi/180, 200)
if lines1 is not None:
    for line in lines1:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 使用cv2.HoughLinesP()检测直线
lines2 = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
if lines2 is not None:
    for line in lines2:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('HoughLines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()