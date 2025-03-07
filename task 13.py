import cv2

# 打开摄像头，参数0通常代表默认的内置摄像头，如果有外接摄像头，可尝试1、2等编号
cap = cv2.VideoCapture(0)

while True:
    # 读取摄像头的一帧图像，ret表示是否读取成功，frame为读取到的图像帧（以BGR格式存储）
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头图像，可能摄像头未正常连接或出现其他问题")
        break

    # 将彩色图像帧转换为灰度图像，便于后续的边缘检测等操作
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用高斯模糊对灰度图像进行降噪处理，减少图像噪声对轮廓检测的干扰
    # 这里的(5, 5)表示高斯核的大小，数值越大，模糊效果越强，但也会丢失更多细节，0为标准差，会自动根据核大小计算
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 采用Canny边缘检测算法检测图像中的边缘，得到二值化的边缘图像
    # 100和200是阈值参数，可根据实际情况调整，低于100的梯度值被设为0（非边缘），高于200的肯定是边缘，中间的值根据是否与边缘相连判断
    edges = cv2.Canny(blurred, 100, 200)

    # 查找图像中的轮廓，cv2.RETR_EXTERNAL表示只检测最外层轮廓，cv2.CHAIN_APPROX_SIMPLE对轮廓进行压缩，只保留端点信息
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历找到的所有轮廓
    for contour in contours:
        # 计算轮廓的周长，用于后续判断轮廓是否符合人像轮廓的大致特征
        perimeter = cv2.arcLength(contour, True)
        # 对轮廓进行多边形逼近，epsilon是逼近精度，这里取周长的0.02倍，可根据实际调整
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # 根据多边形逼近后的顶点数量等特征大致判断是否为人像轮廓（这里简单以顶点数量来大致区分）
        if len(approx) >= 5:
            # 在原始的彩色图像frame上绘制找到的人像轮廓，轮廓颜色为绿色（0, 255, 0），轮廓线粗细为2像素
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

    # 显示绘制好轮廓的图像，窗口标题为"Human Figure Contours"
    cv2.imshow("Human Figure Contours", frame)

    # 等待1毫秒，获取用户按键操作，如果按下键盘上的'q'键（ASCII码值为113），则退出循环
    key = cv2.waitKey(1)
    if key == 113:
        break

# 释放摄像头资源，关闭所有打开的图像窗口
cap.release()
cv2.destroyAllWindows()