import cv2
import numpy as np


# 定义一个函数用于绘制匹配结果，将匹配的特征点用线连接起来
def draw_matches(img1, img2, keypoints1, keypoints2, matches):
    # 创建一个空白图像用于绘制连线结果，高度为两张图像高度之和，宽度为两张图像中较宽的宽度
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    result = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    result[:h1, :w1] = img1
    result[:h2, w1:w1 + w2] = img2

    # 遍历匹配结果，在图像上绘制连线
    for match in matches:
        idx1 = match.queryIdx
        idx2 = match.trainIdx
        pt1 = tuple(map(int, keypoints1[idx1].pt))
        pt2 = tuple(map(int, keypoints2[idx2].pt))
        cv2.line(result, pt1, (pt2[0] + w1, pt2[1]), (0, 255, 0), 1)

    return result


# 读取两张相似的图像，这里假设图像路径为相对路径，你可根据实际情况替换为正确的路径
img1 = cv2.imread('E:\ggboy.jpg')
img2 = cv2.imread('E:\ggboy2.jpg')

# 1. FAST特征检测与匹配
fast = cv2.FastFeatureDetector_create()
keypoints1_fast = fast.detect(img1, None)
keypoints2_fast = fast.detect(img2, None)
orb = cv2.ORB_create()
des1_fast = orb.compute(img1, keypoints1_fast)
des2_fast = orb.compute(img2, keypoints2_fast)
bf = cv2.BFMatcher()
matches_fast = bf.match(des1_fast, des2_fast)
result_fast = draw_matches(img1, img2, keypoints1_fast, keypoints2_fast, matches_fast)

# 2. STAR特征检测与匹配
star = cv2.xfeatures2d.StarDetector_create()
keypoints1_star = star.detect(img1, None)
keypoints2_star = star.detect(img2, None)
des1_star = orb.compute(img1, keypoints1_star)
des2_star = orb.compute(img2, keypoints2_star)
matches_star = bf.match(des1_star, des2_star)
result_star = draw_matches(img1, img2, keypoints1_star, keypoints2_star, matches_star)

# 3. SIFT特征检测与匹配（需调用xfeature2d库）
sift = cv2.xfeatures2d.SIFT_create()
keypoints1_sift = sift.detect(img1, None)
keypoints2_sift = sift.detect(img2, None)
des1_sift = sift.compute(img1, keypoints1_sift)
des2_sift = sift.compute(img2, keypoints2_sift)
matches_sift = bf.match(des1_sift, des2_sift)
result_sift = draw_matches(img1, img2, keypoints1_sift, keypoints2_sift, matches_sift)

# 4. SURF特征检测与匹配（需调用xfeature2d库）
surf = cv2.xfeatures2d.SURF_create()
keypoints1_surf = surf.detect(img1, None)
keypoints2_surf = surf.detect(img2, None)
des1_surf = surf.compute(img1, keypoints1_surf)
des2_surf = surf.compute(img2, keypoints2_surf)
matches_surf = bf.match(des1_surf, des2_surf)
result_surf = draw_matches(img1, img2, keypoints1_surf, keypoints2_surf, matches_surf)

# 5. ORB特征检测与匹配（需调用xfeature2d库）
orb_detector = cv2.xfeatures2d.ORB_create()
keypoints1_orb = orb_detector.detect(img1, None)
keypoints2_orb = orb_detector.detect(img2, None)
des1_orb = orb_detector.compute(img1, keypoints1_orb)
des2_orb = orb_detector.compute(img2, keypoints2_orb)
matches_orb = bf.match(des1_orb, des2_orb)
result_orb = draw_matches(img1, img2, keypoints1_orb, keypoints2_orb, matches_orb)

# 6. MSER特征检测与匹配
mser = cv2.MSER_create()
keypoints1_mser = mser.detect(img1, None)
keypoints2_mser = mser.detect(img2, None)
des1_mser = orb.compute(img1, keypoints1_mser)
des2_mser = orb.compute(img2, keypoints2_mser)
matches_mser = bf.match(des1_mser, des2_mser)
result_mser = draw_matches(img1, img2, keypoints1_mser, keypoints2_mser, matches_mser)

# 7. GFTT特征检测与匹配
gftt = cv2.GoodFeaturesToTrackDetector_create()
keypoints1_gftt = gftt.detect(img1, None)
keypoints2_gftt = gftt.detect(img2, None)
des1_gftt = orb.compute(img1, keypoints1_gftt)
des2_gftt = orb.compute(img2, keypoints2_gftt)
matches_gftt = bf.match(des1_gftt, des2_gftt)
result_gftt = draw_matches(img1, img2, keypoints1_gftt, keypoints2_gftt, matches_gftt)

# 8. HARRIS特征检测与匹配（配合Harris detector）
# 先计算Harris角点响应，这里示例使用一个简单的阈值来筛选角点作为特征点，实际应用中可更精细调整
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
dst1 = cv2.cornerHarris(gray1, 2, 3, 0.04)
dst2 = cv2.cornerHarris(gray2, 2, 3, 0.04)
keypoints1_harris = []
keypoints2_harris = []
for y in range(dst1.shape[0]):
    for x in range(dst1.shape[1]):
        if dst1[y, x] > 0.01 * dst1.max():
            keypoints1_harris.append(cv2.KeyPoint(x, y, 3))
for y in range(dst2.shape[0]):
    for x in range(dst2.shape[1]):
        if dst2[y, x] > 0.01 * dst2.max():
            keypoints2_harris.append(cv2.KeyPoint(x, y, 3))
des1_harris = orb.compute(img1, keypoints1_harris)
des2_harris = orb.compute(img2, keypoints2_harris)
matches_harris = bf.match(des1_harris, des2_harris)
result_harris = draw_matches(img1, img2, keypoints1_harris, keypoints2_harris, matches_harris)

# 9. Dense特征检测与匹配
dense = cv2.DenseFeatureDetector_create()
keypoints1_dense = dense.detect(img1, None)
keypoints2_dense = dense.detect(img2, None)
des1_dense = orb.compute(img1, keypoints1_dense)
des2_dense = orb.compute(img2, keypoints2_dense)
matches_dense = bf.match(des1_dense, des2_dense)
result_dense = draw_matches(img1, img2, keypoints1_dense, keypoints2_dense, matches_dense)

# 10. SimpleBlob特征检测与匹配
simple_blob = cv2.SimpleBlobDetector_create()
keypoints1_simple_blob = simple_blob.detect(img1, None)
keypoints2_simple_blob = simple_blob.detect(img2, None)
des1_simple_blob = orb.compute(img1, keypoints1_simple_blob)
des2_simple_blob = orb.compute(img2, keypoints2_simple_blob)
matches_simple_blob = bf.match(des1_simple_blob, des2_simple_blob)
result_simple_blob = draw_matches(img1, img2, keypoints1_simple_blob, keypoints2_simple_blob, matches_simple_blob)

# 展示各种方法的结果
cv2.imshow('FAST Result', result_fast)
cv2.imshow('STAR Result', result_star)
cv2.imshow('SIFT Result', result_sift)
cv2.imshow('SURF Result', result_surf)
cv2.imshow('ORB Result', result_orb)
cv2.imshow('MSER Result', result_mser)
cv2.imshow('GFTT Result', result_gftt)
cv2.imshow('HARRIS Result', result_harris)
cv2.imshow('DENSE Result', result_dense)
cv2.imshow('SIMPLE BLOB Result', result_simple_blob)
cv2.waitKey(0)
cv2.destroyAllWindows()
