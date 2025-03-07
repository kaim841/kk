import cv2
import dlib
import numpy as np


class FaceBeauty:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        # 美颜参数
        self.slim_factor = 0
        self.eye_factor = 0
        self.mouth_factor = 0

        # 妆容颜色
        self.lip_color = (0, 0, 255)  # 红色
        self.blush_color = (180, 130, 255)  # 粉色
        self.hair_color = (150, 120, 80)  # 棕色

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        for face in faces:
            landmarks = self.predictor(gray, face)

            # 瘦脸
            if self.slim_factor > 0:
                face_points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(17)])
                center = np.mean(face_points, axis=0).astype(int)
                frame = self.slim_face(frame, face_points, center, self.slim_factor)

            # 大眼
            if self.eye_factor > 0:
                left_eye = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(36, 42)])
                right_eye = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(42, 48)])
                frame = self.enlarge_eyes(frame, left_eye, right_eye, self.eye_factor)

            # 嘴巴
            mouth_points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(48, 68)])
            frame = self.apply_lipstick(frame, mouth_points)

            # 腮红
            left_cheek = (landmarks.part(1).x, landmarks.part(1).y)
            right_cheek = (landmarks.part(15).x, landmarks.part(15).y)
            frame = self.apply_blush(frame, left_cheek, right_cheek)

            # 换发色
            frame = self.change_hair_color(frame, face)

        return frame

    def slim_face(self, image, points, center, factor):
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [points], (255, 255, 255))
        return cv2.addWeighted(image, 1 - factor, mask, factor, 0)

    def enlarge_eyes(self, image, left_eye, right_eye, factor):
        for eye in [left_eye, right_eye]:
            center = np.mean(eye, axis=0).astype(int)
            radius = int(np.linalg.norm(eye[0] - eye[3]) * (1 + factor))
            cv2.circle(image, tuple(center), radius, (255, 255, 255), 2)
        return image

    def apply_lipstick(self, image, mouth_points):
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [mouth_points], self.lip_color)
        return cv2.addWeighted(image, 1, mask, 0.4, 0)

    def apply_blush(self, image, left, right):
        mask = np.zeros_like(image)
        cv2.circle(mask, left, 40, self.blush_color, -1)
        cv2.circle(mask, right, 40, self.blush_color, -1)
        mask = cv2.GaussianBlur(mask, (7, 7), 5)
        return cv2.addWeighted(image, 1, mask, 0.3, 0)

    def change_hair_color(self, image, face):
        mask = np.zeros_like(image)
        y = face.top() - 50
        if y > 0:
            mask[0:y, :] = self.hair_color
        return cv2.addWeighted(image, 0.8, mask, 0.2, 0)


def main():
    cap = cv2.VideoCapture(0)
    beauty = FaceBeauty()

    # 创建控制窗口和滑动条
    cv2.namedWindow('Controls')

    # 美颜控制
    cv2.createTrackbar('瘦脸程度', 'Controls', 0, 100, lambda x: setattr(beauty, 'slim_factor', x / 100))
    cv2.createTrackbar('大眼程度', 'Controls', 0, 100, lambda x: setattr(beauty, 'eye_factor', x / 100))
    cv2.createTrackbar('嘴形调整', 'Controls', 0, 100, lambda x: setattr(beauty, 'mouth_factor', x / 100))

    # 妆容颜色控制
    # 口红颜色
    cv2.createTrackbar('口红-红', 'Controls', 0, 255,
                       lambda x: setattr(beauty, 'lip_color', (x, beauty.lip_color[1], beauty.lip_color[2])))
    cv2.createTrackbar('口红-绿', 'Controls', 0, 255,
                       lambda x: setattr(beauty, 'lip_color', (beauty.lip_color[0], x, beauty.lip_color[2])))
    cv2.createTrackbar('口红-蓝', 'Controls', 0, 255,
                       lambda x: setattr(beauty, 'lip_color', (beauty.lip_color[0], beauty.lip_color[1], x)))

    # 腮红颜色
    cv2.createTrackbar('腮红-红', 'Controls', 180, 255,
                       lambda x: setattr(beauty, 'blush_color', (x, beauty.blush_color[1], beauty.blush_color[2])))
    cv2.createTrackbar('腮红-绿', 'Controls', 130, 255,
                       lambda x: setattr(beauty, 'blush_color', (beauty.blush_color[0], x, beauty.blush_color[2])))
    cv2.createTrackbar('腮红-蓝', 'Controls', 255, 255,
                       lambda x: setattr(beauty, 'blush_color', (beauty.blush_color[0], beauty.blush_color[1], x)))

    # 发色控制
    cv2.createTrackbar('发色-红', 'Controls', 150, 255,
                       lambda x: setattr(beauty, 'hair_color', (x, beauty.hair_color[1], beauty.hair_color[2])))
    cv2.createTrackbar('发色-绿', 'Controls', 120, 255,
                       lambda x: setattr(beauty, 'hair_color', (beauty.hair_color[0], x, beauty.hair_color[2])))
    cv2.createTrackbar('发色-蓝', 'Controls', 80, 255,
                       lambda x: setattr(beauty, 'hair_color', (beauty.hair_color[0], beauty.hair_color[1], x)))

    # 创建控制面板背景
    control_panel = np.zeros((400, 600, 3), dtype=np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = beauty.process_frame(frame)

        # 更新控制面板
        control_panel.fill(50)  # 灰色背景
        cv2.putText(control_panel, "Beauty Controls", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 显示结果和控制面板
        cv2.imshow('Beauty Effect', result)
        cv2.imshow('Controls', control_panel)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()