import cv2
import mediapipe as mp
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(57600, 4)
        # self.fc2 = nn.Linear(64, 4)
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = self.relu(self.conv4(x))
        x = self.maxpool(x)
        x = self.relu(self.conv5(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

class HandGestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1)
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        self.save_dir = "hand_images"  # 设置保存目录
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 加载CNN模型
        self.model = ConvNet()
        self.model.load_state_dict(torch.load('model/best_model.pt'))
        self.model.eval()

        # 定义图像转换
        self.transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((500, 500)),
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
        ])

    def detect_gesture(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 绘制手部关键点
                    # self.mp_drawing.draw_landmarks(
                    #     frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # 获取手部框选框的坐标
                    x_min, y_min = frame.shape[1], frame.shape[0]
                    x_max, y_max = 0, 0
                    for lm in hand_landmarks.landmark:
                        x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                        if x < x_min:
                            x_min = x
                        if x > x_max:
                            x_max = x
                        if y < y_min:
                            y_min = y
                        if y > y_max:
                            y_max = y
                    # 增大矩形框的大小
                    padding = 20
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(frame.shape[1], x_max + padding)
                    y_max = min(frame.shape[0], y_max + padding)

                    # 提取手部图像
                    hand_image = frame[y_min:y_max, x_min:x_max]
                    # 对手部图像进行预处理
                    hand_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)
                    hand_image = Image.fromarray(hand_image)
                    hand_image = self.transform(hand_image)
                    hand_image = torch.unsqueeze(hand_image, 0)

                    # 使用CNN模型进行预测
                    with torch.no_grad():
                        outputs = self.model(hand_image)
                        _, predicted_class = torch.max(outputs, 1)
                    label_dict = {"0": "five", "1": "One", "2": "Three", "3": "Two"}
                    # label_dict = {"0": "0", "1": "1", "2": "3", "3": "5"}
                    # 在绘制手势识别结果时，使用预测到的类别获取对应的标签
                    predicted_label = label_dict.get(str(predicted_class.item()), "Unknown")
                    # 绘制手势识别结果
                    cv2.putText(frame, f'{predicted_label}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            cv2.imshow('Hand Gesture Detection', frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = HandGestureDetector()
    detector.detect_gesture()
