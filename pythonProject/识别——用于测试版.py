import sys
import os
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pygame
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QProgressBar
import time

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
        # x = self.fc2(x)
        return x

class MusicPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('音乐播放器')
        self.setGeometry(100, 100, 400, 200)

        self.music_label = QLabel('')
        self.progress_bar = QProgressBar()

        vbox = QVBoxLayout()
        vbox.addWidget(self.music_label)
        vbox.addWidget(self.progress_bar)

        self.setLayout(vbox)

    def set_music_name(self, name):
        self.music_label.setText(name)

    def set_progress(self, value):
        self.progress_bar.setValue(value)

class HandGestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1)
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        self.save_dir = "hand_images"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.model = ConvNet()
        self.model.load_state_dict(torch.load('model/best_model.pt'))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((500, 500)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        pygame.init()
        pygame.mixer.init()
        self.music_dir = "F:\CloudMusic/"
        self.music_files = [os.path.join(self.music_dir, file) for file in os.listdir(self.music_dir) if file.endswith(".mp3")]
        self.current_music_index = 0
        self.load_current_music()

        self.app = QApplication(sys.argv)
        self.music_player = MusicPlayer()

        self.last_detection_time = time.time()  # 记录上次手势检测的时间
        self.last_gesture = None  # 记录上次的手势
        self.music_playing = False  # 记录音乐是否正在播放

    def load_current_music(self):
        if self.current_music_index < len(self.music_files):
            pygame.mixer.music.load(self.music_files[self.current_music_index])

    def play_current_music(self):
        pygame.mixer.music.play()
        self.music_playing = True

    def pause_music(self):
        pygame.mixer.music.pause()
        self.music_playing = False

    def next_music(self):
        self.current_music_index = (self.current_music_index + 1) % len(self.music_files)
        self.load_current_music()
        self.play_current_music()

    def previous_music(self):
        self.current_music_index = (self.current_music_index - 1) % len(self.music_files)
        self.load_current_music()
        self.play_current_music()

    def detect_gesture(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

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

                    padding = 20
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(frame.shape[1], x_max + padding)
                    y_max = min(frame.shape[0], y_max + padding)

                    hand_image = frame[y_min:y_max, x_min:x_max]
                    hand_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)
                    hand_image = Image.fromarray(hand_image)
                    hand_image = self.transform(hand_image)
                    hand_image = torch.unsqueeze(hand_image, 0)

                    with torch.no_grad():
                        outputs = self.model(hand_image)
                        _, predicted_class = torch.max(outputs, 1)
                    label_dict = {"0": "play", "1": "pause", "2": "next", "3": "last"}

                    predicted_gesture = label_dict.get(str(predicted_class.item()), "unknow")

                    self.music_player.set_music_name(os.path.basename(self.music_files[self.current_music_index]))
                    progress = 50  # 示例进度值（0-100）
                    self.music_player.set_progress(progress)
                    self.music_player.show()

                    cv2.putText(frame, f'{predicted_gesture}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    if time.time() - self.last_detection_time >= 2 :#如果当前手势与上次不同
                        self.last_gesture = predicted_gesture
                        self.last_detection_time = time.time()

                        # 根据手势执行相应的操作
                        if predicted_gesture == "play":
                            self.play_current_music()
                        elif predicted_gesture == "pause":
                            self.pause_music()
                        elif predicted_gesture == "next":
                            self.next_music()
                        elif predicted_gesture == "last":
                            self.previous_music()

            cv2.imshow('Hand Gesture Detection', frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break


        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = HandGestureDetector()
    detector.detect_gesture()
    sys.exit(detector.app.exec_())
