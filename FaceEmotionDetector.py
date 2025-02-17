import cv2
import torch
import numpy as np
from torchvision import models, transforms
from torch import nn
from PIL import Image


class FaceEmotionDetector:
    """
    FaceEmotionDetector 类用于检测输入图像中的人脸并识别情绪标签。
    该类依赖于预训练的 EfficientNet B0 模型和 OpenCV 的人脸检测算法。
    """
    def __init__(self, model_path = "visionModel/best_model_epoch20_acc73.9.pth", device=None):
        """
        初始化方法
        参数:
            model_path: 预训练模型的权重文件路径
            device: torch 使用的设备，默认为 'cuda' (如果可用) 或 'cpu'
        """
        # if device is None:
        #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = device
        self.device = 'cpu'
        
        # 定义情绪标签映射，共8类
        self.emotion_labels = [
            "Angry",    # 0
            "Disgust",  # 1
            "Fear",     # 2
            "Happy",    # 3
            "Sad",      # 4
            "Surprise", # 5
            "Neutral",  # 6
            "Contempt"  # 7
        ]
        
        # 定义图像预处理变换，与训练时保持一致
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 加载预训练模型
        self.model = self._load_model(model_path)
        
        # 加载 OpenCV 人脸检测级联分类器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def _load_model(self, model_path):
        """
        加载预训练的 EfficientNet B0 模型，并修改最后一层以输出8类情绪
        参数:
            model_path: 模型权重文件路径
        返回:
            加载好的模型对象
        """
        # 使用 EfficientNet B0 并加载 ImageNet 预训练权重
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 8)
        
        # 加载训练时保存的模型权重
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model

    def detect_emotions(self, frame):
        """
        检测输入图像中的人脸并识别情绪标签
        参数:
            frame: 输入的图像（BGR格式，来自 OpenCV 摄像头）
        返回:
            results: list，每个元素为字典，包含 'box' (x, y, w, h) 和 'emotion' 标签
        """
        # 将图像转换为灰度图以提升人脸检测效率
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 使用级联分类器检测人脸
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        results = []
        
        for (x, y, w, h) in faces:
            # 提取人脸区域
            face_img = frame[y:y+h, x:x+w]
            
            # 将 BGR 图像转换为 RGB，并转换为 PIL Image
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            
            # 对图像进行预处理
            input_tensor = self.transform(face_pil)
            input_tensor = input_tensor.unsqueeze(0).to(self.device)
            
            # 使用模型进行情绪预测
            with torch.no_grad():
                output = self.model(input_tensor)
                pred = torch.argmax(output, dim=1).item()
                emotion = self.emotion_labels[pred]
            
            # 将检测结果保存
            results.append({
                'box': (x, y, w, h),
                'emotion': emotion
            })
        
        return results 