import cv2
import torch
import numpy as np
from torchvision import models, transforms
from torch import nn
from PIL import Image

# 定义情绪标签映射（共8类，根据训练时设置的num_classes=8）
emotion_labels = [
    "Angry",       # 0
    "Disgust",     # 1
    "Fear",        # 2
    "Happy",       # 3
    "Sad",         # 4
    "Surprise",    # 5
    "Neutral",     # 6
    "Contempt"     # 7
]

# 定义图像预处理管道，与训练时的valid_transform类似
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_trained_model(model_path, device):
    """加载训练好的模型，并修改最后一层为8类输出"""
    # 使用EfficientNet B0并加载ImageNet预训练权重
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 8)

    # 加载训练时保存的模型权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 指定训练好的模型权重路径，请根据实际情况修改
    model_path = 'visionModel/best_model_epoch30_acc74.1.pth'
    model = load_trained_model(model_path, device)
    
    # 加载OpenCV的人脸检测分类器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('无法打开摄像头')
        return
    
    # 增加命名窗口，确保cv2.imshow()正常工作
    cv2.namedWindow('Face Emotion Detection', cv2.WINDOW_NORMAL)
    
    print("摄像头已打开，按 'q' 键退出...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print('无法获取摄像头画面')
            break
        
        # 灰度化供人脸检测使用
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        # 遍历检测到的人脸
        for (x, y, w, h) in faces:
            # 在原图上绘制人脸矩形
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # 裁剪人脸区域，并转为RGB，再转换为PIL Image
            face_img = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            
            # 预处理人脸图像
            input_tensor = img_transform(face_pil)
            input_tensor = input_tensor.unsqueeze(0).to(device)
            
            # 模型预测
            with torch.no_grad():
                output = model(input_tensor)
                pred = torch.argmax(output, dim=1).item()
                emotion = emotion_labels[pred]
            
            # 在帧上显示预测的情绪标签
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (36, 255, 12), 2)
            
            # 在控制台打印情绪标签
            print('检测到的人脸情绪:', emotion)
        
        cv2.imshow('Face Emotion Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main() 