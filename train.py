# train.py

# ----------------------
# 1. 基础导入和环境配置
# ----------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
# 设置代理和下载目录
os.environ["HTTP_PROXY"] = "http://127.0.0.1:10809"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10809"
os.environ["TORCH_HOME"] = "."
from tqdm.auto import tqdm
import psutil
import gc
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
import cv2
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time

# ----------------------
# 2. 基础配置
# ----------------------
# 2.1 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # 加速卷积运算
if device.type == 'cuda':
    print("Using GPU for training.")
else:
    print("Warning: GPU not available. Training on CPU.")

# 2.2 训练参数配置
config = {
    "data_dir": "archive/data_relabeled_balanced_1x/train",
    "batch_size": 32,
    "grad_accum_steps": 4,
    "num_epochs": 100,
    "learning_rate": 1e-3,
    "num_classes": 8,
    "input_size": 224,
    "valid_ratio": 0.15,
    "seed": 42,
    "num_workers": 0,
    "max_grad_norm": 1.0,
    "mixup_alpha": 0.2,
    "label_smoothing": 0.1,
    "model_dir": "visionModel",
    "freeze_blocks": 2,
    "patience": 10,
    "min_lr": 1e-6,
    "weight_decay": 5e-4
}

# 2.3 固定随机种子
def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
seed_everything(config['seed'])

# ----------------------
# 3. 模型选择函数
# ----------------------
def list_available_models():
    """列出 visionModel 目录下的所有可用模型"""
    model_files = glob.glob('visionModel/*.pth')
    if not model_files:
        print("No pre-trained models found in visionModel directory.")
        return None
    
    print("\nAvailable pre-trained models:")
    for i, model_path in enumerate(model_files, 1):
        model_name = os.path.basename(model_path)
        print(f"{i}. {model_name}")
    
    while True:
        try:
            choice = int(input("\nSelect a model number (0 to train from scratch): "))
            if choice == 0:
                return None
            if 1 <= choice <= len(model_files):
                return model_files[choice-1]
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

# ----------------------
# 4. 数据集和数据加载
# ----------------------
class OptimizedDataset(Dataset):
    """优化的数据集类，支持内存缓存"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = sorted([d for d in os.listdir(root_dir) 
                             if os.path.isdir(os.path.join(root_dir, d)) and '_gray' not in d])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []
        self.transform = transform
        
        # 预加载图像路径
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            self.samples.extend([
                (os.path.join(cls_dir, fname), self.class_to_idx[cls]) 
                for fname in os.listdir(cls_dir)
                if fname.lower().endswith(('png', 'jpg', 'jpeg'))
            ])
        
        # 内存优化：预加载小尺寸图像
        self.cache = {}
        if psutil.virtual_memory().available > 8*1024**3:  # 仅当内存>8GB时启用
            for idx in tqdm(range(len(self.samples)), desc="预加载图像"):
                img_path, label = self.samples[idx]
                self.cache[idx] = (Image.open(img_path).convert('RGB'), label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx in self.cache:
            img, label = self.cache[idx]
        else:
            img_path, label = self.samples[idx]
            img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = np.array(img)
            augmented = self.transform(image=img)
            img = augmented['image']
            
        return img, label

# 数据增强定义
train_transform = A.Compose([
    A.Resize(config['input_size'], config['input_size']),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
    ], p=0.5),
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0)),
        A.GaussianBlur(blur_limit=(3, 7)),
        A.MotionBlur(blur_limit=(3, 7)),
    ], p=0.3),
    A.CoarseDropout(
        max_holes=8,
        max_height=32,
        max_width=32,
        min_holes=5,
        fill_value=0,
        p=0.3
    ),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

valid_transform = A.Compose([
    A.Resize(config['input_size'], config['input_size']),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def create_data_loaders():
    """创建训练和验证数据加载器"""
    full_dataset = OptimizedDataset(config['data_dir'], transform=train_transform)
    train_size = int((1 - config['valid_ratio']) * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, valid_size])
    valid_dataset.dataset.transform = valid_transform
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=False
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=config['batch_size']*2,
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    return train_loader, valid_loader

# ----------------------
# 5. 模型创建函数
# ----------------------
def create_model():
    """创建并初始化模型"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 创建基础模型
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features, config['num_classes'])
            )
            
            # 获取预训练模型路径
            pretrained_path = list_available_models()
            
            # 如果指定了预训练模型路径，加载权重
            if pretrained_path:
                print(f"Loading pre-trained weights from: {pretrained_path}")
                checkpoint = torch.load(pretrained_path)
                model.load_state_dict(checkpoint['model_state_dict'])
            
            # 冻结指定层
            for param in model.features[:config['freeze_blocks']].parameters():
                param.requires_grad = False
            
            model = model.to(device, memory_format=torch.channels_last)
            print("Model created successfully.")
            return model
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            torch.cuda.empty_cache()
            gc.collect()
            if attempt == max_retries - 1:
                raise e

# ----------------------
# 6. 训练工具函数
# ----------------------
def print_memory_usage():
    """打印显存使用情况"""
    if device.type == 'cuda':
        mem = torch.cuda.memory_reserved(device) / 1e9
        print(f"当前显存占用: {mem:.2f}GB")
    else:
        print("Training on CPU, no GPU memory usage available.")

def mixup_data(x, y, alpha=1.0):
    """Mixup数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# ----------------------
# 7. 训练和验证函数
# ----------------------
def train_one_epoch(model, train_loader, optimizer, criterion, scaler, max_retries=3):
    """训练一个epoch，添加重试机制"""
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    
    progress_bar = tqdm(train_loader, 
                       desc='Training',
                       leave=True,
                       bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
                       dynamic_ncols=True)
    
    optimizer.zero_grad()
    
    for step, (inputs, labels) in enumerate(progress_bar):
        retry_count = 0
        while retry_count < max_retries:
            try:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Mixup数据增强
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, config['mixup_alpha'])
                
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(inputs)
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                    loss = loss / config['grad_accum_steps']
                
                scaler.scale(loss).backward()
                
                if (step + 1) % config['grad_accum_steps'] == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                running_loss = loss.item() * config['grad_accum_steps']
                _, predicted = outputs.max(1)
                correct += (lam * predicted.eq(targets_a).sum().item() + 
                           (1 - lam) * predicted.eq(targets_b).sum().item())
                total += labels.size(0)
                train_acc = 100. * correct / total
                
                mem_str = f"{torch.cuda.memory_reserved(device)/1e9:.1f}G" if device.type == 'cuda' else "N/A"
                
                progress_bar.set_postfix({
                    'loss': f"{running_loss:.3f}",
                    'acc': f"{train_acc:.1f}%",
                    'lr': f"{optimizer.param_groups[0]['lr']:.1e}",
                    'mem': mem_str
                }, refresh=True)
                
                break  # 如果成功执行，跳出重试循环
                
            except Exception as e:
                retry_count += 1
                print(f"\nError in training step {step}: {str(e)}")
                print(f"Retry attempt {retry_count}/{max_retries}")
                
                if retry_count == max_retries:
                    print(f"Failed after {max_retries} attempts, skipping this batch")
                    continue
                
                # 清理显存并等待一小段时间
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(1)  # 等待1秒后重试
    
    return running_loss, train_acc

def validate(model, valid_loader, criterion, max_retries=3):
    """验证模型，添加重试机制"""
    model.eval()
    valid_loss = 0.0
    correct = 0
    total = 0
    
    for retry in range(max_retries):
        try:
            with torch.no_grad(), autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                for inputs, labels in tqdm(valid_loader, 
                                          desc='Validating', 
                                          leave=False,
                                          bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    valid_loss += loss.item()
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
            
            return valid_loss / len(valid_loader), 100. * correct / total
            
        except Exception as e:
            print(f"\nError in validation: {str(e)}")
            print(f"Retry attempt {retry + 1}/{max_retries}")
            
            if retry == max_retries - 1:
                print("Validation failed after all retries")
                return float('inf'), 0.0
            
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(1)

# ----------------------
# 8. 可视化函数
# ----------------------
def plot_training_history(history):
    """绘制训练历史"""
    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['valid_loss'], label='Valid')
    plt.title('Training/Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['valid_acc'], label='Valid')
    plt.title('Training/Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    save_path = os.path.join(config['model_dir'], 'training_metrics.png')
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved training metrics plot to: {save_path}")

# ----------------------
# 9. 主训练循环
# ----------------------
def main():
    """主训练函数，添加重试机制"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            os.makedirs(config['model_dir'], exist_ok=True)
            
            train_loader, valid_loader = create_data_loaders()
            model = create_model()
            scaler = GradScaler()
            criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
            
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay'],
                betas=(0.9, 0.999)
            )
            
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,
                T_mult=2,
                eta_min=config['min_lr']
            )
            
            early_stopping = EarlyStopping(patience=config['patience'])
            best_acc = 0.0
            history = {'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []}
            
            for epoch in range(config['num_epochs']):
                print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
                
                # 训练和验证都有自己的重试机制
                train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
                valid_loss, valid_acc = validate(model, valid_loader, criterion)
                
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['valid_loss'].append(valid_loss)
                history['valid_acc'].append(valid_acc)
                
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    save_path = os.path.join(
                        config['model_dir'], 
                        f'best_model_epoch{epoch+1}_acc{valid_acc:.1f}.pth'
                    )
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_acc': best_acc,
                        'history': history,  # 保存训练历史，方便恢复
                    }, save_path)
                    print(f"Saved best model to: {save_path}")
                
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Learning rate: {current_lr:.1e}")
                
                early_stopping(valid_loss)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break
                
                torch.cuda.empty_cache()
                gc.collect()
            
            plot_training_history(history)
            break  # 如果成功完成训练，跳出重试循环
            
        except Exception as e:
            print(f"\nTraining failed on attempt {attempt + 1}/{max_retries}: {str(e)}")
            if attempt == max_retries - 1:
                print("Training failed after all retries")
                raise e
            
            print("Cleaning up and retrying...")
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(5)  # 等待5秒后重试

# 添加早停类
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

if __name__ == "__main__":
    main()
