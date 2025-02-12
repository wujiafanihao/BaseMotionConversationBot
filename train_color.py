# train_color.py

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
from datetime import datetime

class EarlyStopping:
    """早停机制
    
    当验证集损失在指定的耐心值内没有改善时，提前停止训练
    
    Args:
        patience (int): 等待改善的轮数
        min_delta (float): 最小改善阈值
    """
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

# 关键参数配置 (根据6GB显存优化)
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
    "patience": 15,        # 增加耐心值
    "min_lr": 1e-6,
    "weight_decay": 5e-4,
    "T_0": 10,            # 余弦退火周期
    "T_mult": 2,          # 周期倍增因子
}

# 2.3 固定随机种子
def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
seed_everything(config['seed'])

# ----------------------
# 内存优化数据集
# 3. 数据集和数据加载
# ----------------------
# 3.1 数据集类定义
class OptimizedDataset(Dataset):
    """优化的数据集类，支持内存缓存"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = sorted([d for d in os.listdir(root_dir) 
                             if os.path.isdir(os.path.join(root_dir, d)) and '_gray' not in d]) # 过滤掉灰度数据集
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
            # img = self.transform(img) # 转换为tensor
            img = np.array(img)  # 转换为numpy array
            augmented = self.transform(image=img)
            img = augmented['image']
            
        return img, label

# 3.2 数据增强定义
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.Resize(config['input_size'], config['input_size']),
    A.HorizontalFlip(p=0.5),
    # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    # 将ShiftScaleRotate替换为 Affine
    A.Affine(
        scale=(0.9, 1.1), # 允许图像随机缩放，范围为 0.9 到 1.1
        translate_percent=(-0.1, 0.1),  # 允许图像随机平移，范围为 -10% 到 10%
        rotate=(-15, 15), # 允许图像随机旋转，范围为 -15 到 15 度 
        shear=(-10, 10),              # 加入轻微shear变换，范围设为 -10 到 10 度
        interpolation=cv2.INTER_LINEAR, # 使用cv2.INTER_LINEAR进行线性插值
        border_mode=cv2.BORDER_REFLECT_101, # 使用cv2.BORDER_REFLECT_101方式填充边缘
        p=0.5 # 应用概率为 0.5
    ),
    A.RandomBrightnessContrast(p=0.3),
    # A.CoarseDropout(max_holes=3, max_height=20, max_width=20, p=0.3),
    # 修改 CoarseDropout 参数
    A.CoarseDropout(
        num_holes_range=(3, 6),           # 每次随机遮挡 3-6 个区域
        hole_height_range=(0.1, 0.2),     # 高度为图像高度的 10%-20%
        hole_width_range=(0.1, 0.2),      # 宽度为图像宽度的 10%-20%
        fill=0,                     # 使用黑色填充
        p=0.3                             # 应用概率为 0.3
    ),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

valid_transform = A.Compose([
    A.Resize(config['input_size'], config['input_size']),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# 3.3 创建数据加载器
def create_data_loaders():
    """创建训练和验证数据加载器"""
    # 创建数据集
    full_dataset = OptimizedDataset(config['data_dir'], transform=train_transform)
    train_size = int((1 - config['valid_ratio']) * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, valid_size])
    valid_dataset.dataset.transform = valid_transform
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        # persistent_workers=True
        persistent_workers=False # 关闭持续workers（多进程）
    )
    
    # 验证数据加载器
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=config['batch_size']*2,
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # 返回训练和验证数据加载器
    return train_loader, valid_loader

# ----------------------
# 4. 模型相关
# ----------------------
# 4.1 模型创建函数
def create_model():
    """创建并初始化模型（带重试机制）"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 使用EfficientNet B0并加载ImageNet预训练权重
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_features = model.classifier[1].in_features
            # 使用新的 classifier，加入 dropout 层以降低过拟合
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features, config['num_classes'])
            )
            
            # 冻结前 freeze_blocks 个阶段的参数以减少模型复杂度
            for param in model.features[:config['freeze_blocks']].parameters():
                param.requires_grad = False
            
            # 显存优化配置
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
# 5. 训练工具函数
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
# 6. 训练和验证函数
# ----------------------
def train_one_epoch(model, train_loader, optimizer, criterion, scaler):
    """训练一个epoch"""
    model.train()
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, 
                       desc='Training',
                       bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    
    optimizer.zero_grad()
    
    for step, (inputs, labels) in enumerate(progress_bar):
        try:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Mixup数据增强
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, config['mixup_alpha'])
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(inputs)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                loss = loss / config['grad_accum_steps']  # 梯度累积
            
            scaler.scale(loss).backward()
            
            if (step + 1) % config['grad_accum_steps'] == 0 or (step + 1) == len(train_loader):
                # 梯度裁剪
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # 计算准确率
            _, predicted = outputs.max(1)
            correct += (lam * predicted.eq(targets_a).sum().item() + 
                       (1 - lam) * predicted.eq(targets_b).sum().item())
            total += labels.size(0)
            train_acc = 100. * correct / total
            
            # 计算显存使用情况
            mem_str = f"{torch.cuda.memory_reserved(device)/1e9:.1f}G" if device.type == 'cuda' else "N/A"
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss.item()*config['grad_accum_steps']:.3f}",
                'acc': f"{train_acc:.1f}%",
                'lr': f"{optimizer.param_groups[0]['lr']:.1e}",
                'mem': mem_str
            })
        except Exception as e:
            print(f"Error in training step: {str(e)}")
            continue
    
    return loss.item(), train_acc

def calculate_confusion_matrix(preds, labels, num_classes):
    """计算混淆矩阵"""
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.int)
    for p, t in zip(preds, labels):
        matrix[t, p] += 1
    return matrix

def validate(model, valid_loader, criterion):
    """验证模型并收集详细指标"""
    model.eval()
    valid_loss = 0.0
    correct = 0
    total = 0
    
    # 收集预测结果和真实标签
    all_preds = []
    all_labels = []
    class_correct = [0] * config['num_classes']
    class_total = [0] * config['num_classes']
    
    try:
        with torch.no_grad(), autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            for inputs, labels in tqdm(valid_loader, desc='Validating', leave=False):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                
                # 收集预测结果
                all_preds.extend(predicted.cpu())
                all_labels.extend(labels.cpu())
                
                # 计算每个类别的准确率
                for label, pred in zip(labels, predicted):
                    label_idx = label.item()
                    if label_idx == pred.item():
                        class_correct[label_idx] += 1
                    class_total[label_idx] += 1
                
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        # 计算每个类别的准确率
        class_accuracies = [
            100 * correct / total if total > 0 else 0 
            for correct, total in zip(class_correct, class_total)
        ]
        
        # 计算混淆矩阵
        conf_matrix = calculate_confusion_matrix(
            torch.tensor(all_preds), 
            torch.tensor(all_labels), 
            config['num_classes']
        )
        
        return (valid_loss / len(valid_loader), 100. * correct / total, 
                conf_matrix, class_accuracies)
    except Exception as e:
        print(f"Error in validation: {str(e)}")
        return float('inf'), 0.0, None, [0] * config['num_classes']

# ----------------------
# 7. 主训练循环
# ----------------------
def main():
    """主训练函数"""
    os.makedirs(config['model_dir'], exist_ok=True)
    
    # 创建训练结果保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(config['model_dir'], f'training_results_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    train_loader, valid_loader = create_data_loaders()
    model = create_model()
    scaler = GradScaler()
    
    # 使用标签平滑的交叉熵损失
    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['T_0'],
        T_mult=config['T_mult'],
        eta_min=config['min_lr']
    )
    
    early_stopping = EarlyStopping(patience=config['patience'])
    best_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [], 
        'valid_loss': [], 'valid_acc': [],
        'learning_rates': [], 'class_accuracies': [],
        'final_confusion_matrix': None
    }
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        valid_loss, valid_acc, conf_matrix, class_accuracies = validate(
            model, valid_loader, criterion)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        history['class_accuracies'].append(class_accuracies)
        
        # 在最后一个epoch或提前停止时保存混淆矩阵
        if epoch == config['num_epochs']-1 or early_stopping.counter >= config['patience']:
            history['final_confusion_matrix'] = conf_matrix
            
            # 打印每个类别的准确率
            print("\nPer-class accuracy:")
            for i, acc in enumerate(class_accuracies):
                print(f"Class {i}: {acc:.1f}%")
        
        if valid_acc > best_acc:
            best_acc = valid_acc
            save_path = os.path.join(
                save_dir, 
                f'best_model_epoch{epoch+1}_acc{valid_acc:.1f}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'history': history,
            }, save_path)
            print(f"Saved best model to: {save_path}")
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.1e}")
        
        # 每5个epoch绘制一次训练图
        if (epoch + 1) % 5 == 0:
            plot_training_history(history, save_dir)
        
        early_stopping(valid_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
        
        torch.cuda.empty_cache()
        gc.collect()
    
    # 训练结束后绘制最终图表
    plot_training_history(history, save_dir)
    
    # 保存完整训练历史
    history_path = os.path.join(save_dir, 'training_history.pth')
    torch.save(history, history_path)
    print(f"Saved training history to: {history_path}")

# ----------------------
# 8. 可视化函数
# ----------------------
def plot_training_history(history, save_dir):
    """绘制详细的训练历史"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 损失和准确率图
    plt.figure(figsize=(15, 10))
    
    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['valid_loss'], label='Valid')
    plt.title('Training/Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['valid_acc'], label='Valid')
    plt.title('Training/Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # 学习率曲线
    plt.subplot(2, 2, 3)
    plt.plot(history['learning_rates'])
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    
    # 每类准确率曲线
    plt.subplot(2, 2, 4)
    for i, class_acc in enumerate(zip(*history['class_accuracies'])):
        plt.plot(class_acc, label=f'Class {i}')
    plt.title('Per-Class Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_metrics_{timestamp}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 混淆矩阵
    if history['final_confusion_matrix'] is not None:
        plt.figure(figsize=(10, 8))
        plt.imshow(history['final_confusion_matrix'], cmap='Blues')
        plt.colorbar()
        
        # 添加数值标注
        for i in range(config['num_classes']):
            for j in range(config['num_classes']):
                plt.text(j, i, history['final_confusion_matrix'][i, j].item(),
                        ha='center', va='center')
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(save_dir, f'confusion_matrix_{timestamp}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

# ----------------------
# 9. 程序入口
# ----------------------
if __name__ == "__main__":
    main()