"""
训练模块

该模块提供了模型训练的完整流程，包括：
1. 渐进式微调策略 (set_train_layers)。
2. 数据加载与预处理 (load_dataset)。
3. 训练与验证循环 (train)。
4. 模型保存与早停。
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import ImageFile
from typing import Tuple

# 防止截断图片报错
ImageFile.LOAD_TRUNCATED_IMAGES = True

from model import create_model
from utils import FlowerDataset, split_data, load_config

# 全局配置和变量
config = load_config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_train_layers(model: nn.Module, stage: int = 1) -> None:
    """设置模型的可训练层，实现渐进式微调。

    Args:
        model (nn.Module): 待训练的模型。
        stage (int): 训练阶段。
            - 1: 冻结 Backbone，只训练分类头 (Head)。
            - 2: 解冻最后几层 Transformer/Conv 层。
            - 3: 解冻所有参数 (全量微调)。
    """
    backbone = model.features
    
    # 辅助函数：冻结/解冻参数
    def freeze(module, requires_grad=False):
        for param in module.parameters():
            param.requires_grad = requires_grad

    if stage == 1:
        # === 阶段1: 冻结 Backbone，只训练 Head ===
        freeze(backbone, False)
        freeze(model.head, True)
        print("阶段1: 冻结 Backbone，只训练分类头")

    elif stage == 2:
        # === 阶段2: 解冻部分 Transformer 层 ===
        freeze(backbone, False) # 先全部冻结

        # 尝试查找 Transformer 层
        transformer_layers = getattr(backbone, 'layer', None)
        if transformer_layers is None and hasattr(backbone, 'encoder'):
            # DINOv2 / ViT 常见结构
            transformer_layers = getattr(backbone.encoder, 'layer', None)

        if transformer_layers is not None:
            total_layers = len(transformer_layers)
            # 解冻最后 3 层
            start_layer = max(0, total_layers - 3)
            print(f"阶段2: 解冻 Transformer Layer {start_layer} 到 {total_layers - 1}")
            
            for i in range(start_layer, total_layers):
                freeze(transformer_layers[i], True)
        else:
            print("警告: 没找到 Transformer 层，尝试解冻最后部分参数 (兜底策略)")
            # 兜底：解冻整个 backbone (或者可以做得更细致，但保留原逻辑是解冻所有)
            freeze(backbone, True)

        # Head 始终解冻
        freeze(model.head, True)

    else:
        # === 阶段3: 全量微调 ===
        freeze(model, True)
        print("阶段3: 全解冻")


def load_dataset(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """加载并预处理数据集。

    Args:
        batch_size (int): 批大小。

    Returns:
        Tuple[DataLoader, DataLoader]: (训练DataLoader, 验证DataLoader)。
    """
    csv_path = config['csv_path']
    img_dir = config['img_dir']
    
    train_df, val_df = split_data(csv_path)
    
    # 标准化参数
    normalize = transforms.Normalize([0.485, 0.456, 0.406], 
                                     [0.229, 0.224, 0.225])
    
    input_size = config['input_size']
    
    # 定义 Transform
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        normalize
    ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        normalize
    ])

    # 实例化数据集
    train_dataset = FlowerDataset(train_df, img_dir, train_transform)
    val_dataset = FlowerDataset(val_df, img_dir, val_transform)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def train(model: nn.Module, criterion: nn.Module, stage: int, base_lr: float, best_acc: float) -> float:
    """执行单个阶段的训练循环。

    Args:
        model (nn.Module): 模型。
        criterion (nn.Module): 损失函数。
        stage (int): 训练阶段 (1, 2, 3)。
        base_lr (float): 基础学习率。
        best_acc (float): 之前的最佳验证准确率。

    Returns:
        float: 更新后的最佳验证准确率。
    """
    # 根据阶段设置超参数
    if stage == 1:
        lr = base_lr * 0.5
        epochs = 25
        batch = 16
    elif stage == 2:
        lr = base_lr * 0.1
        epochs = 13
        batch = 8
    else: # stage 3
        lr = base_lr * 0.01
        epochs = 30
        batch = 8

    train_loader, val_loader = load_dataset(batch)
    
    # 优化器与调度器
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    for epoch in range(epochs):
        # ==================== Training ====================
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Stage {stage}] Train')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # 实时更新进度条显示 Loss
            pbar.set_postfix({'loss': loss.item()})

        train_acc = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / train_total

        # ==================== Validation ====================
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Val', leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100.0 * val_correct / val_total
        
        print(f'Stage {stage} Epoch {epoch + 1}: '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Acc: {val_acc:.2f}%')

        # ==================== Save Best Model ====================
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs('../model', exist_ok=True)
            save_path = config['save_model_path']
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_classes': config['num_classes'],
                'model_name': config['model_name'],
                'best_val_acc': best_acc,
                'stage': stage,
                'epoch': epoch
            }, save_path)
            
            print(f'>>> 新的最佳模型已保存! 验证准确率: {best_acc:.2f}%')

        scheduler.step()
        
    return best_acc


def train_model():
    """模型训练主入口。"""
    num_classes = config['num_classes']
    model_name = config['model_name']
    
    print(f"开始训练流程: Model={model_name}, Classes={num_classes}")

    # 初始化模型
    model = create_model(num_classes, model_name, predicted=True).to(device)
    
    # 标签平滑 CrossEntropy
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    
    # 按阶段依次训练
    for stage in range(1, 4):
        print(f"\n{'='*20} 进入第 {stage} 阶段训练 {'='*20}")
        set_train_layers(model, stage)
        best_acc = train(model, criterion, stage, config['learning_rate'], best_acc)
        
    print(f"\n所有阶段训练完成。最佳验证准确率: {best_acc:.2f}%")


if __name__ == '__main__':
    try:
        train_model()
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        raise