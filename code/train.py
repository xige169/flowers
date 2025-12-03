import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from model import create_model
from utils import FlowerDataset, split_data
from utils import load_config

config = load_config()
# 基本设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def set_train_layers(model, stage=1):
    backbone = model.features

    if stage == 1:
        # === 阶段1: 冻结所有层，只训练 Head ===
        for param in backbone.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True
        print("阶段1: 冻结 Backbone，只训练分类头")

    elif stage == 2:
        # 先冻结所有
        for param in backbone.parameters():
            param.requires_grad = False

        transformer_layers = getattr(backbone, 'layer', None)

        # 如果没找到，尝试找 encoder.layer (DINOv2 标准结构)
        if transformer_layers is None and hasattr(backbone, 'encoder'):
            transformer_layers = getattr(backbone.encoder, 'layer', None)

        if transformer_layers is not None:
            total_layers = len(transformer_layers)
            # 解冻最后 3 层
            start_layer = max(0, total_layers - 3)

            print(f"阶段2: 解冻 Transformer Layer {start_layer} 到 {total_layers - 1}")
            for i in range(start_layer, total_layers):
                for param in transformer_layers[i].parameters():
                    param.requires_grad = True
        else:
            print("警告: 没找到 Transformer 层，尝试解冻最后部分参数")
            # 兜底策略：如果都找不到，解冻所有参数
            for param in backbone.parameters():
                param.requires_grad = True

        # 确保 Head 始终解冻
        for param in model.head.parameters():
            param.requires_grad = True

    else:
        # === 阶段3: 全量微调 ===
        for param in model.parameters():
            param.requires_grad = True
        print("阶段3: 全解冻")

    return

def load_dataset(batch_size):
    # 数据路径
    csv_path = config['csv_path']
    img_dir = config['img_dir']
    train_df, val_df = split_data(csv_path)
    # 数据变换
    train_transform = transforms.Compose([
        transforms.Resize((config['input_size'], config['input_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((config['input_size'], config['input_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 创建数据集
    train_dataset = FlowerDataset(train_df, img_dir, train_transform)
    val_dataset = FlowerDataset(val_df, img_dir, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def train(model, criterion, stage, base_lr, best_acc):
    if stage==1:
        lr=base_lr * 0.5
        epochs =25
        batch = 16
    elif stage ==2:
        lr = base_lr * 0.1
        epochs = 13
        batch = 8
    else:
        lr = base_lr*0.01
        epochs = 30
        batch = 8

    train_loader, val_loader = load_dataset(batch)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.1) 

    # 训练
    for epoch in range(epochs):
        # ===== 训练阶段 =====
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} Train'):
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
        train_acc = 100 * train_correct / train_total
        train_loss = train_loss / train_total

        # ===== 验证阶段 =====
        model.eval()
        val_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Val'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_acc = 100 * val_correct / test_total

        print(f'Epoch {epoch + 1}: Train loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs('../model', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_classes': config['num_classes'],
                'model_name': config['model_name'],
                'best_val_acc': best_acc
            }, config['save_model_path'])
            print(f'保存最佳模型，验证准确率: {best_acc:.2f}%')

        scheduler.step()
    return best_acc

def train_model():
    # 划分数据

    num_classes = config['num_classes']

    # 创建模型
    model_name = config['model_name']
    model = create_model(num_classes, model_name,predicted=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing= 0.1)

    best_acc = 0.0
    for i in range(2,3):
        stage = i
        set_train_layers(model,stage)
        best_acc = train(model, criterion, stage, config['learning_rate'], best_acc)


if __name__ == '__main__':
    train_model()