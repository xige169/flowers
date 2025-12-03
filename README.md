# 花卉分类项目

基于 DINOv2 的花卉图像分类系统

## 项目简介

本项目使用深度学习技术对花卉图像进行分类，支持多种预训练模型架构，包括：
- DINOv2
- ResNet50
- EfficientNet
- ConvNeXt (Tiny/Small/Base)
- Vision Transformer (ViT)
- Swin Transformer V2 (T/S/B)

## 项目结构

```
submission/
├── code/                  # 源代码
│   ├── model.py          # 模型定义
│   └── requirements.txt  # Python 依赖
├── model/                # 模型配置
│   └── config.json       # 配置文件
└── train_labels.csv      # 训练标签
```

## 环境要求

- Python 3.8+
- PyTorch 
- torchvision
- transformers
- pandas
- PIL

安装依赖：
```bash
pip install -r code/requirements.txt
```

## 使用说明

### 模型配置

编辑 `model/config.json` 配置文件，设置模型参数和路径。

### 创建模型

```python
from code.model import create_model

# 创建 DINOv2 模型
model = create_model(
    num_classes=176,  # 类别数
    model_name='facebook/dinov2-base',
    predicted=True
)
```

## 支持的模型

| 模型名称 | 描述 |
|---------|------|
| `resnet50` | ResNet-50 |
| `facebook/dinov2-base` | DINOv2 Base |
| `efficientnet_b0` | EfficientNet-B0 |
| `convnext_tiny` | ConvNeXt Tiny |
| `convnext_small` | ConvNeXt Small |
| `convnext_base` | ConvNeXt Base |
| `vit_b_16` | Vision Transformer Base |
| `swin_v2_t` | Swin Transformer V2 Tiny |
| `swin_v2_s` | Swin Transformer V2 Small |
| `swin_v2_b` | Swin Transformer V2 Base |

## 许可证

本项目仅供学习和研究使用。
