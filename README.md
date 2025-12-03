# 花卉分类项目

基于 DINOv2 的花卉图像分类系统

## 项目简介

本项目使用深度学习技术对花卉图像进行分类，支持多种预训练模型架构，包括：
- **DINOv2** - Meta 的自监督学习视觉模型
- **ResNet50** - 经典残差网络
- **EfficientNet** - 高效卷积神经网络
- **ConvNeXt** (Tiny/Small/Base) - 现代化卷积网络
- **Vision Transformer (ViT)** - 视觉Transformer
- **Swin Transformer V2** (T/S/B) - 层级视觉Transformer

## 项目结构

```
submission/
├── code/                  # 源代码
│   ├── model.py          # 模型定义
│   ├── train.py          # 训练脚本
│   ├── predict.py        # 预测脚本
│   ├── utils.py          # 工具函数
│   └── requirements.txt  # Python 依赖
├── model/                # 模型配置
│   └── config.json       # 配置文件
└── .gitignore           # Git 忽略规则
```

## 数据集格式

训练数据使用 CSV 格式的标签文件，示例如下：

**train_labels.csv**
```
filename	category_id	chinese_name	english_name
img_000051.jpg	164	紫叶竹节秋海棠（紫竹梅）	Tradescantia pallida
img_000052.jpg	164	紫叶竹节秋海棠（紫竹梅）	Tradescantia pallida
img_000053.jpg	164	紫叶竹节秋海棠（紫竹梅）	Tradescantia pallida
```

## 环境要求

- Python 3.8+
- PyTorch 
- torchvision
- transformers
- pandas
- PIL
- tqdm

安装依赖：
```bash
cd code
pip install -r requirements.txt
```

## 使用说明

### 1. 配置模型

编辑 `model/config.json` 配置文件，设置模型参数和路径：
```json
{
  "num_classes": 176,
  "model_name": "facebook/dinov2-base",
  "input_size": 600,
  "learning_rate": 0.001,
  ...
}
```

### 2. 训练模型

```bash
cd code
python train.py
```

训练脚本支持多阶段训练策略：
- **阶段1**: 冻结主干网络，只训练分类头
- **阶段2**: 解冻部分层进行微调
- **阶段3**: 全模型微调

### 3. 预测

```bash
python predict.py <测试图片目录> <输出CSV路径>
```

示例：
```bash
python predict.py ../test_dataset ../results/predictions.csv
```

### 4. 创建模型（代码示例）

```python
from model import create_model

# 创建 DINOv2 模型
model = create_model(
    num_classes=176,
    model_name='facebook/dinov2-base',
    predicted=True
)

# 创建其他模型
model = create_model(num_classes=176, model_name='convnext_base')
model = create_model(num_classes=176, model_name='swin_v2_b')
```

## 支持的模型

| 模型名称 | 描述 | 特点 |
|---------|------|-----|
| `resnet50` | ResNet-50 | 经典架构，稳定可靠 |
| `facebook/dinov2-base` | DINOv2 Base | 强大的自监督学习模型 |
| `efficientnet_b0` | EfficientNet-B0 | 高效的卷积网络 |
| `convnext_tiny` | ConvNeXt Tiny | 现代化卷积架构 |
| `convnext_small` | ConvNeXt Small | 中等规模模型 |
| `convnext_base` | ConvNeXt Base | 较大规模模型 |
| `vit_b_16` | Vision Transformer Base | Transformer架构 |
| `swin_v2_t` | Swin Transformer V2 Tiny | 层级Transformer |
| `swin_v2_s` | Swin Transformer V2 Small | 中等规模 |
| `swin_v2_b` | Swin Transformer V2 Base | 较大规模 |

## 训练策略

- **损失函数**: CrossEntropyLoss with Label Smoothing (0.1)
- **优化器**: AdamW
- **学习率调度**: CosineAnnealingLR
- **数据增强**: Resize, Normalize
- **验证策略**: 分层划分 (80/20)

## 预测输出

预测结果保存为 CSV 文件，格式如下：
```
filename,category_id,confidence
test_001.jpg,164,0.9856
test_002.jpg,23,0.8934
...
```

## 注意事项

1. 训练前请确保已下载预训练模型权重
2. 大型模型文件（.pth）不包含在仓库中，需要自行训练
3. 训练数据和测试数据不包含在仓库中
4. 建议使用 GPU 进行训练以加快速度

## 许可证

本项目仅供学习和研究使用。
