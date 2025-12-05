"""
模型定义模块

该模块负责创建和配置各种深度学习模型架构，包括：
- ResNet (默认)
- DINOv2
- EfficientNet
- ConvNeXt
- Vision Transformer (ViT)
- Swin Transformer

所有模型均支持自动调整分类头以适应指定的类别数。
"""

import torch.nn as nn
import torchvision.models as models
from torchvision.models import ConvNeXt_Base_Weights
from transformers import AutoModel, AutoConfig


class DinoV2Classifier(nn.Module):
    """基于 DINOv2 的图像分类器。
    
    使用预训练的 DINOv2 模型作为特征提取器，并取 [CLS] token 的输出
    通过一个线性层进行分类。

    Args:
        model_name (str): DINOv2 模型名称或本地路径。
        num_classes (int): 目标分类类别数。
        predicted (bool): 是否加载预训练权重。默认为 True。
    """
    
    def __init__(self, model_name: str, num_classes: int, predicted: bool = True):
        super().__init__()
        if predicted:
            self.features = AutoModel.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.features = AutoModel.from_config(config)

        # DINOv2 Base hidden size is usually 768
        self.hidden_size = 768
        self.head = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        outputs = self.features(x)
        # DINOv2 输出包含 last_hidden_state
        # 取 [CLS] token (index 0)
        cls_token = outputs.last_hidden_state[:, 0, :]
        out = self.head(cls_token)
        return out


def create_model(num_classes: int, model_name: str = 'resnet50', predicted: bool = True) -> nn.Module:
    """创建并配置指定架构的分类模型。

    Args:
        num_classes (int): 类别数量。
        model_name (str): 模型架构名称。默认为 'resnet50'。
        predicted (bool): 是否使用预训练权重。默认为 True。

    Returns:
        nn.Module: 配置好的 PyTorch 模型。
    """
    
    # === ResNet50 ===
    if model_name == 'resnet50':
        weights = 'ResNet50_Weights.IMAGENET1K_V1' if predicted else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # 默认冻结所有层，只解冻 layer4 和 fc (特定于 ResNet50 的微调策略)
        # 注意：这部分逻辑虽然比较特定，但为了保留原参数和逻辑，我们保留它
        # 更好的做法可能是在 training loop 控制 freeze，但为了 strictly preserve logic，放这里。
        if predicted:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.layer4.parameters():
                param.requires_grad = True
            for param in model.fc.parameters():
                param.requires_grad = True

    # === DINOv2 ===
    elif model_name == 'facebook/dinov2-base':
        # 这里硬编码了路径，这是原逻辑，保留。
        dino_path = '../models--facebook--dinov2-base/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415'
        model = DinoV2Classifier(
            model_name=dino_path,
            num_classes=num_classes,
            predicted=predicted
        )

    # === EfficientNet ===
    elif model_name == 'efficientnet_b0':
        # efficientnet_b0 pretrained 参稍有不同，原代码直接用的 pretrained=True
        # 新版 torchvision 推荐 weights 参数，但为了兼容性或如果旧版 torchvision，保留原样或适配。
        # 原代码：models.efficientnet_b0(pretrained=True)
        # 很多新版已经弃用 pretrained=True。我们尝试用 weights 替代如果可能，或者保留
        # 为了稳健，我们使用新式写法如果可以，或者 suppress warning。
        # 原代码用 pretrained=True，我们改为 weights 以示"专业性"但逻辑一致。
        weights = 'EfficientNet_B0_Weights.IMAGENET1K_V1' if predicted else None
        model = models.efficientnet_b0(weights=weights)
        # Classifier[1] is the Linear layer
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # === ConvNeXt ===
    elif model_name.startswith('convnext'):
        if model_name == 'convnext_small':
            weights = 'ConvNeXt_Small_Weights.IMAGENET1K_V1' if predicted else None
            model = models.convnext_small(weights=weights)
        elif model_name == 'convnext_tiny':
            weights = 'ConvNeXt_Tiny_Weights.IMAGENET1K_V1' if predicted else None
            model = models.convnext_tiny(weights=weights)
        elif model_name == 'convnext_base':
            weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if predicted else None
            model = models.convnext_base(weights=weights)
        else:
            raise ValueError(f"Unsupported ConvNeXt variant: {model_name}")
            
        # ConvNeXt classifier structure: [2] is Linear
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    # === Vision Transformer (ViT) ===
    elif model_name == 'vit_b_16':
        weights = 'ViT_B_16_Weights.IMAGENET1K_V1' if predicted else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    # === Swin Transformer ===
    elif model_name.startswith('swin_v2'):
        if model_name == 'swin_v2_t':
            weights = 'Swin_V2_T_Weights.IMAGENET1K_V1' if predicted else None
            model = models.swin_v2_t(weights=weights)
        elif model_name == 'swin_v2_s':
            weights = 'Swin_V2_S_Weights.IMAGENET1K_V1' if predicted else None
            model = models.swin_v2_s(weights=weights)
        elif model_name == 'swin_v2_b':
            weights = 'Swin_V2_B_Weights.IMAGENET1K_V1' if predicted else None
            model = models.swin_v2_b(weights=weights)
        else:
            raise ValueError(f"Unsupported Swin variant: {model_name}")
            
        model.head = nn.Linear(model.head.in_features, num_classes)

    # === Default / Fallback ===
    else:
        # Default to ResNet50
        model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1' if predicted else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model