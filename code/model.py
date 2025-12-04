"""
模型定义模块

该模块提供了多种预训练模型的创建和配置功能，包括 ResNet、DINOv2、EfficientNet、
ConvNeXt、ViT 和 Swin Transformer 等架构。
"""

import torch.nn as nn
import torchvision.models as models
from torchvision.models import ConvNeXt_Base_Weights
from transformers import AutoModel, AutoConfig


class DinoV2Classifier(nn.Module):
    """
    DINOv2 图像分类器
    
    基于 DINOv2 预训练模型的分类器，使用 [CLS] token 进行分类。
    
    Args:
        model_name: DINOv2 模型名称或路径
        num_classes: 分类类别数
        predicted: 是否使用预训练权重，默认为 True
    """
    
    def __init__(self, model_name, num_classes, predicted=True):
        super().__init__()
        if predicted:
            self.features = AutoModel.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.features = AutoModel.from_config(config)

        hidden_size = 768
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像张量
            
        Returns:
            分类 logits
        """
        outputs = self.features(x)
        # DINOv2 的输出包含 last_hidden_state
        # 取 [CLS] token，即序列的第一个 token
        cls_token = outputs.last_hidden_state[:, 0, :]
        out = self.head(cls_token)
        return out


def create_model(num_classes, model_name='resnet50', predicted=True):
    """
    创建指定架构的分类模型
    
    支持多种预训练模型架构，并自动配置分类头以适应目标类别数。
    
    Args:
        num_classes: 目标分类类别数
        model_name: 模型架构名称，默认为 'resnet50'
        predicted: 是否使用预训练权重，默认为 True
        
    Returns:
        配置好的 PyTorch 模型
    """
    
    # ==================== ResNet50 ====================
    if model_name == 'resnet50':
        model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True

    # ==================== DINOv2 ====================
    elif model_name == 'facebook/dinov2-base':
        model = DinoV2Classifier(
            model_name='../models--facebook--dinov2-base/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415',
            num_classes=num_classes,
            predicted=predicted
        )

    # ==================== EfficientNet ====================
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # ==================== ConvNeXt ====================
    elif model_name == 'convnext_small':
        model = models.convnext_small(weights='ConvNeXt_Small_Weights.IMAGENET1K_V1')
        num_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_features, num_classes)

    elif model_name == 'convnext_tiny':
        model = models.convnext_tiny(weights='ConvNeXt_Tiny_Weights.IMAGENET1K_V1')
        num_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_features, num_classes)

    elif model_name == 'convnext_base':
        model = models.convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        num_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_features, num_classes)

    # ==================== Vision Transformer ====================
    elif model_name == 'vit_b_16':
        model = models.vit_b_16(weights='ViT_B_16_Weights.IMAGENET1K_V1')
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    # ==================== Swin Transformer ====================
    elif model_name == 'swin_v2_t':
        model = models.swin_v2_t(weights='Swin_V2_T_Weights.IMAGENET1K_V1')
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)

    elif model_name == 'swin_v2_s':
        model = models.swin_v2_s(weights='Swin_V2_S_Weights.IMAGENET1K_V1')
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)

    elif model_name == 'swin_v2_b':
        model = models.swin_v2_b(weights='Swin_V2_B_Weights.IMAGENET1K_V1')
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)

    # ==================== 默认模型 ====================
    else:
        # 默认使用resnet50
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model