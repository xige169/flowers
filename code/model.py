import torch.nn as nn
import torchvision.models as models
from torchvision.models import ConvNeXt_Base_Weights
from transformers import AutoModel, AutoConfig


class DinoV2Classifier(nn.Module):
    def __init__(self, model_name, num_classes,predicted=True):
        super().__init__()
        if(predicted):
            self.features = AutoModel.from_pretrained(
                model_name,
            )
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.features = AutoModel.from_config(config)

        hidden_size = 768

        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        outputs = self.features(x)
        # DINOv2 的输出包含 last_hidden_state
        # 取 [CLS] token，即序列的第一个 token
        cls_token = outputs.last_hidden_state[:, 0, :]
        out = self.head(cls_token)
        return out

def create_model(num_classes, model_name='resnet50',predicted=True):
    """创建模型"""
    if model_name == 'resnet50':
        model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        for param in model.parameters():
            param.requires_grad=False
        for param in model.layer4.parameters():
            param.requires_grad =True
        for param in model.fc.parameters():
            param.requires_grad = True

    elif model_name == 'facebook/dinov2-base':
        model = DinoV2Classifier(
            model_name='../models--facebook--dinov2-base/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415',
            num_classes=num_classes,
            predicted=predicted
        )

    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name== 'convnext_small':
        model = models.convnext_small(weights= 'ConvNeXt_Small_Weights.IMAGENET1K_V1')
        num_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_features, num_classes)
        # print(model)
        # for param in model.parameters():
        #     param.requires_grad = False
        # for param in model.features[6:].parameters():
        #     param.requires_grad = True
        # for param in model.classifier.parameters():
        #     param.requires_grad = True

    elif model_name== 'convnext_tiny':
        model = models.convnext_tiny(weights= 'ConvNeXt_Tiny_Weights.IMAGENET1K_V1')
        num_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_features, num_classes)


    elif model_name == 'convnext_base':
        model = models.convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        num_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_features, num_classes)

    #     for param in model.features.parameters():
    #         param.requires_grad = False
        # for param in model.features[6:].parameters():
        #     param.requires_grad = True
        # for param in model.features.parameters():
        #      param.requires_grad =False
        # for param in model.classifier.parameters():
        #     param.requires_grad = True
    elif model_name == 'vit_b_16':
        model = models.vit_b_16(weights='ViT_B_16_Weights.IMAGENET1K_V1')
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    elif model_name == 'swin_v2_t':
        model = models.swin_v2_t(weights= 'Swin_V2_T_Weights.IMAGENET1K_V1')
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        # print(model)
        # for param in model.parameters():
        #     param.requires_grad = False
    elif model_name == 'swin_v2_s':
        model = models.swin_v2_s(weights= 'Swin_V2_S_Weights.IMAGENET1K_V1')
        in_features = model.head.in_features
        model.head = nn.Linear(in_features,num_classes)
        # print(model)
        # for param in model.parameters():
        #     param.requires_grad = False

    elif model_name == 'swin_v2_b':
        model = models.swin_v2_b(weights= 'Swin_V2_B_Weights.IMAGENET1K_V1')
        in_features = model.head.in_features
        model.head = nn.Linear(in_features,num_classes)
        # for param in model.parameters():
        #     param.requires_grad = False

    # elif model_name == 'maxvit_t':
    #     model = timm.create_model(model_name= 'maxvit_tiny_tf_512.in1k', pretrained= False, num_classes= num_classes)
    #     in_features = model.head.fc.in_features
    #     model.head.fc = nn.Linear(in_features, num_classes)
        # for param in model.parameters():
        #     param.requires_grad = False
    else:
        # 默认使用resnet50
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model