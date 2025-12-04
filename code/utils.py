"""
工具模块

该模块提供了数据集处理、数据划分和配置加载等实用功能。
"""

import os
import json
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class FlowerDataset(Dataset):
    """
    花卉图片数据集
    
    用于训练和验证的数据集类，支持数据增强和标签转换。
    
    Args:
        data_df: 包含图片文件名和类别的 DataFrame
        img_dir: 图片目录路径
        transform: 数据变换/增强操作
    """
    
    def __init__(self, data_df, img_dir, transform=None):
        self.data_df = data_df
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f'损坏图片路径:{img_path}')
        
        if self.transform:
            image = self.transform(image)
        
        config = load_config()
        label = config['id_label'][str(row['category_id'])]
        return image, label


def split_data(csv_path, test_size=0.2):
    """
    按类别分层划分数据集
    
    将数据集划分为训练集和验证集，确保每个类别都有代表性的样本在验证集中。
    
    Args:
        csv_path: CSV 文件路径
        test_size: 验证集比例，默认为 0.2
        
    Returns:
        train_df: 训练集 DataFrame
        val_df: 验证集 DataFrame
    """
    df = pd.read_csv(csv_path)
    
    # 按类别分层划分
    train_data = []
    val_data = []
    
    for category in df['category_id'].unique():
        category_data = df[df['category_id'] == category]
        n_val = max(1, int(len(category_data) * test_size))
        
        val_data.append(category_data.sample(n=n_val, random_state=42))
        train_data.append(category_data.drop(val_data[-1].index))
    
    train_df = pd.concat(train_data).reset_index(drop=True)
    val_df = pd.concat(val_data).reset_index(drop=True)
    
    return train_df, val_df


def load_config(config_path="../model/config.json"):
    """
    加载模型配置文件
    
    Args:
        config_path: 配置文件路径，默认为 "../model/config.json"
        
    Returns:
        config: 配置字典
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config
