"""
通用工具模块

该模块提供了项目所需的通用功能，包括：
1. FlowerDataset: 统一的数据集类，支持训练（DataFrame）和推理（图片路径列表）模式。
2. 数据划分: 保持原有的分层划分逻辑。
3. 配置加载: 标准化的配置加载函数。
"""

import os
import json
import logging
from typing import List, Tuple, Dict, Any, Optional, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = "../model/config.json") -> Dict[str, Any]:
    """加载模型配置文件。

    Args:
        config_path (str): 配置文件路径，默认为 "../model/config.json"。

    Returns:
        Dict[str, Any]: 配置字典。
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        logger.error(f"配置文件未找到: {config_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"配置文件格式错误: {config_path}")
        raise


class FlowerDataset(Dataset):
    """通用花卉数据集类。

    该类统一了训练和预测的数据加载逻辑。
    - 训练模式: 传入 pandas DataFrame (包含 'filename' 和 'category_id')。
    - 预测模式: 传入图片文件名列表 (List[str])。

    Args:
        data_source (Union[pd.DataFrame, List[str]]): 数据源。
            - 如果是 DataFrame，必须包含 'filename' 和 'category_id' 列。
            - 如果是列表，则是文件名字符串列表。
        img_dir (str): 图片所在的根目录路径。
        transform (Optional[transforms.Compose]): 图片预处理转换操作。
    """

    def __init__(self, 
                 data_source: Union[pd.DataFrame, List[str]], 
                 img_dir: str, 
                 transform: Optional[transforms.Compose] = None):
        self.data_source = data_source
        self.img_dir = img_dir
        self.transform = transform
        
        # 判断模式：如果数据源是 DataFrame，则为训练模式（有标签）
        self.is_train = isinstance(data_source, pd.DataFrame)
        
        # 缓存配置以加速标签查找 (仅训练模式需要)
        self.config = load_config() if self.is_train else None

    def __len__(self) -> int:
        return len(self.data_source)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, str]]:
        """获取单个样本。

        Returns:
            训练模式: (image_tensor, label_int)
            预测模式: (image_tensor, filename_str)
        """
        if self.is_train:
            # DataFrame 模式
            row = self.data_source.iloc[idx]
            filename = row['filename']
        else:
            # 列表模式
            filename = self.data_source[idx]

        img_path = os.path.join(self.img_dir, filename)

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"无法读取图片或图片损坏: {img_path}, 错误: {e}")
            # 返回一个全黑的 dummy 图片防止程序崩溃
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        if self.is_train:
            # 获取标签
            # 根据原代码逻辑，这里返回的是 mapped label
            # 确保返回 int 类型以供 CrossEntropyLoss 使用
            try:
                label = int(self.config['id_label'][str(row['category_id'])])
            except (KeyError, ValueError) as e:
                logger.error(f"标签映射失败: category_id={row['category_id']}, error={e}")
                label = 0 # Fallback
            return image, label
        else:
            # 预测模式返回文件名，方便生成结果
            return image, filename


def split_data(csv_path: str, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按类别分层划分数据集为训练集和验证集。

    Args:
        csv_path (str): 原始 CSV 数据文件的路径。
        test_size (float): 验证集所占比例，默认为 0.2。

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (训练集 DataFrame, 验证集 DataFrame)。
    """
    df = pd.read_csv(csv_path)
    
    train_data_list = []
    val_data_list = []
    
    # 获取所有唯一类别
    categories = df['category_id'].unique()
    
    for category in categories:
        category_data = df[df['category_id'] == category]
        
        # 确保至少有一个验证样本
        n_val = max(1, int(len(category_data) * test_size))
        
        # 随机采样验证集
        val_sample = category_data.sample(n=n_val, random_state=42)
        train_sample = category_data.drop(val_sample.index)
        
        val_data_list.append(val_sample)
        train_data_list.append(train_sample)
    
    # 合并并重置索引
    train_df = pd.concat(train_data_list).reset_index(drop=True)
    val_df = pd.concat(val_data_list).reset_index(drop=True)
    
    logger.info(f"数据划分完成 - 训练集: {len(train_df)} 张, 验证集: {len(val_df)} 张")
    
    return train_df, val_df
