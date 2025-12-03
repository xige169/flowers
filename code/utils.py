import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import json
import os

class FlowerDataset(Dataset):
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
    """简单划分数据"""
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
    """加载模型配置"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config

