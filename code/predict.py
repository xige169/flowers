import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from model import create_model
from utils import load_config
import torch.nn.functional as F
import argparse

class FlowerDataset(Dataset):
    def __init__(self, image_files, img_dir, transform=None):
        self.image_files = image_files
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.img_dir, filename)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f'损坏图片路径:{img_path}，文件名为：{filename}')

        if self.transform:
            image = self.transform(image)

        return image, filename


def load_model(model_path):
    """加载模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)

    config = load_config()
    model = create_model(checkpoint['num_classes'], checkpoint['model_name'],predicted=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, device


def predict_images(model_path, test_dir, output_path):
    """预测图片并保存结果"""
    # 加载模型
    model, device = load_model(model_path)

    # 获取测试图片列表
    image_files = [f for f in os.listdir(test_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 创建数据集和加载器
    dataset = FlowerDataset(image_files, test_dir, transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # 预测
    predictions = []
    filenames = []
    confidences = []
    config = load_config()

    with torch.no_grad():
        for images, batch_filenames in tqdm(loader, desc='预测中'):
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

            predicted = [config['label_id'][str(p.item())] for p in predicted]
            confidences.extend(confidence.cpu().numpy().tolist())
            predictions.extend(predicted)
            filenames.extend(batch_filenames)

    # 保存结果
    results_df = pd.DataFrame({
        'filename': filenames,
        'category_id': predictions,
        'confidence': confidences
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"预测完成，结果保存到: {output_path}")


if __name__ == '__main__':
    # 使用示例
    parser = argparse.ArgumentParser(description='花卉分类模型预测')

    # 位置参数：测试集文件夹和输出文件
    parser.add_argument('test_img_dir', type=str,
                        help='测试图片目录')
    parser.add_argument('output_path', type=str,
                        help='预测结果输出路径 (CSV文件)')
    args = parser.parse_args()

    model_path = '../model/best_model.pth'
    test_dir = args.test_img_dir
    output_path = args.output_path
    print(f'测试集目录: {args.test_img_dir}')
    print(f'输出文件: {args.output_path}')

    predict_images(model_path, test_dir, output_path)