
import json
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TEST_DATA_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

class EndoDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.transform = TEST_DATA_TRANSFORM

        labels = os.listdir(data_dir)
        self.data = []
        for label in labels:
            label_dir = os.path.join(data_dir, label)
            for file in os.listdir(label_dir):
                if file.endswith('.jpg'):
                    self.data.append({
                        'image': os.path.join(label_dir, file),
                        'label': label
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_item = self.data[idx]
        # if img_item["image_type"] == "Ultrasound":  # 去除ULT数据
        #     return self.__getitem__(idx + 1)
        image = Image.open(img_item['image']).convert('RGB')

        label_text = img_item['label']
        label = 0 if 'no' in label_text else 1 
        if self.transform:
            image = self.transform(image)
        return image, label