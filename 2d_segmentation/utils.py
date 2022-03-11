import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CityscapeDataset(Dataset):
    def __init__(self,img_dir):
        self.img_dir = img_dir
        self.img_fnms = os.listdir(self.img_dir)
        
        def genLabelModel():
            num_items=5000
            num_classes = 10
            color_arr = np.random.choice(range(256),3*num_items).reshape(-1,3)
            label_model = KMeans(n_clusters=num_classes)
            label_model.fit(color_arr)
            return label_model
        
        self.label_model = genLabelModel()
        
    def __len__(self):
        return len(self.img_fnms)
    
    def __getitem__(self,index):
        img_fnm = self.img_fnms[index]
        img_fp = os.path.join(self.img_dir,img_fnm)
        img = Image.open(img_fp).convert('RGB')
        img = np.array(img)
        cityscape,label = self.img_split(img)
        label_class = self.label_model.predict(label.reshape(-1,3)).reshape(256,256)
        cityscape = self.transform(cityscape)
        label_class = torch.Tensor(label_class).long()
        return cityscape,label_class
    
    def img_split(self,img):
        img = np.array(img)
        cityscape,label = img[:,:256,:], img[:,256:,:]
        return cityscape,label
    
    def transform(self,img):
        transforms_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
        ])
        return transforms_ops(img)
