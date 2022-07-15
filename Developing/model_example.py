from urllib.request import ProxyBasicAuthHandler
import pandas as pd
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#current directoryを指定
os.chdir("C:/Users/西岡大翔/OneDrive - Kyushu University/デスクトップ/MIC/signate/tanatyo_20220731/Developing")

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.batch_norm = nn.BatchNorm2d(3)
        #3*512*512
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,8,15),
            #8*498*498
            nn.MaxPool2d(6,6),
            #8*83*83
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(8,16,7),
            #16*77*77
            nn.MaxPool2d(7,7),
            #16*11*11
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16,32,4),
            #32*8*8
            nn.MaxPool2d(2),
            #32*4*4
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.drop = nn.Dropout(p=0.5)
        #全結合
        self.fanc1 = nn.Linear(4*4*32,512)
        self.relu1 = nn.ReLU()
        self.fanc2 = nn.Linear(512,122)
        self.soft = nn.Softmax()
    def forward(self,x):
        #x = self.batch_norm(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.drop(x)
        #x = self.flatten(x)
        x = x.view(-1)
        x=self.relu1(self.drop(self.fanc1(x)))
        x=self.fanc2(x)
        return self.soft(x)
model = MyNet()
model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(),lr=0.001, momentum=0.9, weight_decay=0.005)

if os.path.isfile("model02_414_acc-0.22105263157894736.tar"):
    checkpoint = torch.load("model02_414_acc-0.22105263157894736.tar")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
model.eval()
transform = transforms.Compose([
                                #正方形にする
                                transforms.Pad([504,0],padding_mode= "edge"),
                                #解像度を下げて、データ量を下げる。
                                transforms.Resize((512,512))
                                ])

df = pd.read_csv("../datasets/eval_data_set_1.csv")
answers = []
probs = []

for idx,path in enumerate(df["image_path"]):
    img_path = "../downloaded/train" +"/"+ df.iloc[idx][0]
    image = Image.open(img_path)
    image = transform(image)
    image = np.transpose(np.array(image),(2,0,1))
    image = np.double(image)
    
    xmean = np.zeros((3,512,512))
    xmean[:,:,:] = image.mean(axis=(1,2))[:,None,None]
    xstd  = np.zeros((3,512,512))
    xstd[:,:,:] = np.std(image,axis=(1,2))[:,None,None] 
    image = (image-xmean)/xstd
    
    image = torch.Tensor(image).float()
    y_pred = model(image.to(device))
    prob = max(list(y_pred)).item()
    idx = list(y_pred).index(prob)
    print(f"確信度:{prob},type:{idx}")
    probs = probs + [prob]
    answers = answers + [idx]

df = pd.concat([df,pd.DataFrame({"probabilities":probs,"predictions":answers})],axis=1)
df["predictions"].value_counts()
df["type"].value_counts()
df.to_csv("RESULT.csv",index=False)