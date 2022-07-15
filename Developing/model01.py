from time import time
import pandas as pd
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_set_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(data_set_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_dir +"/"+ self.img_labels.iloc[idx][0]
        self.image = Image.open(img_path)
        if self.transform:
            self.image = self.transform(self.image)
        self.image = torch.Tensor(np.transpose(np.array(self.image),(2,0,1))).float()
        
        self.label = int(self.img_labels.iloc[idx][3])
        
        return self.image,self.label


#DataLoaderを設定。
Batch_Size = 64
#transformを設定
transform = transforms.Compose([
                                #正方形にする
                                transforms.Pad([504,0],padding_mode= "edge"),
                                #解像度を下げて、データ量を下げる。
                                transforms.Resize((512,512))
                                ])

train_data = ImageDataset("train_data_set.csv","./downloaded/train",transform)
train_dataLoader = torch.utils.data.DataLoader(train_data,batch_size=Batch_Size,shuffle=True)
eval_Data = ImageDataset("eval_data_set.csv","./downloaded/train",transform)
eval_dataLoader = torch.utils.data.DataLoader(eval_Data,batch_size=Batch_Size,shuffle=True)


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
        self.drop = nn.Dropout(p=0.3)
        #全結合
        self.fanc1 = nn.Linear(4*4*32,512)
        self.relu1 = nn.ReLU()
        self.fanc2 = nn.Linear(512,122)
        
    def forward(self,x):
        x = self.batch_norm(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.drop(x)
        x = self.flatten(x)
        x=self.relu1(self.drop(self.fanc1(x)))
        x=self.fanc2(x)
        return x
model = MyNet()
model = model.to(device)


#https://venoda.hatenablog.com/entry/2020/10/03/075322
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=0.005)
# エポック数
num_epochs = 200

# 学習時と検証時で分けるためディクショナリを用意
dataloaders_dict = {
    'train': train_dataLoader,
    'val': eval_dataLoader
}

#前回の進捗をロード
if os.path.isfile("model01_0708.tar"):
    checkpoint = torch.load("model01_0707.tar")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

cur_time = time()

for epoch in range(1,num_epochs+1):
    cur_time = time()
    print('-------------')
    print('Epoch {}/{}'.format(epoch, num_epochs))
    
    
    for phase in ['train', 'val']:
        
        if phase == 'train':
            # モデルを訓練モードに設定
            model.train()
        else:
            # モデルを推論モードに設定
            model.eval()
        
        # 損失和
        epoch_loss = 0.0
        # 正解数
        epoch_corrects = 0
        
        # DataLoaderからデータをバッチごとに取り出す
        for inputs, labels in dataloaders_dict[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # optimizerの初期化
            optimizer.zero_grad()
            
            # 学習時のみ勾配を計算させる設定にする
            with torch.set_grad_enabled(phase == 'train'):
                
                outputs = model(inputs)
                # 損失を計算
                loss = criterion(outputs, labels)
                
                # ラベルを予測
                _, preds = torch.max(outputs, 1)
                
                # 訓練時はバックプロパゲーション
                if phase == 'train':
                    # 逆伝搬の計算
                    loss.backward()
                    # パラメータの更新
                    optimizer.step()
                
                # イテレーション結果の計算
                # lossの合計を更新
                # PyTorchの仕様上各バッチ内での平均のlossが計算される。
                # データ数を掛けることで平均から合計に変換をしている。
                # 損失和は「全データの損失/データ数」で計算されるため、
                # 平均のままだと損失和を求めることができないため。
                epoch_loss += loss.item() * inputs.size(0)
                
                # 正解数の合計を更新
                epoch_corrects += torch.sum(preds == labels.data)

        # epochごとのlossと正解率を表示
        epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
        epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        
        if phase == "train":
            train_losses = train_losses + [epoch_loss]
            train_accuracies = train_accuracies + [epoch_acc.cpu()]
        else:
            val_losses = val_losses + [epoch_loss]
            val_accuracies = val_accuracies + [epoch_acc.cpu()]
    
    lapsed_time = time()
    print(lapsed_time - cur_time)
    
    torch.save(
    {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    "model01.tar",
    )

    # 学習途中の状態を読み込む。
    checkpoint = torch.load("model01.tar")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]

plt.plot(train_losses,label = "train_loss")
plt.plot(train_accuracies,label = "train_acc")
plt.plot(val_losses,label = "val_loss")
plt.plot(val_accuracies,label = "val_acc")
plt.show()