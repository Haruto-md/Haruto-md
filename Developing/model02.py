from time import time
import pandas as pd
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as T
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#current directoryを指定
os.chdir("C:/Users/西岡大翔/OneDrive - Kyushu University/デスクトップ/MIC/signate/tanatyo_20220731/Developing")

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_set_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(data_set_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_dir +"/"+ self.img_labels.iloc[idx][0]
        
        self.image = np.transpose(np.array(Image.open(img_path)),(2,0,1))
        self.image = torch.Tensor(self.image).float().to(device)
        
        if self.transform:
            self.image = self.transform(self.image)
        
        self.label = int(self.img_labels.iloc[idx][3])
        
        return self.image, self.label


#DataLoaderを設定。
Batch_Size = 64

#transformを設定
image_size = 512
t1 = T.Compose([T.Pad([int(image_size/4),0],padding_mode= "edge"), 
                T.Pad([0,int(image_size/4)],padding_mode= "edge"),
                T.Resize((image_size,image_size))])
com_t = T.Compose([t1,
                T.RandomAffine([-180,180],translate=(0.6,0.6)),
                T.Resize((image_size,image_size))])

train_data = ImageDataset("../datasets/train_data_set_1.csv","../downloaded/train",com_t)
train_dataLoader = torch.utils.data.DataLoader(train_data,batch_size=Batch_Size,shuffle=True)
eval_Data = ImageDataset("../datasets/eval_data_set_1.csv","../downloaded/train",t1)
eval_dataLoader = torch.utils.data.DataLoader(eval_Data,batch_size=Batch_Size,shuffle=False)


class MyNet_512(nn.Module):
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
    
class MyNet_1024(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.batch_norm = nn.BatchNorm2d(3)
        #3*1024*1024
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,16,25),
            #16*1000*1000
            nn.MaxPool2d(8,8),
            #16*125*125
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,64,27),
            #16*100*100
            nn.MaxPool2d(7,9),
            #64*11*11
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64,128,5),
            #128*9*9
            nn.MaxPool2d(3,3),
            #128*3*3
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.drop = nn.Dropout(p=0.3)
        #全結合
        self.fanc1 = nn.Linear(128*3*3,256)
        self.relu1 = nn.ReLU()
        self.fanc2 = nn.Linear(256,122)
        
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

#モデルを選択
model = MyNet_512()
model = model.to(device)


#https://venoda.hatenablog.com/entry/2020/10/03/075322
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.005)
# エポック数
num_epochs = 466

# 学習時と検証時で分けるためディクショナリを用意
dataloaders_dict = {
    'train': train_dataLoader,
    'val': eval_dataLoader
}


#train_losses, train_accuracies, val_losses, val_accuracies
learn_flow = [[],[],[],[]]
epoch_best_scores = [[0,0]]

start_epoch = 1

#前回の進捗をロード
pre_trained_model = "./savefiles/model02_latest.tar"
if os.path.isfile(pre_trained_model):
    print(f"loaded model:{pre_trained_model}")
    checkpoint = torch.load(pre_trained_model)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    start_epoch = epoch + 1
    if "learn_flow" in checkpoint:
        learn_flow = checkpoint["learn_flow"]
    if "epoch_best_scores" in checkpoint:
        epoch_best_scores = checkpoint["epoch_best_scores"]

for epoch in range(start_epoch,num_epochs+1):
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
            learn_flow[0].append(epoch_loss)
            learn_flow[1].append(epoch_acc.cpu())
        else:
            learn_flow[2].append(epoch_loss)
            learn_flow[3].append(epoch_acc.cpu())
            
            if (epoch_acc > epoch_best_scores[-1][0]):
                
                epoch_best_scores.append([epoch_acc,epoch])
                
                now_savefile_loc = f"./savefiles/model02_{epoch}_acc-{epoch_acc}.tar"
                
                if len(epoch_best_scores) > 10:
                    if os.path.isfile(f"./savefiles/model02_{epoch_best_scores[0][1]}_acc-{epoch_best_scores[0][0]}.tar"):
                        os.remove(f"./savefiles/model02_{epoch_best_scores[0][1]}_acc-{epoch_best_scores[0][0]}.tar")
                    del epoch_best_scores[0]
                    
                torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "learn_flow":learn_flow,
                    "epoch_best_scores":epoch_best_scores,
                },
                now_savefile_loc,
                )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "learn_flow":learn_flow,
                    "epoch_best_scores":epoch_best_scores,
                },
                f"./savefiles/model02_latest.tar",
                )

    lapsed_time = time()
    print("time:",time() - cur_time)

plt.plot(learn_flow[0],label = "train_loss")
plt.plot(learn_flow[2],label = "val_loss")
plt.show()
plt.plot(learn_flow[1],label = "train_acc")
plt.plot(learn_flow[3],label = "val_acc")
plt.show()