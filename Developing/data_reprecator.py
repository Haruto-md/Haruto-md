import numpy as np
from torchvision import transforms as T
import pandas as pd
from PIL import Image


path = "C:/Users/西岡大翔/OneDrive - Kyushu University/デスクトップ/MIC/signate/tanatyo_20220731/downloaded/train"

transform = T.Compose([
                        #正方形にする
                        T.Pad([512,0],padding_mode= "edge"),
                        T.Pad([0,512],padding_mode= "edge"),
                        #解像度を下げて、データ量を下げる。
                        T.Resize((1024,1024))
                                ])

t1 = T.Compose([T.Pad([256,0],padding_mode= "edge"),
                T.Pad([0,256],padding_mode= "edge"),
                T.Resize((1024,1024)),
                T.RandomAffine([-180,180],translate=(0.6,0.6)),T.Resize((1024,1024))])

def Aval(n):
    for i in range(n):
        B = list(range(n))
        A = Image.open(path+"/"+"000/0.jpg")
        B[i] = t1(transform(A))
        B[i].save(f"./arg_data/{i}.jpg")