# %%
#あらかたインポート
import pandas as pd
import numpy as np
import os
import json

os.chdir("C:/Users/西岡大翔/OneDrive - Kyushu University/デスクトップ/MIC/signate/tanatyo_20220731")
# %%
#train_abs_pathにはtrain(画像が格納されてるフォルダ）の絶対パスを入れてください。"\"は"/"に変えないとエラーが出ることがある。
train_abs_path = "C:/Users/西岡大翔/OneDrive - Kyushu University/デスクトップ/MIC/signate/tanatyo_20220731/downloaded/train"

def Get_Meta_Data(meta_data,type):
    
    color, category = meta_data[type]["color"],meta_data[type]["category"]
    
    return [color, category]
#pathsをつくる。
json_open = open("./downloaded/train_meta.json",'r')
meta_data = json.load(json_open)



# %%
df = pd.DataFrame()

for curDir, dirs, files in os.walk(train_abs_path):
    for file in files:
        df = pd.concat(axis=0,objs=[df,pd.DataFrame([[os.path.join(curDir[-3:],file), Get_Meta_Data(meta_data,curDir[-3:])[0],Get_Meta_Data(meta_data,curDir[-3:])[1] ,curDir[-3:]]],columns=["image_path","color","category","type"])])


# %%
df.index = np.arange(df["color"].count())

#csvファイルとして出力index=Falseにしないと、read_csvのときにunnamed列ができちゃう。
df.to_csv("data_set.csv",index=False)

# %%
df = pd.read_csv("./data_set.csv")
colors = []
for i in range(122):
    colors.append(Get_Meta_Data(meta_data,str(i).zfill(3))[0])
colors = set(colors)
colors = list(colors)

# %%
for i,color in enumerate(colors):
    df = df.replace(color,i)

# %%
df = df.replace("クレセント",0)
df = df.replace("戸車",1)

# %% [markdown]
# colorは['YW', 'YB', 'WM', 'CC', 'DG', 'A3', 'YK', 'E5', 'CH', 'RB', 'YS']のインデックスに変換（0-10）
# categoryは["クレセント","戸車"]のインデックスに変換(0-1)

# %%
df.to_csv("data_set_1.csv",index=False)


# %% [markdown]
# 訓練用と評価用に分ける。



# %%
data_path = "data_set_1.csv"
df = pd.read_csv(data_path)
df = df.sample(frac=1)

#train/train+evalを入力
train_eval_rate = 0.75

train_num=int(df["type"].count()*train_eval_rate)
train_df = df[:train_num]
eval_df = df[train_num:]
train_df.to_csv(f"train_{str(data_path)}",index=False)
eval_df.to_csv(f"eval_{str(data_path)}",index=False)