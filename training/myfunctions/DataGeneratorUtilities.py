
# ----- Other Libraries _____
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import time
import os
import gc
import cv2
import copy
import time
import torch
import random
import string
import joblib
import tifffile
import numpy as np 
import pandas as pd 
import torch.nn as nn
import seaborn as sns
from random import randint
from torchvision import models
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pickle
import warnings; warnings.filterwarnings("ignore")
gc.enable()



# ------- Seeding __________ DO NOT CHANGE THESE!!!!!
def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
seed = 42
seed_everything(seed)






# ______________ DataFrames Functions ________________

def label_encoder(raw_label_names, train_labels):
    train_labels_ind = np.copy(train_labels)
    l_int = int(0)
    for l_name in raw_label_names:
        ids = np.argwhere(train_labels == l_name)
        train_labels_ind[ids] = l_int
        l_int += 1
        print (str(l_name), len(ids))
    return train_labels_ind



def replace_img_paths_format (in_df, in_format, out_format, in_dir, out_dir):
    out_df = [] 
    for path in in_df:
        path = path.replace(in_format, out_format)
        path = path.replace(in_dir, out_dir)
        out_df.append(path)
    return out_df
    

def Balance_Train_Dataframe (train_df):
    max_count = max(train_df.label.value_counts())
    for label in train_df.label.unique():
        df = train_df.loc[train_df.label == label]
        while(train_df.label.value_counts()[label] < max_count):
            train_df = pd.concat([train_df, df.head(max_count - train_df.label.value_counts()[label])], axis = 0)
    return train_df


def Enchance_Train_Dataframe (df):
    def enhance_df(df):
        df["image_size"]   = df.image_path.apply(lambda x: Image.open(x).size)
        df["image_pixels"] = df["image_size"].apply(lambda x: int(x[0]*int(x[1])))    
        df["image_width"]  = df["image_size"].apply(lambda x: int(x[0]))
        df["image_height"] = df["image_size"].apply(lambda x: int(x[1]))
        df["aspect_ratio"] = df["image_width"]/df["image_height"]
        return df

    cols = df.columns
    # Getting image paths into the df
    df['image_path'] = df.image_id.apply(lambda x: os.path.join("../input/mayo-clinic-strip-ai/train/", x+".tif"))
    df=enhance_df(df)
    print('enhanced train dataframe ready')
    #train_labels, train_paths = train_csv.label, train_csv.image_path
    return df




# ______________ Data Loaders Functions ________________
class ImgDataset(Dataset):
    def __init__(self, df, img_size):
        self.df = df
        self.img_size = img_size  
    def __len__(self): return len(self.df)    
    def __getitem__(self, index):
        image = cv2.imread(self.df.iloc[index].image_path)
        if len(image.shape) == 5:
            image = image.squeeze().transpose(1, 2, 0)
        image = cv2.resize(image, (self.img_size, self.img_size)).transpose(2, 0, 1)
        l = self.df.iloc[index].label
        return image, l