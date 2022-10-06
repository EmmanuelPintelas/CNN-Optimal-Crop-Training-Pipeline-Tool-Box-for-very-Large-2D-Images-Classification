import os
import gc
import cv2
import copy
import time
import random
import string
import joblib
import tifffile
import numpy as np 
import pandas as pd 
import torch
from torch import nn
import seaborn as sns
from torchvision import models
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from torch.optim import lr_scheduler
import warnings
warnings.filterwarnings("ignore")
gc.enable()
import os
import pandas as pd
import numpy as np
# import skimage.io as io
from openslide import OpenSlide
import cv2
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
print('libraries imported')
from skimage import io
#! pip install timm
#import timm
import torch
import torch.nn as nn
from torch.nn import Sequential, AdaptiveAvgPool2d, Identity, Module
from typing import Iterable
from torch.nn.modules.flatten import Flatten
#from timm.models.resnet import ResNet
import pickle
import time
import random
from time import gmtime, strftime
import requests
from skimage import exposure
from skimage.transform import resize
#import ot
from torchvision import transforms
from torch import optim
import torch.nn.functional as F
from typing import Iterable, Any, Tuple, List
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier # PassiveAggressiveClassifier(max_iter=1000, random_state=0) # 0.508
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import tensorflow as tf
from skimage.color import rgb2gray
from skimage import exposure
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
print('libraries imported')


# ----- External Imports _____
import sys
sys.path.append('../input/einops')
sys.path.append('../input/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master')
###from einops import rearrange
###from einops.layers.torch import Rearrange
from efficientnet_pytorch import EfficientNet


print('libraries imported')






# ------- Seeding __________
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

# ------- Device __________
DEVICE = torch.device('cuda') #  





def normalize_preds (preds):
    min_pr = preds.min()
    if min_pr < 0:
        preds = preds - min_pr
        preds = preds/preds.max()
    else:
        preds = preds/preds.max()
    return preds


def Slide_Open_Resize(path_image):
    
    try:
        image_resized = np.array(cv2.imread(path_image))
        print ('common')
  
    except:
        print("Image too big!!!, slide approach used instead") 
        im_data = OpenSlide(path_image)

        W_init, H_init = im_data.dimensions
        print(W_init, H_init)
        if H_init > 50000 or W_init > 50000:
            ratio = 10 # desired resizing scale e.g.: resized_image = init_image/ratio
                        # this is the pre-init resizing in order to store and handle the image,
                        # at second step, e.g. when need to feed a CNN model, then we can resize again
                        # in the desired CNN input dimensions
            tiles_per_side = 5
            print('ratio: ', ratio)
        elif  H_init > 25000 or W_init > 25000:
            ratio = 5
            tiles_per_side = 5
            print('ratio: ', ratio)
        elif  H_init > 10000 or W_init > 10000:
            ratio = 2
            tiles_per_side = 5
            print('ratio: ', ratio)
        else:
            ratio = 1
            tiles_per_side = 5
            print('ratio: ', ratio)


        tile_size = int(H_init/tiles_per_side), int(W_init/tiles_per_side)

        image_tiles = []
        for i in range(0,H_init-tile_size[0]+1,tile_size[0]):
                blank_row, row, blank_row_map = [], [], []
                for j in range(0,W_init-tile_size[1]+1,tile_size[1]):
                    tile = np.array(im_data.read_region((j,i),0, (tile_size[1], tile_size[0])))
                    ###plt.imshow(tile); plt.show()
                    tile = cv2.resize(tile, dsize = (int(tile_size[1]/ratio), int(tile_size[0]/ratio)), interpolation=cv2.INTER_CUBIC) # INTER_CUBIC   INTER_NEAREST 
                    row.append(tile)
                image_tiles.append(row)

        image_tiles= np.array(image_tiles)

        image_resized = []
        for i in range(np.shape(image_tiles)[0]):
                row = []
                for j in range(np.shape(image_tiles)[1]):
                        row.append(image_tiles[i,j])
                row = np.concatenate(row, axis = 1)
                image_resized.append(row)
        image_resized = np.concatenate(image_resized, axis = 0)
        
        image_resized = cv2.resize(image_resized, (1000, 1000))
        image_resized = image_resized[:,:,0:3]

    return image_resized



# def report_gpu(): 
#     print(torch.cuda.list_gpu_processes()) 
#     gc.collect() 
#     torch.cuda.empty_cache()



def blank_tile_detector (tile, r):
        av, std = np.mean(tile), np.std(tile)
        th = av*r# low: 0.05 - high: 0.35  # a high r can detect mainly the dense object areas, while a low r is mainly for tossing out the total blank areas
        if std <= th:                      # and keeping the less dense areas
            return -1 #blank tile detected
        else:
            return 0


def Slide_Cut_off_Resize(path_image, size):
    
    im_data = OpenSlide(path_image)
    
    W_init, H_init = im_data.dimensions
    if H_init > 50000 or W_init > 50000:
        ratio = 10 # desired resizing scale e.g.: resized_image = init_image/ratio
                    # this is the pre-init resizing in order to store and handle the image,
                    # at second step, e.g. when need to feed a CNN model, then we can resize again
                    # in the desired CNN input dimensions
        tiles_per_side = 6
        print('ratio: ', ratio)
    elif  H_init > 25000 or W_init > 25000:
        ratio = 5
        tiles_per_side = 6
        print('ratio: ', ratio)
    elif  H_init > 10000 or W_init > 10000:
        ratio = 2
        tiles_per_side = 6
        print('ratio: ', ratio)
    else:
        ratio = 1
        tiles_per_side = 5
        print('ratio: ', ratio)
    
    tile_size = int(H_init/tiles_per_side), int(W_init/tiles_per_side)

    image_tiles = []
    blank_map = []
    for i in range(0,H_init-tile_size[0]+1,tile_size[0]):
            blank_row, row, blank_row_map = [], [], []
            for j in range(0,W_init-tile_size[1]+1,tile_size[1]):
                tile = np.array(im_data.read_region((j,i),0, (tile_size[1], tile_size[0])))
                
#                 print('tile init')
#                 plt.imshow(tile); plt.show()
                
                blank_row_map.append(blank_tile_detector (tile, 0.05)) # <<<<<< blank tile detection

                
                tile_r = cv2.resize(tile, dsize = (int(tile_size[1]/ratio), int(tile_size[0]/ratio)), interpolation=cv2.INTER_CUBIC) # INTER_CUBIC   INTER_NEAREST 
                row.append(tile_r)
            
                #print('tile res')
                #plt.imshow(tile_r); plt.show()
            
                del tile # free memory
                gc.collect()  
                
                del tile_r # free memory
                gc.collect()  
            
            blank_map.append(blank_row_map)
            image_tiles.append(row)

            del row # free memory
            gc.collect() 
            
    blank_map = np.array(blank_map)
    image_tiles= np.array(image_tiles)

    # cut off function
    image_cut_off = []
    for i in range(np.shape(blank_map)[0]):
        if list(blank_map[i]).count(-1) != np.shape(blank_map)[1]:
            row = []
            for j in range(np.shape(blank_map)[1]):
                if list(blank_map[:,j]).count(-1) != np.shape(blank_map)[0]:
                    row.append(image_tiles[i,j])
            row = np.concatenate(row, axis = 1)
            image_cut_off.append(row)
            
            del row # free memory
            gc.collect() 
            
    image_cut_off = np.concatenate(image_cut_off, axis = 0)
    
    del image_tiles # free memory
    gc.collect()
    
    del im_data
    gc.collect()   
    
    ###report_gpu()
    
    image_cut_off = cv2.resize(image_cut_off, (size, size))
    image_cut_off = image_cut_off[:,:,0:3]
    
    return image_cut_off#, blank_map





def DataFrame_Creation ():
    def enhance_df(df):
        df["image_size"]   = df.image_path.apply(lambda x: Image.open(x).size)
        df["image_pixels"] = df["image_size"].apply(lambda x: int(x[0]*int(x[1])))    
        df["image_width"]  = df["image_size"].apply(lambda x: int(x[0]))
        df["image_height"] = df["image_size"].apply(lambda x: int(x[1]))
        df["aspect_ratio"] = df["image_width"]/df["image_height"]
        return df

    train_csv = pd.read_csv('../input/mayo-clinic-strip-ai/train.csv')
    cols = train_csv.columns
    # Getting image paths into the df
    train_csv['image_path'] = train_csv.image_id.apply(lambda x: os.path.join("../input/mayo-clinic-strip-ai/train/", x+".tif"))
    train_csv=enhance_df(train_csv)
    print('enhanced train dataframe ready')
    train_labels, train_paths = train_csv.label, train_csv.image_path
    return train_labels, train_paths, train_csv




def Test_DataFrame_Creation ():
    def enhance_df(df):
        df["image_size"]   = df.image_path.apply(lambda x: Image.open(x).size)
        df["image_pixels"] = df["image_size"].apply(lambda x: int(x[0]*int(x[1])))    
        #df["image_width"]  = df["image_size"].apply(lambda x: int(x[0]))
        #df["image_height"] = df["image_size"].apply(lambda x: int(x[1]))
        #df["aspect_ratio"] = df["image_width"]/df["image_height"]
        return df

    test_df = pd.read_csv("../input/mayo-clinic-strip-ai/test.csv")
    cols = test_df.columns
    # Getting image paths into the df
    test_df['image_path'] = test_df.image_id.apply(lambda x: os.path.join("../input/mayo-clinic-strip-ai/test/", x+".tif"))
    test_df=enhance_df(test_df)
    print('enhanced train dataframe ready')
    test_paths = test_df.image_path
    return test_df





    
def tiles_detector_extractor (img):
                    # detects only the most informative tiles!
                    stds, ids = [], []
                    for j in range(4):
                        for i in range(4):
                            tile = img[1000*i:1000*(i+1),1000*j:1000*(j+1)]
                            stds.append(np.std(tile))
                            ids.append([i,j])
                    ids, stds = np.array(ids), np.array(stds)

                    sort_stds = np.sort(stds)

#                     forth_id = ids[np.argwhere(stds == sort_stds[-4])[0][0]]
#                     third_id = ids[np.argwhere(stds == sort_stds[-3])[0][0]] 
#                     sec_id = ids[np.argwhere(stds == sort_stds[-2])[0][0]]
                    best_id = ids[np.argwhere(stds == sort_stds[-1])[0][0]]                        
                    #for idd in [forth_id, third_id, sec_id, best_id]:
                        #i,j  = idd[0], idd[1]
                        #tile = img[1000*i:1000*(i+1),1000*j:1000*(j+1)]
                        #plt.imshow(tile); plt.show()
                    i,j  = best_id[0], best_id[1]
                    best_tile = img[1000*i:1000*(i+1),1000*j:1000*(j+1)]
#                     i,j  = sec_id[0], sec_id[1]
#                     sec_tile = img[1000*i:1000*(i+1),1000*j:1000*(j+1)]
#                     i,j  = third_id[0], third_id[1]
#                     third_tile = img[1000*i:1000*(i+1),1000*j:1000*(j+1)]
#                     i,j  = forth_id[0], forth_id[1]
#                     forth_tile = img[1000*i:1000*(i+1),1000*j:1000*(j+1)]
                    
##                     plt.imshow(cv2.resize(img, (1000, 1000))); plt.show()
                    ##plt.imshow(best_tile); plt.show()
                    
                    return best_tile #, sec_tile, third_tile, forth_tile
        









class ImgDataset(Dataset):
    def __init__(self, df):
        self.df = df 
        self.train = 'label' in df.columns
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        if(generate_new):
            paths = ["../test/", "../train/"]
        else:
            paths = ["../input/jpg-images-strip-ai/test/", "../input/jpg-images-strip-ai/train/"]
        try:
            image = cv2.imread(paths[self.train] + self.df.iloc[index].image_id + ".jpg")
        except:
            image = np.zeros((3, 1000, 1000))
            print('error read')
        label = 0
        try:
            if len(image.shape) == 5:
                image = image.squeeze().transpose(1, 2, 0)
            image = cv2.resize(image, (1000, 1000)).transpose(2, 0, 1)
        except:
            image = np.zeros((3, 1000, 1000))
            print('error read 2')
        if(self.train):
            label = {"CE" : 0, "LAA": 1}[self.df.iloc[index].label]
        patient_id = self.df.iloc[index].patient_id
        
        return image, label, patient_id
    
    
    
def predict(model, dataloader):
    #model.cuda()
    model.eval()
    dataloader = dataloader
    outputs = []
    s = nn.Softmax(dim=1)
    ids = []
    for item in tqdm(dataloader, leave=False):
        patient_id = item[2][0]
        try:
            images = item[0].cuda().float()
            ids.append(patient_id)
            output = model(images)
            outputs.append(output.cpu()[:,:2][0].detach().numpy())
        except:
            ids.append(patient_id)
            outputs.append(s(torch.tensor([[1, 1]]).float())[0].detach().numpy())
    return np.array(outputs), ids



