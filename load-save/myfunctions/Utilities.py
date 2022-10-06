import os
import gc
gc.enable()
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
from openslide import OpenSlide
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from skimage import io
#! pip install timm
#import timm
from torch.nn import Sequential, AdaptiveAvgPool2d, Identity, Module
from typing import Iterable
from torch.nn.modules.flatten import Flatten
from time import gmtime, strftime
import requests
from skimage import exposure
from torchvision import transforms
from torch import optim
import torch.nn.functional as F
from typing import Iterable, Any, Tuple, List
from skimage import exposure
print('libraries imported')






def Rotate (image):
    max_intensity = np.max(image)
    image = rotate(image, 90)
    image = image*max_intensity
    return image


def contrast (image):
    v_min, v_max = np.percentile(np.array(image), (5, 95))  #0.2 99.8
    image = exposure.rescale_intensity(np.array(image), in_range=(v_min, v_max))
    return image

def adjust_gamma (image):
    image = exposure.adjust_gamma(np.array(image), gamma=0.4, gain=0.9) # gamma=0.4, gain=0.9
    return image

def adapthist(image):
    #image = image.astype(float)
    max_intensity = np.max(image)
    image = exposure.equalize_adapthist(np.array(image))
    image = image*max_intensity
    return image




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
    
    #report_gpu()
    
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




def img_read_resize (df, index, dirs):

        img_id = df.iloc[index].image_id
        img_sz = df.iloc[index].image_pixels # --> we need this info (number of pixels) for the very large images, 
                                             #     in order to activate openslide read
        
        if img_sz <= 3928229285: # this is nearly the limit of tiff file read
            print(1)
            img = cv2.resize(tifffile.imread(dirs + img_id + ".tif"), (1000, 1000))
            
        else:
            print('raw image too big, slipe approach used instead')
            img = Slide_Cut_off_Resize(dirs + img_id + ".tif", 1000)

        return img




# Given a large image, we extract the four most informative tiles
# this approach can help a DL model to focus on local informative areas with much high resolution, 
# comparing if we fed it with the initial image

def most_informative_tiles_generator (df, index, dirs):
        # df : a dataframe which contains info of data like images id, path, etc.
        # dirs: the path of dir where the raw images are stores
        # index: the index corresponding to the loadded image
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

                    forth_id = ids[np.argwhere(stds == sort_stds[-4])[0][0]]
                    third_id = ids[np.argwhere(stds == sort_stds[-3])[0][0]] 
                    sec_id = ids[np.argwhere(stds == sort_stds[-2])[0][0]]
                    best_id = ids[np.argwhere(stds == sort_stds[-1])[0][0]]                        
                    #for idd in [forth_id, third_id, sec_id, best_id]:
                        #i,j  = idd[0], idd[1]
                        #tile = img[1000*i:1000*(i+1),1000*j:1000*(j+1)]
                        #plt.imshow(tile); plt.show()
                    i,j  = best_id[0], best_id[1]
                    best_tile = img[1000*i:1000*(i+1),1000*j:1000*(j+1)]
                    i,j  = sec_id[0], sec_id[1]
                    sec_tile = img[1000*i:1000*(i+1),1000*j:1000*(j+1)]
                    i,j  = third_id[0], third_id[1]
                    third_tile = img[1000*i:1000*(i+1),1000*j:1000*(j+1)]
                    i,j  = forth_id[0], forth_id[1]
                    forth_tile = img[1000*i:1000*(i+1),1000*j:1000*(j+1)]
                    return best_tile, sec_tile, third_tile, forth_tile
        
        img_id = df.iloc[index].image_id
        img_sz = df.iloc[index].image_pixels # --> we need this info (number of pixels) for the very large images, 
                                             #     in order to activate openslide read
        
        if img_sz <= 3928229285: # this is nearly the limit of tiff file read
            print(1)
            img = cv2.resize(tifffile.imread(dirs + img_id + ".tif"), (4096, 4096))
            #
            best_tile, sec_tile, third_tile, forth_tile = tiles_detector_extractor (img)
        else:
            print('raw image too big, slipe approach used instead')
            img = Slide_Cut_off_Resize(dirs + img_id + ".tif", 4096)
            best_tile, sec_tile, third_tile, forth_tile = tiles_detector_extractor (img)
            
            
#         plt.imshow(cv2.resize(img, (1000, 1000))); plt.show()
#         plt.imshow(best_tile); plt.show()
#         plt.imshow(sec_tile); plt.show()
#         plt.imshow(third_tile); plt.show()
#         plt.imshow(forth_tile); plt.show()
#         cv2.imwrite("init.jpg", cv2.resize(img, (1000, 1000)))
#         cv2.imwrite("best_tile.jpg", best_tile)
#         cv2.imwrite("sec_tile.jpg", sec_tile)
#         cv2.imwrite("third_tile.jpg", third_tile)
#         cv2.imwrite("forth_tile.jpg", forth_tile)
        
            
        return img_id, best_tile, sec_tile, third_tile, forth_tile 
    
def read_save_most_informative (train_df):
    path_to_read = "../input/mayo-clinic-strip-ai/train/"
    path_to_write = ""

    os.mkdir("mayojpgCrops1")
    os.mkdir("mayojpgCrops2")
    os.mkdir("mayojpgCrops3")
    os.mkdir("mayojpgCrops4")

    for i in tqdm(range(train_df.shape[0])):
        img_id, best_tile, sec_tile, third_tile, forth_tile = most_informative_tiles_generator (train_df, i, path_to_read)

        cv2.imwrite("mayojpgCrops1/"+img_id+".jpg", best_tile)
        cv2.imwrite("mayojpgCrops2/"+img_id+".jpg", sec_tile)
        cv2.imwrite("mayojpgCrops3/"+img_id+".jpg", third_tile)
        cv2.imwrite("mayojpgCrops4/"+img_id+".jpg", forth_tile)




def read_save_most_informative (train_df):
    path_to_read = "../input/mayo-clinic-strip-ai/train/"
    path_to_write = ""

    os.mkdir("mayojpg")

    for i in tqdm(range(train_df.shape[0])):
        img_id, img = img_read_resize (train_df, i, path_to_read)
        cv2.imwrite("mayojpg/"+img_id+".jpg", img)
