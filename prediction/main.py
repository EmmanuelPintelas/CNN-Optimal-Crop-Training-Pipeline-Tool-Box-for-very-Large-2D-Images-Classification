


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



from myfunctions.Utilities import *



# we load the raw and in a temporar jpg file in order to load and feed the trained model and predict
#   We trained with the best detected tiles , so we have to predict with the best-tile config.
try:
    os.mkdir("../test/")
except:
    pass

start = time.time()
for i in tqdm(range(test_df.shape[0])):

    img_id = test_df.iloc[i].image_id
    img_sz = test_df.iloc[i].image_pixels

    if img_sz <= 3728229285:
        #img = np.zeros((1000,1000,3), np.uint8)
        print(1)
        img = cv2.resize(tifffile.imread(dirs[1] + img_id + ".tif"), (4096, 4096))
        img = tiles_detector_extractor (img) ## it returns a 1000x1000 tile (the best tile)

    else:
        print('raw image too big, slipe approach used instead')
        img = Slide_Cut_off_Resize(dirs[1] + img_id + ".tif", 4096)
        img = tiles_detector_extractor (img) ## it returns a 1000x1000 tile

    cv2.imwrite(f"../test/{img_id}.jpg", img)
    del img
    gc.collect()
    
print('data load time: ', time.time() - start)  
    


batch_size = 1
test_loader = DataLoader(ImgDataset(test_df), batch_size=batch_size, shuffle=False, num_workers=1)

# for _ in test_loader:
#     plt.imshow(_[0][0][0]); plt.show()


start = time.time()
Output, ids = predict(model_crop, test_loader)
print('cnn1 pred time: ', time.time() - start)


tmin, tmax =  0.01, 0.99
Output_scaled = ((Output - Output.min())/(Output.max() - Output.min()))*(tmax-tmin) + tmin



anss = Output_scaled ###(anss1 + anss2)/2.0

prob = pd.DataFrame({"CE" : anss[:,0], "LAA" : anss[:,1], "id" : ids}).groupby("id").mean()
submission = pd.read_csv("../input/mayo-clinic-strip-ai/sample_submission.csv")

submission.CE = prob.CE.to_list()
submission.LAA = prob.LAA.to_list()

submission.to_csv("submission.csv", index = False)