

from myfunctions.OtherUtilities import *
from myfunctions.DataGeneratorUtilities import *
from myfunctions.TrainingPipelines import *




# ----- External Imports _____
import sys
sys.path.append('../input/einops')
sys.path.append('../input/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master')
from einops import rearrange
from einops.layers.torch import Rearrange
from efficientnet_pytorch import EfficientNet

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
# ------- Seeding __________ DO NOT CHANGE THESE!!!!!


# ------- Device __________
DEVICE = torch.device('cuda') #  




# --------- DataFrames Creation --------
total_train_df = pd.read_csv("../input/mayo-clinic-strip-ai/train.csv")#.head(1000) #<---- initial df input 
#test_df = pd.read_csv("../input/mayo-clinic-strip-ai/test.csv")
dir_raw_train_path = "../input/mayo-clinic-strip-ai/train/" #<---- initial dir input 
#dir_raw_test_path = "../input/mayo-clinic-strip-ai/test/"
dir_jpg_train_path = "../input/mayojpg/mayojpg/mayojpg/"            #<----jpg saved input 

# below are the augments dir paths, 
# we prefer to save them and pay storage, 
# instead of computing online during training so as we wont lose training time

dir_jpg_crop_train_path  = "../input/mayojpg/augm/augm/mayojpgCrops/"
dir_jpg_crop2_train_path = "../input/mayojpg/mayojpgCrops2/mayojpgCrops2/"
dir_jpg_crop3_train_path = "../input/mayojpg/mayojpgCrops3/mayojpgCrops3/"
dir_jpg_crop4_train_path = "../input/mayojpg/mayojpgCrops4/mayojpgCrops4/"

dir_jpg_rotate_train_path    = "../input/mayojpg/augm/augm/mayojpgrotate/"
dir_jpg_contrast_train_path  = "../input/mayojpg/augm/augm/mayojpgContrast/"
dir_jpg_adapthist_train_path = "../input/mayojpg/augm/augm/mayojpgadapthist/"
dir_jpg_gamma_train_path     = "../input/mayojpg/augm/augm/mayojpggamma/"

total_train_df = Balance_Train_Dataframe (total_train_df)
total_train_df = Enchance_Train_Dataframe (total_train_df) # We enchance with extra metadata info provided by the tiff files, like dimensions of imageges etc.
display(total_train_df)





# Now we have to update the raw input images paths with the corresponding ones to the jpg folder
in_format, out_format = '.tif','.jpg'
total_train_img_paths = replace_img_paths_format (total_train_df.image_path , in_format, out_format, dir_raw_train_path, dir_jpg_train_path)
# augments
total_train_crop_img_paths      = replace_img_paths_format (total_train_df.image_path , in_format, out_format, dir_raw_train_path, dir_jpg_crop_train_path)
total_train_crop2_img_paths     = replace_img_paths_format (total_train_df.image_path , in_format, out_format, dir_raw_train_path, dir_jpg_crop2_train_path)
total_train_crop3_img_paths     = replace_img_paths_format (total_train_df.image_path , in_format, out_format, dir_raw_train_path, dir_jpg_crop3_train_path)
total_train_crop4_img_paths     = replace_img_paths_format (total_train_df.image_path , in_format, out_format, dir_raw_train_path, dir_jpg_crop4_train_path)


total_train_rotate_img_paths    = replace_img_paths_format (total_train_df.image_path , in_format, out_format, dir_raw_train_path, dir_jpg_rotate_train_path)
total_train_contrast_img_paths  = replace_img_paths_format (total_train_df.image_path , in_format, out_format, dir_raw_train_path, dir_jpg_contrast_train_path)
total_train_adapthist_img_paths = replace_img_paths_format (total_train_df.image_path , in_format, out_format, dir_raw_train_path, dir_jpg_adapthist_train_path)
total_train_gamma_img_paths     = replace_img_paths_format (total_train_df.image_path , in_format, out_format, dir_raw_train_path, dir_jpg_gamma_train_path)



# Now we have to encode the string raw labels to integers
raw_label_names = ['CE', 'LAA']
total_train_labels = label_encoder(raw_label_names, np.array(total_train_df.label))
total_n_instances = len(total_train_labels)
print ('total_n_instances: ', total_n_instances)


# update dataframes
total_train_df.image_path, total_train_df.label = total_train_img_paths, total_train_labels

# augments
total_train_crop_df       = total_train_df.copy()
total_train_crop2_df      = total_train_df.copy()
total_train_crop3_df      = total_train_df.copy()
total_train_crop4_df      = total_train_df.copy()
total_train_rotate_df     = total_train_df.copy()
total_train_contrast_df   = total_train_df.copy()
total_train_adapthist_df  = total_train_df.copy()
total_train_gamma_df      = total_train_df.copy()

total_train_crop_df.image_path, total_train_df.label       = total_train_crop_img_paths, total_train_labels
total_train_crop2_df.image_path, total_train_df.label      = total_train_crop2_img_paths, total_train_labels
total_train_crop3_df.image_path, total_train_df.label      = total_train_crop3_img_paths, total_train_labels
total_train_crop4_df.image_path, total_train_df.label      = total_train_crop4_img_paths, total_train_labels

total_train_rotate_df.image_path, total_train_df.label     = total_train_rotate_img_paths, total_train_labels 
total_train_contrast_df.image_path, total_train_df.label   = total_train_contrast_img_paths, total_train_labels 
total_train_adapthist_df.image_path, total_train_df.label  = total_train_adapthist_img_paths, total_train_labels 
total_train_gamma_df.image_path, total_train_df.label      = total_train_gamma_img_paths, total_train_labels 

display(total_train_df)





# Now we have to shuffle the data (the seed keep it stable)
indices = [i for i in range(total_n_instances)]
random.shuffle(indices)

total_train_df = total_train_df.iloc[indices]
# augments
total_train_crop_df       = total_train_crop_df.iloc[indices]
total_train_crop2_df      = total_train_crop2_df.iloc[indices]
total_train_crop3_df      = total_train_crop3_df.iloc[indices]
total_train_crop4_df      = total_train_crop4_df.iloc[indices]


total_train_rotate_df    = total_train_rotate_df.iloc[indices]
total_train_contrast_df  = total_train_contrast_df.iloc[indices]
total_train_adapthist_df = total_train_adapthist_df.iloc[indices]
total_train_gamma_df     = total_train_gamma_df.iloc[indices]

display(total_train_df)




# Now we have to split into train - val 
split = int(total_n_instances*0.8) # train, val = train_test_split(train_df, test_size=0.2, random_state=42, stratify = train_df.label)
train_df, val_df = total_train_df[:split], total_train_df[split:]

# augments
train_crop_df,       val_crop_df       = total_train_crop_df[:split],  total_train_crop_df[split:]
train_crop2_df,      val_crop2_df      = total_train_crop2_df[:split], total_train_crop2_df[split:]
train_crop3_df,      val_crop3_df      = total_train_crop3_df[:split], total_train_crop3_df[split:]
train_crop4_df,      val_crop4_df      = total_train_crop4_df[:split], total_train_crop4_df[split:]

train_rotate_df,    val_rotate_df    = total_train_rotate_df[:split], total_train_rotate_df[split:]
train_contrast_df,  val_contrast_df  = total_train_contrast_df[:split], total_train_contrast_df[split:]
train_adapthist_df, val_adapthist_df = total_train_adapthist_df[:split], total_train_adapthist_df[split:]
train_gamma_df,     val_gamma_df     = total_train_gamma_df[:split], total_train_gamma_df[split:]




# Now we have to built our image loaders
batch_size = 2
img_size = 1000

train_loader       = DataLoader(ImgDataset(train_df, img_size), batch_size=batch_size, shuffle=False, num_workers=1)
val_loader         = DataLoader(ImgDataset(val_df, img_size), batch_size=batch_size, shuffle=False, num_workers=1)

train_loader_crop  = DataLoader(ImgDataset(train_crop_df, img_size), batch_size=batch_size, shuffle=False, num_workers=1)
val_loader_crop    = DataLoader(ImgDataset(val_crop_df, img_size), batch_size=batch_size, shuffle=False, num_workers=1)

train_loader_crop2 = DataLoader(ImgDataset(train_crop2_df, img_size), batch_size=batch_size, shuffle=False, num_workers=1)
val_loader_crop2   = DataLoader(ImgDataset(val_crop2_df, img_size), batch_size=batch_size, shuffle=False, num_workers=1)

train_loader_crop3 = DataLoader(ImgDataset(train_crop3_df, img_size), batch_size=batch_size, shuffle=False, num_workers=1)
val_loader_crop3   = DataLoader(ImgDataset(val_crop3_df, img_size), batch_size=batch_size, shuffle=False, num_workers=1)

train_loader_crop4 = DataLoader(ImgDataset(train_crop4_df, img_size), batch_size=batch_size, shuffle=False, num_workers=1)
val_loader_crop4   = DataLoader(ImgDataset(val_crop4_df, img_size), batch_size=batch_size, shuffle=False, num_workers=1)

train_loader_rotate    = DataLoader(ImgDataset(train_rotate_df, img_size), batch_size=batch_size, shuffle=False, num_workers=1)
train_loader_contrast  = DataLoader(ImgDataset(train_contrast_df, img_size), batch_size=batch_size, shuffle=False, num_workers=1)
train_loader_adapthist = DataLoader(ImgDataset(train_adapthist_df, img_size), batch_size=batch_size, shuffle=False, num_workers=1)
train_loader_gamma     = DataLoader(ImgDataset(train_gamma_df, img_size), batch_size=batch_size, shuffle=False, num_workers=1)


# train_loader_dict = {'train':train_loader,
#                      'train_crop':train_loader_crop,
#                      'train_rotate':train_loader_rotate,
#                      'train_contrast':train_loader_contrast,
#                      'train_adapthist':train_loader_adapthist,
#                      'train_gamma':train_loader_gamma,
#                      'val':val_loader,
#                      'val_crop':val_loader_crop}



# ---- crop-view training -----
train_loader_dict = {'train':train_loader,
                     
                     'train_crop':train_loader_crop,
                     'train_crop2':train_loader_crop2,
                     'train_crop3':train_loader_crop3,
                     'train_crop4':train_loader_crop4,
                     
                     'train_rotate':train_loader_rotate,
                     'train_contrast':train_loader_contrast,
                     'train_adapthist':train_loader_adapthist,
                     'train_gamma':train_loader_crop,
                     
                     'val':val_loader,
                     
                     'val_crop':val_loader_crop,
                     'val_crop2':val_loader_crop2,
                     'val_crop3':val_loader_crop3,
                     'val_crop4':val_loader_crop4}







# Now we have to load or build/define a model architecture
model = EfficientNet.from_name("efficientnet-b4")
#checkpoint = torch.load('../input/efficientnet-pytorch/efficientnet-b4-e116e8b3.pth')
checkpoint = torch.load('../input/mymayomodels/cropaugm2.pth')
model.load_state_dict(checkpoint)

#classifier = nn.Softmax(4,2)

#model_size (model)
## softmax =  nn.Softmax(dim=1) # LogSoftmax #model = torch.load('../input/mymayomodels/model1.pth')
#classifier = torch.nn.Linear(2000, 2)






# Define criterion and optimizer and pass the parameters of models

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-4)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.AdamW([{"params": model.parameters()},{"params": classifier.parameters(), "lr": 1e-3}], lr=1e-4)







# Now Lets Train our model!!!!
time_to_train = 1 #31000 # sec
path_to_save = ''
labels_n = 2
#Train_Loop(model, train_loader_dict, criterion, optimizer, time_to_train, path_to_save, labels_n)
Train_Loop_Augments(model, train_loader_dict, criterion, optimizer, time_to_train, path_to_save, labels_n)
