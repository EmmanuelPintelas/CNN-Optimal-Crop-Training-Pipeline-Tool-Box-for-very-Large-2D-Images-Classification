



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
# ------- Ting __________ DO NOT CHANGE THESE!!!!!


# ------- Device __________
DEVICE = torch.device('cuda') #  



# _____________ Training Functions _________________

def train_per_epoch (model, dataloader, criterion, optimizer, num_epochs, labels_n):
            model.train()
            ###classifier.train()
            data_size = len(dataloader.dataset)
            epoch_loss, epoch_acc = 0.0, 0
            for item in tqdm(dataloader, leave=False):
                    images = item[0].float().to(DEVICE) 
                    classes = item[1].long().to(DEVICE) 
                    optimizer.zero_grad()                
                    with torch.set_grad_enabled(True):
                        ###plt.imshow(images[0]); plt.show()
                        output = model(images)#[:,:labels_n]
                        ###print(output)
                        loss = criterion(output, classes)
                        _, preds = torch.max(output, 1)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item() * len(output)
                        epoch_acc += torch.sum(preds == classes.data)  
            epoch_loss = epoch_loss / data_size
            epoch_acc = epoch_acc.double() / data_size
            print(f'Epoch {num_epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')    
            return epoch_acc

def val_per_epoch (model, dataloader, criterion, best_acc, labels_n):
            model.eval()
            ###classifier.eval()
            data_size = len(dataloader.dataset)
            epoch_loss, epoch_acc = 0.0, 0
            for item in tqdm(dataloader, leave=False):
                    images = item[0].float().to(DEVICE) 
                    classes = item[1].long().to(DEVICE) 
                    optimizer.zero_grad()                
                    with torch.set_grad_enabled(False):
                        output = model(images)[:,:labels_n]
                        loss = criterion(output, classes)
                        _, preds = torch.max(output, 1)
                        epoch_loss += loss.item() * len(output)
                        ###print(classes.data)
                        epoch_acc += torch.sum(preds == classes.data)                      
            epoch_loss = epoch_loss / data_size
            epoch_acc = epoch_acc.double() / data_size
            print(f'Val_Loss: {epoch_loss:.4f} | Val_Acc: {epoch_acc:.4f}')  
            return epoch_acc
   
                
                
def Train_Loop(model,  
                        dataloader_dict, 
                        criterion, 
                        optimizer, 
                        time_to_train, 
                        path_to_save, 
                        labels_n):

    train_loader, val_loader = dataloader_dict['train'], dataloader_dict['val']

    model.to(DEVICE)
    ##classifier.to(DEVICE)

    best_acc, num_epochs = 0.0, 0
    start = time.time()
    time_left = time_to_train
    while time_left >= 300:
        time_left = time_left - (time.time() - start)
        print('time_left: ', time_left)
        start = time.time()
        num_epochs += 1

        tr_acc  = train_per_epoch (model, train_loader, criterion, optimizer, num_epochs, labels_n)
        val_acc = val_per_epoch (model, val_loader, criterion, best_acc, labels_n)

        if val_acc > best_acc:
            best_acc = val_acc
            print ('best_acc: ', best_acc.cpu().numpy())
            torch.save(model.state_dict(), os.path.join(path_to_save, "cnn.pth"))

#             # --------------------- LOAD ---------------------------
#             checkpoint = torch.load('../models/param.pth')
#             model = EfficientNet.from_name("efficientnet-b4")
#             model.load_state_dict(checkpoint)
            
            
#KANE SAVE TO BEST PARAM



def Train_Loop_Augments(model, 
                                dataloader_dict, 
                                criterion, 
                                optimizer, 
                                time_to_train, 
                                path_to_save, 
                                labels_n):
    
#     train_loader_dict = {'train':train_loader_crop,
#                          'train_crop':train_loader_crop,
#                          'train_rotate':train_loader_rotate,
#                          'train_contrast':train_loader_contrast,
#                          'train_adapthist':train_loader_adapthist,
#                          'train_gamma':train_loader_crop,
#                          'val':val_loader_crop,
#                          'val_crop':val_loader_crop}
    
    augm_names = ['train_crop', 
                  'train', 
                  'train_contrast', 
                  'train_adapthist', 
                  'train']
    
    model.to(DEVICE)
    ###classifier.to(DEVICE) 
    best_acc, num_epochs = 0.0, 0
    tr_acc = 0
    start = time.time()
    time_left = time_to_train 
    while time_left >= 300:
        print ('')
        print ('------------------------------- Start of Epoch ---------------------------------------')
        time_left = time_left - (time.time() - start)
        print('time_left: ', time_left)
        start = time.time()
        num_epochs += 1
        
        if best_acc >= 0.8580:
            print('old max acc reached!')
            augm_names = ['train_rotate', 
                          'train_contrast', 
                          'train']
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        
        
        if tr_acc <= 0.99:
            train_loader, val_loader = dataloader_dict['train'], dataloader_dict['val']
        elif tr_acc > 0.99 and best_acc >= 0.835:
            print('-- augm activated -->')
            augm_name = augm_names[random.randint(0,len(augm_names)-1)]
            print(augm_name)
            train_loader, val_loader = dataloader_dict[augm_name], dataloader_dict['val']
        else:
            train_loader, val_loader = dataloader_dict['train'], dataloader_dict['val']
        
            
        tr_acc  = train_per_epoch (model, train_loader, criterion, optimizer, num_epochs, labels_n)
        val_acc = val_per_epoch (model, val_loader, criterion, best_acc, labels_n)
        
        if val_acc > best_acc:
            best_acc = val_acc
            print ('best_acc: ', best_acc.cpu().numpy())
            torch.save(model.state_dict(), os.path.join(path_to_save, "cropaugm.pth"))
            
        print ('------------------------------- End of Epoch ---------------------------------------')


    
def Train_Loop_Augments2(model, 
                                dataloader_dict, 
                                criterion, 
                                optimizer, 
                                time_to_train, 
                                path_to_save, 
                                labels_n):

#     train_loader_dict = {'train':train_loader,

#                          'train_crop':train_loader_crop,
#                          'train_crop2':train_loader_crop2,
#                          'train_crop3':train_loader_crop3,
#                          'train_crop4':train_loader_crop4,

#                          'train_rotate':train_loader_rotate,
#                          'train_contrast':train_loader_contrast,
#                          'train_adapthist':train_loader_adapthist,
#                          'train_gamma':train_loader_crop,

#                          'val':val_loader,

#                          'val_crop':val_loader_crop,
#                          'val_crop2':val_loader_crop2,
#                          'val_crop3':val_loader_crop3,
#                          'val_crop4':val_loader_crop4}
    

    val_augm_names = ['val_crop',
                      'val_crop',
                      'val_crop',
                      'val_crop2',
                      'val_crop2']

    augm_names = ['train',
                  'train_crop',
                  'train_crop',
                  'train_crop3', 
                  'train_crop4', 
                  'train_adapthist']

    augm_names2 = ['train',
                   'train',
                   'train_crop',
                   'train_crop',
                   'train_crop',
                   'train_crop2',
                   'train_crop2',
                   'train_crop3', 
                   'train_crop4']

    model.to(DEVICE)
    ###classifier.to(DEVICE) 
    best_acc, num_epochs = 0.0, 0
    tr_acc = 0
    start = time.time()
    time_left = time_to_train 
    while time_left >= 300:
        print ('')
        print ('------------------------------- Start of Epoch ---------------------------------------')
        time_left = time_left - (time.time() - start)
        print('time_left: ', time_left)
        start = time.time()
        num_epochs += 1
   
        if best_acc >= 0.8580:
            print('old max acc reached!')
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)


        if tr_acc <= 0.99:
            train_loader, val_loader = dataloader_dict['train_crop2'], dataloader_dict['val_crop2']

        elif tr_acc > 0.99 and best_acc >= 0.790 and best_acc < 0.810:
            print('-- augm activated -->')

            augm_name = augm_names[random.randint(0,len(augm_names)-1)]
            print(augm_name)
            if augm_name == 'train':
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
            elif augm_name == 'train_adapthist':
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

            train_loader, val_loader = dataloader_dict[augm_name], dataloader_dict['val_crop2']

        elif tr_acc > 0.99 and best_acc >= 0.810:
            print('-- val augm activated -->')

            augm_name2 = augm_names2[random.randint(0,len(augm_names2)-1)]
            print(augm_name2)
            val_augm_name = val_augm_names[random.randint(0,len(val_augm_names)-1)]
            print(val_augm_name)
            if augm_name2 == 'train':
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

            train_loader, val_loader = dataloader_dict[augm_name2], dataloader_dict[val_augm_name]
            # IMPORTANT: In case the the best model was achieved for a diffrent val than the originial init,
            #            we have to use this val transformation for the testing phase

        else:
            train_loader, val_loader = dataloader_dict['train_crop2'], dataloader_dict['val_crop2']


        tr_acc  = train_per_epoch (model, train_loader, criterion, optimizer, num_epochs, labels_n)
        val_acc = val_per_epoch (model, val_loader, criterion, best_acc, labels_n)

        if val_acc > best_acc:
            best_acc = val_acc
            print ('best_acc: ', best_acc.cpu().numpy())
            torch.save(model.state_dict(), os.path.join(path_to_save, "cropaugm.pth"))
            
        print ('------------------------------- End of Epoch ---------------------------------------')

#             # --------------------- LOAD ---------------------------
#             checkpoint = torch.load('../models/param.pth')
#             model = EfficientNet.from_name("efficientnet-b4")
#             model.load_state_dict(checkpoint)



