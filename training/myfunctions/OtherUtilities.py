

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




# ______________ OTHER ________________
def model_size (model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))



print (1)