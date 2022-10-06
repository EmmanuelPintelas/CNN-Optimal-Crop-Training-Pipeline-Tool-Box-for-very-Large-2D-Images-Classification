from myfunctions.Utilities import *




# In this project we read/load the huge raw images (tiff format) from a folder and then resize them and save them in jpg format.
# in order to load them we utilize also open-slide tools. We also provide some functions for detecting the most informative
# tiles (crops) and save them also as crops. A croped image can focus on local areas and thus the resolution of important regions
# is much higher. Feeding them in such a way can drastically boost the performance of a CNN model.





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





# Demonstration

_, _, train_df = DataFrame_Creation ()
path_to_read = "../input/mayo-clinic-strip-ai/train/"
index = 0
img_id, best_tile, sec_tile, third_tile, forth_tile = most_informative_tiles_generator (train_df, index, path_to_read)
index = 4
img_id, best_tile, sec_tile, third_tile, forth_tile = most_informative_tiles_generator (train_df, index, path_to_read)
