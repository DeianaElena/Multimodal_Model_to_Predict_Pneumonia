import numpy as np
import pandas as pd
import torch, gc, torchsummary, h5py, PIL
import torchvision, torchtext, transformers, os
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel
import torch.optim as optim
import torch.nn as nn
import time, glob, random
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
#models
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_b4, EfficientNet_B4_Weights
from transformers import T5EncoderModel, LlamaForSequenceClassification
#my imports
from model_utils import save_h5_array, reduce_df
from path_and_parameters import Paths_Configuration, Defining_Parameters

random.seed(28)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#parameters and paths
pars = Defining_Parameters()
par_path = Paths_Configuration()
my_csv = par_path.my_csv; image_dir = par_path.image_dir
draek_path = par_path.draek_path; output_folder = par_path.output_folder
num_workers = pars.num_workers; batch_size = pars.batch_size

# print(my_csv)

def visualize_specific_image(csv_path, image_index):
    df = pd.read_csv(csv_path)

    image_path = draek_path + df.loc[image_index, 'path']
    label = df.loc[image_index, 'label']

    try:
        img = Image.open(image_path)
        # plt.imshow(img)  #default is viridis' colormap
        plt.imshow(img, cmap='gray')  # Specify the colormap as 'gray'
        plt.title(f"Image at index {image_index+1} - Label: {label}")
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Error opening image at index {image_index+1}: {str(e)}")


# #chose index
# desired_image_index = 3
# image = visualize_specific_image(my_csv, desired_image_index)
