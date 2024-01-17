import torch
import pandas as pd
import numpy as np
from image_model2 import Image_Model2
from path_and_parameters import Paths_Configuration, Defining_Parameters
from read_images_and_label import visualize_specific_image
from metric import Metric
from torchvision.transforms.functional import to_pil_image
from matplotlib import colormaps
import numpy as np
import PIL
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
import os

# Parameters and paths
pars = Defining_Parameters()
par_path = Paths_Configuration()
my_csv = par_path.my_csv

draek_path = '/home/elenad/draeck/edeiana/'

# image_index = 22
# image_path = draek_path + df.loc[image_index, 'path']
# label = df.loc[image_index, 'label']  
# image = Image.open(image_path).convert('RGB')


### drop column from my_csv to keep only image id, path and label
df = pd.read_csv(my_csv)
columns_to_keep = ['path', 'dicom_id', 'label']
df2 = df[columns_to_keep]
# print(df2.head())

### create a smaller df based on test_df indeces
test_indices = pd.read_csv('split_indices/test_indices.csv')

# Step 2: Filter the main df based on the indices from 'test_indices'
f_test_df = df2.loc[test_indices['index']].reset_index(drop=True)
f_test_df.to_csv('f_test_df.csv', index=False)