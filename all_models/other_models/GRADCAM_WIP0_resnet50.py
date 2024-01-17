# libraries
import numpy as np
import pandas as pd
import torch.nn as nn
import torch, h5py, os, glob, random
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import dataframe_image as dfi
import datetime
# libraries
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
from torchvision.models import resnet50, ResNet50_Weights
from image_model import Image_Model

from torchvision.transforms.functional import to_pil_image
from matplotlib import colormaps
import numpy as np
import PIL

#parameters and paths
pars = Defining_Parameters()
par_path = Paths_Configuration()
my_csv = par_path.my_csv; image_dir = par_path.image_dir
draek_path = par_path.draek_path; output_folder = par_path.output_folder
num_workers = pars.num_workers; batch_size = pars.batch_size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


lr = 0.0001
input_dim = 1000

model = resnet50(pretrained=True)
model.to(device)

model1 = Image_Model(1000,lr)
best_model_path = '..' #TODO add path
model1.load_state_dict(torch.load(best_model_path)["state_dict"])
model1.to(device)

# model1.eval()
# print(model)
params = list(model.parameters()) + list(model1.parameters())
opt = torch.optim.AdamW(params,lr = lr)

#load test image tensors + for loop over all those
test_image_tensor_file = draek_path+"output_features/full_db/test_image_tensor.pt"
test_image_labels = draek_path+"output_features/full_db/test_labels.npy"
output_folder_gradcam2 = draek_path+'/gradcam_images_resnet50/'
# Load the test image tensor
test_image_tensor = torch.load(test_image_tensor_file)


# defines two global scope variables to store our gradients and activations
gradients = None
activations = None

def backward_hook(module, grad_input, grad_output):
  global gradients # refers to the variable in the global scope
  print('Backward hook running...')
  gradients = grad_output
  # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
  print(f'Gradients size: {gradients[0].size()}') 
  # We need the 0 index because the tensor containing the gradients comes
  # inside a one element tuple.

def forward_hook(module, args, output):
  global activations # refers to the variable in the global scope
  print('Forward hook running...')
  activations = output
  # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
  print(f'Activations size: {activations.size()}')

backward_hook = model.layer4.register_full_backward_hook(backward_hook, prepend=False)
forward_hook = model.layer4.register_forward_hook(forward_hook, prepend=False)

df = pd.read_csv('f_test_df.csv')

# Loop through all the images in the DataFrame
# for image_index in range(len(df)):
for image_index in range(20):
    
    # Get the image path and label from the DataFrame
    image_path = draek_path + df.loc[image_index, 'path']
    label = df.loc[image_index, 'label']
    print(label)

    # Open the image and resize it to the required dimensions
    image = Image.open(image_path).convert('RGB')
    image = image.resize((256, 256), resample=PIL.Image.BICUBIC)
    image = transforms.CenterCrop(224)(image)

    # Perform Grad-CAM on the image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = transform(image)
    # Follow model forward and backward
    y = model(image_tensor.unsqueeze(0).to(device))
    z = model1(y)
    print('this is z: ',z)
    opt.zero_grad()
    label_tensor = torch.tensor(label).unsqueeze(0).to(device)
    loss = model1.criterion(z, label_tensor)
    loss.backward()

    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    for i in range(activations.size()[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    print(heatmap)
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.detach().cpu().numpy()
    # plt.matshow(heatmap)
  

    # Create a figure and plot the original image
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(image)

    # Resize the heatmap to the size of the input image and apply colormap
    overlay = to_pil_image(heatmap, mode='F').resize((224, 224), PIL.Image.BICUBIC)
    cmap = colormaps['jet']
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
    print(overlay)

    # Plot the heatmap on the same axes with transparency
    ax.imshow(overlay, alpha=0.4, interpolation='nearest')

    # Save the Grad-CAM image in the output folder with the name of the index and label
    gradcam_image_path = os.path.join(output_folder_gradcam2, f"index_{image_index}_label_{label}.png")
    plt.savefig(gradcam_image_path)
    # plt.show()

    # Clear the plot for the next image
    plt.clf()

# Optionally, you may close the figure after processing all images
plt.close('all')

# image_path = draek_path + df.loc[image_index, 'path']
# label = df.loc[image_index, 'label']  
# image = Image.open(image_path).convert('RGB')

# # Peform transform for resnet
# image_size = 256

# # Open the image and resize it to the required dimensions
# image = Image.open(image_path).convert('RGB')
# # image = image.resize((256, 256), resample=PIL.Image.BICUBIC)
# # image = transforms.CenterCrop(224)(image)

# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# image_tensor = transform(image)
# # Follow model forward and backward
# y = model(image_tensor.unsqueeze(0).to(device))
# z = model1(y)
# # opt.zero_grad()
# label_tensor = torch.tensor(label).unsqueeze(0).to(device)
# loss = model1.criterion(z, label_tensor)
# loss.backward()


# # pool the gradients across the channels
# pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
# # weight the channels by corresponding gradients
# for i in range(activations.size()[1]):
#     activations[:, i, :, :] *= pooled_gradients[i]
# # average the channels of the activations
# heatmap = torch.mean(activations, dim=1).squeeze()
# # relu on top of the heatmap
# heatmap = F.relu(heatmap)
# # normalize the heatmap
# heatmap /= torch.max(heatmap)
# heatmap = heatmap.detach().cpu().numpy()
# # draw the heatmap
# plt.matshow(heatmap)
# plt.show()


# # Create a figure and plot the first image
# fig, ax = plt.subplots()
# ax.axis('off') # removes the axis markers

# # First plot the original image
# ax.imshow(image)

# # Resize the heatmap to the same size as the input image and defines
# # a resample algorithm for increasing image resolution
# # we need heatmap.detach() because it can't be converted to numpy array while
# # requiring gradients
# overlay = to_pil_image(heatmap, mode='F').resize((224,224), PIL.Image.BICUBIC)

# # Apply any colormap you want
# cmap = colormaps['jet']
# overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)

# # Plot the heatmap on the same axes, 
# # but with alpha < 1 (this defines the transparency of the heatmap)
# ax.imshow(overlay, alpha=0.4, interpolation='nearest')

# # Show the plot
# plt.show()

