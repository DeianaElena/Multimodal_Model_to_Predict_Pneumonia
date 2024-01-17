import numpy as np
import pandas as pd
import torch
import torchsummary
import os
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.models import resnet50, efficientnet_b4
from model_utils import save_h5_array_image
from path_and_parameters import Paths_Configuration, Defining_Parameters
import time, glob, random

random.seed(28)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parameters and paths
pars = Defining_Parameters()
par_path = Paths_Configuration()
my_csv = par_path.my_csv
image_dir = par_path.image_dir
draek_path = par_path.draek_path
output_folder = par_path.output_folder
num_workers = pars.num_workers
batch_size = pars.batch_size

print(my_csv)
df = pd.read_csv(my_csv)

# To not train the models
torch.set_grad_enabled(False)

####### Select IMAGE model #######
# image_model_name = 'resnet50'
# image_model_name = 'efficientnet_b4'
# image_model_name = 'mod_resnet_50'
image_model_name = 'mod_effb4'

######################### -------------------- #######################################
if image_model_name == 'mod_resnet_50':
    # Load the pretrained ResNet50 model
    model = resnet50(pretrained=True)
    class CustomResNet(torch.nn.Module):
        def __init__(self, original_model):
            super(CustomResNet, self).__init__()
            self.features = torch.nn.Sequential(
                # Keep layers up to 'layer4'
                *list(original_model.children())[:-2])         
        def forward(self, x):
            x = self.features(x)
            return x
    image_model = CustomResNet(model)
    resize_size = 256
    crop_size = 224
    
elif image_model_name == 'mod_effb4':
    model = efficientnet_b4(pretrained=True)
    class CustomEffb4(torch.nn.Module):
        def __init__(self, original_model):
            super(CustomEffb4, self).__init__()
            self.features = torch.nn.Sequential(
                # Keep layers without last 2'
                *list(original_model.children())[:-2]) 
        def forward(self, x):
            x = self.features(x)
            return x
    image_model = CustomEffb4(model)
    resize_size = 384
    crop_size = 380
elif image_model_name == 'efficientnet_b4':
    image_model = efficientnet_b4(pretrained=True)
    resize_size = 384
    crop_size = 380
elif image_model_name == 'resnet50':
    image_model = resnet50(pretrained=True)
    resize_size = 256
    crop_size = 224
    
######################### -------------------- #######################################
transform = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Freeze the parameters of the image model, except for the last fully connected layer
for param in image_model.parameters():
    param.requires_grad = False

# Set the image model to evaluation mode
image_model.eval()

class Clinical_Dataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)    

    def __getitem__(self, idx):
        # get the subject_id and the study_id from the corresponding columns
        full_path = self.image_dir + self.df.loc[idx, "path"]
        label = self.df.loc[idx, 'label']
        image_file = full_path
        image = Image.open(image_file).convert('RGB')  # Open image file in PIL
        if self.transform:
            image = self.transform(image)
        
        return {"image": image, 'label': label}

# Create dataset
dataset = Clinical_Dataset(df, image_dir, transform)
print('Length dataset: ', len(dataset))

# Create data loader
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, prefetch_factor=2)

image_model.to(device)

# Initialize empty numpy array to store the image features
image_outputs = []

start_time = time.time()
total_iterations = len(data_loader)

# Helper function to resize the model output tensor to [6, 6]
def resize_images(images, target_size=(6, 6)):
    return torch.nn.functional.adaptive_avg_pool2d(images, target_size)

# Process batches of data and store image features
for i, batch in enumerate(data_loader):
    image = batch["image"].to(device)

    image_output = image_model(image)  # Pass image data through image_model

    # Resize the images in the model output tensor to [6, 6]
    resized_image_output = resize_images(image_output, target_size=(6, 6))

    if i == 0:
        image_outputs = resized_image_output.detach().cpu()
    else:
        image_outputs = torch.cat((image_outputs, resized_image_output.detach().cpu()), 0)
        
    # Time info
    iteration_time = time.time() - start_time
    average_iteration_time = iteration_time / (i + 1)
    remaining_iterations = total_iterations - (i + 1)
    estimated_remaining_time = remaining_iterations * average_iteration_time
    print(
        f"Iteration: {i + 1}/{total_iterations} | Avg. Iteration Time: {average_iteration_time:.2f}s | Estimated Remaining Time: {estimated_remaining_time:.2f}s"
    )

total_time = time.time() - start_time  # total execution time
print(f"Total Time: {total_time:.2f}s")

# Convert tensor to numpy array
image_outputs = image_outputs.numpy()

# Save the image features
save_h5_array_image(draek_path+output_folder, image_outputs, image_model_name=image_model_name)

print("STORING COMPLETED")




############## -------------------------------- ##############3
# import numpy as np
# import pandas as pd
# import torch
# import torchsummary
# import os
# from tqdm import tqdm
# from PIL import Image
# from torch.utils.data import DataLoader, Dataset, TensorDataset
# from torchvision import transforms
# from torchvision.models import resnet50, efficientnet_b4
# from model_utils import save_h5_array
# from path_and_parameters import Paths_Configuration, Defining_Parameters
# import time, glob, random
# random.seed(28)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # parameters and paths
# pars = Defining_Parameters()
# par_path = Paths_Configuration()
# my_csv = par_path.my_csv
# image_dir = par_path.image_dir
# draek_path = par_path.draek_path
# output_folder = par_path.output_folder
# num_workers = pars.num_workers
# batch_size = pars.batch_size

# print(my_csv)
# df = pd.read_csv(my_csv)

# # To not train the models
# torch.set_grad_enabled(False)

# ####### Select IMAGE model #######
# # image_model_name = 'resnet50'
# # image_model_name = 'efficientnet_b4'
# # image_model_name = 'mod_resnet_50'
# image_model_name = 'mod_effb4'

# ######################### -------------------- #######################################
# if image_model_name == 'mod_resnet_50':
#     # Load the pretrained ResNet50 model
#     model = resnet50(pretrained=True)
#     class CustomResNet(torch.nn.Module):
#         def __init__(self, original_model):
#             super(CustomResNet, self).__init__()
#             self.features = torch.nn.Sequential(
#                 # Keep layers up to 'layer4'
#                 *list(original_model.children())[:-2])         
#         def forward(self, x):
#             x = self.features(x)
#             return x
#     image_model = CustomResNet(model)
#     resize_size = 256
#     crop_size = 224
    
# elif image_model_name == 'mod_effb4':
#     model = efficientnet_b4(pretrained=True)
#     class CustomEffb4(torch.nn.Module):
#         def __init__(self, original_model):
#             super(CustomEffb4, self).__init__()
#             self.features = torch.nn.Sequential(
#                 # Keep layers without last 2'
#                 *list(original_model.children())[:-2]) 
#         def forward(self, x):
#             x = self.features(x)
#             return x
#     image_model = CustomEffb4(model)
#     resize_size = 384
#     crop_size = 380
# elif image_model_name == 'efficientnet_b4':
#     image_model = efficientnet_b4(pretrained=True)
#     resize_size = 384
#     crop_size = 380
# elif image_model_name == 'resnet50':
#     image_model = resnet50(pretrained=True)
#     resize_size = 256
#     crop_size = 224
    
# ######################### -------------------- #######################################
# transform = transforms.Compose([
#     transforms.Resize(resize_size),
#     transforms.CenterCrop(crop_size),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # Freeze the parameters of the image model, except for the last fully connected layer
# for param in image_model.parameters():
#     param.requires_grad = False

# # Set the image model to evaluation mode
# image_model.eval()

# class Clinical_Dataset(Dataset):
#     def __init__(self, df, image_dir, transform=None):
#         self.df = df
#         self.image_dir = image_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.df)    

#     def __getitem__(self, idx):
#         # get the subject_id and the study_id from the corresponding columns
#         full_path = self.image_dir + self.df.loc[idx, "path"]
#         label = self.df.loc[idx, 'label']
#         image_file = full_path
#         image = Image.open(image_file).convert('RGB')  # Open image file in PIL
#         if self.transform:
#             image = self.transform(image)
        
#         return {"image": image, 'label': label}

# # Create dataset
# dataset = Clinical_Dataset(df, image_dir, transform)
# print('Length dataset: ', len(dataset))

# # Create data loader
# data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, prefetch_factor=2)

# image_model.to(device)
# # print(image_model)
# # raise

# # Initialize empty numpy array to store the image features
# image_outputs = []

# start_time = time.time()
# total_iterations = len(data_loader)

# # Process batches of data and store image features
# for i, batch in enumerate(data_loader):
#     image = batch["image"].to(device)

#     image_output = image_model(image)  # Pass image data through image_model
#     # print(image_output.size())
#     # raise
#     if i == 0:
#         image_outputs = image_output.detach().cpu()
#     else:
#         image_outputs = torch.cat((image_outputs, image_output.detach().cpu()), 0)
        
#     # Time info
#     iteration_time = time.time() - start_time
#     average_iteration_time = iteration_time / (i + 1)
#     remaining_iterations = total_iterations - (i + 1)
#     estimated_remaining_time = remaining_iterations * average_iteration_time
#     print(
#         f"Iteration: {i + 1}/{total_iterations} | Avg. Iteration Time: {average_iteration_time:.2f}s | Estimated Remaining Time: {estimated_remaining_time:.2f}s"
#     )

# total_time = time.time() - start_time  # total execution time
# print(f"Total Time: {total_time:.2f}s")

# # Convert tensor to numpy array
# image_outputs = image_outputs.numpy()

# # Save the image features
# save_h5_array(draek_path+output_folder, image_outputs, text_outputs=None, text_name=None, image_model_name=image_model_name)

# print("STORING COMPLETED")
