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

print(my_csv)
df = pd.read_csv(my_csv)
#In case we want to test less images we reduce the csv file
# reduce_df(df, image_dir)

# To not train the models
torch.set_grad_enabled(False)

####### Select TEXT model #######   
# text_model_name =  "Clinical-T5-Base"   
# text_model_name =  "Clinical-T5-Scratch"    
# text_model_name =  "t5-base"
text_model_name = "yikuan8/Clinical-Longformer"
# text_model_name = "distilbert-base-uncased"
# text_model_name = 'emilyalsentzer/Bio_Discharge_Summary_BERT'
text_name = text_model_name.split('/')[-1]

if text_model_name == "t5-base":
    text_model = T5EncoderModel.from_pretrained(text_model_name) #from documentation
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
# elif text_model_name ==  "Clinical-T5-Base" or "Clinical-T5-Scratch":
elif text_model_name ==  "Clinical-T5-Base":
    t5_model_path = f"/home/elenad/draeck/edeiana/clinical-t5/{text_model_name}"
    tokenizer = AutoTokenizer.from_pretrained(t5_model_path)
    text_model = T5EncoderModel.from_pretrained(t5_model_path)
else:
    text_model = AutoModel.from_pretrained(text_model_name)
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)

####### Select IMAGE model #######
image_model_name = 'resnet50'
# image_model_name = 'efficientnet_b4'
# image_model_name = 'mod_resnet_50'
# image_model_name = 'mod_effb4'

######################### -------------------- #######################################
if image_model_name == 'mod_resnet_50':
    # Load the pretrained ResNet50 model
    model = resnet50(pretrained=True)
    # new_model = torch.nn.Sequential()
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
    image_model = efficientnet_b4(EfficientNet_B4_Weights)
    resize_size = 384
    crop_size = 380
elif image_model_name == 'resnet50':
    image_model = resnet50(ResNet50_Weights)
    resize_size = 256
    crop_size = 224
    
######################### -------------------- #######################################
# print(image_model)
#or >> file.txt to save the outputs from terminal
# raise
transform = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Freeze the parameters of the image model, except for the last fully connected layer
for param in image_model.parameters():
    param.requires_grad = False

#debugging
# summary= torchsummary.summary(text_model)
###################################################
# Set the image and text model to evaluation mode
image_model.eval()
text_model.eval()

class Clinical_Dataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)    

    def __getitem__(self, idx):
        text = self.df.loc[idx, "filtered_text"]
        text = tokenizer.encode(text, max_length=None, padding="max_length", truncation=True)
        text = torch.tensor(text)

        # get the subject_id and the study_id from the corresponding columns
        full_path = self.image_dir + self.df.loc[idx, "path"]
        label = self.df.loc[idx, 'label']
        image_file = full_path
        image = None
        image = Image.open(image_file).convert('RGB')  # Open image file in PIL
        if self.transform:
            image = self.transform(image)
        
        return {"text": text, "image": image, 'label': label}

# Create dataset
dataset = Clinical_Dataset(df, image_dir, transform)
print('Length dataset: ', len(dataset))

# Create data loader
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, prefetch_factor=2)

image_model.to(device)
text_model.to(device)

# Initialize empty numpay array to store the outputs
# image_outputs = []
text_outputs = []
# text_outputs = np.array()

start_time = time.time()
total_iterations = len(data_loader)

# Process batches of data and store outputs
for i, batch in enumerate(data_loader):
    text = batch["text"].to(device)
    image = batch["image"].to(device)

    text_output = text_model(text)  # Pass text data through text_model
    #debugging
    # print('THE KEYS: ', text_output.keys())
    # raise
    image_output = image_model(image)  # Pass image data through image_model
    # print(image_output.size()) #torch.Size([16, 2048, 7, 7])
  
    if i == 0:
        image_outputs = image_output.detach().cpu()
    else:
        image_outputs = torch.cat((image_outputs, image_output.detach().cpu()), 0)

    # image_outputs.append(image_output.detach().cpu().numpy())  # Move image output to CPU and store in list
    # print(image_outputs.size())  #torch.Size([16, 2048, 7, 7])
    #Different outputs based on model

    if text_model_name == "yikuan8/Clinical-Longformer": #in case we want to choose pooler output
        # THE KEYS:  odict_keys(['last_hidden_state', 'pooler_output'])
        # print(text_output[1].size())  #torch.Size([8, 768])
        # pooled_output = torch.mean(text_output[1], dim=1)   #average pooling
        pooled_output = torch.mean(text_output["last_hidden_state"], dim=1)
        # pooled_output = torch.mean(text_output["pooler_output"], dim=1)
        text_outputs.append(pooled_output.detach().cpu().numpy())  
    else:
        pooled_output = torch.mean(text_output["last_hidden_state"], dim=1)
        text_outputs.append(pooled_output.detach().cpu().numpy())


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

# Convert lists to numpy arrays
image_outputs = image_outputs.numpy()
text_outputs = np.concatenate(text_outputs)


save_h5_array(draek_path+output_folder, image_outputs, text_outputs,text_name, image_model_name)

print("STORING COMPLETED")
