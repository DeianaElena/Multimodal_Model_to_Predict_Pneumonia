# libraries
import torch
import pandas as pd
import numpy as np
from image_model3 import Image_Model3
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
image_dir = par_path.image_dir
draek_path = par_path.draek_path
output_folder = par_path.output_folder
num_workers = pars.num_workers
batch_size = pars.batch_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = 0.0001
best_model_path = './checkpoints/mod3_resnet_50/best-model-select_image_model_name=0epoch=02-val_loss=0.62.ckpt'

draek_path = '/home/elenad/draeck/edeiana/'
output_folder_gradcam = draek_path+'/gradcam_images_mod3_resnet50/'
model = Image_Model3(2048,lr)
df = pd.read_csv('f_test_df.csv')

model.load_state_dict(torch.load(best_model_path)["state_dict"])
model.to(device)
# print(model)
opt = model.configure_optimizers()

#load test image tensors + for loop over all those
test_image_tensor_file = draek_path+"output_features/full_db/test_image_tensor_mod3_resnet_50.pt"
test_image_labels = draek_path+"output_features/full_db/test_labels_mod3_resnet_50.npy"
# Load the test image tensor
test_image_tensor = torch.load(test_image_tensor_file)
# test_image_label = torch.load(test_image_labels)
# print(test_image_label)
# raise

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

backward_hook = model.conv_block.register_full_backward_hook(backward_hook, prepend=False)
forward_hook = model.conv_block.register_forward_hook(forward_hook, prepend=False)

print('here')

############ new code ###############


# Loop through all the images in the DataFrame
# for image_index in range(len(df)):
for image_index in range(4):
    # Get the image path and label from the DataFrame
    image_path = draek_path + df.loc[image_index, 'path']
    label = df.loc[image_index, 'label']

    # Open the image and resize it to the required dimensions
    image = Image.open(image_path).convert('RGB')
    image = image.resize((256, 256), resample=PIL.Image.BICUBIC)
    image = transforms.CenterCrop(224)(image)

    # Perform Grad-CAM on the image
    y = model(test_image_tensor[image_index].unsqueeze(0).to(device))
    print('this is y: ', y)
    opt.zero_grad()
    label_tensor = torch.tensor(label).unsqueeze(0).to(device)
    loss = model.criterion(y, label_tensor)
    loss.backward()

    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    for i in range(activations.size()[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.detach().cpu().numpy()

    # Create a figure and plot the original image
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(image)

    # Resize the heatmap to the size of the input image and apply colormap
    overlay = to_pil_image(heatmap, mode='F').resize((224, 224), PIL.Image.BICUBIC)
    cmap = colormaps['jet']
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)

    # Plot the heatmap on the same axes with transparency
    ax.imshow(overlay, alpha=0.4, interpolation='nearest')

    # Save the Grad-CAM image in the output folder with the name of the index and label
    gradcam_image_path = os.path.join(output_folder_gradcam, f"index_{image_index}_label_{label}.png")
    plt.savefig(gradcam_image_path)

    # Clear the plot for the next image
    plt.clf()

# Optionally, you may close the figure after processing all images
plt.close('all')
