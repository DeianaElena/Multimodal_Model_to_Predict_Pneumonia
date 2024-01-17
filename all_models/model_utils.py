#libraries
import pandas as pd
import torch, h5py, os, random
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split, Subset
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import dataframe_image as dfi
import csv
import os
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit

# my imports
from image_model import Image_Model
from image_model2 import Image_Model2
# from image_model3 import Image_Model3
from image_model_mod_effb4 import Image_Model_effb4
from text_model import Text_Model
from fusion_model import Fusion_Model
# from CM_fusion_model import Fusion_Model #TODO change it back
from fusion_model_mod2_resnet import Fusion_Model_mod2_resnet
from fusion_model_mod_effb4 import Fusion_Model_mod_eff4
# from image_model_bce import Image_Model_bce
# from focalBCE_image_model import Image_Model_bce_focal
from path_and_parameters import all_text_models, all_image_models, select_image_model_name, select_text_model_name, select_model_name 
random.seed(28)

#In case we want to test less images we reduce the csv file
def reduce_df(df, image_dir):
    # Create a new column that indicates if the file exists or not
    df["file_exists"] = df["path"].apply(lambda x: os.path.exists(image_dir + x))
    # Drop the rows that have False values in the file_exists column
    df = df.drop(df[df["file_exists"] == False].index)
    # Drop the file_exists column as it is no longer needed
    df = df.drop("file_exists", axis=1)

#To save features in h5 format
def save_h5_array_image(output_folder, image_outputs, image_model_name):
    output_path = os.path.join(output_folder, f"image_outputs_frontal_{image_model_name}.h5")
    with h5py.File(output_path, "w") as hf:
        hf.create_dataset("image_outputs", data=image_outputs)


#To save features in h5 format
def save_h5_array(output_folder, image_outputs, text_outputs, text_model_name, image_model_name):
    output_path = os.path.join(output_folder, f"image_outputs_frontal_{image_model_name}.h5")
    with h5py.File(output_path, "w") as hf:
        hf.create_dataset("image_outputs", data=image_outputs)

    output_path = os.path.join(output_folder, f"text_outputs_frontal_{text_model_name}.h5")
    with h5py.File(output_path, "w") as hf:
        hf.create_dataset("text_outputs", data=text_outputs)


# To load the features in h5 format
def load_h5_array(image_file, text_file, labels):
    # image_features = []
    # text_features = []
    # Load the image and text features from the h5 file
    with h5py.File(image_file, 'r') as hf:
        image_features = hf['image_outputs'][:]
    with h5py.File(text_file, 'r') as hf:
        text_features = hf['text_outputs'][:]

    # Convert features and labels to tensors
    image_features = torch.tensor(image_features)
    text_features = torch.tensor(text_features)
    labels = torch.tensor(labels)

    return image_features, text_features, labels


#To select the model to train and test
def select_dataset_and_model(log_path, select_model_name, learning_rate, image_features, text_features, labels):
    #image model
    if select_model_name in all_image_models:
        dataset = TensorDataset(image_features, labels)
        # input_dim_image = image_features.shape[1] #Use number of channels, not total number of elements.

        # print(image_features.size())  #torch.Size([45103, 2048, 7, 7])
        if select_model_name == 'mod2_resnet_50':
            image_input_dim = 2048
            chosen_model = Image_Model2(image_input_dim, learning_rate)  #testing new model  ford grandcam
        # elif select_model_name == 'mod3_resnet_50':
        #     image_input_dim = 2048
        #     chosen_model = Image_Model3(image_input_dim, learning_rate)  #testing new model  ford grandcam
        elif select_model_name == 'mod_effb4':
            image_input_dim = 1792 #1792 or 448
            chosen_model = Image_Model_effb4(image_input_dim, learning_rate)  #testing new model  ford grandcam
        
        else:
            # image_input_dim = 2048
            image_input_dim = image_features.view(image_features.size(0), -1).shape[1]  #1000
            chosen_model = Image_Model(image_input_dim, learning_rate)
            # chosen_model = Image_Model_bce(image_input_dim, learning_rate)
            # chosen_model = Image_Model_bce_focal(image_input_dim, learning_rate)

        tb_logger = pl_loggers.TensorBoardLogger(log_path+'logs/', name=f'image_score_{select_model_name}')
    #text model
    elif select_model_name in all_text_models:
        dataset = TensorDataset(text_features, labels)
        input_dim_text = text_features.view(text_features.size(0), -1).shape[1]
        chosen_model = Text_Model(input_dim_text, learning_rate)
        tb_logger = pl_loggers.TensorBoardLogger(log_path+'logs/', name=f'text_score_{select_model_name}')
    #fusion model
    elif select_model_name == select_text_model_name + '_' + select_image_model_name:
        dataset = TensorDataset(image_features, text_features, labels)
        # image_input_dim = image_features.shape[1] 
        text_input_dim = text_features.view(text_features.size(0), -1).shape[1]
        if select_image_model_name == 'mod2_resnet_50':
             image_input_dim = image_features.shape[1] 
             chosen_model = Fusion_Model_mod2_resnet(image_input_dim, text_input_dim, learning_rate)  #both image and text
        elif select_image_model_name == 'mod_effb4':
             image_input_dim = image_features.shape[1] 
             chosen_model = Fusion_Model_mod_eff4(image_input_dim, text_input_dim, learning_rate)  #both image and text
        else:
            image_input_dim = image_features.view(image_features.size(0), -1).shape[1]
            chosen_model = Fusion_Model(image_input_dim, text_input_dim, learning_rate)  #both image and text
        tb_logger = pl_loggers.TensorBoardLogger(log_path+'logs/', name=f'fusion_score_{select_model_name}')

    return chosen_model, dataset, tb_logger


#############################################################################
def load_indices_from_csv(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        indices = [int(idx) for idx in next(reader)]
    return indices

def load_datasets_from_indices(dataset, train_indices, val_indices, test_indices):
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset



#######################  corrected splitting? #############
def split_dataset(dataset, labels, train_ratio, val_ratio, test_ratio, model):
    random_seed = 28
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    if os.path.exists('split_indices/train_indices.csv') and os.path.exists('split_indices/val_indices.csv') and os.path.exists('split_indices/test_indices.csv'):
        train_indices, val_indices, test_indices = load_indices_from_file(model)
        print("Loaded split indices.")
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)

    else:
        print("Performing dataset split and saving indices...")
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size + val_size, random_state=random_seed)
        train_indices, temp_indices = next(splitter.split(dataset, labels))

        temp_dataset = Subset(dataset, temp_indices)
        temp_labels = [labels[idx] for idx in temp_indices]

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size / (test_size + val_size), random_state=random_seed)
        val_indices, test_indices = next(splitter.split(np.array(range(len(temp_dataset))), temp_labels))
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(temp_dataset, val_indices)
        test_dataset = Subset(temp_dataset, test_indices)

        save_indices_to_file(model, train_indices, val_indices, test_indices, 'split_indices')
        print("Split done. Check idx")

    # print("Expected sizes:")
    # print(train_size)
    # print(val_size)
    # print(test_size)
    print("Actual sizes:")
    print("Train dataset length:", len(train_dataset))
    print("Val dataset length:", len(val_dataset))
    print("Test dataset length:", len(test_dataset))
    
    return train_dataset, val_dataset, test_dataset




#############################
def save_indices_to_file(model, train_indices, val_indices, test_indices, save_path='split_indices'):
    os.makedirs(save_path, exist_ok=True)
    
    train_df = pd.DataFrame({'index': train_indices})
    val_df = pd.DataFrame({'index': val_indices})
    test_df = pd.DataFrame({'index': test_indices})

    train_df.to_csv(os.path.join(save_path, f'train_indices_{model}.csv'), index=False)
    val_df.to_csv(os.path.join(save_path, f'val_indices_{model}.csv'), index=False)
    test_df.to_csv(os.path.join(save_path, f'test_indices_{model}.csv'), index=False)


def load_indices_from_file(model, save_path='split_indices'):
    train_df = pd.read_csv(os.path.join(save_path, f'train_indices_{model}.csv'))
    val_df = pd.read_csv(os.path.join(save_path, f'val_indices_{model}.csv'))
    test_df = pd.read_csv(os.path.join(save_path, f'test_indices_{model}.csv'))

    train_indices = train_df['index'].tolist()
    val_indices = val_df['index'].tolist()
    test_indices = test_df['index'].tolist()

    return train_indices, val_indices, test_indices
#############################################################################

##--------------------------------------------------

# To save results to df
def save_results(test_results, result_path, select_model_name, n_epochs, lr, batch_size):
    result_file = result_path + 'test_results.csv'  # Unique CSV file name
    df = pd.DataFrame(test_results).round(4)
    df['model'] = select_model_name
    df['epochs'] = n_epochs
    df['lr'] = lr
    df['batch_size'] = batch_size
    # Check if the file already exists
    if os.path.isfile(result_file):
        existing_df = pd.read_csv(result_file)
        updated_df = pd.concat([existing_df, df], ignore_index=True)  # Append the new rows
    else:
        updated_df = df
    updated_df.to_csv(result_file, index=False, mode='a', header=not os.path.isfile(result_file))  # Append data to the existing file (mode='a')


# def save_best_model(chosen_model, checkpoint_path, model_name):
#     # Save the best model's state in .pth format
#     save_dir = 'Best_Models'
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     # Find the latest numbered checkpoint for the current model
#     existing_models = os.listdir(save_dir)
#     latest_number = max([int(model_file.split('_')[-1].split('.')[0]) for model_file in existing_models if model_file.startswith(f"best_model_{model_name}_")], default=0)

#     # Increment the latest number and use it for the new model file
#     best_model_path = os.path.join(save_dir, f"best_model_{model_name}_{latest_number + 1}.pth")

#     chosen_model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
#     torch.save(chosen_model.state_dict(), best_model_path)
def save_confusion_matrix_as_image(confusion_matrix, result_path, model_name):
    plt.figure(figsize=(8, 6))
    classes = ['Negative', 'Positive']  # Replace with your class names
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(confusion_matrix[i, j]), ha='center', va='center')

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    save_path = os.path.join(result_path, f'confusion_matrix_{model_name}.png')
    plt.savefig(save_path)
    plt.close()