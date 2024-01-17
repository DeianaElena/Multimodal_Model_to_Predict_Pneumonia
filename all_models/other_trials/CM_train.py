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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

### my imports ###
from model_utils import save_confusion_matrix_as_image, load_h5_array, select_dataset_and_model, split_dataset, save_results, load_datasets_from_indices, load_indices_from_csv
from image_model_mod_effb4 import Image_Model_effb4
from text_model import Text_Model
from CM_fusion_model import Fusion_Model
from image_model_bce import Image_Model_bce
from focalBCE_image_model import Image_Model_bce_focal
from path_and_parameters import Paths_Configuration, Defining_Parameters
from path_and_parameters import all_text_models, all_image_models, select_image_model_name, select_text_model_name, select_model_name 

def train_model(params, paths):
    random_seed = random.seed(28)
    start_time = datetime.datetime.now()
    print(f"Start time: {start_time}")

    my_csv = paths.my_csv
    df = pd.read_csv(my_csv)
    labels = df['label'].values

    image_file = os.path.join(paths.output_dir, f"image_outputs_frontal_{select_image_model_name}.h5")
    text_file = os.path.join(paths.output_dir, f"text_outputs_frontal_{select_text_model_name}.h5")

    #loading features
    image_features, text_features, labels = load_h5_array(image_file, text_file, labels)
    
    # select dataset and model
    chosen_model, dataset, tb_logger = select_dataset_and_model(paths.log_path, select_model_name, params.lr, image_features, text_features, labels)

    #Splitting dataset
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, labels, params.train_ratio, params.val_ratio, params.test_ratio, select_model_name)
    print('Split done. Check idx')

    #Adding early stop to fight overfitting
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=params.early_stop_patience,  
        verbose=False,
        mode='min')

    # Define the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=paths.draek_path +f'checkpoints/{select_model_name}/',
        filename='best-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=5,  # -1 to save all model, otherwise the best k models according to what selected
        mode='min'  # for val loss
    )

    # Define the trainer with the checkpoint callback
    trainer = pl.Trainer(
        max_epochs=params.n_epochs,
        logger=tb_logger,     # Use the TensorBoardLogger
        callbacks=[early_stop_callback, checkpoint_callback])

    # Fit the model
    trainer.fit(
        chosen_model,
        DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, prefetch_factor=2),
        DataLoader(val_dataset, batch_size=params.batch_size, num_workers=params.num_workers, prefetch_factor=2))

    return trainer, test_dataset, checkpoint_callback

def test_model(trainer, test_dataset, checkpoint_callback, params, paths):
    # Test the best model
    test_results = trainer.test(ckpt_path=checkpoint_callback.best_model_path, dataloaders=DataLoader(test_dataset, batch_size=params.batch_size))

    # Save the results in df and png
    save_results(test_results, paths.result_path, select_model_name, params.n_epochs, params.lr, params.batch_size)

    return test_results

def compute_and_plot_cm(test_results):
    # Extract the predictions and labels from the test results
    all_preds = []
    all_labels = []
    for result in test_results:
        all_preds.append(result['preds'])
        all_labels.append(result['labels'])

    # Concatenate all the predictions and labels
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Print it as a table
    print("Confusion Matrix:")
    print(pd.DataFrame(cm))

    # Plot it as a heatmap
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save the figure
    plt.savefig("confusion_matrix.png")

if __name__ == "__main__":
    pars = Defining_Parameters()
    par_path = Paths_Configuration()
    trainer, test_dataset, checkpoint_callback = train_model(pars, par_path)
    test_results = test_model(trainer, test_dataset, checkpoint_callback, pars, par_path)
    compute_and_plot_cm(test_results)
