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
import matplotlib.pyplot as plt

### my imports ###
from model_utils import load_h5_array, select_dataset_and_model, split_dataset, save_results, load_datasets_from_indices, load_indices_from_csv
# from BCE_image_model import Image_Model_bce

from image_model_mod_effb4 import Image_Model_effb4
from text_model import Text_Model
from fusion_model import Fusion_Model
# from CM_fusion_model import Fusion_Model
# from image_model_bce import Image_Model_bce
# from focalBCE_image_model import Image_Model_bce_focal
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
    # image_file = os.path.join(paths.output_dir, f"image_outputs_frontal_mod2_resnet_50.h5")  ####using same as resnet50 without convolutional layer!!!!!!!! 
    text_file = os.path.join(paths.output_dir, f"text_outputs_frontal_{select_text_model_name}.h5")

    #loading features
    image_features, text_features, labels = load_h5_array(image_file, text_file, labels)
    
    # select dataset and model
    chosen_model, dataset, tb_logger = select_dataset_and_model(paths.log_path, select_model_name, params.lr, image_features, text_features, labels)
    # print(chosen_model)
    # raise
    #Splitting dataset
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, labels, params.train_ratio, params.val_ratio, params.test_ratio, select_model_name)
    print('Split done. Check idx')
    # print(type(test_dataset))  #<class 'torch.utils.data.dataset.Subset'>
    # raise
    # Extract test labels
    test_labels = np.array([sample[1] for sample in test_dataset])  

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
        callbacks=[early_stop_callback, checkpoint_callback]
        )

    # Fit the model
    trainer.fit(
        chosen_model,
        DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, prefetch_factor=2),
        DataLoader(val_dataset, batch_size=params.batch_size, num_workers=params.num_workers, prefetch_factor=2))

    # Test the best model
    test_results = trainer.test(ckpt_path=checkpoint_callback.best_model_path, dataloaders=DataLoader(test_dataset, batch_size=params.batch_size))


    # Append to CSV (create the file if it doesn't exist)
    with open(f'predictions_labels_{select_text_model_name}.csv', 'w') as f:
        chosen_model.df2.to_csv(f, header=f.tell() == 0, index=False)


    # Save the results in df and png
    save_results(test_results, paths.result_path, select_model_name, params.n_epochs, params.lr, params.batch_size)


    ###############
    # # Save the test image tensor
    # test_image_tensor = torch.stack([sample[0] for sample in test_dataset])
    # test_image_tensor_file = os.path.join(paths.output_dir2, f"test_image_tensor_{select_image_model_name}.pt")
    # torch.save(test_image_tensor, test_image_tensor_file)

    # # Save the test labels to a file
    # test_labels_file = os.path.join(paths.output_dir2, f"test_labels_{select_image_model_name}.npy")
    # np.save(test_labels_file, test_labels)
    ##############


    trainer.logger.experiment.close()

    # track of time
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    minutes = duration.total_seconds() / 60

    print(f"Completed at: {end_time}")
    print(f"Total time (in minutes): {minutes}")
### ----------------------------------------------------------###

if __name__ == "__main__":
    pars = Defining_Parameters()
    par_path = Paths_Configuration()
    test_results = train_model(pars, par_path)

