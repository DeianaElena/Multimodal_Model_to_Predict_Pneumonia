import os

#### MODELS ####
all_text_models = ['Clinical-Longformer', 
                   'distilbert-base-uncased', 't5-base',
                   'Bio_Discharge_Summary_BERT', 'Clinical-T5-Base']
all_image_models = ['resnet50', 'efficientnet_b4', 'mod2_resnet_50', 'mod3_resnet_50', 'mod_effb4']

############## Manually change this accordingly #############
# select_image_model_name = 'resnet50'
# select_image_model_name = 'efficientnet_b4'
# select_image_model_name  = 'mod2_resnet_50'
# select_image_model_name  = 'mod3_resnet_50'   #TODO fix issue with this 
select_image_model_name = 'mod_effb4'

# select_text_model_name = "distilbert-base-uncased"
# select_text_model_name = 'Clinical-Longformer'
# select_text_model_name = 't5-base'
select_text_model_name = "Clinical-T5-Base"
# select_text_model_name = "Clinical-T5-Scratch" #TODO to be tested



# ###### Model selection ######
select_model_name = select_image_model_name
# select_model_name = select_text_model_name
# select_model_name = select_text_model_name + '_' + select_image_model_name

######################################################

########### Folders and path selection ###########
class Paths_Configuration:
    def __init__(self):
        # Path (Windows path, Linux path)
        self.draek_path = '/home/elenad/draeck/edeiana/'
        self.repo_path = '/home/elenad/Documents/gh/thesis_code_final/'
        self.image_dir = self.draek_path

        # folder with all csv files
        self.path = self.draek_path + 'verified'

        ##### FULL output features #####
        self.output_dir = '/home/elenad/draeck/edeiana/output_features/full_db'
        self.output_dir2 = '/home/elenad/draeck/edeiana/output_features'
        self.my_csv = '{0}/{1}'.format(self.path, 'FRONTAL_gxray_dicom_discharge_filt_text.csv')  # full db with only frontal images

        ##### 4k output features #####
        # self.output_dir = '/home/elenad/draeck/edeiana/output_features/4k_output'
        # self.my_csv = '{0}/{1}'.format(self.path, 'sub4k_gxray_dicom_discharge_filt_text.csv')

        # for training and testing
        self.log_path = self.repo_path + 'all_models/'  # to save logs in all models folder
        self.result_path = self.repo_path + 'Results/'  # to save results in general folder Results
        self.output_folder = 'output_features'
       

############ Parameters ############
class Defining_Parameters:
    def __init__(self):
        self.train_ratio = 0.8  # 80% training ratio
        self.val_ratio = 0.1
        self.test_ratio = 0.1

        self.lr = 0.0001    #0.0001 maybe too fast
        self.n_epochs = 20  #25 number of epochs
        self.batch_size = 32  # 16 in paper but depend on what the machine can support
        self.num_workers = 4

        self.early_stop_patience = 5  #to change later