#libraries
import pandas as pd
import matplotlib.pyplot as plt
import random
import re

#my imports
from text_utils import normalize_df_column, filter_text
from sections import accepted_list, rejected_list

#paths
draek_path = '/home/elenad/draeck/edeiana/'
discharge_note_path = draek_path+'/mimic-iv-note/2.2/note/discharge.csv'

PATH_verified = draek_path + "verified/preprocessing_"
PATH_preprocess = draek_path + "verified/preprocessing/"

full_df = pd.read_csv(discharge_note_path)
# small_df = full_df.head(10)
# small_df.to_csv('small_df.csv')


frontal_no_text = pd.read_csv(PATH_verified+"FRONTAL_gxray_dicom_discharge_filt__no_filtered_text.csv")

######## REMOVE SECTIONS ########
few_sections_df = filter_text(full_df, accepted_list, rejected_list)
# print(few_sections_df.head())

########### NORMALIZE ###########
filtered_discharge_df = normalize_df_column(few_sections_df) 
filtered_discharge_df.to_csv(PATH_preprocess+'filtered_discharge_df_full_with_text_col2.csv')
print('Filtered text done!')

###### DROP text col ######
# filtered_discharge_df = pd.read_csv(PATH_preprocess+'filtered_discharge_df_full_with_text_col.csv')
# filtered_discharge_df_without_text_col  = filtered_discharge_df. drop(['text'], axis=1)
# filtered_discharge_df_without_text_col.to_csv(PATH_preprocess+'filtered_discharge_df_without_text_col.csv')

# filtered_discharge_df_without_text_col = pd.read_csv(PATH_preprocess+'filtered_discharge_df_without_text_col.csv')
# FRONTAL_gxray_dicom_discharge_filt_text = pd.read_csv(PATH_verified+"FRONTAL_gxray_dicom_discharge_filt_text.csv")
# #drop previous text filtered_column
# FRONTAL_gxray_dicom_discharge_filt_text.drop(['filtered_text'], axis=1)

# # #merging
# norm_filtered_frontal = pd.merge(FRONTAL_gxray_dicom_discharge_filt_text, filtered_discharge_df_without_text_col[['subject_id', 'hadm_id','filtered_text']], on=['subject_id', 'hadm_id'], how='inner')
# norm_filtered_frontal.to_csv(PATH_preprocess+'norm_filtered_frontal.csv')
# print(norm_filtered_frontal.head())
