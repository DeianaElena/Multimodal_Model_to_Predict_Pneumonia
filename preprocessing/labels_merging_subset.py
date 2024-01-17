#libraries
import pandas as pd
import re
from collections import Counter

#my functions
from text_utils import create_smaller_db, normalize_df_column, extract_recurrent_note_categories, filter_text, from_code_add_label_col

#old paths #TODO to be verified 
PATH = "../../data/db/mimic_note/"  #contains 2 files of discharge notes with and without filtered text
PATH_M = "../../data/db/merged/"
PATH_G = "../../data/db/"

#new paths
draek_path = '/home/elenad/draeck/edeiana/'
PATH_verified = draek_path + "verified/preprocessing_"
PATH_cxr = draek_path + "mimic_cxr_sr/"


gxray_dicom_discharge = pd.read_csv(PATH_M + 'gxray_dicom_discharge.csv')
diagnoses_icd = pd.read_csv(PATH_G + 'diagnoses_icd.csv')
# print('len diagnoses_icd: ', len(diagnoses_icd))  #4756326
# print('duplicates diagnoses_icd: ', diagnoses_icd.duplicated().sum())  #0

path_dicom_df = pd.read_csv(PATH_cxr + 'cxr-record-list.csv')

# ### temporarly dropping text column for faster merging
gxray_dicom_discharge = gxray_dicom_discharge.drop(['text'], axis=1)

####### ADDING ICD CODES ########
def adding_icd_codes(df, pneumonia_code_list):
    # Merge to have icd code
    df_icd = pd.merge(df, diagnoses_icd[['subject_id', 'hadm_id', 'seq_num', 'icd_code']], on=['subject_id', 'hadm_id'], how='inner')
    # print('len : ', len(df_icd))  #3647479 BECAUSE MORE ICD CODE ARE REGISTERED IN SAME PATIENT AND ADMISSION
    # print('duplicates : ', df_icd.duplicated().sum())  #176!!!!!!!!!!!!
    df_icd_no_dupl_all_secondary_icd = df_icd.drop_duplicates()
    #ADDING LABELS IN CASE THERE IS PNEUMONIA OR NOT INDEPENDLY OF SEQ_NUM
    discharge_pneumo_label_all_secondary_icd = from_code_add_label_col(df_icd_no_dupl_all_secondary_icd, pneumonia_code_list)
    return discharge_pneumo_label_all_secondary_icd

###### FILTERING BASED ON LABELS ANS SEQ NUM ######
def filter_interesting_icd(df, PATH_verified):
    # Filter based on labels and seq_num, to select 
    mask = (df['label'] == 1) | (df['seq_num'] == 1)
    interesting_icd_df = df[mask].reset_index(drop=True)
    interesting_icd_df.to_csv(PATH_verified+'interesting_icd_df2.csv')  #212366 (before 196068)
    return interesting_icd_df

###### ADD JPG PATH ######
def add_jpg_path(df):
    # Merge with path_dicom_df and replace path extension
    discharge_pneumo_label_path = pd.merge(df, path_dicom_df[['subject_id', 'dicom_id', 'path']], on=['subject_id', 'dicom_id'])
    #replace the path name from dicom to jpg
    discharge_pneumo_label_path['path'] = discharge_pneumo_label_path['path'].str.replace('.dcm', '.jpg')
    return discharge_pneumo_label_path

def filter_duplicates(df):
    # For label == 1, only take that admission once (the latest) 
    df.drop_duplicates(subset='dicom_id', keep='last', inplace=True)
    return df

def filter_on_poe_seq(df):
    # Create a new column 'min_poe_id' that contains the minimum 'poe_id' for each group
    df['min_poe_id'] = df.groupby(['subject_id', 'hadm_id'])['poe_id'].transform('min')
    # Filter the DataFrame to keep only the rows where 'poe_id' matches the minimum 'poe_id' for each group
    df = df[df['poe_id'] == df['min_poe_id']].drop(columns=['min_poe_id'])
    return df

def save_dataframe(df, path):
    # Save the dataframe to a CSV file
    df.to_csv(path, index=False)

# -----------------------------------------------------------------
# Set the DataFrame name and pneumonia code list
df_name = pd.read_csv(PATH_M +'gxray_dicom_discharge.csv')
pneumonia_code_list = ['486']

# Adding ICD codes
df_icd = adding_icd_codes(df_name, pneumonia_code_list)

# Filtering based on labels and seq_num
# interesting_icd_df = filter_interesting_icd(PATH_verified+df_icd)
interesting_icd_df = filter_interesting_icd(df_icd, PATH_verified)

# Add JPG path
discharge_pneumo_label_path = add_jpg_path(interesting_icd_df)

# Filter duplicates
filtered_df = filter_duplicates(discharge_pneumo_label_path)

# Filter based on poe seq
filtered_icd_duplicates = filter_on_poe_seq(filtered_df)

# Save the modified DataFrame
save_dataframe(filtered_icd_duplicates, PATH_verified + 'filtered_icd_duplicates.csv')
print('Complete')
# -----------------------------------------------------------------

filtered_icd_duplicates = pd.read_csv(PATH_verified + 'filtered_icd_duplicates.csv')
print('Labels count: ', filtered_icd_duplicates['label'].value_counts())
# # Positie labels considering seq num and poe id 
# # 0    103904
# # 1      7474  ~7.19%

# ####### REMOVING duplicates based on admission #####
filtered_icd_duplicates.drop_duplicates(subset='hadm_id', keep='last', inplace=True)  #keping the last row for same admsission
print('Labels df before splitting: ', filtered_icd_duplicates['label'].value_counts()) 
# # 0    58043
# # 1     4196 ~7.2%

# ############### SELECTING ONLY FRONTAL XRAY ############# 
def filter_frontal_df(df):
    frontal_df = df[df['ViewCodeSequence_CodeMeaning'].isin(['antero-posterior', 'postero-anterior'])]
    # print('len frontal_df: ', len(frontal_df)) #45103
    frontal_df.to_csv(PATH_verified + 'frontal_df.csv')
    return frontal_df

# #################### LAST MERGING ################

# Filter frontal_df
frontal_df = filter_frontal_df(filtered_icd_duplicates)
print('len frontal_df: ', len(frontal_df))   #len frontal_df:  45103
print('Labels count only frontal: ', frontal_df['label'].value_counts()) 
# Labels count only frontal:  label
# 0    41960
# 1     3143

print('Duplicates for subset of frontal_df: ', frontal_df.duplicated(subset=['hadm_id']).sum())  #0
frontal_df.to_csv(PATH_verified+"FRONTAL_gxray_dicom_discharge_filt__no_filtered_text.csv")

# Merge filtered text with frontal_df
# FRONTAL_gxray_dicom_discharge_filt_text = add_back_filtered_text(frontal_df, PATH+"disharge_full_only_filtered_text.csv")
# FRONTAL_gxray_dicom_discharge_filt_text.to_csv(PATH_verified+"FRONTAL_gxray_dicom_discharge_filt_text.csv")

# print('len FRONTAL_gxray_dicom_discharge_filt_text: ', len(FRONTAL_gxray_dicom_discharge_filt_text))
# print('Duplicates for subset of FRONTAL_gxray_dicom_discharge_filt_text: ', FRONTAL_gxray_dicom_discharge_filt_text.duplicated(subset=['hadm_id']).sum())
# print('Duplicates for full FRONTAL_gxray_dicom_discharge_filt_text: ', FRONTAL_gxray_dicom_discharge_filt_text.duplicated().sum())

# ########### Create smaller database ###########
# sub4k_gxray_dicom_discharge_filt_text = create_smaller_db(FRONTAL_gxray_dicom_discharge_filt_text, 4000, 0.2, 0.8)
# print('Len:', len(sub4k_gxray_dicom_discharge_filt_text))
# print('Labels count: ', sub4k_gxray_dicom_discharge_filt_text['label'].value_counts())
# # Labels count:  label
# # 0    3728
# # 1     272
