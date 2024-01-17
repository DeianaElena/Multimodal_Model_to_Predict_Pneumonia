import pandas as pd
import matplotlib.pyplot as plt

draek_path = '/home/elenad/draeck/edeiana/verified/'
frontal_csv = "FRONTAL_gxray_dicom_discharge_filt_text.csv"

age_sex = pd.read_csv(draek_path+"sex_age_q.csv")

# Create or load your DataFrame
data = pd.read_csv(draek_path+frontal_csv)
# Drop the 'col1' column
data_without_text = data.drop('filtered_text', axis=1)

# Print the DataFrame without 'col1'
print(len(data_without_text)) #45103


#  Merge the two dataframes on 'subject_id' and 'hadm_id'
merged_data = data.merge(age_sex[['subject_id', 'hadm_id', 'gender', 'age']], on=['subject_id', 'hadm_id'], how='left')

# Drop duplicate rows
merged_data.drop_duplicates(subset=['subject_id', 'hadm_id'], inplace=True)
print(len(merged_data))
# Save the merged data to a new CSV file
merged_data.to_csv(draek_path+'frontal+sex_age.csv', index=False)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.pyplot as plt

# Map 'F' to 'Female' and 'M' to 'Male'
merged_data['sex_mapped'] = merged_data['gender'].map({'F': 'Female', 'M': 'Male'})

# Set style
sns.set(style="whitegrid", font_scale=1.8)  # Increase the font scale

# Create a box plot
plt.figure(figsize=(8, 10))  # Increase figure size for taller plot
my_pal = {"Female": "m", "Male": "g"}
sns.boxplot(x="sex_mapped", y="age", data=merged_data, palette=my_pal, width=0.4)  # Decreased width for thinner boxes

plt.title('Age and Sex Distribution', fontsize=18)
plt.xlabel('Sex', fontsize=18)
plt.ylabel('Age', fontsize=18)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.gca().spines['left'].set_linewidth(2)   # Increase left y-axis thickness
plt.gca().spines['bottom'].set_linewidth(2) # Increase x-axis thickness

plt.tight_layout()
plt.show()


print(merged_data.columns)

# # Get the top 15 ICD codes
# top_icd_codes = merged_data['icd_code'].value_counts().head(15).index

# # Filter the data for the top 15 ICD codes
# filtered_data = merged_data[merged_data['icd_code'].isin(top_icd_codes)]
# # Print the names of the top 15 ICD codes
# print("Top 15 ICD Codes:")


unique_subjects = merged_data['subject_id'].nunique()
print(f'There are {unique_subjects} unique subjects in the dataset.')  #28550



# Define a dictionary to map ICD codes to short descriptions (populate this)
icd_short_description_mapping = {
    '486': 'Pneumonia',
    '0389': 'Septicemia',
    'A419': 'Sepsis, unspecified',
    '41401': 'Coronary atherosclerosis',
    '42833': 'Acute on chronic diastolic heart failure',
    '41071': 'Subendocardial infarction, initial episode',
    '42823': 'Acute on chronic systolic heart failure',
    '5849': 'Acute kidney failure',
    '42731': 'Atrial fibrillation',
    '5990': 'Urinary tract infection, site not specified',
    '78659': 'Chest pain, unspecified',
    '7802': 'Syncope and collapse',
    '431': 'Intracerebral hemorrhage',
    '43411': 'Cerebral embolism',
    'I214': 'Other acute ischemic heart diseases'
}

import pandas as pd
import matplotlib.pyplot as plt

# Count the frequency of each code
icd_counts = merged_data['icd_code'].value_counts(ascending=False).head(10)  # Select the top 15

# Prepare the labels using the mapping dictionary
labels = [f"{code}: {icd_short_description_mapping.get(code, code)}" for code in icd_counts.index]

# Create a colormap
colormap = plt.cm.get_cmap('tab20', 10)

plt.figure(figsize=(10, 6))
bars = plt.bar(icd_counts.index, icd_counts.values, color=[colormap(i) for i in range(10)]) # Using codes as x labels

plt.xlabel('ICD Code', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.title('Distribution of Top 10 ICD Codes', fontsize=20)
plt.xticks(rotation=90, fontsize=18)  # Rotates the x labels for better visibility
plt.yticks(fontsize=18)
plt.legend(bars, labels, loc='upper right', fontsize=16)  # Places the legend in the upper right
plt.tight_layout()  # Ensures that labels fit into the figure
plt.show()









