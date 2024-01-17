import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2_contingency

# from path_and_parameters import Paths_Configuration, Defining_Parameters
# from path_and_parameters import all_text_models, all_image_models, select_image_model_name, select_text_model_name, select_model_name 

# # 'Clinical-T5 + EfB4-2' 
# # 'Efficientnet B4 v2'
# # 'Clinical-T5'


file_t = 'predictions_labels_Clinical-T5-Base.csv'
file_i = f'predictions_labels_mod_effb4.csv'
file_f = 'predictions_labels_mod_effb4_Clinical-T5-Base.csv'

# Step 2: Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_i)

# Assuming the columns are named "predictions" and "labels"
predicted_labels = df['predictions']
actual_labels = df['labels']

# Step 3: Create the confusion matrix
conf_matrix = confusion_matrix(actual_labels, predicted_labels)

# # Step 4: Create a heatmap plot of the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlOrBr', cbar=False) # Change 'Blues' to 'YlGnBu'
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.xticks(ticks=[0.5, 1.5], labels=['Normal', 'Pneumonia'])  # Set x-axis tick labels
# plt.yticks(ticks=[0.5, 1.5], labels=['Normal', 'Pneumonia'])  # Set y-axis tick labels
# plt.title(f'Confusion Matrix of Clinical-T5 + EfB4-2')
# # plt.title(f'Confusion Matrix of Efficientnet B4 v2')
# # plt.title(f'Confusion Matrix of Clinical-T5')

# # Step 5: Save the confusion matrix as a CSV file
# # Convert confusion matrix to a DataFrame
# conf_matrix_df = pd.DataFrame(conf_matrix, index=['Normal', 'Pneumonia'], columns=['Normal', 'Pneumonia'])
# conf_matrix_csv_file = f'CM/conf_matrix_fusion.csv'
# conf_matrix_df.to_csv(conf_matrix_csv_file)

# # Step 6: Save the plot as an image

# plt.savefig(f'CM/conf_matrix_Clinical-T5+EfB4-2.png', bbox_inches='tight')
# # plt.savefig(f'CM/conf_matrix_EfB4-2.png', bbox_inches='tight')
# # plt.savefig(f'CM/conf_matrix_Clinical-T5.png', bbox_inches='tight')

from scipy.stats import chi2_contingency
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from path_and_parameters import Paths_Configuration, Defining_Parameters
from path_and_parameters import all_text_models, all_image_models, select_image_model_name, select_text_model_name, select_model_name

# Increase the default font size for the entire plot
plt.rcParams.update({'font.size': 20})


# Step 4: Create the confusion matrix
conf_matrix = confusion_matrix(actual_labels, predicted_labels)

# Convert confusion matrix to a DataFrame
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Normal', 'Pneumonia'], columns=['Normal', 'Pneumonia'])

# Step 5: Save the confusion matrix as a CSV file
conf_matrix_csv_file = f'CM/conf_matrix_img.csv'
conf_matrix_df.to_csv(conf_matrix_csv_file)

# Step 6: Create a heatmap plot of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlOrBr', cbar=False,
            annot_kws={'size': 32})  # Change 'Blues' to 'YlGnBu', adjust font size as needed
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks(ticks=[0.5, 1.5], labels=['Normal', 'Pneumonia'])  # Set x-axis tick labels
plt.yticks(ticks=[0.5, 1.5], labels=['Normal', 'Pneumonia'])  # Set y-axis tick labels
# plt.title(f'Confusion Matrix of Clinical-T5 + EfB4-2')
plt.title(f'Confusion Matrix of Efficientnet B4 v2')
# plt.title(f'Confusion Matrix of Clinical-T5')

# Step 7: Save the plot as an image
# plt.savefig(f'CM/conf_matrix_Clinical-T5+EfB4-2.png', bbox_inches='tight')
plt.savefig(f'CM/conf_matrix_EfB4-2.png', bbox_inches='tight')
# plt.savefig(f'CM/conf_matrix_Clinical-T5.png', bbox_inches='tight')


plt.show()


