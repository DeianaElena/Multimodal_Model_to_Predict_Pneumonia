import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Creating a dictionary to represent your data
data = {'Model': ['BERT Base', 'Clinical-Longformer (CL)', 'T5', 'Clinical-T5',
                  'Resnet-50 (R50)', 'Resnet-50 v2 (R50-2)', 'Efficientnet B4 (EfB4)', 'Efficientnet B4 v2 (EfB4-2)',
                  'BERT + R50', 'BERT + EfB4', 'CL + EfB4', 'T5 + EfB4',
                  'Clinical-T5 + R50', 'Clinical-T5 + R50-2', 'Clinical-T5 + EfB4', 'Clinical-T5 + EfB4-2'],
        'Sensitivity': [46, 34, 54, 61, 46, 35, 47, 61, 45, 52, 46, 54, 56, 54, 63, 53],
        'Type': ['Text', 'Text', 'Text', 'Text',
                 'Image', 'Image', 'Image', 'Image',
                 'Fusion', 'Fusion', 'Fusion', 'Fusion',
                 'Fusion', 'Fusion', 'Fusion', 'Fusion']}

# Creating a DataFrame from the dictionary
df = pd.DataFrame(data)

# Define custom colors for each type of model
colors = {'Text': 'skyblue', 'Image': 'lightgreen', 'Fusion': 'lightcoral'}

# Set Seaborn style
sns.set(style="whitegrid")

# Create a list to store the original order of the models
original_order = data['Model']

# Create a bar plot of the data using Seaborn
plt.figure(figsize=(10, 6))  # Adjust size as needed
sns.barplot(x='Sensitivity', y='Model', hue='Type', data=df, palette=colors, order=original_order, dodge=False)

plt.xlabel('Sensitivity')
plt.title('Model Performance Based on Sensitivity')
plt.legend(title='Model Type')
plt.tight_layout()  # Adjust layout for better alignment
plt.show()
