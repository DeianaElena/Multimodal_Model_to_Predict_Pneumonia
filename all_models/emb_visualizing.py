import h5py
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D

from path_and_parameters import Paths_Configuration, Defining_Parameters

output_dir = '/home/elenad/draeck/edeiana/output_features/full_db/'

# select_text_model_name = 'text_outputs_frontal_distilbert-base-uncased'
select_text_model_name = 'text_outputs_frontal_Clinical-T5-Base'
# select_text_model_name = 'text_outputs_frontal_t5-base'

select_image_model_name = 'image_outputs_frontal_resnet50'


text_file= output_dir + select_text_model_name + ".h5"
image_file =  output_dir + select_image_model_name + ".h5"


# Load the h5 file
with h5py.File(text_file, 'r') as hf:
    text_outputs = hf['text_outputs'][:]

with h5py.File(image_file, 'r') as hf:
    image_outputs = hf['image_outputs'][:]

# with h5py.File('image_outputs_frontal_text_model.h5', 'r') as hf:
#     image_outputs = hf['image_outputs'][:]

par_path = Paths_Configuration()
my_csv = par_path.my_csv; 

df = pd.read_csv(my_csv)

#Theory 
# For word-level embeddings, each point in the plot corresponds to a word. 
# You can label the points with their corresponding words, which allows you 
# to see which words have similar embeddings (and are therefore considered 
# "similar" by the model). For example, you might find that synonyms are located close together in the
# embedding space.

# For text-level embeddings, each point in the plot corresponds to a longer 
# piece of text (like a sentence or document). It's less common to label
# these points because the labels (the entire texts) would be too long. 
# However, if each text has a shorter label or category (like "positive" 
# or "negative" for sentiment analysis), you could use those labels in the plot.

######################### VISUALIZE EMBEDDINGS #########################
############ working in 2d ################

labels = df['label'].values  

tsne = TSNE(n_components=2, random_state=42)
# embeddings_2d = tsne.fit_transform(text_outputs)
embeddings_2d = tsne.fit_transform(image_outputs)
# Define label mapping
label_mapping = {0: 'No Pneumonia', 1: 'Pneumonia'}

# Visualize the 2D embeddings
plt.figure(figsize=(8, 8))

for label in np.unique(labels):
    idx = labels == label
    plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=label_mapping[label])

plt.legend()
plt.show()


## ------------------------------------------------- 

# labels = df['label'].values  

# # Perform t-SNE for 3 components
# tsne = TSNE(n_components=3, random_state=42)
# embeddings_3d = tsne.fit_transform(text_outputs)

# # Define label mapping
# label_mapping = {0: 'No Pneumonia', 1: 'Pneumonia'}

# # Visualize the 3D embeddings
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')

# for label in np.unique(labels):
#     idx = labels == label
#     ax.scatter(embeddings_3d[idx, 0], embeddings_3d[idx, 1], embeddings_3d[idx, 2], label=label_mapping[label])

# plt.legend()
# plt.show()

## -------------------------------------------------
# ########### 3D with plotly express but big dots ###########
# import numpy as np
# from sklearn.manifold import TSNE
# import plotly.express as px
# import pandas as pd

# # Assume df is your DataFrame, and image_outputs, text_outputs are your embeddings
# labels = df['label'].values  # replace 'label' with the actual column name for the labels

# # Perform t-SNE for 3 components
# tsne = TSNE(n_components=2, random_state=42)
# embeddings_3d = tsne.fit_transform(text_outputs)

# # Define label mapping
# label_mapping = {0: 'No Pneumonia', 1: 'Pneumonia'}
# mapped_labels = [label_mapping[label] for label in labels]

# # Create a DataFrame for Plotly
# df_plotly = pd.DataFrame(embeddings_3d, columns=['x', 'y', 'z'])
# df_plotly['label'] = mapped_labels
# df_plotly['size'] = 0.5 

# # Visualize the 3D embeddings
# fig = px.scatter_3d(df_plotly, x='x', y='y', z='z', color='label', title='t-SNE visualization of embeddings')
# fig.show()
