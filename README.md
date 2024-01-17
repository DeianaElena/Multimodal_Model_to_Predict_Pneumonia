# To run experiments:
Main code is found in "all_models".
The experiment can be run in two separate steps: 
1. Store of the Features
2. Train/test

# To know before running experiments:
- storing_all_features.py: after selecting by commenting and uncommmenting the selected models, run this to store the Features
        - storing_only_img_features: useful only for image Features

- path_and_parameters.py: make sure to uncomment which image, text model you want to use and also if you want to do only image,text or combined model.

- train.py:run this after selecting path and parameters to train the sepcific model. 

# GRADCAM:
Either use gradcam_mod_efficientnet for gradcam on efficientnet or gradcam_mod2_resnet50 for using resnet 50. 
