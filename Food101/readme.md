# Food101 Dataset

## Description
Based on **Food101** model of https://www.kaggle.com/datasets/dansbecker/food-101 .
Dataset is a variation of original dataset from kaggle. Consist of 7500 images for training and 2500 images for validation.
All images of training and testing datasets are equally distributed at 10 labels (categories).

A Convolution Neural Network used based on **Functional API** of **tensorflow.keras.applications** module and particularly 
the **EfficientNetV2B0** architecture.

### A. cnn_transfer_learining_models_food_101:
Performed simulations on 3 different model architectures:
1. **model_0**: Attach base_model (EfficientNetV2B0) with **untrained layers** and **original** images on model.
2. **model_1**: Attach base_model with **untrained layers** and **augmented** images on model.
3. **model_1 (fine-tuning)**: Fine-tuning of **model_1**, **enabling train at last 10 layers** of base_model, with augmented images, continuing the training of **model_1**.

### B. food_vision_food101:
In this case performed simulations with whole training dataset and 101 different classes.
The same methedology was excecuded, by creating a base model with Functional API, and impoved step by step enabling at the end all layers of **efficientnet** functional model.

After simulations of each model a visualisation following by plots of loss and accuracy curves and comparison of model of model before and after feature extraction / fine-tuning.

## Notes
1. The script can be used on different datasets, but the dataset of images must be as the structure will follow:
   * root (dataset_directory)
     * train 
       * label_1 (directory contains images representing object of label_1)
       * label_2 (directory contains images representing object of label_2)
       * ...
       * label_N (directory contains images representing object of label_N)
     * test
       * label_1 (directory contains images representing object of label_1)
       * label_2 (directory contains images representing object of label_2)
       * ...
       * label_N (directory contains images representing object of label_N)
2. The cell which proceeds to download the dataset for simulations, can be commented after the download. Furthermore, the URL of !wget command can be replaced with your own. Note that if the downloaded dataset is not as the structure explained at **Note 1**, then it is necessary to modified it !
3. In order to run at resonable time the model, it is necessary a system with **NVIDIA GPU**, because the tensor oparations are a lot for CPU to handle them.


