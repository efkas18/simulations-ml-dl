import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random, itertools, os, datetime, zipfile
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

## Walks through an image classification directory and find out how many files (images) are in each subdirectory.
def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    ----
    Args:
      dir_path (str): target directory
    ----
    Returns:
      A print out of:
        number of subdiretories in dir_path
        number of images (files) in each subdirectory
        name of each subdirectory
    ----        
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

## Visualisation of images.
def view_random_image(target_dir, target_class):
    """
    Returns a random image of class from specific directory.
    ----
    Args:
      target_dir (str): path of target directory.
      target_class (str): name of directory represents a class.
    ----
    Returns:
      Image object.
    ----      
    """    
    # Setting up the target directory.
    target_folder = target_dir + target_class
    # Gets random Image Path.
    random_image = random.sample(os.listdir(target_folder), 1)
    print(f"{random_image}\n")
    # Reads the image and plot it using matplotlb.
    img = mpimg.imread(target_folder + "/" + random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off")
    print(f"Image shape: {img.shape}")
    
    return img

## Plot the validation and training data separately
def plot_loss_curves(history):
    """
    Returns separate loss curves for training and validation metrics.
    ----
    Args: 
        history (obj): The history object of neural network model.
    ----        
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    
    epochs = range(len(history.history['loss']))
    
    # Plot loss
    plt.figure(figsize = (14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label = 'training_loss')
    plt.plot(epochs, val_loss, label = 'val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend();
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label = 'training_accuracy')
    plt.plot(epochs, val_accuracy, label = 'val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();


## Calculating significant metrics binary - classification.
def calculate_binary_classification_metrics(y_true, y_pred):
    """
    Calculates accucacy, precision, recall and f1-score of binary classification.
    ----
    Args:
        y_true (array): Array of actual targets.
        y_pred (array): Array of predicted targets.
    ----
    Return:
        Returns result dataframe of model.
    ----        
    """
    # Calcualating the metrics to evaluate the succession of each classifier.
    results = []
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average = 'weighted')
    precision = precision_score(y_true, y_pred, average = 'weighted')
    recall = recall_score(y_true, y_pred, average = 'weighted')
    results.append({
        "accuracy (%)": round(accuracy, 2) * 100,
        "f1-Score": round(f1, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
    })
    return pd.DataFrame(results)

## A custom version of Confusion Matrix using custom function.
def custom_confusion_matrix(y_true, y_pred, title = "Confusion Matrix"):
    """
    A custom version of confusion metrix.
    ----
    Args:
        y_true (array): Array of actual targets.
        y_pred (array): Array of predicted targets.    
        title (str): Title of figure.
    ----        
    """
    cf_matrix = confusion_matrix(y_true, y_pred)
    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    
    ax = plt.axes()
    sns.heatmap(cf_matrix,
                annot = labels, 
                fmt = "", 
                cmap = "crest",
                linewidth = 1);
    ax.set_title(title);
