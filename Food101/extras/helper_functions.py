import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random, itertools, os, datetime, zipfile


# Create function to unzip a zipfile into current working directory
# (since we're going to be downloading and unzipping a few files)
def unzip_data(filename):
    """
    Unzips filename into the current working directory.
    After unzip, removes zipfile.

    Args:
    filename (str): a filename to a target zip folder to be unzipped.
    """
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('.')
    os.remove(filename)

# Walk through an image classification directory and find out how many files (images)
# are in each subdirectory.
def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.

    Args:
      dir_path (str): target directory

    Returns:
      A print out of:
        number of subdiretories in dir_path
        number of images (files) in each subdirectory
        name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

# Gets the names of labels based on subfolders' names of root path.
def print_dir_labels(dir_path):
    """
    Walks through dir_path and prints out the label of each subdirectory,
    which represents tha name of category images could belong to.

    Args:
        dir_path (str): target directory
    """
    cat_names = []
    for label in os.listdir(dir_path):
        cat_names.append(label)
    print(cat_names)

# Gets a random image path.
def random_image(dir_path, classes):
    """
    Walks through dir_path, choose a random image from a random subdirectory
    and return its path.
    Args:
         dir_path (str): target directory
         classes (list): a list of possible classes of images.
    Returns:
        (list): a list of random category name and path.
    """
    rand_cat = random.choice(classes)
    rand_img = random.choice(os.listdir(dir_path + rand_cat + '/'))
    path = dir_path + rand_cat + '/' + rand_img

    return [rand_cat, path]

def create_tensorboard_callback(dir_name, experiment_name):
    """
    Creates a TensorBoard callback instance to store log files.

    Stores log files with the filepath:
    "dir_name/experiment_name/current_datetime/"

    Args:
    dir_name: target directory to store TensorBoard log files
    experiment_name: name of experiment directory (e.g. efficientnet_model_1)
    """
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")

    return tensorboard_callback

# Plots the loss curves and accuracy based on training and validation results.
def plot_loss_curves(history):
    """
    Returns separate loss curves for training and validation metrics.

    Args:
    history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Plot loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();

# Compares history of original model and after fine-tuning.
def compare_histories(original_history, new_history, initial_epochs = 5):
    """
    Compares two TensorFlow model History objects.

    Args:
      original_history: History object from original model (before new_history)
      new_history: History object from continued model training (after original_history)
      initial_epochs: Number of epochs in original_history (new_history plot starts from here)
    """

    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs - 1, initial_epochs - 1],
             plt.ylim(), label='Start Fine Tuning')  # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs - 1, initial_epochs - 1],
             plt.ylim(), label='Start Fine Tuning')  # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
