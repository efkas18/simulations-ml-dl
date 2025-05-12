import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random, itertools, os, datetime, zipfile, time

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


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
    
    return cat_names;

# Gets a random image path.
def random_image(dir_path, classes):
    """
    Walks through a root dir_path, choosing a random image from a random subdirectory
    and returns list consist of class and path of image was chosen.
    Args:
         dir_path (str): root of target directory.
         classes (list): a list of possible classes of images.
    Returns:
        (list): a list of random category name and path.
    """
    rand_cat = random.choice(classes)
    rand_img = random.choice(os.listdir(dir_path + rand_cat + '/'))
    path = dir_path + rand_cat + '/' + rand_img

    return [rand_cat, path]

def create_tensorboard_callback(dir_name, experiment_name, keras_version = 3):
    """
    Creates a TensorBoard callback instance to store log files.

    Stores log files with the filepath:
    "dir_name/experiment_name/current_datetime/"

    Args:
    dir_name: target directory to store TensorBoard log files
    experiment_name: name of experiment directory (e.g. efficientnet_model_1)
    keras_version (default == 3): if "tf_keras" module is used (e.g. transfer learning), change keras_version == 2.
    """
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if keras_version == 3:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir)
    else:
        tensorboard_callback = tf_keras.callbacks.TensorBoard(log_dir = log_dir)        
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

# Creates checkpoint callback to save Keras model or model weights at some frequency.
def model_checkpoint_callback(checkpoint_path):
    """
    Saves only the keras model's best weights to a filepath, at some frequency.
    Args:
         checkpoint_path (str): target filepath to save the model's weights.
    """
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                             save_weights_only = True,  # saves only the model weights.
                                                             monitor = "val_accuracy", # saves the model weights which score the best validation accuracy.
                                                             save_best_only = True)  # only keeps the best model weights on file (delete the rest).

    return checkpoint_callback

# Creates a sequential keras layer for random augmentation of data.
def data_augmentation(layer_name = "augmentation_data", rescaling_flag: bool = False):
    """
    Returns an augmented sequential model, could be attached functionally on a base_model.

    Args:
        layer_name (str): name of the layer to augment (optional).
        rescaling_flag (Boolean): adds a rescaling layer to the model, in case need it (optional).
    """
    if not isinstance(rescaling_flag, bool):
        raise TypeError(f"rescaling_flag must be a boolean, got {type(rescaling_flag)}")

    augmentation_data = tf.keras.models.Sequential([
        tf.keras.layers.RandomFlip("horizontal"), # maybe without random flip, there is strong possibility model performs better.
        tf.keras.layers.RandomHeight(0.2),
        tf.keras.layers.RandomWidth(0.2),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ], name = layer_name)
    if rescaling_flag:
        # If a model which not implements rescaling procedure on data will be used (e.g. RestNet), a Rescaling layer must be added for data normalization.
        augmentation_data.add(tf.keras.layers.Rescaling(1./255))

    return augmentation_data

# A confusion matrix based on sklearn confusion_matrix module,
def confusion_matrix_custom(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False):
    """
    Makes a labelled confusion matrix comparing predictions and ground truth labels.

    If classes is passed, confusion matrix will be labelled, if not, integer class values
    will be used.

    Args:
        y_true: Array of truth labels (must be same shape as `y_pred`).
        y_pred: Array of predictions (must be same shape as `y_true`).
        classes: Array of class labels (e.g. string form). If 'None', integer labels will be used.
        figsize: Size of output figure (default (10,10) ).
        text_size: Size of text (default 15).
        norm: Normalize values or not (default = False).
        savefig: If True, save the confusion matrix to file (default = False).

    Returns:
        A labeled confusion matrix plot comparing predictions and ground truth labels.

    Example:
        confusion_matrix_custom(y_true = test_labels,
                                y_pred = y_preds,
                                classes = class_names,
                                figsize = (20, 20),
                                text_size = 15)
    """
    # Creates the confusion matrix.
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    n_classes = cm.shape[0]  # finds the number of classes.

    # Plots the figure and makes it pretty.
    fig, ax = plt.subplots(figsize = figsize)
    cax = ax.matshow(cm, cmap = plt.cm.Blues)  # colors will represent how 'correct' a class is, darker == better.
    fig.colorbar(cax)

    # Are there a list of classes ?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes.
    ax.set(title = "Confusion Matrix",
            xlabel = "Predicted label",
            ylabel = "True label",
            xticks = np.arange(n_classes),  # create enough axis slots for each class.
            yticks = np.arange(n_classes),
            xticklabels = labels,  # axes will labeled with class names (if they exist) or ints.
            yticklabels = labels)

    # Makes x-axis labels appear on bottom.
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    ### Added: Rotate x-ticks for readability & increase font size (required due to such a large confusion matrix)
    plt.xticks(rotation = 70, fontsize = text_size)
    plt.yticks(fontsize = text_size)

    # Sets the threshold for different colors.
    threshold = (cm.max() + cm.min()) / 2.

    # Plots the text on each cell.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
                      horizontalalignment = "center",
                      color = "white" if cm[i, j] > threshold else "black",
                      size = text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                      horizontalalignment = "center",
                      color = "white" if cm[i, j] > threshold else "black",
                      size = text_size)

    # Save the figure to the current working directory
    if savefig:
        fig.savefig("confusion_matrix.png")

# Loads and prepare the image for predictions
def load_and_prep_image(filename, img_shape = 256, scale=True):
    """
    Reads in an image from filename, turns it into a tensor and reshapes into
    (256, 256, 3).

    Parameters
    ----------
    filename (str): string filename of target image
    img_shape (int): size to resize target image to, default 256
    scale (bool): whether to scale pixel values to range(0, 1), default True
    """

    # Reads in the image.
    img = tf.io.read_file(filename)
    # Decodes it into a tensor.
    img = tf.io.decode_image(img)
    # Resizes the image.
    img = tf.image.resize(img, [img_shape, img_shape])
    if scale:
        # Rescales the image (get all values between 0 and 1)
        return img / 255.
    else:
        return img

# A function for preprocessing images tensors.
def preprocess_img(image, label, img_shape = 224, scaling = False):
  """
  Converts image datatype from 'unit8' -> 'float32' and reshapes
  image to [img_shape, img_shape, colour_channels]
  
  Args:
    image: the selected image which will be casted.
    label (int): the id of image's class_name.
    img_shape (int): size of image
    scaling (bool): a variable for scaling of image's values (e.g. ResNet => True, EfficientNet => False).
  """
  image = tf.image.resize(image, [img_shape, img_shape])
  if scaling == True: 
    image = image / 255.
  return tf.cast(image, tf.float32), label # return(floata32_image, label) tuple


# Creating function evaluates the succession of predictions of a binary model.
def calculate_results(y_true, y_pred):
    """
    Calculates model accuracy, precision, recall and f1 score of a binary classification model.
    Args:
        y_true = true labels in the form of a 1D array
        y_pred = predicted labels in the form of a 1D array
    Return:
        A dictionary of accuracy, precision, recall, f1-score.
    """
    # Calculates model accuracy.
    model_accuracy = accuracy_score(y_true, y_pred) * 100

    # Calculates model precision, recall and f1 score using "weighted" average.
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average = "weighted")
    model_results = {"accuracy": model_accuracy,
                     "precision": model_precision,
                     "recall": model_recall,
                     "f1": model_f1}
    return model_results


# Comparing the baseline model's results against new model which was made to beat the baseline model's results.
def compare_baseline_to_new_model_results(bs_model_res, new_model_res):
    """
    Compares the results (in dictionary type) of baseline model (or initial model) against a new (or updated) model, and calculates the difference between them.
    Args:
        bs_model_res (dict): Results of baseline / initial model.
        new_model_res (dict): Results of secondary model.
    """
    for k, v in new_model_res.items():
        result = new_model_res[k] - bs_model_res[k]
        if new_model_res[k] > bs_model_res[k]:
            print(f"The \"{k}\" of new model '{new_model_res[k]:.2f}', is BETTER than baseline's '{bs_model_res[k]:.2f}' => {result:.2f}.\n")
        else:
            print(f"The \"{k}\" of new model '{new_model_res[k]:.2f}', is LOWER than baseline's '{bs_model_res[k]:.2f}' => {result:.2f}.\n")


# Calculates the time of predictions.
def pred_timer(model, samples):
    """
    Times how long a model takes to make predictions on samples.
    ----
    Args:
        model = a trained model
        sample = a list of samples
    Returns:
        total_time = total elapsed time for model to make predictions on samples
        time_per_pred = time in seconds per single sample
    """
    start_time = time.perf_counter() # gets start time.
    model.predict(samples) # makes predictions.
    end_time = time.perf_counter() # gets finish time.
    total_time = end_time - start_time # calculates how long predictions took to make.
    time_per_pred = total_time / len(samples) # finds prediction time per sample.
    
    return total_time, time_per_pred


##### ================================================================================================================================================================== ####

# Function to read the lines of a document.
def get_lines(filename):
    """
    Reads filename (a text file) and returns the lines of text as a list.
    
    Args:
      filename: a string containing the target filepath to read.
    
    Returns:
      A list of strings with one string per line from the target filename.
      For example:
      ["this is the first line of filename",
       "this is the second line of filename",
       "..."]
    """
    with open(filename, 'r') as f:
        return f.readlines()

# Function preprocessing the text of each line of filename.
def preprocess_text_with_line_numbers(filename):
    """
    Takes in filename, reads ot contents and sorts through each line,
    extracting things like the target label, the text of the sentence,
    how many sentences are in the current abstract and what sentence 
    number the target line is.
    
    Args:
      filename: a string of the target text file to read and extract line data from.    
    Rerurns:
        A list of dictionaries of abstract line data.
    """
    input_lines = get_lines(filename) # gets all lines from filename
    abstract_lines = "" # creates an empty abstract
    abstract_samples = [] # creates an empty list of abstracts

    # Looping through each line in the target file.
    for line in input_lines:
        if line.startswith("###"): # checking to see of there is an ID line
            abstract_id = line
            abstract_lines = "" # resets the abstract string if the line is an ID line
        elif line.isspace(): # recognizes a single "\n" as an empty space
            abstract_line_split = abstract_lines.splitlines() # splits abstract into separates lines
            
            # Iterates through each line in a single abstract and count the at the same time.
            for abstract_line_number, abstract_line in enumerate(abstract_line_split):
                line_data = {} # creates an empty dictionary for each line
                target_text_split = abstract_line.split("\t") # splits target label from text
                line_data["target"] = target_text_split[0] # gets the target label
                line_data["text"] = target_text_split[1].lower() # gets the text and lower it
                line_data["line_number"] = abstract_line_number # what number line does the line appear in the abstract?                
                line_data["total_lines"] = len(abstract_line_split) - 1 # how many total lines are in the abstract? (start from 0)
                abstract_samples.append(line_data) # add line data to abstract samples list
        else: # if the above conditions aren't fulfilled, the line contains a labelled sentence
          abstract_lines += line
        
    return abstract_samples

