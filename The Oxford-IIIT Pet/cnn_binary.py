# Convolutional Neural Network - The Oxford-IIIT Pet

import warnings
warnings.filterwarnings('ignore')

### Imports the libraries.
import pandas as pd
import numpy as np
import tensorflow as tf
import os
# print(tf.__version__)

## Part 1 - Data Preprocessing
### Manipulates files and images before create training and testing datasets.
# Loads annotations in order to create a DataFrame with filename, breed and is_dog as columns names.
dataset_df = pd.read_csv(filepath_or_buffer = "datasets/annotations/list.txt", 
                         delimiter = " ", 
                         skiprows = 6,
                         names = ["filename", "class_id", "species", "breed_id"],
                         header = None)

# Extracts images filenames and labels directly.
dataset_df.columns = ["filename", "class_id", "species", "breed_id"]
image_paths = [os.path.join("datasets/images/", fname + ".jpg") for fname in dataset_df["filename"]]
labels = dataset_df["species"].values

# Moves the categorical column cat/dog at the end of DataFrame.
column_to_move = dataset_df.pop("species")
dataset_df.insert(dataset_df.shape[1],"species", column_to_move)

# Converts species labels into binary. '0' represents "dog" and '1' represents "cat".
labels = np.array([0 if label == 2 else label for label in labels])

# Loading and preprocessing images.
from tensorflow.keras.preprocessing import image
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size = (IMAGE_HEIGHT, IMAGE_WIDTH))
    image_array = image.img_to_array(img)
    image_array = image_array / 255.0
    return image_array

images = np.array([preprocess_image(path) for path in image_paths]) 

# Splits the dataset to training and testing dataset.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2, random_state = 1)


## Part 2 - Building the CNN
### Initialising the CNN
cnn = tf.keras.models.Sequential()

### Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters = 32, 
                               kernel_size = 3, 
                               activation = 'relu', 
                               input_shape = [IMAGE_HEIGHT, IMAGE_WIDTH, 3]))

### Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))

### Adding a second convolutional layer.
cnn.add(tf.keras.layers.Conv2D(filters = 32, 
                               kernel_size = 3, 
                               activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))

### Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

### Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units = 64, activation = 'relu'))
cnn.add(tf.keras.layers.Dropout(0.5))
cnn.add(tf.keras.layers.Dense(units = 32, activation = 'relu'))
cnn.add(tf.keras.layers.Dropout(0.5))

### Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

## Part 3 - Training the CNN
### Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

### Training the CNN on the training set.
cnn.fit(x = X_train, y = y_train, epochs = 25)

## Evaluation of CNN.
y_pred = cnn.predict(X_test)
y_pred = np.round(y_pred)

from sklearn.metrics import accuracy_score, confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"The accuracy of testing dataset is {round(accuracy_score(y_test, y_pred) * 100, 2)}%")
