import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical

import cv2
import glob

global inputShape, size

def kerasModel4():
    model = Sequential()
    model.add(Conv2D(16, (8, 8), strides=(4, 4), padding='valid', input_shape=(size, size, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (5, 5), padding="same"))
    model.add(Activation('relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(512))
    model.add(Dropout(0.1))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model

size = 300

# Load Training data: pothole
potholeTrainImages = glob.glob("F:/My Dataset/train/Pothole/*.jpg")
potholeTrainImages.extend(glob.glob("F:/My Dataset/train/Pothole/*.jpeg"))
potholeTrainImages.extend(glob.glob("F:/My Dataset/train/Pothole/*.png"))

train1 = [cv2.imread(img, 0) for img in potholeTrainImages if img is not None]
for i in range(0, len(train1)):
    train1[i] = cv2.resize(train1[i], (size, size))
temp1 = np.asarray(train1)

# Load Training data: non-pothole
nonPotholeTrainImages = glob.glob("F:/My Dataset/train/Plain/*.jpg")
nonPotholeTrainImages.extend(glob.glob("F:/My Dataset/train/Plain/*.jpeg"))
nonPotholeTrainImages.extend(glob.glob("F:/My Dataset/train/Plain/*.png"))

train2 = [cv2.imread(img, 0) for img in nonPotholeTrainImages if img is not None]
for i in range(0, len(train2)):
    train2[i] = cv2.resize(train2[i], (size, size))
temp2 = np.asarray(train2)

X_train = np.concatenate((temp1, temp2))
y_train1 = np.ones([temp1.shape[0]], dtype=int)
y_train2 = np.zeros([temp2.shape[0]], dtype=int)
y_train = np.concatenate((y_train1, y_train2))

# Shuffle the dataset and labels
X_train, y_train = shuffle(X_train, y_train)

# Convert labels to categorical
y_train = to_categorical(y_train)

print("Train shape X:", X_train.shape)
print("Train shape y:", y_train.shape)

# Create and compile the model
inputShape = (size, size, 1)
model = kerasModel4()
X_train = X_train / 255
model.compile('adam', 'categorical_crossentropy', ['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.1)

# Load Testing data: potholes
potholeTestImages = glob.glob("F:/My Dataset/test/Pothole/*.jpg")
potholeTestImages.extend(glob.glob("F:/My Dataset/test/Pothole/*.jpeg"))
potholeTestImages.extend(glob.glob("F:/My Dataset/test/Pothole/*.png"))

test1 = [cv2.imread(img, 0) for img in potholeTestImages if img is not None]
for i in range(0, len(test1)):
    test1[i] = cv2.resize(test1[i], (size, size))
temp3 = np.asarray(test1)

# Load Testing data: non-pothole
nonPotholeTestImages = glob.glob("F:/My Dataset/test/Plain/*.jpg")
nonPotholeTestImages.extend(glob.glob("F:/My Dataset/test/Plain/*.jpeg"))
nonPotholeTestImages.extend(glob.glob("F:/My Dataset/test/Plain/*.png"))

test2 = [cv2.imread(img, 0) for img in nonPotholeTestImages if img is not None]
for i in range(0, len(test2)):
    test2[i] = cv2.resize(test2[i], (size, size))
temp4 = np.asarray(test2)

X_test = np.concatenate((temp3, temp4))
y_test1 = np.ones([temp3.shape[0]], dtype=int)
y_test2 = np.zeros([temp4.shape[0]], dtype=int)
y_test = np.concatenate((y_test1, y_test2))

# Convert labels to categorical for the testing set
y_test = to_categorical(y_test)

print("Test shape X:", X_test.shape)
print("Test shape y:", y_test.shape)

print("")

# Evaluate the model on the testing set
metricsTest = model.evaluate(X_test, y_test)
print("Testing Accuracy: ", metricsTest[1] * 100, "%")

# Save the model weights and configuration
model.save('latest_full_model.h5')
print("Saved model to disk")

# Save the training history
history_df = pd.DataFrame(history.history)
history_df.to_pickle('history.pkl')
print("Saved training history to disk")

# Plot training and validation accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Save the accuracy plot as an image
plt.savefig('accuracy_plot.png')

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Save the loss plot as an image
plt.savefig('loss_plot.png')

# Display the plots
plt.show()
