import numpy as np
import cv2
import glob
from keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Global variable
size = 300

# Load the pre-trained model
model = load_model('F:/pothole-detection-system-using-convolution-neural-networks-master/pothole-detection-system-using-convolution-neural-networks-master/Real-time Files/full_model.h5')

# Load Testing data: non-pothole
nonPotholeTestImages = glob.glob("F:/pothole-detection-system-using-convolution-neural-networks-master/pothole-detection-system-using-convolution-neural-networks-master/My Dataset/test/Plain/*.jpg")
test2 = [cv2.imread(img, 0) for img in nonPotholeTestImages]
test2 = [cv2.resize(img, (size, size)) for img in test2]
temp4 = np.asarray(test2)

# Load Testing data: potholes
potholeTestImages = glob.glob("F:/pothole-detection-system-using-convolution-neural-networks-master/pothole-detection-system-using-convolution-neural-networks-master/My Dataset/test/Pothole/*.jpg")
test1 = [cv2.imread(img, 0) for img in potholeTestImages]
test1 = [cv2.resize(img, (size, size)) for img in test1]
temp3 = np.asarray(test1)

# Combine the datasets
X_test = np.concatenate((temp3, temp4))
X_test = X_test.reshape(X_test.shape[0], size, size, 1)

# Create labels for the test set
y_test1 = np.ones([temp3.shape[0]], dtype=int)
y_test2 = np.zeros([temp4.shape[0]], dtype=int)
y_test = np.concatenate((y_test1, y_test2))
y_test = to_categorical(y_test)

# Normalize the input data
X_test = X_test / 255

# Predictions
predictions = model.predict(X_test)

# Display predictions with probabilities
for i, prediction in enumerate(predictions):
    pothole_probability = prediction[1]  # Probability for class 'Pothole'
    predicted_class = 1 if pothole_probability > 0.5 else 0
    print(f">>> Predicted for sample {i}: {'Pothole' if predicted_class == 1 else 'Non-Pothole'} with probability {pothole_probability * 100:.2f}%")

# Evaluate the model
print("")
metrics = model.evaluate(X_test, y_test)
print("Test Accuracy: ", metrics[1] * 100, "%")
