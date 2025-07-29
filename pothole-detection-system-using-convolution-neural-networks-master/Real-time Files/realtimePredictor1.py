import cv2
import imutils
from keras.models import load_model
import numpy as np

# Global variables
loadedModel = None
size = 300

def predict_pothole(currentFrame):
    currentFrame = cv2.resize(currentFrame, (size, size))
    currentFrame = currentFrame.reshape(1, size, size, 1).astype('float') / 255.0
    raw_predictions = loadedModel.predict(currentFrame)
    predicted_class = np.argmax(raw_predictions)
    max_prob = raw_predictions[0, predicted_class]

    if max_prob > 0.90:
        return predicted_class, max_prob
    return "None", 0

if __name__ == '__main__':
    try:
        # Load the pre-trained model
        loadedModel = load_model(r'F:\pothole-detection-system-using-convolution-neural-networks-master\pothole-detection-system-using-convolution-neural-networks-master\Real-time Files\full_model.h5')

        # Open the camera
        camera = cv2.VideoCapture(0)

        show_pred = False

        while True:
            grabbed, frame = camera.read()
            if not grabbed:
                print("Camera not available or disconnected.")
                break

            frame = imutils.resize(frame, width=700)
            frame = cv2.flip(frame, 1)

            clone = frame.copy()
            (height, width) = frame.shape[:2]

            grayClone = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)

            pothole, prob = predict_pothole(grayClone)

            keypress_toshow = cv2.waitKey(1)

            if keypress_toshow == ord("e"):
                show_pred = not show_pred

            if show_pred:
                label = "Pothole" if pothole == 1 else "Non-Pothole"
                text = f"{label} ({prob * 100:.2f}%)"
                color = (0, 255, 0) if pothole == 1 else (0, 0, 255)
                cv2.putText(clone, text, (30, 30), cv2.FONT_HERSHEY_DUPLEX, 1, color, 1)

            cv2.imshow("GrayClone", grayClone)
            cv2.imshow("Video Feed", clone)

            keypress = cv2.waitKey(1) & 0xFF

            if keypress == ord("q"):
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Release the camera
        camera.release()
        cv2.destroyAllWindows()
