import cv2
import numpy as np
import tensorflow as tf

# Load your trained Teachable Machine model (TF Keras)
model = tf.keras.models.load_model("keras_model.h5")

# Labels (must match your model's labels.txt)
CLASS_NAMES = ["Diseased Lung", "Healthy Lung"]

# Path to lung image (X-ray or CT scan)
image_path = "lung_xray.png"
frame = cv2.imread(image_path)

if frame is None:
    print("Error: Could not load image.")
else:
    # Preprocess the image for Teachable Machine
    img = cv2.resize(frame, (224, 224))  # model input size
    img_array = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    confidence = predictions[0][class_index] * 100
    label = f"{CLASS_NAMES[class_index]} ({confidence:.2f}%)"

    # Display label on the image
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show image
    cv2.imshow("Lung Disease Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
