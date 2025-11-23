"""
Real‑time face mask detector
---------------------------------
This script opens the default webcam, detects faces with OpenCV's Haar Cascade,
and classifies each detected face into three categories using a Keras model:

1) mask_weared_incorrect
2) with_mask
3) without_mask

For each face, it overlays a label and a color-coded rectangle on the frame:
 - Green for properly worn mask
 - Red for no mask
 - Orange for incorrectly worn mask

Press 'q' to exit. The script assumes there is a trained model file
`my_mask_detector_1.model` in the working directory.
"""

import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Resolve the Haar Cascade path using OpenCV's installed data directory.
# This avoids hard-coding a relative path and works across environments.
cascPath = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt2.xml")
faceCascade = cv2.CascadeClassifier(cascPath)

# Load the trained mask detector model.
# The model is expected to produce three probabilities per face in the order:
#   [mask_weared_incorrect, with_mask, without_mask]
model = load_model("my_mask_detector_1.model")

# Open the default webcam. Use index 0; change if multiple cameras exist.
cap = cv2.VideoCapture(0)
while True:
    # Grab a single frame from the webcam stream.
    ret, frame = cap.read()
    if not ret:
        break
    # Convert to grayscale for Haar detection (it expects grayscale images).
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces. Tweak these parameters for recall/precision tradeoffs:
    #  - scaleFactor: image pyramid scaling between iterations
    #  - minNeighbors: detection stability (higher = fewer false positives)
    #  - minSize: minimum face box size to consider
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    # Collect per-face tensors, then batch into a single array.
    faces_batch = []
    for (x, y, w, h) in faces:
        # Crop the face ROI, convert to RGB (model expects RGB),
        # resize to 224x224 (MobileNetV2 input size), then preprocess.
        face_frame = frame[y:y+h, x:x+w]
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face_frame = cv2.resize(face_frame, (224, 224))
        face_frame = img_to_array(face_frame)  # shape: (224, 224, 3)
        face_frame = preprocess_input(face_frame)
        faces_batch.append(face_frame)

    # Run prediction once per frame on a batched tensor: (N, 224, 224, 3).
    # Batching reduces overhead versus predicting per-face in a loop.
    if len(faces_batch) > 0:
        faces_batch = np.array(faces_batch, dtype=np.float32)
        preds = model.predict(faces_batch)
    else:
        preds = []

    for (x, y, w, h), pred in zip(faces, preds):
        (mask_weared_incorrect, with_mask, without_mask) = pred
        # Select the label and color based on the highest probability.
        if (with_mask > without_mask and with_mask > mask_weared_incorrect):
            label = "Mask Worn Properly :)"
            color = (0, 255, 0) 
        elif (without_mask > with_mask and without_mask > mask_weared_incorrect):
            label = "No Mask! (please wear)"
            color = (0, 0, 255)    
        else:
            label = "Wear Mask Properly!"
            color = (255, 140, 0)
        # Append the max class probability to the label for transparency.
        label = "{}: {:.2f}%".format(label,
                                     max(with_mask, mask_weared_incorrect, without_mask) * 100)
        # Overlay text and rectangle around the face region.
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Display the annotated frame in a window titled 'Video'.
    cv2.imshow('Video', frame)

    # Exit on 'q' key press to terminate the loop cleanly.
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

# Release resources (camera and windows) to avoid locking the device.
cap.release()
cv2.destroyAllWindows()