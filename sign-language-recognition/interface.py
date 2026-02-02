
# After training and saving your model (sign_cnn.h5)
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("sign_cnn.h5")
IMG_SIZE = 28  # or 128, depending on training
CLASS_NAMES = [chr(ord('A')+i) for i in range(model.output_shape[-1])]

cap = cv2.VideoCapture(0)

def preprocess_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    norm = resized.astype('float32') / 255.0
    return norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define a region of interest (top-left corner box)
    x, y, w, h = 50, 50, 200, 200
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    roi = frame[y:y+h, x:x+w]

    inp = preprocess_roi(roi)
    preds = model.predict(inp, verbose=0)[0]
    idx = np.argmax(preds)
    label = CLASS_NAMES[idx]
    conf = preds[idx]
    
    # ADD: choose a simple confidence threshold
    THRESH = 0.60  # adjust based on your model; try 0.5–0.8

    # ADD: decide display text based on confidence
    if conf >= THRESH:
        display_text = f"{label} ({conf:.2f})"
        color = (0, 255, 0)  # green for confident predictions
    else:
        display_text = "No sign detected"
        color = (0, 0, 255)  # red for no detection

    cv2.putText(frame, display_text, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Sign Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
