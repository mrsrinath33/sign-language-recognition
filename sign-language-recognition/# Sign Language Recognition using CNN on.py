# Sign Language Recognition using CNN on SignMNIST
# TensorFlow 2.x | Python 3.9+

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# ----------------------
# 1) Load CSV files
# ----------------------
train_csv = "sign_mnist_train.csv"
test_csv  = "sign_mnist_test.csv"

train_df = pd.read_csv(train_csv)
test_df  = pd.read_csv(test_csv)

X_train = train_df.drop('label', axis=1).values
y_train_raw = train_df['label'].values

X_test = test_df.drop('label', axis=1).values
y_test_raw = test_df['label'].values

# ----------------------
# 2) FIX: Remap labels (CRITICAL)
# ----------------------
def remap_labels(y):
    unique_labels = sorted(np.unique(y))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y_mapped = np.array([label_map[label] for label in y])
    return y_mapped, label_map

y_train, label_map = remap_labels(y_train_raw)
y_test, _ = remap_labels(y_test_raw)

num_classes = len(label_map)   # = 24

print("Number of classes:", num_classes)
print("Label mapping:", label_map)

# ----------------------
# 3) Preprocess images
# ----------------------
IMG_SIZE = 28

X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0
X_test  = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0

y_train_cat = to_categorical(y_train, num_classes)
y_test_cat  = to_categorical(y_test, num_classes)

# ----------------------
# 4) Data augmentation
# ----------------------
data_augment = tf.keras.Sequential([
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.08),
    layers.RandomTranslation(0.08, 0.08)
], name="augment")

# ----------------------
# 5) Build CNN model
# ----------------------
def build_model(input_shape=(28, 28, 1), num_classes=24):
    inputs = layers.Input(shape=input_shape)
    x = data_augment(inputs)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = build_model(num_classes=num_classes)
model.summary()

# ----------------------
# 6) Train model
# ----------------------
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=5e-5
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=7, restore_best_weights=True
    )
]

history = model.fit(
    X_train, y_train_cat,
    validation_split=0.15,
    epochs=100,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

# ----------------------
# 7) Evaluate
# ----------------------
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# ----------------------
# 8) Save model
# ----------------------
model.save("sign_cnn.h5")
print("\nModel saved as sign_cnn.h5")

