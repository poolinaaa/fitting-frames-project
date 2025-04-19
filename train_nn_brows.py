from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import numpy as np

df = pd.read_csv("brows_raw.csv")

X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.1, random_state=42, stratify=y_encoded
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dense(len(label_encoder.classes_), activation="softmax")
])


model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


checkpoint = ModelCheckpoint("best_brows_classifier_model.h5",
                             monitor="val_accuracy",
                             save_best_only=True,
                             mode="max",
                             verbose=1)

model.fit(X_train, y_train, epochs=50, batch_size=8,
          validation_data=(X_test, y_test),
          callbacks=[checkpoint])

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

model.save("brows_classifier_model3.h5")

with open("label_encoder3.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

with open("scaler3.pkl", "wb") as f:
    pickle.dump(scaler, f)
