import tensorflow as tf
from tensorflow.keras import layers, models

from data_utils import load_data_tf, label_int_to_str


class SimpleCNN:
    def __init__(self):
        self.model = models.Sequential()
        self.model.add(
            layers.Conv2D(64, (3, 3), activation="relu", input_shape=(64, 64, 3))
        )
        self.model.add(layers.MaxPooling2D((3, 3)))
        self.model.add(layers.Conv2D(128, (3, 3), activation="relu"))
        self.model.add(layers.MaxPooling2D((3, 3)))
        self.model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        self.model.add(layers.Flatten())
        # self.model.add(layers.Dense(64, activation="relu"))
        self.model.add(layers.Dense(4, activation="softmax"))

        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

    def fit(self, X_train, y_train, epochs=10):
        history = self.model.fit(X_train, y_train, epochs=epochs)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        # softmax = tf.nn.softmax(y_pred)
        y_probs = tf.reduce_max(y_pred, -1)
        return y_pred, y_probs
