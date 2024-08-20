import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_probability as tfp

from data_utils import load_data_tf, label_int_to_str


class BayesianCNNLL:
    def __init__(self):
        pass

    def init_model(self, n_train_examples, n_classes):
        kl_divergence_function = lambda q, p, _: tfp.distributions.kl_divergence(
            q, p
        ) / tf.cast(n_train_examples, dtype=tf.float32)
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
        self.model.add(
            tfp.layers.DenseReparameterization(
                n_classes,
                kernel_divergence_fn=kl_divergence_function,
                activation=tf.nn.softmax,
            )
        )

        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

    def fit(self, X_train, y_train, epochs=10):
        n_train_examples = X_train.shape[0]
        n_classes = y_train.shape[1]
        self.init_model(n_train_examples, n_classes)
        history = self.model.fit(X_train, y_train, epochs=epochs)

    def predict(self, X_test, n_inference=20):
        y_preds = []
        for i in range(n_inference):
            y_preds.append(self.model.predict(X_test))
        y_preds = np.array(y_preds)
        y_pred = np.mean(y_preds, 0)
        y_stds = np.std(y_preds, 0)

        # y_pred_max = np.argmax(y_pred, -1)
        # y_std = y_stds[np.arange(y_pred.shape[0]), y_pred_max]

        return y_pred, y_stds


class BayesianCNN:
    def __init__(self):
        pass

    def init_model(self, n_train_examples, n_classes):
        kl_divergence_function = lambda q, p, _: tfp.distributions.kl_divergence(
            q, p
        ) / tf.cast(n_train_examples, dtype=tf.float32)
        self.model = models.Sequential()
        self.model.add(
            tfp.layers.Convolution2DReparameterization(
                64,
                kernel_size=(3, 3),
                # padding="SAME",
                kernel_divergence_fn=kl_divergence_function,
                activation=tf.nn.relu,
            )
        )
        self.model.add(layers.MaxPooling2D((3, 3)))
        self.model.add(
            tfp.layers.Convolution2DReparameterization(
                128,
                kernel_size=(3, 3),
                # padding="SAME",
                kernel_divergence_fn=kl_divergence_function,
                activation=tf.nn.relu,
            )
        )
        self.model.add(layers.MaxPooling2D((3, 3)))
        self.model.add(
            tfp.layers.Convolution2DReparameterization(
                64,
                kernel_size=(3, 3),
                # padding="SAME",
                kernel_divergence_fn=kl_divergence_function,
                activation=tf.nn.relu,
            )
        )
        self.model.add(layers.Flatten())
        # self.model.add(
        #     tfp.layers.DenseReparameterization(
        #         64,
        #         kernel_divergence_fn=kl_divergence_function,
        #         activation=tf.nn.softmax,
        #     )
        # )
        self.model.add(
            tfp.layers.DenseReparameterization(
                n_classes,
                kernel_divergence_fn=kl_divergence_function,
                activation=tf.nn.softmax,
            )
        )

        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

    def fit(self, X_train, y_train, epochs=10):
        n_train_examples = X_train.shape[0]
        n_classes = y_train.shape[1]
        self.init_model(n_train_examples, n_classes)
        history = self.model.fit(X_train, y_train, epochs=epochs)

    def predict(self, X_test, n_inference=10):
        y_preds = []
        for i in range(n_inference):
            y_preds.append(self.model.predict(X_test))
        y_preds = np.array(y_preds)
        y_pred = np.mean(y_preds, 0)
        y_stds = np.std(y_preds, 0)

        return y_pred, y_stds
