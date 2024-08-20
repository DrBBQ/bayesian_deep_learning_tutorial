import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import cv2

hue_order = ["trafficlight", "stop", "speedlimit", "crosswalk"]

LABEL_MAP = {
    "trafficlight": 0,
    "stop": 1,
    "speedlimit": 2,
    "crosswalk": 3,
}

INVERSE_MAP = {0: "trafficlight", 1: "stop", 2: "speedlimit", 3: "crosswalk,"}


def load_data_tf(dir=""):
    inputs = np.load("inputs.npy")
    outputs = np.load("outputs.npy")

    X_train, X_test, y_train, y_test = train_test_split(
        inputs, outputs, test_size=0.4)

    X_train = tf.convert_to_tensor(X_train)
    X_test = tf.convert_to_tensor(X_test)

    y_train = tf.one_hot(y_train, depth=len(np.unique(outputs)))
    y_test = tf.one_hot(y_test, depth=len(np.unique(outputs)))

    return X_train, X_test, y_train, y_test




def label_int_to_str(label_ints):
    label_ints = np.array(label_ints)
    labels = []
    for val in label_ints:
        labels.append(INVERSE_MAP[val])
    return np.array(labels)

def load_ood_data(directory="ood_examples"):
    data = []
    labels = []
    for filename in os.listdir(directory):
        img = cv2.imread(os.path.join(directory, filename))
        img = cv2.resize(
            img, (64, 64), interpolation=cv2.INTER_LINEAR
        )
        label = filename[:-4]
        data.append(img)
        labels.append(label)
    data = tf.convert_to_tensor(np.array(data)/255)
    labels = np.array(labels)
    return data, labels
