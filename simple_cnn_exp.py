import numpy as np
import pickle
import tensorflow as tf
import matplotlib

matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import seaborn as sns

from simple_cnn import SimpleCNN
from data_utils import load_data_tf, label_int_to_str, hue_order, load_ood_data

X_train, X_test, y_train, y_test = load_data_tf()

model = SimpleCNN()

model.fit(X_train, y_train, epochs=4)
y_pred, y_probs = model.predict(X_test)

ood_data, ood_labels = load_ood_data("ood_examples")
y_pred_ood, y_probs_ood = model.predict(ood_data)

loss_function = tf.keras.losses.CategoricalCrossentropy(
    from_logits=False, reduction="none"
)

loss = loss_function(y_test, y_pred)

y_int_labels_true = tf.argmax(y_test, -1)
y_int_labels_pred = tf.argmax(y_pred, -1)

class_labels_true = label_int_to_str(y_int_labels_true)
class_labels_pred = label_int_to_str(y_int_labels_pred)
class_labels_correct = class_labels_true == class_labels_pred

labels_correct = np.empty_like(class_labels_true)
labels_correct[class_labels_correct] = "correct"
labels_correct[np.logical_not(class_labels_correct)] = "incorrect"

results_data = {
    "loss": loss,
    "y_pred": y_pred,
    "y_test": y_test,
    "prediction_probability": y_probs,
    "prediction_uncertainty": 1 - y_probs,
    "class_label_true": class_labels_true,
    "class_label_pred": class_labels_pred,
    "classification": labels_correct,
    "y_pred_ood": y_pred_ood,
    "y_probs_ood": y_probs_ood,
    "prediction_uncertainty_ood": 1 - y_probs_ood,
    "ood_labels": ood_labels,
}

with open("simple_cnn_results.pkl", "wb") as f:
    pickle.dump(results_data, f)

plt.subplot(1, 2, 1)
sns.kdeplot(data=results_data, x="loss", y="prediction_probability")
sns.scatterplot(
    data=results_data, x="loss", y="prediction_probability", hue="classification"
)
plt.ylim(0.4, 1.0)
plt.xlim(0.5, 1.8)
plt.subplot(1, 2, 2)
sns.scatterplot(
    data=results_data,
    x="loss",
    y="prediction_probability",
    hue="class_label_true",
    hue_order=hue_order,
)
plt.ylim(0.4, 1.0)
plt.xlim(0.5, 1.8)
fig = plt.gcf()
fig.set_size_inches(15, 6)
plt.savefig("simple_cnn_plot.png")
plt.close()
