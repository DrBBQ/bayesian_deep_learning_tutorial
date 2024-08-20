import numpy as np
import pickle
import tensorflow as tf
import matplotlib

matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import seaborn as sns

from bayesian_cnn import BayesianCNN, BayesianCNNLL
from data_utils import load_data_tf, label_int_to_str, hue_order, load_ood_data

X_train, X_test, y_train, y_test = load_data_tf()

model = BayesianCNN()

model.fit(X_train, y_train, epochs=40)
y_pred, y_stds = model.predict(X_test)

y_pred_max = np.argmax(y_pred, -1)
y_std = y_stds[np.arange(y_pred.shape[0]), y_pred_max]

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

ood_data, ood_labels = load_ood_data("ood_examples")
y_pred_ood, y_stds_ood = model.predict(ood_data)
y_pred_ood_max = np.argmax(y_pred_ood, -1)
y_std_ood = y_stds_ood[np.arange(y_pred_ood.shape[0]), y_pred_ood_max]

results_data = {
    "loss": loss,
    "y_pred": y_pred,
    "y_test": y_test,
    "y_stds": y_stds,
    "prediction_std": y_std,
    "class_label_true": class_labels_true,
    "class_label_pred": class_labels_pred,
    "classification": labels_correct,
    "y_pred_ood": y_pred_ood,
    "prediction_std_ood": y_std_ood,
    "prediction_stds_ood": y_stds_ood,
    "ood_labels": ood_labels,
}

with open("bayesian_cnn_results.pkl", "wb") as f:
    pickle.dump(results_data, f)

plt.subplot(1, 2, 1)
sns.kdeplot(data=results_data, x="loss", y="prediction_std")
sns.scatterplot(data=results_data, x="loss", y="prediction_std", hue="classification")
plt.ylim(0, 0.5)
plt.xlim(0.5, 1.8)
plt.subplot(1, 2, 2)
sns.scatterplot(
    data=results_data,
    x="loss",
    y="prediction_std",
    hue="class_label_true",
    hue_order=hue_order,
)
plt.ylim(0, 0.5)
plt.xlim(0.5, 1.8)
fig = plt.gcf()
fig.set_size_inches(15, 6)
plt.savefig("bayesian_cnn_plot.png")
plt.close()
