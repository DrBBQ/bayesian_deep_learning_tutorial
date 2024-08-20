import numpy as np
import matplotlib

matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import tensorflow_probability as tfp
import os


def likelihood(y_true, y_pred, y_var):
    v_noise = 1e-10
    diff = np.sum(results_bayes["y_pred"] - results_bayes["y_test"], -1)
    likelihood = np.mean(
        -0.5 * np.log(2 * np.pi * (y_var + v_noise))
        - 0.5 * diff ** 2 / (y_var + v_noise)
    )
    return likelihood

results_dir = "results_pkls_4"
results_dir = ""
with open(os.path.join(results_dir, "bayesian_cnn_results.pkl"), "rb") as f:
    results_bayes = pickle.load(f)

with open(os.path.join(results_dir, "simple_cnn_results.pkl"), "rb") as f:
    results_simple = pickle.load(f)

ece_bayes = tfp.stats.expected_calibration_error(
    10,
    logits=results_bayes["y_pred"],
    labels_true=np.argmax(results_bayes["y_test"], -1),
    labels_predicted=np.argmax(results_bayes["y_pred"], -1),
    name=None,
)

ece_simple = tfp.stats.expected_calibration_error(
    10,
    logits=results_simple["y_pred"],
    labels_true=np.argmax(results_simple["y_test"], -1),
    labels_predicted=np.argmax(results_simple["y_pred"], -1),
    name=None,
)

results_bayes["prediction_std"] = results_bayes["prediction_std"]

ll_bayes = likelihood(
    results_bayes["y_test"],
    results_bayes["y_pred"],
    results_bayes["prediction_std"]**2,
)
ll_simple = likelihood(
    results_simple["y_test"],
    results_simple["y_pred"],
    results_simple["prediction_uncertainty"],
)

print(f"ECE Bayes: {ece_bayes}")
print(f"ECE Simple: {ece_simple}")
print(f"LL Bayes: {ll_bayes}")
print(f"LL Simple: {ll_simple}")

plt.subplot(1, 2, 1)
sns.kdeplot(
    data=results_simple, x="loss", y="prediction_uncertainty", fill=False, alpha=0.5
)
sns.scatterplot(
    data=results_simple,
    x="loss",
    y="prediction_uncertainty",
    hue="classification",
    hue_order=["correct", "incorrect"],
)
plt.hlines(0.1, 0, np.max(results_simple["loss"]))
# plt.ylim(0, 0.5)
# plt.xlim(0.5, 1.8)

plt.subplot(1, 2, 2)
sns.kdeplot(data=results_bayes, x="loss", y="prediction_std", fill=False, alpha=0.5)
sns.scatterplot(
    data=results_bayes,
    x="loss",
    y="prediction_std",
    hue="classification",
    hue_order=["correct", "incorrect"],
)
plt.hlines(0.1, 0, np.max(results_bayes["loss"]))
# plt.ylim(0, 0.5)
# plt.xlim(0.5, 1.8)

fig = plt.gcf()
fig.set_size_inches(15, 6)
plt.savefig("combined_distribution_plot.png")
plt.close()


ood_comparison_simple = {
    "prediction_uncertainty": np.concatenate([results_simple["prediction_uncertainty"],
                                   results_simple["prediction_uncertainty_ood"]]),
    "data_distribution": np.concatenate([results_simple["classification"],
                              ["OOD"]*len(results_simple["prediction_uncertainty_ood"])]),
}

ood_comparison_bayes= {
    "prediction_uncertainty": np.concatenate([results_bayes["prediction_std"],
                                   results_bayes["prediction_std_ood"]]),
    "data_distribution": np.concatenate([results_bayes["classification"],
                              ["OOD"]*len(results_bayes["prediction_std_ood"])]),
}


plt.subplot(1, 2, 1)
sns.boxplot(data=ood_comparison_simple, x="data_distribution", y="prediction_uncertainty", order=["correct", "incorrect", "OOD"])
plt.ylabel("Uncertainty")
plt.title("Simple CNN")
# plt.xlim(0.5, 1.8)

plt.subplot(1, 2, 2)
sns.boxplot(data=ood_comparison_bayes, x="data_distribution", y="prediction_uncertainty", order=["correct", "incorrect", "OOD"])
plt.title("Bayesian CNN")
# plt.ylim(0, 0.5)
# plt.xlim(0.5, 1.8)

fig = plt.gcf()
fig.set_size_inches(15, 6)
plt.savefig("ood_comparison.png")
plt.close()
