import numpy as np
import matplotlib

matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import seaborn as sns


def sample_die(n_samples, sides=6):
    return np.array([np.random.randint(6) + 1
            for i in range(n_samples)])


samples = sample_die(10) + 3
sns.histplot(x=samples, stat="probability",
             bins=6)
plt.title("Dice example: 100 samples")
plt.locator_params(nbins=6)
plt.xticks([4, 5, 6, 7, 8, 9])



fig = plt.gcf()
fig.set_size_inches(10, 6)
plt.savefig("die_example_100.png")
plt.close()


samples = sample_die(1000) + 3
sns.histplot(x=samples, stat="probability", bins=6)
plt.title("Dice example: 1000 samples")
plt.locator_params(nbins=6)
plt.xticks([4, 5, 6, 7, 8, 9])
fig = plt.gcf()
fig.set_size_inches(10, 6)
plt.savefig("die_example_1000.png")
plt.close()


samples = sample_die(100000) + 3
sns.histplot(x=samples, stat="probability", bins=6)
plt.title("Dice example: 100,000 samples")
plt.locator_params(nbins=6)
plt.xticks([4, 5, 6, 7, 8, 9])
fig = plt.gcf()
fig.set_size_inches(10, 6)
plt.savefig("die_example_100000.png")
plt.close()