import matplotlib.pyplot as plt
import seaborn as sns


def plot_samples(samples):
    for k, v in samples.items():
        v = v[..., 0]
        plt.title(k)
        sns.histplot(v)
        plt.show()
