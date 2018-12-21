import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from neupy import algorithms, utils


utils.reproducible()


def plot_scattermatrix(data, target):
    df = pd.DataFrame(data)
    df['target'] = target
    return sns.pairplot(df, hue='target', diag_kind='hist')


if __name__ == '__main__':
    dataset = datasets.load_iris()
    data, target = dataset.data, dataset.target

    lvqnet = algorithms.LVQ3(
        # number of features
        n_inputs=4,

        # number of data points that we want
        # to have at the end
        n_subclasses=30,

        # number of classes
        n_classes=3,

        verbose=True,
        show_epoch=20,

        step=0.001,
        n_updates_to_stepdrop=150 * 100,
    )
    lvqnet.train(data, target, epochs=100)

    plot_scattermatrix(data, target)
    plot_scattermatrix(data=lvqnet.weight, target=lvqnet.subclass_to_class)
    plt.show()
