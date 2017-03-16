from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
from neupy import algorithms, environment


environment.reproducible()


def plot_scattermatrix(data, target):
    df = pd.DataFrame(data)
    df['target'] = target
    return sns.pairplot(df, hue='target')


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
