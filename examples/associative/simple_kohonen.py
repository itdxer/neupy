"""
Attribute Information:
1. Age of patient at time of operation (numerical)
2. Patient's year of operation (year - 1900, numerical)
3. Number of positive axillary nodes detected (numerical)
4. Survival status (class attribute)
     1 = the patient survived 5 years or longer
     2 = the patient died within 5 year
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from neupy import algorithms, layers


current_dir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(current_dir, 'data')
data = pd.read_csv(os.path.join(datadir, 'survival.csv'))

# scatter_matrix(data)
# plt.show()

from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics, svm, ensemble, preprocessing
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
x_train, x_test, y_train, y_test = train_test_split(data[data.columns[:-1]],
                                                    data[data.columns[-1]],
                                                    train_size=0.8)
import numpy as np
def balanced_subsample(x,y,subsample_size=1.0):
    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys
x_train, y_train = balanced_subsample(x_train.values, y_train.values)
# tree = ensemble.RandomForestClassifier(10000)
# tree.fit(x_train, y_train)
#
# predicted = tree.predict(x_test)
# y_test = y_test.values
#
# print(metrics.roc_auc_score(predicted - 1, y_test- 1))
# print(metrics.confusion_matrix(predicted, y_test))

# knet = algorithms.Kohonen(
#     layers.Linear(2) > layers.CompetitiveOutput(3)
# )

# scaler = preprocessing.MinMaxScaler((-2, 2))
# x_train = x_train.astype(np.float)
# x_test = x_test.astype(np.float)
# x_train = scaler.fit_transform(x_train)
# y_train = y_train - 1
# x_test = scaler.transform(x_test)
# y_test = y_test - 1
#
# hnet = algorithms.Hessian(
#     [
#         layers.Sigmoid(3, init_method='ortho'),
#         layers.Sigmoid(20, init_method='ortho'),
#         layers.RoundedOutput(1),
#     ],
#     inv_penalty_const=1,
#     verbose=True,
#     error='binary_crossentropy'
# )
# hnet.train(x_train, y_train, x_test, y_test, epochs=100)
# predicted = hnet.predict(x_test)
# print(metrics.roc_auc_score(predicted.round(), y_test))
# print(metrics.confusion_matrix(predicted.round(), y_test))
