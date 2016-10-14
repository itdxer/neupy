"""
Music/Speech classification using PNN
-------------------------------------

A similar dataset which was collected for the purposes of
music/speech discrimination. The dataset consists of 120 tracks,
each 30 seconds long. Each class (music/speech) has 60 examples.
The tracks are all 22050Hz Mono 16-bit audio files in .wav format.

Dataset page: http://marsyasweb.appspot.com/download/data_sets/
Dataset file: http://opihi.cs.uvic.ca/sound/music_speech.tar.gz
"""
import numpy as np
from neupy import algorithms
from sklearn import preprocessing, model_selection, metrics, decomposition
import matplotlib.pyplot as plt
from librosa.feature import mfcc
from sklearn.utils import shuffle

from getdata import train_test_data, parser


plt.style.use('ggplot')
parser.add_argument('--pca', '-p', dest='apply_pca', default=False,
                    action='store_true',
                    help="Apply PCA for the train data set visualization")

x_train, x_test, y_train, y_test = train_test_data()


def extract_features(data, n_fft=2048):
    res = []
    for row in data:
        centroid = mfcc(row, n_fft=n_fft, sr=22050)
        res.append([
            np.min(centroid),
            np.max(centroid),
            np.median(centroid),
        ])

    return np.array(res)


print("> Data preprocessing procedure")

args = parser.parse_args()

if args.seed is not None:
    np.random.seed(args.seed)

std = 0.2
n_fft = 128

print("STD = {}".format(std))
print("#FFT = {}".format(n_fft))

scaler = preprocessing.MinMaxScaler()
x_train = scaler.fit_transform(extract_features(x_train, n_fft=n_fft))
x_test = scaler.transform(extract_features(x_test, n_fft=n_fft))
x_train, y_train = shuffle(x_train, y_train)

if args.apply_pca:
    pca = decomposition.PCA(2)
    plt.scatter(*pca.fit_transform(x_train).T, c=y_train, s=100)
    plt.show()
    print("PCA explain {:.2%}".format(pca.explained_variance_ratio_.sum()))

print("\n> Train prediction")

skf = model_selection.StratifiedKFold(n_splits=5)
skf_iterator = skf.split(x_train, y_train)
scores = []

for i, (train_index, test_index) in enumerate(skf_iterator, start=1):
    print("\nK-fold #{}".format(i))
    pnnet = algorithms.PNN(std=std, verbose=False)

    x_fold_train, x_fold_test = x_train[train_index], x_train[test_index]
    y_fold_train, y_fold_test = y_train[train_index], y_train[test_index]

    pnnet.fit(x_fold_train, y_fold_train)
    y_predicted = pnnet.predict(x_fold_test)
    score = metrics.roc_auc_score(y_predicted, y_fold_test)
    accurucy = metrics.accuracy_score(y_predicted, y_fold_test)
    scores.append(score)

    print("ROC AUC score: {:.4f}".format(score))
    print("Accurucy: {:.2%}".format(accurucy))
    print(metrics.confusion_matrix(y_predicted, y_fold_test))

print("Average ROC AUC score: {:.4f}".format(np.mean(scores)))

print("\n> Test prediction")
pnnet = algorithms.PNN(std=std, verbose=False)
pnnet.fit(x_train, y_train)
y_predicted = pnnet.predict(x_test)
test_accurucy = metrics.roc_auc_score(y_predicted, y_test)
print("Test data accurucy: {:.4f}".format(test_accurucy))
