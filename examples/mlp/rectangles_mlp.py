from sklearn import model_selection, metrics
from skdata.larochelle_etal_2007 import dataset
from neupy import algorithms, layers, environment


environment.reproducible()

rectangle_dataset = dataset.Rectangles()
rectangle_dataset.fetch(download_if_missing=True)

data, target = rectangle_dataset.classification_task()
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    data, target, train_size=0.5
)

network = algorithms.MinibatchGradientDescent(
    [
        layers.Input(784),
        layers.Sigmoid(20),
        layers.Sigmoid(1),
    ],
    error='binary_crossentropy',
    verbose=True,
    show_epoch=1,
    batch_size=1,
)
network.train(x_train, y_train, x_test, y_test, epochs=10)

y_predicted = network.predict(x_test).round()
print(metrics.classification_report(y_test, y_predicted))

roc_score = metrics.roc_auc_score(y_test, y_predicted)
print("ROC score: {}".format(roc_score))

accuracy = metrics.accuracy_score(y_test, y_predicted)
print("Accuracy: {:.2%}".format(accuracy))
