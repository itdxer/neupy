import pandas as pd
from sklearn import datasets, preprocessing, metrics
from sklearn.cross_validation import train_test_split
from neuralpy import algorithms, layers, ensemble
from neuralpy.functions import rmsle

from base import BaseTestCase


class PandasCompatibilityTestCase(BaseTestCase):
    def test_pandas_for_bp(self):
        dataset = datasets.load_diabetes()

        input_scaler = preprocessing.MinMaxScaler()
        target_scaler = preprocessing.MinMaxScaler()

        n_features = dataset.data.shape[1]
        input_columns = ['column_' + str(i) for i in range(n_features)]

        pandas_data = pd.DataFrame(dataset.data, columns=input_columns)
        pandas_data['target'] = target_scaler.fit_transform(dataset.target)
        pandas_data[input_columns] = input_scaler.fit_transform(
            pandas_data[input_columns]
        )

        x_train, x_test, y_train, y_test = train_test_split(
            pandas_data[input_columns],
            pandas_data['target'],
            train_size=0.85
        )

        bpnet = algorithms.Backpropagation(
            connection=[
                layers.SigmoidLayer(10),
                layers.SigmoidLayer(40),
                layers.OutputLayer(1),
            ],
            use_bias=True,
            show_epoch=100
        )
        bpnet.train(x_train, y_train, epochs=1000)
        y_predict = bpnet.predict(x_test)

        error = rmsle(target_scaler.inverse_transform(y_test),
                      target_scaler.inverse_transform(y_predict).round())
        self.assertAlmostEqual(0.4477, error, places=4)
