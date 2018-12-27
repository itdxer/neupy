import pandas as pd
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split

from neupy import algorithms, layers
from neupy.utils import asfloat
from neupy.algorithms.gd import objectives

from base import BaseTestCase


class PandasCompatibilityTestCase(BaseTestCase):
    def test_pandas_for_bp(self):
        dataset = datasets.load_diabetes()
        target = dataset.target.reshape(-1, 1)

        input_scaler = preprocessing.MinMaxScaler()
        target_scaler = preprocessing.MinMaxScaler()

        n_features = dataset.data.shape[1]
        input_columns = ['column_' + str(i) for i in range(n_features)]

        pandas_data = pd.DataFrame(dataset.data, columns=input_columns)
        pandas_data['target'] = target_scaler.fit_transform(target)
        pandas_data[input_columns] = input_scaler.fit_transform(
            pandas_data[input_columns]
        )

        x_train, x_test, y_train, y_test = train_test_split(
            asfloat(pandas_data[input_columns]),
            asfloat(pandas_data['target']),
            test_size=0.15
        )

        bpnet = algorithms.GradientDescent(
            [
                layers.Input(10),
                layers.Sigmoid(30),
                layers.Sigmoid(1),
            ],
            batch_size=None,
        )
        bpnet.train(x_train, y_train, epochs=50)
        y_predict = bpnet.predict(x_test).reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        error = objectives.rmsle(
            target_scaler.inverse_transform(y_test),
            target_scaler.inverse_transform(y_predict).round()
        )
        error = self.eval(error)
        self.assertAlmostEqual(0.48, error, places=2)
