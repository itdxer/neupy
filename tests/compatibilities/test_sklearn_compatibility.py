import numpy as np
from sklearn import datasets, preprocessing, model_selection
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from neupy import algorithms, layers
from neupy.utils import asfloat
from neupy.algorithms.gd import objectives

from base import BaseTestCase


class SklearnCompatibilityTestCase(BaseTestCase):
    def test_pipeline(self):
        dataset = datasets.load_diabetes()
        target_scaler = preprocessing.MinMaxScaler()
        target = dataset.target.reshape(-1, 1)

        x_train, x_test, y_train, y_test = train_test_split(
            asfloat(dataset.data),
            asfloat(target_scaler.fit_transform(target)),
            test_size=0.15
        )

        network = algorithms.GradientDescent(
            connection=[
                layers.Input(10),
                layers.Sigmoid(25),
                layers.Sigmoid(1),
            ],
            batch_size='all',
            show_epoch=100,
            verbose=False,
        )
        pipeline = Pipeline([
            ('min_max_scaler', preprocessing.MinMaxScaler()),
            ('gd', network),
        ])
        pipeline.fit(x_train, y_train, gd__epochs=50)
        y_predict = pipeline.predict(x_test)

        error = objectives.rmsle(
            target_scaler.inverse_transform(y_test),
            target_scaler.inverse_transform(y_predict).round()
        )
        error = self.eval(error)
        self.assertAlmostEqual(0.48, error, places=2)

    def test_grid_search(self):
        def scorer(network, X, y):
            y = asfloat(y)
            result = asfloat(network.predict(X))
            return self.eval(objectives.rmsle(result[:, 0], y))

        dataset = datasets.load_diabetes()
        x_train, x_test, y_train, y_test = train_test_split(
            dataset.data, dataset.target, test_size=0.3
        )

        grnnet = algorithms.GRNN(std=0.5, verbose=False)
        grnnet.train(x_train, y_train)
        error = scorer(grnnet, x_test, y_test)

        self.assertAlmostEqual(0.513, error, places=3)

        random_search = model_selection.RandomizedSearchCV(
            grnnet,
            param_distributions={'std': np.arange(1e-2, 0.1, 1e-4)},
            n_iter=10,
            scoring=scorer,
            random_state=self.random_seed,
            cv=3,
        )
        random_search.fit(dataset.data, dataset.target)
        scores = random_search.cv_results_

        best_score = min(scores['mean_test_score'])
        self.assertAlmostEqual(0.4266, best_score, places=3)

    def test_transfrom_method(self):
        dataset = datasets.load_diabetes()

        grnnet = algorithms.GRNN(std=0.5, verbose=False)
        grnnet.train(dataset.data, dataset.target)

        y_predicted = grnnet.predict(dataset.data)
        y_transformed = grnnet.transform(dataset.data)

        np.testing.assert_array_almost_equal(y_predicted, y_transformed)
