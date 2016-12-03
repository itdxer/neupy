import numpy as np
from sklearn import datasets, preprocessing, metrics, model_selection
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from neupy import algorithms, layers
from neupy.estimators import rmsle

from base import BaseTestCase


class SklearnCompatibilityTestCase(BaseTestCase):
    def test_pipeline(self):
        dataset = datasets.load_diabetes()
        target_scaler = preprocessing.MinMaxScaler()
        target = dataset.target.reshape(-1, 1)

        x_train, x_test, y_train, y_test = train_test_split(
            dataset.data,
            target_scaler.fit_transform(target),
            train_size=0.85
        )

        network = algorithms.GradientDescent(
            connection=[
                layers.Input(10),
                layers.Sigmoid(25),
                layers.Sigmoid(1),
            ],
            show_epoch=100,
            verbose=False,
        )
        pipeline = Pipeline([
            ('min_max_scaler', preprocessing.MinMaxScaler()),
            ('gd', network),
        ])
        pipeline.fit(x_train, y_train, gd__epochs=50)
        y_predict = pipeline.predict(x_test)

        error = rmsle(target_scaler.inverse_transform(y_test),
                      target_scaler.inverse_transform(y_predict).round())
        self.assertAlmostEqual(0.48, error, places=2)

    def test_ensemble(self):
        data, target = datasets.make_classification(300, n_features=4,
                                                    n_classes=2)
        x_train, x_test, y_train, y_test = train_test_split(
            data, target, train_size=0.7
        )

        dan = algorithms.DynamicallyAveragedNetwork([
            algorithms.RPROP((4, 5, 1), step=0.1, maxstep=1),
            algorithms.GradientDescent((4, 5, 1), step=0.1),
            algorithms.ConjugateGradient((4, 5, 1), step=0.01),
        ])

        pipeline = Pipeline([
            ('min_max_scaler', preprocessing.StandardScaler()),
            ('dan', dan),
        ])
        pipeline.fit(x_train, y_train, dan__epochs=100)

        result = pipeline.predict(x_test)
        ensemble_result = metrics.accuracy_score(y_test, result)
        self.assertAlmostEqual(0.9444, ensemble_result, places=4)

    def test_grid_search(self):
        def scorer(network, X, y):
            result = network.predict(X)
            return rmsle(result[:, 0], y)

        dataset = datasets.load_diabetes()
        x_train, x_test, y_train, y_test = train_test_split(
            dataset.data, dataset.target, train_size=0.7
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
            random_state=self.random_seed
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
