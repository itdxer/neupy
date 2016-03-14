import numpy as np
from sklearn import datasets, preprocessing, cross_validation
from neupy import algorithms, layers
from neupy.estimators import rmsle

from base import BaseTestCase


class MixtureOfExpertsTestCase(BaseTestCase):
    def test_handle_errors(self):
        networks = [
            algorithms.GradientDescent(
                (1, 20, 1),
                step=0.2,
                verbose=False
            ),
            algorithms.GradientDescent(
                (1, 20, 1),
                step=0.2,
                verbose=False
            ),
        ]

        with self.assertRaises(ValueError):
            # Ivalid network (not GradientDescent)
            algorithms.MixtureOfExperts(
                networks=networks + [
                    algorithms.GRNN(verbose=False)
                ],
                gating_network=algorithms.GradientDescent(
                    layers.Sigmoid(1) > layers.Output(3),
                    verbose=False,
                )
            )

        with self.assertRaises(ValueError):
            # Ivalid number of outputs in third network
            algorithms.MixtureOfExperts(
                networks=networks + [
                    algorithms.GradientDescent(
                        (1, 20, 2),
                        step=0.2,
                        verbose=False
                    )
                ],
                gating_network=algorithms.GradientDescent(
                    layers.Sigmoid(1) > layers.Output(3),
                    verbose=False,
                )
            )

        with self.assertRaises(ValueError):
            # Ivalid gating network output layer size
            algorithms.MixtureOfExperts(
                networks=networks,
                gating_network=algorithms.GradientDescent(
                    layers.Softmax(1) > layers.Output(1),
                    verbose=False,
                )
            )

        with self.assertRaises(ValueError):
            # Ivalid gating network input layer
            algorithms.MixtureOfExperts(
                networks=networks,
                gating_network=algorithms.GradientDescent(
                    layers.Sigmoid(1) > layers.Output(2),
                    verbose=False,
                )
            )

        with self.assertRaises(ValueError):
            # Ivalid gating network output layer
            algorithms.MixtureOfExperts(
                networks=networks,
                gating_network=algorithms.GradientDescent(
                    layers.Softmax(1) > layers.RoundedOutput(2),
                    verbose=False,
                )
            )

        with self.assertRaises(ValueError):
            # Ivalid network error function
            algorithms.MixtureOfExperts(
                networks=networks + [
                    algorithms.GradientDescent(
                        (1, 20, 1),
                        step=0.2,
                        error='rmsle',
                        verbose=False,
                    )
                ],
                gating_network=algorithms.GradientDescent(
                    layers.Sigmoid(1) > layers.Output(3),
                    verbose=False,
                ),
            )

        with self.assertRaises(ValueError):
            moe = algorithms.MixtureOfExperts(
                # Ivalid gating error function
                networks=networks,
                gating_network=algorithms.GradientDescent(
                    layers.Softmax(1) > layers.Output(2),
                    error='rmsle',
                    verbose=False
                ),
            )

        moe = algorithms.MixtureOfExperts(
            # Ivalid gating network output layer
            networks=networks,
            gating_network=algorithms.GradientDescent(
                layers.Softmax(1) > layers.Output(2),
                verbose=False
            ),
        )
        with self.assertRaises(ValueError):
            # Wrong number of train input features
            moe.train(np.array([[1, 2]]), np.array([[0]]))

        with self.assertRaises(ValueError):
            # Wrong number of train output features
            moe.train(np.array([[1]]), np.array([[0, 0]]))

    def test_mixture_of_experts(self):
        dataset = datasets.load_diabetes()
        data, target = dataset.data, dataset.target
        insize, outsize = data.shape[1], 1

        input_scaler = preprocessing.MinMaxScaler((-1 ,1))
        output_scaler = preprocessing.MinMaxScaler()
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(
            input_scaler.fit_transform(data),
            output_scaler.fit_transform(target.reshape(-1, 1)),
            train_size=0.8
        )

        n_epochs = 10
        scaled_y_test = output_scaler.inverse_transform(y_test)
        scaled_y_test = scaled_y_test.reshape((y_test.size, 1))

        # -------------- Train single GradientDescent -------------- #

        bpnet = algorithms.GradientDescent(
            (insize, 20, outsize),
            step=0.1,
            verbose=False
        )
        bpnet.train(x_train, y_train, epochs=n_epochs)
        network_output = bpnet.predict(x_test)
        network_error = rmsle(output_scaler.inverse_transform(network_output),
                              scaled_y_test)

        # -------------- Train ensemlbe -------------- #

        moe = algorithms.MixtureOfExperts(
            networks=[
                algorithms.Momentum(
                    (insize, 20, outsize),
                    step=0.1,
                    batch_size=1,
                    verbose=False
                ),
                algorithms.Momentum(
                    (insize, 20, outsize),
                    step=0.1,
                    batch_size=1,
                    verbose=False
                ),
            ],
            gating_network=algorithms.Momentum(
                layers.Softmax(insize) > layers.Output(2),
                step=0.1,
                verbose=False
            )
        )
        moe.train(x_train, y_train, epochs=n_epochs)
        ensemble_output = moe.predict(x_test)
        ensemlbe_error = rmsle(
            output_scaler.inverse_transform(ensemble_output),
            scaled_y_test
        )

        self.assertGreater(network_error, ensemlbe_error)
