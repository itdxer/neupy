import numpy as np
from sklearn import datasets, preprocessing, cross_validation
from neupy import algorithms, layers, ensemble
from neupy.functions import rmsle

from base import BaseTestCase


class MixtureOfExpertsTestCase(BaseTestCase):
    def test_handle_errors(self):
        networks = [
            algorithms.Backpropagation((1, 20, 1), step=0.2),
            algorithms.Backpropagation((1, 20, 1), step=0.2),
        ]

        with self.assertRaises(ValueError):
            # Ivalid network (not Backpropagation)
            ensemble.MixtureOfExperts(
                networks=networks + [
                    algorithms.GRNN()
                ],
                gating_network=algorithms.Backpropagation(
                    layers.SigmoidLayer(1) > layers.OutputLayer(3),
                )
            )

        with self.assertRaises(ValueError):
            # Ivalid number of outputs in third network
            ensemble.MixtureOfExperts(
                networks=networks + [
                    algorithms.Backpropagation((1, 20, 2), step=0.2)
                ],
                gating_network=algorithms.Backpropagation(
                    layers.SigmoidLayer(1) > layers.OutputLayer(3),
                )
            )

        with self.assertRaises(ValueError):
            # Ivalid gating network output layer size
            ensemble.MixtureOfExperts(
                networks=networks,
                gating_network=algorithms.Backpropagation(
                    layers.SoftmaxLayer(1) > layers.OutputLayer(1),
                )
            )

        with self.assertRaises(ValueError):
            # Ivalid gating network input layer
            ensemble.MixtureOfExperts(
                networks=networks,
                gating_network=algorithms.Backpropagation(
                    layers.SigmoidLayer(1) > layers.OutputLayer(2),
                )
            )

        with self.assertRaises(ValueError):
            # Ivalid gating network output layer
            ensemble.MixtureOfExperts(
                networks=networks,
                gating_network=algorithms.Backpropagation(
                    layers.SoftmaxLayer(1) > layers.RoundOutputLayer(2)
                )
            )

        with self.assertRaises(ValueError):
            # Ivalid network error function
            ensemble.MixtureOfExperts(
                networks=networks + [
                    algorithms.Backpropagation(
                        (1, 20, 1), step=0.2, error=rmsle
                    )
                ],
                gating_network=algorithms.Backpropagation(
                    layers.SigmoidLayer(1) > layers.OutputLayer(3),
                )
            )

        with self.assertRaises(ValueError):
            moe = ensemble.MixtureOfExperts(
                # Ivalid gating error function
                networks=networks,
                gating_network=algorithms.Backpropagation(
                    layers.SoftmaxLayer(1) > layers.OutputLayer(2),
                    error=rmsle
                )
            )

        moe = ensemble.MixtureOfExperts(
            # Ivalid gating network output layer
            networks=networks,
            gating_network=algorithms.Backpropagation(
                layers.SoftmaxLayer(1) > layers.OutputLayer(2)
            )
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
            output_scaler.fit_transform(target),
            train_size=0.8
        )

        n_epochs = 300
        scaled_y_test = output_scaler.inverse_transform(y_test).reshape(
            (y_test.size, 1)
        )

        # -------------- Train single Backpropagation -------------- #

        moe = algorithms.Backpropagation((insize, 20, outsize), step=0.1)
        moe.train(x_train, y_train, epochs=n_epochs)
        network_output = moe.predict(x_test)
        network_error = rmsle(output_scaler.inverse_transform(network_output),
                              scaled_y_test)

        # -------------- Train ensemlbe -------------- #

        moe = ensemble.MixtureOfExperts(
            networks=[
                algorithms.Backpropagation((insize, 20, outsize), step=0.1),
                algorithms.Backpropagation((insize, 20, outsize), step=0.1),
            ],
            gating_network=algorithms.Backpropagation(
                layers.SoftmaxLayer(insize) > layers.OutputLayer(2),
                step=0.1
            )
        )
        moe.train(x_train, y_train, epochs=n_epochs)
        ensemble_output = moe.predict(x_test)
        ensemlbe_error = rmsle(
            output_scaler.inverse_transform(ensemble_output), scaled_y_test
        )

        self.assertGreater(network_error, ensemlbe_error)
