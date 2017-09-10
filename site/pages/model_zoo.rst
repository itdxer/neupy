Model Zoo
=========

ImageNet classification
-----------------------

These modes are trained to perform classification in the ImageNet ILSVRC challenge data. The goal of the competition is to build a model that classifies image into one of the 1,000 categories. Categories include animals, objects, transports and so on.

.. csv-table::
    :header: "Name", "Number of parameters", "Pre-trained model", "Code example"

    "ResNet50", "~25.5 millions", `resnet50.pickle <http://neupy.s3.amazonaws.com/imagenet-models/resnet50.pickle>`_, `resnet50.py <https://github.com/itdxer/neupy/blob/master/examples/cnn/resnet50.py>`_
    "SqueezeNet", "~1.2 million", `squeezenet.pickle <http://neupy.s3.amazonaws.com/imagenet-models/squeezenet.pickle>`_, `squeezenet.py <https://github.com/itdxer/neupy/blob/master/examples/cnn/squeezenet.py>`_
    "VGG16", "~138 million", `vgg16.pickle <http://neupy.s3.amazonaws.com/imagenet-models/vgg16.pickle>`_, `vgg16.py <https://github.com/itdxer/neupy/blob/master/examples/cnn/vgg16.py>`_
    "VGG19", "~143 million", `vgg19.pickle <http://neupy.s3.amazonaws.com/imagenet-models/vgg19.pickle>`_, `vgg19.py <https://github.com/itdxer/neupy/blob/master/examples/cnn/vgg19.py>`_
    "AlexNet", "~61 million", `alexnet.pickle <http://neupy.s3.amazonaws.com/imagenet-models/alexnet.pickle>`_, `alexnet.py <https://github.com/itdxer/neupy/blob/master/examples/cnn/alexnet.py>`_

Value Iteration Network (VIN)
-----------------------------

VINs can learn to plan, and are suitable for predicting outcomes that involve planning-based reasoning, such as policies for reinforcement learning. NeuPy has 3 models pre-trained for the path-searching task in arthificialy created environments with different grid sizes.

.. csv-table::
    :header: "Grid size", "Pre-trained parameters"

    "8x8", `pretrained-VIN-8.pickle <https://github.com/itdxer/neupy/blob/master/examples/reinforcement_learning/vin/models/pretrained-VIN-8.pickle?raw=true>`_
    "16x16", `pretrained-VIN-16.pickle <https://github.com/itdxer/neupy/blob/master/examples/reinforcement_learning/vin/models/pretrained-VIN-16.pickle?raw=true>`_
    "28x28", `pretrained-VIN-28.pickle <https://github.com/itdxer/neupy/blob/master/examples/reinforcement_learning/vin/models/pretrained-VIN-28.pickle?raw=true>`_

Project that include everything related to VIN is avaliable on Github: `examples/reinforcement_learning/vin <https://github.com/itdxer/neupy/tree/master/examples/reinforcement_learning/vin/>`_
