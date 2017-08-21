import os

import theano
from six.moves import cPickle as pickle
from neupy.utils import asfloat
from neupy import layers, storage, architectures, environment

from imagenet_tools import (CURRENT_DIR, FILES_DIR, load_image,
                            print_top_n, download_file, read_image)


RESNET50_WEIGHTS_FILE = os.path.join(FILES_DIR, 'resnet50.pickle')
IMAGENET_MEAN_FILE = os.path.join(FILES_DIR, 'resnet50-imagenet-means.pickle')


def prepare_image(fname):
    with open(IMAGENET_MEAN_FILE, 'rb') as f:
        # Mean values is the average image accros all training dataset.
        # if dataset (1000, 3, 224, 224) then mean image shape
        # is (3, 224, 224) and computes as data.mean(axis=0)
        mean_values = pickle.load(f)

    image = read_image(fname, image_size=(256, 256), crop_size=(224, 224))
    # Convert RGB to BGR
    image[:, (0, 1, 2), :, :] = image[:, (2, 1, 0), :, :]
    return asfloat(image - mean_values)


environment.speedup()
resnet50 = architectures.resnet50()

if not os.path.exists(RESNET50_WEIGHTS_FILE):
    download_file(
        url="http://neupy.s3.amazonaws.com/imagenet-models/resnet50.pickle",
        filepath=RESNET50_WEIGHTS_FILE,
        description='Downloading weights')

storage.load(resnet50, RESNET50_WEIGHTS_FILE)
predict = resnet50.compile()

dog_image = prepare_image(os.path.join(CURRENT_DIR, 'images', 'dog2.jpg'))
output = predict(dog_image)
print_top_n(output, n=5)
