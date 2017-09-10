import os

import theano
from neupy import layers, storage, architectures

from imagenet_tools import (CURRENT_DIR, FILES_DIR, load_image,
                            print_top_n, download_file)


theano.config.floatX = 'float32'
VGG16_WEIGHTS_FILE = os.path.join(FILES_DIR, 'vgg16.pickle')

vgg16 = architectures.vgg16()

if not os.path.exists(VGG16_WEIGHTS_FILE):
    download_file(
        url="http://neupy.s3.amazonaws.com/imagenet-models/vgg16.pickle",
        filepath=VGG16_WEIGHTS_FILE,
        description='Downloading weights')

storage.load(vgg16, VGG16_WEIGHTS_FILE)

dog_image = load_image(
    os.path.join(CURRENT_DIR, 'images', 'dog.jpg'),
    image_size=(256, 256),
    crop_size=(224, 224))

predict = vgg16.compile()
output = predict(dog_image)
print_top_n(output, n=5)
