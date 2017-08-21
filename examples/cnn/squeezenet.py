import os

import theano
import theano.tensor as T
from neupy import layers, storage, architectures

from imagenet_tools import (CURRENT_DIR, FILES_DIR, load_image,
                            print_top_n, download_file)


theano.config.floatX = 'float32'
SQUEEZENET_WEIGHTS_FILE = os.path.join(FILES_DIR, 'squeezenet.pickle')

# Networks weight ~4.8 Mb
squeezenet = architectures.squeezenet()

if not os.path.exists(SQUEEZENET_WEIGHTS_FILE):
    download_file(
        url="http://neupy.s3.amazonaws.com/imagenet-models/squeezenet.pickle",
        filepath=SQUEEZENET_WEIGHTS_FILE,
        description='Downloading weights')

storage.load(squeezenet, SQUEEZENET_WEIGHTS_FILE)

monkey_image = load_image(
    os.path.join(CURRENT_DIR, 'images', 'titi-monkey.jpg'),
    image_size=(227, 227),
    crop_size=(227, 227),
    use_bgr=True)

predict = squeezenet.compile()
output = predict(monkey_image)
print_top_n(output, n=5)
