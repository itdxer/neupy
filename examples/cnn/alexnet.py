import os

import theano
from neupy import layers, storage, architectures

from imagenet_tools import (CURRENT_DIR, FILES_DIR, load_image, print_top_n,
                            download_file)


theano.config.floatX = 'float32'
ALEXNET_WEIGHTS_FILE = os.path.join(FILES_DIR, 'alexnet.pickle')

alexnet = architectures.alexnet()

if not os.path.exists(ALEXNET_WEIGHTS_FILE):
    download_file(
        url=(
            "http://srv70.putdrive.com/putstorage/DownloadFileHash/"
            "F497B1D43A5A4A5QQWE2295998EWQS/alexnet.pickle"
        ),
        filepath=ALEXNET_WEIGHTS_FILE,
        description='Downloading weights'
    )

storage.load(alexnet, ALEXNET_WEIGHTS_FILE)

dog_image = load_image(
    os.path.join(CURRENT_DIR, 'images', 'dog.jpg'),
    image_size=(256, 256),
    crop_size=(227, 227),
    use_bgr=False)

predict = alexnet.compile()
output = predict(dog_image)

print_top_n(output[0], n=5)
