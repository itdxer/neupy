import os

from neupy import storage, architectures

from imagenet_tools import (CURRENT_DIR, FILES_DIR, load_image,
                            print_top_n, download_file)


RESNET50_WEIGHTS_FILE = os.path.join(FILES_DIR, 'resnet50.hdf5')
DOG_IMAGE_PATH = os.path.join(CURRENT_DIR, 'images', 'german-shepherd.jpg')

resnet50 = architectures.resnet50()

if not os.path.exists(RESNET50_WEIGHTS_FILE):
    download_file(
        url="http://neupy.s3.amazonaws.com/tensorflow/imagenet-models/resnet50.hdf5",
        filepath=RESNET50_WEIGHTS_FILE,
        description='Downloading weights')

storage.load(resnet50, RESNET50_WEIGHTS_FILE)

dog_image = load_image(
    DOG_IMAGE_PATH,
    image_size=(256, 256),
    crop_size=(224, 224))

output = resnet50.predict(dog_image)
print_top_n(output, n=5)
