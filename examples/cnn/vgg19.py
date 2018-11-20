import os

from neupy import layers, storage, architectures

from imagenet_tools import (CURRENT_DIR, FILES_DIR, load_image,
                            print_top_n, download_file)


VGG19_WEIGHTS_FILE = os.path.join(FILES_DIR, 'vgg19.pickle')
DOG_IMAGE_PATH = os.path.join(CURRENT_DIR, 'images', 'german-shepherd.jpg')
vgg19 = architectures.vgg19()

if not os.path.exists(VGG19_WEIGHTS_FILE):
    download_file(
        url="http://neupy.s3.amazonaws.com/imagenet-models/vgg19.pickle",
        filepath=VGG19_WEIGHTS_FILE,
        description='Downloading weights')

storage.load(vgg19, VGG19_WEIGHTS_FILE)

dog_image = load_image(
    DOG_IMAGE_PATH,
    image_size=(256, 256),
    crop_size=(224, 224))

output = vgg19.predict(dog_image)
print_top_n(output, n=5)
