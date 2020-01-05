import os

from neupy import storage, architectures

from imagenet_tools import CURRENT_DIR, FILES_DIR, load_image, print_top_n, download_file


RESNET50_WEIGHTS_FILE = os.path.join(FILES_DIR, 'resnet50.hdf5')
DOG_IMAGE_PATH = os.path.join(CURRENT_DIR, 'images', 'german-shepherd.jpg')


def download_resnet50_weights():
    if not os.path.exists(RESNET50_WEIGHTS_FILE):
        download_file(
            url="http://neupy.s3.amazonaws.com/tensorflow/imagenet-models/resnet50.hdf5",
            filepath=RESNET50_WEIGHTS_FILE,
            description='Downloading weights')

    print("File with ResNet-50 weights: {}".format(RESNET50_WEIGHTS_FILE))
    return RESNET50_WEIGHTS_FILE


if __name__ == '__main__':
    resnet50_weights_filename = download_resnet50_weights()
    resnet50 = architectures.resnet50()

    print("Recovering ResNet-50 parameters...")
    storage.load(resnet50, resnet50_weights_filename)

    print("Making prediction...")
    dog_image = load_image(
        DOG_IMAGE_PATH,
        image_size=(256, 256),
        crop_size=(224, 224))

    output = resnet50.predict(dog_image)
    print_top_n(output, n=5)
