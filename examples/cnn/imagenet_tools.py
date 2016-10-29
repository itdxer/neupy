import os

import requests
from tqdm import tqdm
import numpy as np
from scipy.misc import imread, imresize


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
FILES_DIR = os.path.join(CURRENT_DIR, 'files')
IMAGENET_CLASSES_FILE = os.path.join(FILES_DIR, 'imagenet_classes.txt')


def download_file(url, filepath):
    response = requests.get(url, stream=True)
    chunk_size = int(1e7)

    with open(filepath, "wb") as handle:
        for data in tqdm(response.iter_content(chunk_size)):
            handle.write(data)


def load_image(image_name, image_size=None, crop_size=None):
    image = imread(image_name)

    if image_size is not None:
        image = imresize(image, image_size)

    if crop_size is not None:
        image = image[
            slice(
                (image_size[0] - crop_size[0]) // 2,
                (image_size[0] + crop_size[0]) // 2,
            ),
            slice(
                (image_size[1] - crop_size[1]) // 2,
                (image_size[1] + crop_size[1]) // 2,
            ),
            :,
        ]

    # RGB -> BGR
    image[:, :, (0, 1, 2)] = image[:, :, (2, 1, 0)]
    image = image.astype('float32')

    # Normalize channels (based on the pretrained VGG16 configurations)
    image[:, :, 0] -= 123.68
    image[:, :, 1] -= 116.779
    image[:, :, 2] -= 103.939

    # (height, width, channel) -> (channel, height, width)
    image = image.transpose((2, 0, 1))
    # (channel, height, width) -> (1, channel, height, width)
    image = np.expand_dims(image, axis=0)

    return image


def top_n(probs, n=5):
    with open(IMAGENET_CLASSES_FILE, 'r') as f:
        class_names = f.read().splitlines()
        class_names = np.array(class_names)

    max_probs_indeces = probs.argsort()[-n:][::-1]
    class_probs = probs[max_probs_indeces]
    top_classes = class_names[max_probs_indeces]

    return top_classes, class_probs


def print_top_n(probs, n=5):
    top_classes, class_probs = top_n(probs, n=5)

    print('-----------------------')
    print('Top-5 predicted classes')
    print('-----------------------')

    for top_class, class_prob in zip(top_classes, class_probs):
        print("{:<80s}: {:.2%}".format(top_class, class_prob))
