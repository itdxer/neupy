from __future__ import division

import os

import requests
from tqdm import tqdm
import numpy as np
from imageio import imread
from skimage import transform

from neupy.utils import asfloat


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
FILES_DIR = os.path.join(CURRENT_DIR, 'files')
IMAGENET_CLASSES_FILE = os.path.join(FILES_DIR, 'imagenet_classes.txt')


def download_file(url, filepath, description=''):
    head_response = requests.head(url)
    filesize = int(head_response.headers['content-length'])

    response = requests.get(url, stream=True)
    chunk_size = int(1e7)

    n_iter = (filesize // chunk_size) + 1

    print(description)
    print('URL: {}'.format(url))
    with open(filepath, "wb") as handle:
        for data in tqdm(response.iter_content(chunk_size), total=n_iter):
            handle.write(data)

    print('Downloaded sucessfully')


def read_image(image_name, image_size=None, crop_size=None):
    image = imread(image_name, pilmode='RGB')

    if image_size is not None:
        height, width, _ = image.shape
        new_height, new_width = image_size

        if height < width:
            # Since width is bigger than height, this scaler
            # factor will say by how much it bigger
            # New width dimension will be scaled in the way
            # that output image will have proportional width and
            # height compae to it's original size
            proportion_scaler = width / height
            image_size = (new_height, int(new_width * proportion_scaler))
        else:
            proportion_scaler = height / width
            image_size = (int(new_height * proportion_scaler), new_width)

        image = transform.resize(
            image, image_size,
            preserve_range=True,
            mode='constant')

    if crop_size is not None:
        height, width, _ = image.shape
        height_slice = slice(
            (height - crop_size[0]) // 2,
            (height + crop_size[0]) // 2)

        width_slice = slice(
            (width - crop_size[1]) // 2,
            (width + crop_size[1]) // 2)

        image = image[height_slice, width_slice, :]

    # (height, width, channel) -> (1, height, width, channel)
    image = np.expand_dims(image, axis=0)
    return asfloat(image)


def process(image, use_bgr):
    # Per channel normalization
    image[:, :, :, 0] -= 123.68
    image[:, :, :, 1] -= 116.78
    image[:, :, :, 2] -= 103.94

    if use_bgr:
        # RGB -> BGR
        image[:, :, :, (0, 1, 2)] = image[:, :, :, (2, 1, 0)]

    return image


def load_image(image_name, image_size=None, crop_size=None, use_bgr=True):
    image = read_image(image_name, image_size, crop_size)
    return process(image, use_bgr)


def deprocess(image):
    image = image.copy()

    # BGR -> RGB
    image[:, :, (0, 1, 2)] = image[:, :, (2, 1, 0)]

    image[:, :, 0] += 123.68
    image[:, :, 1] += 116.78
    image[:, :, 2] += 103.94

    return image.astype(int)


def top_n(probs, n=5):
    if probs.ndim == 2:
        probs = probs[0]  # take probabilities for first image

    with open(IMAGENET_CLASSES_FILE, 'r') as f:
        class_names = f.read().splitlines()
        class_names = np.array(class_names)

    max_probs_indices = probs.argsort()[-n:][::-1]
    class_probs = probs[max_probs_indices]
    top_classes = class_names[max_probs_indices]

    return top_classes, class_probs


def print_top_n(probs, n=5):
    top_classes, class_probs = top_n(probs, n)

    print('-----------------------')
    print('Top-{} predicted classes'.format(n))
    print('-----------------------')

    for top_class, class_prob in zip(top_classes, class_probs):
        print("{:<80s}: {:.2%}".format(top_class, class_prob))

    print('-----------------------')
