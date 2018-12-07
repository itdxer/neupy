import os
import sys
import os.path
from collections import namedtuple

# This code allows to make imagenet_tools visible in the deeplab project
previous_dir = os.path.join(os.path.dirname(__file__), os.path.pardir)
sys.path.append(os.path.abspath(previous_dir))

import numpy as np
from imageio import imread
from scipy.misc import imresize
from tqdm import tqdm

from imagenet_tools import process


BASEPATH = "/Users/itdxer/Downloads/VOCdevkit/VOC2012/"

IMAGESETS = os.path.join(BASEPATH, "ImageSets", "Segmentation")
IMAGES = os.path.join(BASEPATH, "JPEGImages")
ANNOTATIONS = os.path.join(BASEPATH, "SegmentationClass")

TRAIN_SET = os.path.join(IMAGESETS, "train.txt")
TRAINVAL_SET = os.path.join(IMAGESETS, "trainval.txt")
VALIDATION_SET = os.path.join(IMAGESETS, "val.txt")

VOCClass = namedtuple("VOCClass", "index color name")
voc_classes = [
    VOCClass(index=0, color=(0, 0, 0), name="background"),
    VOCClass(index=1, color=(128, 0, 0), name="aeroplane"),
    VOCClass(index=2, color=(0, 128, 0), name="bicycle"),
    VOCClass(index=3, color=(128, 128, 0), name="bird"),
    VOCClass(index=4, color=(0, 0, 128), name="boat"),
    VOCClass(index=5, color=(128, 0, 128), name="bottle"),
    VOCClass(index=6, color=(0, 128, 128), name="bus"),
    VOCClass(index=7, color=(128, 128, 128), name="car"),
    VOCClass(index=8, color=(64, 0, 0), name="cat"),
    VOCClass(index=9, color=(192, 0, 0), name="chair"),
    VOCClass(index=10, color=(64, 128, 0), name="cow"),
    VOCClass(index=11, color=(192, 128, 0), name="diningtable"),
    VOCClass(index=12, color=(64, 0, 128), name="dog"),
    VOCClass(index=13, color=(192, 0, 128), name="horse"),
    VOCClass(index=14, color=(64, 128, 128), name="motorbike"),
    VOCClass(index=15, color=(192, 128, 128), name="person"),
    VOCClass(index=16, color=(0, 64, 0), name="potted plant"),
    VOCClass(index=17, color=(128, 64, 0), name="sheep"),
    VOCClass(index=18, color=(0, 192, 0), name="sofa"),
    VOCClass(index=19, color=(128, 192, 0), name="train"),
    VOCClass(index=20, color=(0, 64, 128), name="tv/monitor"),
]


def read_data(filepath):
    with open(filepath) as f:
        for line in f:
            image_path = os.path.join(IMAGES, line.strip() + '.jpg')
            annotation_path = os.path.join(ANNOTATIONS, line.strip() + '.png')

            yield (
                imread(image_path),
                imread(annotation_path),
            )


def find_pads(difference):
    half_difference = int(difference // 2)

    if difference % 2 == 0:
        return (half_difference, half_difference)

    return (half_difference, half_difference + 1)


def pad_values(image, size, value):
    in_height, in_width, _ = image.shape

    return np.pad(
        image,[
            find_pads(size - in_height),
            find_pads(size - in_width),
            (0, 0),  # no padding for channels
        ],
        mode='constant',
        constant_values=value,
    )


def scale_images(image, annotation, size):
    if image.shape[:-1] != annotation.shape[:-1]:
        raise ValueError(
            "Image and annotation should have same height and width, got "
            "image with shape {} and annotation with shape {}"
            "".format(image.shape, annotation.shape))

    in_height, in_width, _ = image.shape
    largest_side = max(in_height, in_width)
    scale = size / largest_side

    image = imresize(image, scale, interp='nearest')
    annotation = imresize(annotation, scale, interp='nearest')

    return (
        pad_values(image, size, value=0),
        # We make sure that padding will add unknown pixel
        # colors - (33, 33, 33). Because there is no class
        # assiciated with this color we can ignore it as
        # unknown class in the future. We cannot use 0, because it
        # marks background class.
        pad_values(annotation, size, value=33),
    )


def make_annotation_one_hot_encoded(annotation_3d):
    # From: https://github.com/wuhuikai/DeepGuidedFilter
    annotation_3d = annotation_3d[:, :, (0, 1, 2)]  # use only RGB
    height, width, _ = annotation_3d.shape

    n_classes = len(voc_classes)
    annotation_onehot = np.zeros((height, width, n_classes), dtype=np.uint8)

    for voc_class in voc_classes:
        class_color = np.array(voc_class.color).reshape(1, 1, 3)
        mask = np.all(annotation_3d == class_color, axis=2)
        annotation_onehot[mask, voc_class.index] = 1

    return annotation_onehot


def rgb_to_bgr(images):
    images[:, :, :, (0, 1, 2)] = images[:, :, :, (2, 1, 0)]
    return images


def get_data(filepath):
    images = []
    annotations = []

    for image, annotation in tqdm(read_data(filepath)):
        image, annotation = scale_images(image, annotation, size=224)
        annotation = make_annotation_one_hot_encoded(annotation)

        images.append(image)
        annotations.append(annotation)

    return (
        # rgb_to_bgr(np.stack(images, axis=0)),
        process(np.stack(images, axis=0).astype(np.float32), use_bgr=True),
        np.stack(annotations, axis=0),
    )


def get_training_data():
    print("Loading training data...")
    return get_data(TRAIN_SET)


def get_validation_data():
    print("Loading validation data...")
    return get_data(VALIDATION_SET)
