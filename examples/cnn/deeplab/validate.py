import argparse

import numpy as np
from tqdm import tqdm

from data import (VALIDATION_SET, read_data, make_annotation_one_hot_encoded,
                  scale_images, rgb_to_bgr)
from model import create_deeplab_model
from utils import get_confusion_matrix, segmentation_metrics
from resnet50 import download_resnet50_weights


parser = argparse.ArgumentParser()
parser.add_argument('--deeplab-weights', required=True)


if __name__ == '__main__':
    args = parser.parse_args()
    resnet50_weights = download_resnet50_weights()

    resnet50, deeplab = create_deeplab_model(
        resnet50_weights, args.deeplab_weights)

    deeplab = resnet50 > deeplab
    confusion = np.zeros((21, 21))

    print("Start validation")
    for i, (image, annotation) in tqdm(enumerate(read_data(VALIDATION_SET))):
        image, annotation = scale_images(image, annotation, size=224)
        annotation = make_annotation_one_hot_encoded(annotation)

        image = rgb_to_bgr(np.expand_dims(image, axis=0))
        annotation = np.expand_dims(annotation, axis=0)
        segmentation = deeplab.predict(image)

        confusion += get_confusion_matrix(annotation, segmentation)

    accuracy, miou = segmentation_metrics(confusion)
    print("Val accuracy: {:.3f}".format(accuracy))
    print("Val miou: {:.3f}".format(miou))
