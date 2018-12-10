import os
import argparse

import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from neupy import algorithms, storage

from data import get_training_data, get_validation_data
from model import create_deeplab_model
from utils import score_segmentation
from resnet50 import download_resnet50_weights


parser = argparse.ArgumentParser()
parser.add_argument('--storage-folder', '-s', required=True)
parser.add_argument('--epochs', '-e', type=int, default=30)
parser.add_argument('--batch-size', '-b', type=int, default=10)


def iter_batches(images, annotations, batch_size=10):
    images, annotations = shuffle(images, annotations)

    for i in range(0, len(images), batch_size):
        # Note: We will exclude last incomplete batch
        yield images[i:i + batch_size], annotations[i:i + batch_size]


if __name__ == '__main__':
    args = parser.parse_args()
    storage_folder = args.storage_folder

    if not os.path.exists(storage_folder):
        os.mkdir(storage_folder)

    resnet50_weights_filename = download_resnet50_weights()
    resnet50, deeplab = create_deeplab_model(resnet50_weights_filename)

    images, annotations = get_training_data()
    val_images, val_annotations = get_validation_data()

    optimizer = algorithms.Adam(
        deeplab,

        error='categorical_crossentropy',
        step=0.00001,
        verbose=True,

        addons=[algorithms.WeightDecay],
        decay_rate=0.0001,
    )

    for i in range(args.epochs):
        print("Epoch #{}".format(i + 1))

        for x_batch, y_batch in iter_batches(images, annotations, args.batch_size):
            x_batch = resnet50.predict(x_batch)
            optimizer.train(x_batch, y_batch, epochs=1, summary='inline')

        x_val, y_val = next(iter_batches(val_images, val_annotations, 100))
        y_pred = deeplab.predict(resnet50.predict(x_val))

        accuracy, miou = score_segmentation(y_val, y_pred)
        print("Val accuracy: {:.3f}".format(accuracy))
        print("Val miou: {:.3f}".format(miou))

        filename = 'deeplab_{:0>3}_{:.3f}_{:.3f}.hdf5'.format(i, accuracy, miou)
        filepath = os.path.join(storage_folder, filename)

        print("Saved: {}".format(filepath))
        storage.save(deeplab, filepath)
