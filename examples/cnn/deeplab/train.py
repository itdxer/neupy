import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from neupy import algorithms, storage

from data import get_training_data, get_validation_data
from model import create_deeplab_model
from resnet50 import download_resnet50_weights


def iter_batches(images, annotations, batch_size=10):
    images, annotations = shuffle(images, annotations)

    for i in range(0, len(images), batch_size):
        # Note: We will exclude last incomplete batch
        yield images[i:i + batch_size], annotations[i:i + batch_size]


def get_confusion_matrix(y_val, y_pred):
    y_valid_label = y_val.max(axis=-1)
    interested_labels = y_valid_label == 1

    y_expected = y_val.argmax(axis=-1)[interested_labels]
    y_pred = y_pred.argmax(axis=-1)[interested_labels]

    labels = list(range(21))
    return confusion_matrix(y_expected, y_pred, labels=labels)


def segmentation_metrics(confusion):
    tp = np.diag(confusion)
    sum_cols = confusion.sum(axis=1)
    sum_rows = confusion.sum(axis=0)

    union = (sum_cols + sum_rows - tp)
    iou = tp / union
    miou = np.mean(iou[(union > 0) & (sum_cols > 0)])
    accuracy = tp.sum() / confusion.sum()

    return accuracy, miou


def score_segmentation(y_val, y_pred):
    confusion = get_confusion_matrix(y_val, y_pred)
    return segmentation_metrics(confusion)


if __name__ == '__main__':
    batch_size = 10
    n_epochs = 30

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

    for i in range(n_epochs):
        print("Epoch #{}".format(i + 1))

        for x_batch, y_batch in iter_batches(images, annotations, batch_size):
            x_batch = resnet50.predict(x_batch)
            optimizer.train(x_batch, y_batch, epochs=1, summary='inline')

        x_val, y_val = next(iter_batches(val_images, val_annotations, 100))
        y_pred = deeplab.predict(resnet50.predict(x_val))

        accuracy, miou = score_segmentation(y_val, y_pred)
        print("Val accuracy: {:.3f}".format(accuracy))
        print("Val miou: {:.3f}".format(miou))

        filename = 'dump_v7/deeplab_v07_{:0>3}_{:.3f}_{:.3f}.hdf5'.format(i, accuracy, miou)
        print("Saved: {}".format(filename))
        storage.save(deeplab, filename)
