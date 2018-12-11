import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from data import voc_classes


def reverse_indeces(image):
    image = image[0]
    image = image.argmax(axis=-1)

    index2color = np.array([class_.color for class_ in voc_classes])
    return index2color[image.astype(int)]


def show(image, annotation, segmentation):
    plt.subplot(131)
    plt.imshow(image[0])

    plt.subplot(132)
    plt.imshow(reverse_indeces(annotation))

    plt.subplot(133)
    plt.imshow(reverse_indeces(segmentation))

    plt.show()


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
