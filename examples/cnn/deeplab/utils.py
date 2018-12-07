import numpy as np
import matplotlib.pyplot as plt

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
