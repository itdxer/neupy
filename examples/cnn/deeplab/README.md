## Deeplab for semantic segmentation


### Data

1. Download VOC 2012 data from the official website: [http://host.robots.ox.ac.uk/pascal/VOC/voc2012/](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)

2. Unpack data into the `dataset` folder and make sure that directory `dataset/VOCdevkit/VOC2012` exists.

### Training

Traing the model and save partial progress into the `storage` folder.

```
python -i examples/cnn/deeplab/train.py --storage-folder storage
```


### Validation

Use one of the stored deeplab models in order to assess network's performance on the validation dataset.

```
python -i examples/cnn/deeplab/validate.py --deeplab-weights storage/dump.hdf5
```
