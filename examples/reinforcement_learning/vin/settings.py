import os


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'data')
MODELS_DIR = os.path.join(CURRENT_DIR, 'models')

environments = {
    8: dict(
        mat_file=os.path.join(DATA_DIR, 'gridworld_8.mat'),
        train_data_file=os.path.join(DATA_DIR, 'gridworld-8-train.pickle'),
        test_data_file=os.path.join(DATA_DIR, 'gridworld-8-test.pickle'),
        pretrained_network_file=os.path.join(MODELS_DIR,
                                             'pretrained-VIN-8.hdf5'),
        input_image_shape=(8, 8, 2),
        image_size=(8, 8),
        k=10,
        epochs=120,
    ),
    16: dict(
        mat_file=os.path.join(DATA_DIR, 'gridworld_16.mat'),
        train_data_file=os.path.join(DATA_DIR, 'gridworld-16-train.pickle'),
        test_data_file=os.path.join(DATA_DIR, 'gridworld-16-test.pickle'),
        pretrained_network_file=os.path.join(MODELS_DIR,
                                             'pretrained-VIN-16.hdf5'),
        input_image_shape=(16, 16, 2),
        image_size=(16, 16),
        k=20,
        epochs=120,
    ),
    28: dict(
        mat_file=os.path.join(DATA_DIR, 'gridworld_28.mat'),
        train_data_file=os.path.join(DATA_DIR, 'gridworld-28-train.pickle'),
        test_data_file=os.path.join(DATA_DIR, 'gridworld-28-test.pickle'),
        pretrained_network_file=os.path.join(MODELS_DIR,
                                             'pretrained-VIN-28.hdf5'),
        input_image_shape=(28, 28, 2),
        image_size=(28, 28),
        k=36,
        epochs=120,
    ),
}
