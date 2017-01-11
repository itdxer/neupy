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
                                             'pretrained-VIN-8.pickle'),
        input_image_shape=(2, 8, 8),
        image_size=(8, 8),
        k=10,
    ),
    16: dict(
        mat_file=os.path.join(DATA_DIR, 'gridworld_16.mat'),
        train_data_file=os.path.join(DATA_DIR, 'gridworld-16-train.pickle'),
        test_data_file=os.path.join(DATA_DIR, 'gridworld-16-test.pickle'),
        pretrained_network_file=os.path.join(MODELS_DIR,
                                             'pretrained-VIN-16.pickle'),
        input_image_shape=(2, 16, 16),
        image_size=(16, 16),
        k=20,
    ),
    28: dict(
        mat_file=os.path.join(DATA_DIR, 'gridworld_28.mat'),
        train_data_file=os.path.join(DATA_DIR, 'gridworld-28-train.pickle'),
        test_data_file=os.path.join(DATA_DIR, 'gridworld-28-test.pickle'),
        pretrained_network_file=os.path.join(MODELS_DIR,
                                             'pretrained-VIN-28.pickle'),
        input_image_shape=(2, 28, 28),
        image_size=(28, 28),
        k=36,
    ),
}
