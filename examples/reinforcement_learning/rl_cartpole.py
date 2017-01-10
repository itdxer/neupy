import os
import random
import argparse
from collections import deque

import gym
import numpy as np
from neupy import layers, algorithms, environment, storage


environment.reproducible()

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
FILES_DIR = os.path.join(CURRENT_DIR, 'files')
CARTPOLE_WEIGHTS = os.path.join(FILES_DIR, 'cartpole-weights.pickle')


def training_samples(network, memory, gamma=0.9):
    data = np.array(memory, dtype=[
        ('state', np.ndarray),
        ('action', np.int),
        ('reward', np.int),
        ('done', np.bool),
        ('new_state', np.ndarray),
    ])

    state = np.array(data['state'].tolist())
    new_state = np.array(data['new_state'].tolist())

    # Note: Calculating Q for all states at once is much faster
    # that do it per each sample separately
    Q = network.predict(state)
    new_Q = network.predict(new_state)
    max_Q = np.max(new_Q, axis=1)

    n_samples = len(memory)
    row_index = np.arange(n_samples)
    column_index = data['action']

    Q[(row_index, column_index)] = np.where(
        data['done'],
        data['reward'],
        data['reward'] + gamma * max_Q
    )

    return state, Q


def play_game(env, network, n_steps=1000):
    state = env.reset()

    for _ in range(n_steps):
        env.render()

        q = network.predict(state)
        action = int(np.argmax(q[0]))

        state, _, done, _ = env.step(action)

        if done:
            break


def train_network(env, network, memory, n_games=200, max_score=200,
                  epsilon=0.1, gamma=0.9):
    for episode in range(1, n_games + 1):
        state = env.reset()

        for t in range(max_score):
            if random.random() <= epsilon:
                # Select random action with probability
                # equal to the `epsilon`
                action = random.randint(0, 1)
            else:
                # Use action selected by the network
                q = network.predict(state)
                action = int(np.argmax(q[0]))

            new_state, reward, done, info = env.step(action)
            memory.append((state, action, reward, done, new_state))

            if done:
                # We done when network lost the game.
                # Low reward will penalyze network.
                reward = -10

            if len(memory) == memory_size:
                # Train only when we collected enough samples
                x_train, y_train = training_samples(network, memory, gamma)
                network.train(x_train, y_train, epochs=1)
                loss = network.errors.last()

            state = new_state

            if done:
                break

        if len(memory) == memory_size:
            print("Game #{:<3} | Lost after {:<3} iterations | loss: {:.4}"
                  "".format(episode, t + 1, loss))
        else:
            print("Game #{:<3} | Lost after {:<3} iterations"
                  "".format(episode, t + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Trains network to play CartPole game')
    parser.add_argument(
        '-p', '--pretrained', dest='use_pretrained', action='store_true',
        help='load pretrained network from file and play without training')
    args = parser.parse_args()

    network = algorithms.RMSProp(
        [
            layers.Input(4),

            layers.Relu(64),
            layers.Relu(48),
            layers.Relu(32),
            layers.Relu(64) > layers.Dropout(0.2),

            # Expecting two different actions:
            # 1. Move left
            # 2. Move right
            layers.Linear(2),
        ],

        step=0.001,
        error='rmse',
        batch_size=100,

        decay_rate=0.1,
        addons=[algorithms.WeightDecay],
    )

    env = gym.make('CartPole-v0')
    env.seed(0)  # To make results reproducible for the gym

    memory_size = 1000  # Number of samples stored in the memory
    memory = deque(maxlen=memory_size)

    if args.use_pretrained:
        if not os.path.exists(CARTPOLE_WEIGHTS):
            raise OSError("Cannot find file with pretrained weights "
                          "(File name: {})".format(CARTPOLE_WEIGHTS))

        print("Loading pretrained weights")
        storage.load(network, CARTPOLE_WEIGHTS)

    else:
        print("Start training")
        train_network(
            env, network, memory,
            n_games=120,  # Number of games that networks is going to play,
            max_score=200,  # Maximum score that network can achive in the game
            epsilon=0.1,  # Probability to select random action during the game
            gamma=0.99)

        if not os.path.exists(FILES_DIR):
            os.mkdir(FILES_DIR)

        print("Saving parameters")
        storage.save(network, CARTPOLE_WEIGHTS)

    # After the training we can check how network solves the problem
    print("Start playing game")
    play_game(env, network, n_steps=1000)
