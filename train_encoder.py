#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import vizdoom as vzd
import torch
import numpy as np
import random
import itertools as it
import skimage.transform

from vizdoom import Mode
from time import sleep, time
from collections import deque
from tqdm import trange
from AutoEncoder import AutoEncoder

batch_size = 64
resolution = (30, 45)

model_savefile = "encoder-model-doom.pth"
save_model = False
load_model = False

# Configuration file path
config_file_path = "./scenarios/deadly_corridor.cfg"


# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device('cpu')


def preprocess(img):
    """Down samples image to resolution"""
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img


def create_simple_game():
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")

    return game



def run(game, actions, encoder, num_epochs, steps_per_epoch):
    start_time = time()
    dset = deque(maxlen=10000)

    for epoch in range(num_epochs):
        game.new_episode()
        train_scores = []

        for _ in trange(steps_per_epoch, leave=False):
            state = preprocess(game.get_state().screen_buffer)
            action = np.random.choice(len(actions))
            reward = game.make_action(actions[action])
            done = game.is_episode_finished()

            dset.append(state)

            if not done:
                next_state = preprocess(game.get_state().screen_buffer)
                dset.append(next_state)
            else:
                game.new_episode()

            if len(dset) > batch_size:
                batch = random.sample(dset, batch_size)
                batch = torch.tensor(batch).float().to(DEVICE)

                loss_val = encoder.train(batch)
                train_scores.append(loss_val)

        train_scores = np.array(train_scores)

        print("Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()),
              "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

        if save_model:
            print("Saving the network weights to:", model_savefile)
            torch.save(encoder.encoder, model_savefile)
        print("Total elapsed time: %.2f minutes" % ((time() - start_time) / 60.0))

    game.close()

def foo():
    # Initialize game and actions
    game = create_simple_game()
    n = game.get_available_buttons_size()
    actions = np.identity(n, dtype=int).tolist()
    encoder = AutoEncoder(4, resolution, batch_size)

    # Run the training for the set number of epochs
    run(game, actions, encoder, num_epochs=10, steps_per_epoch=500)


if __name__ == '__main__':
    foo()
