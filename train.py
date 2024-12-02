import os
import json
import pickle
import argparse
import torch
import gym
from scipy.io import loadmat
from random import randint, uniform, choice
import numpy as np
import math
from math import sin, cos, atan2, sqrt
import scipy.io
from scipy import interpolate
import torch
import torch.nn as nn
import torch.optim as optim
import torch
import scipy.io
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
from nets import Expert
from nets import My_Model
from gail1 import GAIL1
from gail2 import GAIL2
from gail3 import GAIL3
from Missile import MISSILE


task = ["1", "2", "3"][1]


def main(env_name):
    ckpt_path = "ckpts"
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    if env_name not in ["CartPole-v1", "Pendulum-v0", "BipedalWalker-v3","Missile"]:
        print("The environment name is wrong!")
        return

    expert_ckpt_path = "experts"
    expert_ckpt_path = os.path.join(expert_ckpt_path, env_name)

    with open(os.path.join(expert_ckpt_path, "model_config.json")) as f:
        expert_config = json.load(f)

    ckpt_path = os.path.join(ckpt_path, env_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    with open("config.json") as f:
        config = json.load(f)[env_name]

    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    env = MISSILE()
    env.modify()
    state_dim = len(env.state)
    discrete = False
    action_dim = 1

    device = "cpu"

    if task == "1":
        config = {'save_path': './experts/Expert1.pth'}
        model = GAIL1(state_dim, action_dim, discrete, config).to(device)

    if task == "2":
        config = {'save_path': './experts/Expert2.pth'}
        model = GAIL2(state_dim, action_dim, discrete, config).to(device)

    if task == "3":
        config = {'save_path': './experts/Expert3.pth'}
        model = GAIL3(state_dim, action_dim, discrete, config).to(device)

    expert = My_Model(input_dim=6)
    expert.load_state_dict(torch.load(config['save_path'],map_location='cpu'))
    expert.eval()
    expert.to(device)

    results = model.train(env, expert)


    if hasattr(model, "pi"):
        torch.save(
            model.pi.state_dict(), os.path.join(ckpt_path, "policy.ckpt")
        )
    if hasattr(model, "v"):
        torch.save(
            model.v.state_dict(), os.path.join(ckpt_path, "value.ckpt")
        )
    if hasattr(model, "d"):
        torch.save(
            model.d.state_dict(), os.path.join(ckpt_path, "discriminator.ckpt")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="Missile",
        help="Type the environment name to run. \
            The possible environments are \
                [CartPole-v1, Pendulum-v0, BipedalWalker-v3,Missile]"
    )
    args = parser.parse_args()

    main(**vars(args))