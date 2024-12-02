Missile guidance via Generative Adversarial Imitation Learning (GAIL) and Continuous Learning

This repository contains code for training reinforcement learning models using Generative Adversarial Imitation Learning (GAIL), specifically in the context of missile guidance simulation.
The objective is to train a policy model that can control a missile’s flight dynamics based on expert demonstrations. This is done by using GAIL, which combines imitation learning and adversarial training to match the policy to the expert’s behavior.

Project Structure
The repository consists of the following key files:
funcs.py: Utility functions for mathematical operations, state transitions, and other tasks necessary for the missile environment simulation.
gail1.py, gail2.py, gail3.py: Implementations of different GAIL models. These may vary in terms of network architecture, training procedures, or other details.
Missile.py: Defines the missile simulation environment. It models the missile's flight dynamics and provides functions for interacting with the environment, including taking actions and receiving state transitions.
nets.py: Contains neural network architectures used in the training of the GAIL models.
test.py: This file is for testing the trained models and evaluating their performance.
train.py: The main training script. This file configures the environment, sets up the model, and manages the training process, including saving and loading model checkpoints.
