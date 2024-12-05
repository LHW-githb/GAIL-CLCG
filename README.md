# Missile guidance via Generative Adversarial Imitation Learning (GAIL) and Continuous Learning

Code and models from the paper "Generative Adversarial Imitation Learning-based Continuous Learning Computational Guidance".

## Project Structure

The repository consists of the following key files:

funcs.py: Utility functions for mathematical operations, state transitions, and other tasks necessary for the missile environment simulation.

gail1.py, gail2.py, gail3.py: Implementations of different GAIL models. These may vary in terms of network architecture, training procedures, or other details.

Missile.py: Defines the missile simulation environment. It models the missile's flight dynamics and provides functions for interacting with the environment, including taking actions and receiving state transitions.

nets.py: Contains neural network architectures used in the training of the GAIL models.

test.py: This file is for testing the trained models and evaluating their performance.

train.py: The main training script. This file configures the environment, sets up the model, and manages the training process, including saving and loading model checkpoints.

##  Usage

### Training

To train the model, run the following command:

python train.py --env_name Missile

You can specify the environment name (Missile) and other hyperparameters in the command line arguments：

task = ["1", "2", "3"][1]

###  Testing

To test a trained model, run the following command:

python test.py --env_name Missile --model_path ./path_to_trained_model

(Models that have been trained are stored in . /ckpts/, the model corresponding to scenario 1 is best1.ckpt, the model corresponding to scenario 2 is best2.ckpt, and the model corresponding to scenario 3 is best3.ckpt.)

You can decide which guidance scenario to test by selecting the options in task.：

task = ["1", "2", "3"][1]
