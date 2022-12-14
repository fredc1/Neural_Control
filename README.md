# Neural_Control
* Frederick Cunningham, fc2687, 12/18/2022
* Simply run python3 animate_data.py to view the dynamics output.
* All code was written by Fred Cunningham (fc2687) as this is an individual project
* Data for this project was automatically generated in data_prep.py
* scripts are included in the main directory... ther is logic that depends on them being on the same level as .pkl files
* Tools: scikitlearn, pytorch, matplotlib

dynamics.py: generates data streams for the animator using numerical ivp solver
animate_data.py animates the results
data_prep.py: data generation and scaling logic
train_models.ipynb: a playground for model iteration contains the training functions and neural model definitions
.pkl files represent model checkpoints
* emulator_random_inputs3 is the most successful emulator
* controller_model2.pkl is the most successful controller. 
* when unpickled, these become nn.Sequential objects

Abstract:
The goal of this work is to explore the integration of neural networks with dynamic systems in order to stabilize and control them. Neural networks have shown the ability to learn complex non-linear dynamics without prior knowledge of the system. However, they carry the drawbacks of poor generalization outside regions where they have a dense sampling of the input space to train on. This work achieved a model that predicted system dynamics with ~0.99 cosine error, but could not translate this model into a controller that stabilized the real world dynamics under consideration.
