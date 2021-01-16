import torch
import torch.optim as optim
import random
import numpy as np

from trainings.scripts.model import NeuralNetwork

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 512  # minibatch size
GAMMA = 0.99  # discount factor 0.99
TAU = 1e-3  # for soft update of target parameters
LR = 0.5e-4  # learning rate 0.5e-4 works
UPDATE_EVERY = 10  # how often to update the network

device = torch.device("cpu")

class Agent:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.q_network = NeuralNetwork(input_size, output_size).to(device)
        self.target_network = NeuralNetwork(input_size, output_size).to(device)

        self.optimizer = optim.RMSprop(self.q_network.parameters(), lr=LR)

        self.replay_buffer = ReplayBuffer(output_size, BUFFER_SIZE, BATCH_SIZE)

    def save(self):
        torch.save(self.q_network.state_dict(),"../backup/backup.q")
        torch.save(self.target_network.state_dict(),"../backup/backup.target")

    def load(self):
        self.q_network.load_state_dict(torch.load("../backup/backup.q"))
        self.target_network.load_state_dict(torch.load("../backup/backup.target"))

    def give_action(self, state, epsilon = 0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.q_network.eval()
        with torch.no_grad():
            action = self.q_network(state)
        self.q_network.train()

        if random.random() > epsilon:
            return np.argmax(action.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.output_size))

class ReplayBuffer:
    def __init__(self, output_size, buffer_size, batch_size):
        pass