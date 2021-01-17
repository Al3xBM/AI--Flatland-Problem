import random
from collections import namedtuple, deque, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from script.model import NeuralNetwork

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
        self.t_step = 0

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

    def take_step(self, state, action, reward, next_state, done, train = True):
        self.replay_buffer.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.replay_buffer) > BATCH_SIZE:
                experiences = self.replay_buffer.sample()
                if train:
                    self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        Q_expected = self.q_network(states).gather(1, actions)

        q_best_action = self.q_network(next_states).max(1)[1]
        Q_targets_next = self.target_network(next_states).gather(1, q_best_action.unsqueeze(-1))

        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        loss = F.mse_loss(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.q_network, self.target_network, TAU)

    def soft_update(self, q_model, target_model, tau):
        for target_param, q_param in zip(target_model.parameters(), q_model.parameters()):
            target_model.data.copy_(tau * q_param.data + (1.0 - tau) * target_param.data)

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        # Add a sample
        e = self.experience(np.expand_dims(state, 0), action, reward, np.expand_dims(next_state, 0), done)
        self.memory.append(e)

    def sample(self):
        # Returns a random sample
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(self.__v_stack_impr([e.state for e in experiences if e is not None])) \
            .float().to(device)
        actions = torch.from_numpy(self.__v_stack_impr([e.action for e in experiences if e is not None])) \
            .long().to(device)
        rewards = torch.from_numpy(self.__v_stack_impr([e.reward for e in experiences if e is not None])) \
            .float().to(device)
        next_states = torch.from_numpy(self.__v_stack_impr([e.next_state for e in experiences if e is not None])) \
            .float().to(device)
        dones = torch.from_numpy(self.__v_stack_impr([e.done for e in experiences if e is not None]).astype(np.uint8)) \
            .float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

    def __v_stack_impr(self, states):
        sub_dim = len(states[0][0]) if isinstance(states[0], Iterable) else 1
        np_states = np.reshape(np.array(states), (len(states), sub_dim))
        return np_states
