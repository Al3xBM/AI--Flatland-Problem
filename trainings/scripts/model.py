import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=150):
        super().__init__()

        self.q_input = nn.Linear(input_size, hidden_size)
        self.q_hidden = nn.Linear(hidden_size, hidden_size)
        self.q_output = nn.Linear(hidden_size, 1)

        self.target_input = nn.Linear(input_size, hidden_size)
        self.target_hidden = nn.Linear(hidden_size, hidden_size)
        self.target_output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        q_val = F.relu(self.q_input(x))
        q_val = F.relu(self.q_hidden(q_val))
        q_val = self.q_output(q_val)

        # advantage calculation
        target_val = F.relu(self.target_input(x))
        target_val = F.relu(self.target_hidden(target_val))
        target_val = self.target_output(target_val)
        return q_val + target_val - target_val.mean()
