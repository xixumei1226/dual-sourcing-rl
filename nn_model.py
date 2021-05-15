import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

class Model(nn.Module):
    def __init__(self, state_dim, action_space, hidden_size1):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size1)
        self.relu = nn.ReLU()
        
        self.action_head = nn.Linear(hidden_size1, action_space)
        self.action_output = nn.Softmax(dim = 0)
        
        self.value_output = nn.Linear(hidden_size1, 1)
        
        self.data = []

    def forward(self, s):
        out = self.fc1(s)
        out = self.relu(out)
        
        action_probs = self.action_head(out)
        action_probs = self.action_output(action_probs)
        
        value = self.value_output(out)
        
        return action_probs, value

def nn_policy(env, model, m):
    state = torch.from_numpy(env.state).float()
    action_probs, value = model(state)
    
    dist = Categorical(action_probs)
    action = dist.sample()
    model.data.append([dist.log_prob(action), value])
    
    action = np.asarray([action.item() // m, action.item() % m])
    return action