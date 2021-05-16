import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from nn_model import nn_policy
from TBS_module import TBS

def initialize(env, model, r, S, stepsize=0.01, maxit=10000, seed=0):
    np.random.seed(seed)
    
    state_dim = env.observation_space.shape[0]
    m = env.max_order + 1
    n = env.Lambda + 1
    l = 2 * env.max_order + 1
    
    for i in range(maxit):
        s = np.random.rand(state_dim)
        s = np.floor(s * np.asarray([n]*(state_dim-1) + [l]))
        s[-1] -= env.max_order
        env.state = s
        
        qr, qe = TBS(env, r, S)
        action = m * qr + qe
        action = int(action)
        probs, _ = model(torch.from_numpy(s).float())
        
        model.zero_grad()
        loss = (probs ** 2).sum() - 2 * probs[action]
        loss.backward()

        for name, layer in model.named_modules():
            if type(layer) == nn.Linear and not (name == 'value_output'):
                layer.weight.data -= stepsize * layer.weight.grad
                layer.bias.data -= stepsize * layer.bias.grad

def A2C(env, model, optimizer, maxit, rollout, param = {}):
    if 'gamma' in param:
        gamma = param['gamma']
    else:
        gamma = 0.99
    
    if 'terminal_state' in param:
        terminal_state = param['terminal_state']
    else:
        terminal_state = None
    
    if 'loss_function' in param:
        loss_function = param['loss_function']
    else:
        loss_function = functional.smooth_l1_loss
    
    if 'value_weight' in param:
        value_weight = param['value_weight']
    else:
        value_weight = 1.0
    
    if 'number_of_actors' in param:
        number_of_actors = param['number_of_actors']
    else:
        number_of_actors = 1
    
    if 'proportion' in param:
        proportion = param['proportion']
    else:
        proportion = 1
    
    if 'env_seed' in param:
        env.seed(param['env_seed'])
    if 'torch_seed' in param:
        torch.manual_seed(param['torch_seed'])

    m = env.max_order + 1
    
    for t in range(maxit):
        policy_loss = []
        value_loss = []
        for k in range(number_of_actors):
            state = env.reset()
            model.data = []
            rewards = []
            for t in range(rollout):
                action = nn_policy(env, model, m)
                state, reward, _, _ = env.step(action)
                rewards.append(torch.tensor(reward).float())

                if np.array_equal(state, terminal_state):
                    break
            
            data = model.data
            discounted_reward = 0
            T = len(rewards)
            for i in range(T-1, -1, -1):
                log_prob, value = data[i]
                discounted_reward = rewards[i] + gamma * discounted_reward
                if i < T * proportion:
                    advantage = discounted_reward - value.item()
                    policy_loss.append(- advantage * log_prob / number_of_actors)
                    value_loss.append(loss_function(value, torch.tensor([discounted_reward])) / number_of_actors)

        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum() + value_weight * torch.stack(value_loss).sum()
        loss.backward()
        optimizer.step()