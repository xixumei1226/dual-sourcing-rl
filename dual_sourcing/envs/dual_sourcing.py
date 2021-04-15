import gym
import numpy as np
import sys


class DualSourcing(gym.Env):

    def __init__(self, config):
        
        self.Lr = config['Lr']
        self.Le = config['Le']
        self.cr = config['cr']
        self.ce = config['ce']
        self.Lambda = config['lambda']
        self.h = config['h']
        self.b = config['b']
        self.starting_state = config['starting_state']
        self.max_order = config['max_order']
        self.max_inventory = config['max_inventory']
        
        self.state = np.asarray(self.starting_state)
        self.action_space = gym.spaces.MultiDiscrete([self.max_order+1]*2)
        self.observation_space = gym.spaces.MultiDiscrete([self.max_order+1]*(self.Lr+self.Le)+[self.max_inventory])
        
        metadata = {'render.modes': ['human']}
    
    def seed(self, seed=None):
        np.random.seed(seed)
        self.action_space.np_random.seed(seed)
        return seed
    
    def step(self, action):
        reward = self.r(self.state)
        
        demand = np.random.poisson(self.Lambda)
        newState = self.g(self.state, action)
        newState[-1] = newState[-1] - demand
        self.state = newState.copy()
#         done = (newState[-1] == self.starting_state[-1])
        
        return self.state, reward, demand, {}
    
    # Auxilary function computing the reward
    def r(self, state):
        return -(self.cr*state[self.Lr-1] + self.ce*state[self.Lr+self.Le-1] + 
                 self.h*max(state[-1],0) + self.b*max(-state[-1],0))
    
    # Auxilary function
    def g(self, state, action):
        return np.hstack([state[1:self.Lr], action[0], state[self.Lr+1:self.Lr+self.Le], action[1],
                         state[self.Lr+self.Le]+state[0]+state[self.Lr]]).astype(int)
    
    def render(self, mode='human'):
        outfile = sys.stdout if mode == 'human' else super(DualSourcing, self).render(mode=mode)
        outfile.write(np.array2string(self.state)+'\n')

    def reset(self):
        self.state = np.asarray(self.starting_state)
        return self.state