import numpy as np
import matplotlib.pyplot as plt

def convergence_test(env, numiters, policy, *args):

    cum_reward = np.zeros(numiters)
    av_reward = np.zeros(numiters)
    
    env.reset() # reset environment
    for t in range(numiters-1):
        action = policy(*args)
        state, reward, demand, info = env.step(action)
        cum_reward[t+1] = cum_reward[t] + reward
        av_reward[t+1] = cum_reward[t+1] / (t+1)
        
    plt.plot(range(numiters), av_reward)
    plt.xlabel('time step')
    plt.ylabel('average reward')
    
    print('Average reward: ' + str(av_reward[-1]))

def evaluate(env, n_episodes, numiters, policy, *args):
    # env: gym environment
    # n_episodes: number of total episodes to run (outer iteration)
    # numiters: number of time steps (inner iteration)
    # policy: policy function
    # *args: arguments in the policy function
    env.seed(0)
    av_reward = np.zeros(n_episodes)
    
    for i in range(n_episodes):
        av_r = 0
        env.reset() # reset environment
        for t in range(numiters):
            action = policy(*args)
            state, reward, demand, info = env.step(action)
            if t > 100 and np.abs( av_r / (t+1) - (av_r + reward) / (t+2))  < 1e-4: # convergence is spotted
                break
            av_r = av_r + reward
        av_reward[i] = av_r / (t+1)
        
    return np.mean(av_reward), np.std(av_reward) # return average reward and std