import numpy as np
from scipy.stats import poisson
import utility

def TBS(env, r, S):
    ip = r * env.Le + np.sum(env.state[env.Lr:])
    return r, max(0, S-ip)

def eigenvalue(A, v):
    Av = A.dot(v)
    return v.dot(Av)

def power_iteration(A):
    n, d = A.shape

    v = np.ones(d) / np.sqrt(d)
    ev = eigenvalue(A, v)

    while True:
        Av = A.dot(v)
        v_new = Av / np.linalg.norm(Av)

        ev_new = eigenvalue(A, v_new)
        if np.abs(ev - ev_new) < 0.01:
            break

        v = v_new
        ev = ev_new
    
    return v_new

def find_optimal_TBS(env):
    max_reward = 0
    flag = True
    lam = env.Lambda
    for r in range(lam):
        # find optimal S
        size = 100
        trans_prob = np.zeros(shape = (size, size))
        for i in range(1, size, 1):
            for j in range(1, min(i+r, size), 1):
                trans_prob[i, j] = poisson.pmf(lam, i+r-j)
            k = i + r 
            prob = poisson.pmf(lam, k)
            while prob > 0:
                trans_prob[i, 0] = trans_prob[i, 0] + prob
                k = k + 1
                prob = poisson.pmf(lam, k)
        stat_dist = power_iteration(trans_prob)
        cum_prob = 0
        S = 0
        while cum_prob < env.b / (env.b + env.h):
            cum_prob = cum_prob + stat_dist[S]
            S = S + 1
        
        # compare reward
        average_reward = utility.evaluate(env, 100, 1000, TBS, env, r, S)[0]
        if flag:
            flag = False
            max_reward = average_reward
            opt_r = r
            opt_S = S
        elif max_reward < average_reward:
            max_reward = average_reward
            opt_r = r
            opt_S = S
    
    return opt_r, opt_S