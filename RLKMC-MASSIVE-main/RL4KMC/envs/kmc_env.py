from RL4KMC.envs.kmc import KMCEnv
import random
import torch
from gym import spaces
import numpy as np

class KMCEnvWrap(object):
    def __init__(self, args):
        
        self.env = KMCEnv(args)
        
        
        self.num_agents = args.lattice_v_nums
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(8,))

    def reset(self):
        return self.env.reset()
    
    def step(self, action, episode):
        obs, share_obs, positions, reward, done, info = self.env.step(action, episode)
        return (obs, share_obs, positions, reward, done, info)
    
    def seed(self, seed=None):
        if seed is None:
            random.seed(1)
        else:
            random.seed(seed)
    
    def get_system_stats(self):
        return self.env.get_system_stats()
