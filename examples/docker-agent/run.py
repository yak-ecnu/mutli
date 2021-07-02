"""Implementation of a simple deterministic agent using Docker."""
import numpy as np
import torch
from torch import nn
#from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions import Normal

import pommerman
from pommerman import agents
from pommerman.runner import DockerAgentRunner
import time

import math
import random
import datetime

import random

from IPython import display

# Our own files
from .convertInputMapToTrainingLayers import*
#from convertInputMapToTrainingLayers import *


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic_con = nn.Sequential(
            nn.Conv2d(in_channels=7,
                      out_channels=64, 
                      kernel_size=3, 
                      padding=0),
            nn.Conv2d(in_channels=64,
                      out_channels=64, 
                      kernel_size=3, 
                      padding=0),
            nn.Conv2d(in_channels=64,
                      out_channels=64, 
                      kernel_size=3, 
                      padding=0),
            nn.ReLU()
        )
        self.critic_linear = nn.Sequential(
            nn.Linear(3*3*64, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor_con = nn.Sequential(
            nn.Conv2d(in_channels=7,
                      out_channels=64, 
                      kernel_size=3, 
                      padding=0),
            nn.Conv2d(in_channels=64,
                      out_channels=64, 
                      kernel_size=3, 
                      padding=0),
            nn.Conv2d(in_channels=64,
                      out_channels=64, 
                      kernel_size=3, 
                      padding=0),
            nn.ReLU()
        )
        self.actor_linear = nn.Sequential(
            nn.Linear(3*3*64, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs)
        )
        
        self.log_std = nn.Parameter(torch.ones(num_outputs) * std)
        
        self.apply(init_weights)
        
    def forward(self, x):
        value = self.critic_con(x)
        value = self.critic_linear(value.view(-1, 3*3*64))
        
        mu    = self.actor_con(x)
        mu    = self.actor_linear(mu.view(-1, 3*3*64))
        
        std1  = self.log_std.exp()
        std   = std1.expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value

num_inputs       = 324
num_outputs      = 6
hidden_size      = 1024
lr               = 1e-6
lr_RND           = 1e-3
mini_batch_size  = 5
ppo_epochs       = 4
max_frames       = 1500000
frame_idx        = 0
game_idx         = 0
device           = "cpu" # Hard-coded since we have a GPU, but does not want to use
clip_param       = 0.2



class MyAgent(DockerAgentRunner):
    '''An example Docker agent class'''

    def __init__(self):
        self.model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
        import os
        if os.path.exists("./model.pth"):
            self.model = torch.load("model.pth", map_location='cpu')
        self._agent = agents.SimpleAgent()

    def act(self, observation, action_space):
        obs = self.translate_obs(observation)
        obs = torch.from_numpy(obs).float().to(self.device)
        self.obs_fps.append(obs)
        obs = torch.cat(self.obs_fps[-4:])
        sample = random.random()
        if sample > 0.1:
            re_action = self.model(obs).argmax().item()
            return re_action
        else:
            return self._agent.act(observation, action_space)

    """ def act(self, observation, action_space):
        return self.agent(torch.from_numpy(observation).float().to('cpu'), action_space) """



""" class MyAgent(DockerAgentRunner):
    '''An example Docker agent class'''

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(num_inputs, num_outputs, hidden_size).to(self.device)
        self.obs_width = 11
        import os
        if os.path.exists("../newAI02_from_oldAI04.pth"):
            self.model=torch.load("newAI02_from_oldAI04.pth")
        self._agent = agents.SimpleAgent()
        self.obs_fps = [torch.zeros(366),torch.zeros(366),torch.zeros(366)]

    def init_agent(self, id, game_type):
        return self._agent.init_agent(id, game_type)

    def translate_obs(self, o):
        obs_width = self.obs_width

        board = o['board'].copy()
        agents = np.column_stack(np.where(board > 10))

        for i, agent in enumerate(agents):
            agent_id = board[agent[0], agent[1]]
            if agent_id not in o['alive']:  # < this fixes a bug >
                board[agent[0], agent[1]] = 0
            else:
                board[agent[0], agent[1]] = 11

        obs_radius = obs_width // 2
        pos = np.asarray(o['position'])

        # board
        board_pad = np.pad(board, (obs_radius, obs_radius), 'constant', constant_values=1)
        self.board_cent = board_cent = board_pad[pos[0]:pos[0] + 2 * obs_radius + 1, pos[1]:pos[1] + 2 * obs_radius + 1]

        # bomb blast strength
        bbs = o['bomb_blast_strength']
        bbs_pad = np.pad(bbs, (obs_radius, obs_radius), 'constant', constant_values=0)
        self.bbs_cent = bbs_cent = bbs_pad[pos[0]:pos[0] + 2 * obs_radius + 1, pos[1]:pos[1] + 2 * obs_radius + 1]

        # bomb life
        bl = o['bomb_life']
        bl_pad = np.pad(bl, (obs_radius, obs_radius), 'constant', constant_values=0)
        self.bl_cent = bl_cent = bl_pad[pos[0]:pos[0] + 2 * obs_radius + 1, pos[1]:pos[1] + 2 * obs_radius + 1]

        return np.concatenate((
            board_cent, bbs_cent, bl_cent,
            o['blast_strength'], o['can_kick'], o['ammo']), axis=None)

    def act(self, observation, action_space):
        obs = self.translate_obs(observation)
        obs = torch.from_numpy(obs).float().to(self.device)
        self.obs_fps.append(obs)
        obs = torch.cat(self.obs_fps[-4:])
        sample = random.random()
        if sample > 0.1:
            re_action = self.model(obs).argmax().item()
            return re_action
        else:
            return self._agent.act(observation, action_space)

    def episode_end(self, reward):
        return self._agent.episode_end(reward)

    def shutdown(self):
        return self._agent.shutdown() """


def main():
    '''Inits and runs a Docker Agent'''
    agent = MyAgent()
    agent.run()


if __name__ == "__main__":
    main()
