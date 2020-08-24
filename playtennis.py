from unityagents import UnityEnvironment
import numpy as np
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import ddpg_agent

import time
from datetime import timedelta


env = UnityEnvironment(file_name='Tennis_Linux/Tennis.x86_64')
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=False)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])
random_seed = 0
clip_constant = 1

update_count = 10

action_tensor_size = num_agents * action_size
state_tensor_size = num_agents * state_size

def ddpg(agent_instance, print_every=100):

    agent_obj = [ddpg_agent.Agent(**agent_instance) for _ in range(num_agents)]  # generate list of agents for each agent instance according to number of agents
    count = 0

    for agent in agent_obj:
        count += 1
        agent_obj[count-1].actor_local.load_state_dict(torch.load('checkpoint_actor_%s.pth' % count))
        agent_obj[count-1].critic_local.load_state_dict(torch.load('checkpoint_critic_%s.pth' % count))

    while True:
        env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
        states = env_info.vector_observations                  # get the current
        # state (for each agent)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        for agent in agent_obj:
            agent.reset()

        learn_count = 0

        while True:
            learn_count += 1
            actions = np.array([agent_obj[i].act(states[i], add_noise=False) for i in range(num_agents)])                        # select an action (for each agent)

            #env_info = env.step(actions)[brain_name]           # send all actions to tne environment

            env_info = env.step(np.reshape(np.concatenate((actions[0], actions[1]), axis = 0), (1, action_tensor_size)))[brain_name]           # send all actions to the environment

            next_states = env_info.vector_observations        # get next state (for each agent)
            dones = env_info.local_done                        # see if episode finished

            states = next_states                               # roll over states to next time step


            if np.any(dones):                                  # exit loop if episode finished
                break





agent_instance ={"state_size": state_size, "action_size": action_size, "random_seed": random_seed, "clip_constant": clip_constant}

ddpg(agent_instance)

