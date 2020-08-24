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


env = UnityEnvironment(file_name='Soccer_Linux/Soccer.x86_64')
# get the default brain
brain_name_0 = env.brain_names[0]
brain_name_1 = env.brain_names[1]

brain0 = env.brains[brain_name_0]
brain1 = env.brains[brain_name_1]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name_0]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size0 = brain0.vector_action_space_size
action_size1 = brain1.vector_action_space_size

print('Size of each action:', action_size0)
print('Size of each action:', action_size1)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])
random_seed = 0
clip_constant = 1

update_count = 10

action_tensor_size0 = int(num_agents * action_size0 / 2)
action_tensor_size1 = int(num_agents * action_size1 / 2)

state_tensor_size = num_agents * state_size

half_agents = int(num_agents / 2)

def ddpg(agent_instance0, agent_instance1, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores_deque.append(0)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    n_episodes = 0
    start_time = time.time()                                    # start time for printing
    history = []

    agent_obj0 = [ddpg_agent.Agent(**agent_instance0) for _ in range(half_agents)]  # generate list of agents for each agent instance according to number of agents
    agent_obj1 = [ddpg_agent.Agent(**agent_instance1) for _ in range(half_agents)]  # generate list of agents for each agent instance according to number of agents

    while np.mean(scores_deque) < 10:
        n_episodes += 1
        env_info0 = env.reset(train_mode=True)[brain_name_0]
        env_info1 = env.reset(train_mode=True)[brain_name_1]
        states0 = env_info0.vector_observations                  # get the current
        states1 = env_info1.vector_observations                  # get the current
        # state (for each agent)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        for agent in agent_obj0:
            agent.reset()
        for agent in agent_obj1:
            agent.reset()

        learn_count = 0

        while True:
            learn_count += 1
            actions0 = np.array([agent_obj0[i].act(states0[i], add_noise=False) for i in range(half_agents)])                        # select an action (for each agent)
            actions1 = np.array([agent_obj1[i].act(states1[i], add_noise=False) for i in range(half_agents)])                        # select an action (for each agent)

            #env_info = env.step(actions)[brain_name]           # send all actions to tne environment

            env_info0 = env.step(np.reshape(np.concatenate(([actions0[i] for i in range(half_agents)]), axis = 0), (1, action_tensor_size0)))[brain_name_0]           # send all actions to the environment
            env_info1 = env.step(np.reshape(np.concatenate(([actions1[i] for i in range(half_agents)]), axis = 0), (1, action_tensor_size1)))[brain_name_1]           # send all actions to the environment

            next_states0 = env_info0.vector_observations
            next_states1 = env_info1.vector_observations # get next state (for each agent)

            rewards0 = env_info0.rewards                         # get reward (for each agent)
            rewards1 = env_info1.rewards                         # get reward (for each agent)

            dones0 = env_info0.local_done                        # see if episode finished
            dones1 = env_info1.local_done                        # see if episode finished

            for i in range(half_agents):
                agent_obj0[i].step(states0[i], actions0[i], rewards0[i], next_states0[i], dones0[i], learn_count, update_count)

            for i in range(half_agents):
                agent_obj1[i].step(states1[i], actions1[i], rewards1[i], next_states1[i], dones1[i], learn_count, update_count)
            states0 = next_states0                               # roll over states to next time step
            states1 = next_states1                               # roll over states to next time step

            scores += rewards0                         # update the score (for each agent)
            scores += rewards1                         # update the score (for each agent)

            if np.any(dones0):                                  # exit loop if episode finished
                break
            if np.any(dones1):                                  # exit loop if episode finished
                break
        scores_deque.append(np.max(scores))

        history.append(np.max(scores))

        delta_time = str(timedelta(seconds=time.time() - start_time))  # elapsed time
        count = 0
        for agent in agent_obj:
            count += 1
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_%s.pth' % count)
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_%s.pth' % count)

        if n_episodes % (print_every) == 0:
            print("\rEpisode: {}\tHighest Score {: .2f}\tAverage score for last {} episodes was {: .2f}\tTime: {:.9}".format(n_episodes, np.max(scores_deque), print_every, np.mean(scores_deque), delta_time))

    print("\n\rLast Episode: {}\tHighest Score {: .2f}\tAverage score for last {} episodes was {: .2f}\tTime: {:.9}".format(n_episodes, np.max(scores_deque), print_every, np.mean(scores_deque), delta_time))
    return history

agent_instance0 ={"state_size": state_size, "action_size": action_size0, "random_seed": random_seed, "clip_constant": clip_constant}
agent_instance1 ={"state_size": state_size, "action_size": action_size1, "random_seed": random_seed, "clip_constant": clip_constant}

results = ddpg(agent_instance0, agent_instance1)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(results)+1), results)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
