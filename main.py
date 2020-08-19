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
env_info = env.reset(train_mode=True)[brain_name]

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



def ddpg(agent_instance, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores_deque.append(0)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    n_episodes = 0
    start_time = time.time()                                    # start time for printing
    history = []

    agent_obj = [ddpg_agent.Agent(**agent_instance) for _ in range(num_agents)]  # generate list of agents for each agent instance according to number of agents

    while np.mean(scores_deque) < 30:
        n_episodes += 1
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment
        states = env_info.vector_observations                  # get the current
        # state (for each agent)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        for agent in agent_obj:
            agent.reset()

        learn_count = 0

        while True:
            learn_count += 1
            actions = np.array([agent_obj[i].act(states[i]) for i in range(num_agents)])                        # select an action (for each agent)

            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations        # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished

            for i in range(num_agents):
                agent_obj[i].step(states[i], actions[i], rewards[i], next_states[i], dones[i], learn_count, update_count)

            states = next_states                               # roll over states to next time step

            scores += rewards                         # update the score (for each agent)

            print("\rEpisode: {}\tAvg. Agent Score: {:.2f}\t{} Episode Rolling Average Score: {:.2f}\tStep: {}".format(n_episodes, np.mean(scores), print_every, np.mean(scores_deque), learn_count), end =" ")

            if np.any(dones):                                  # exit loop if episode finished
                break
        scores_deque.append(np.mean(scores))
        history.append(np.mean(scores))

        delta_time = str(timedelta(seconds=time.time() - start_time))   #  elapsed time

        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if n_episodes % (print_every / 100) == 0:
            print("\rEpisode: {}\tAvg. Agent Score: {:.2f}\t{} Episode Rolling Average Score: {:.2f}\tTime: {:.9}".format(n_episodes, np.mean(scores), print_every, np.mean(scores_deque), delta_time))

    print('\n\rFinal Episode: {}\t{} Episode Rolling Average Score: {:.2f}\tTime: {:.9}'.format(n_episodes, print_every, np.mean(scores_deque), delta_time))
    return history

agent_instance ={"state_size": state_size, "action_size": action_size, "random_seed": random_seed, "clip_constant": clip_constant}

results = ddpg(agent_instance)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(results)+1), results)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()