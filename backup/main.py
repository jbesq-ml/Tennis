from unityagents import UnityEnvironment
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import time
from datetime import timedelta

import ddpg_agent

env = UnityEnvironment(file_name='../Tennis_Linux/Tennis.x86_64')
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

update_count = 1

action_tensor_size = num_agents * action_size
state_tensor_size = num_agents * state_size



def ddpg(agent_instance, print_every=500):
    scores_deque = deque(maxlen=print_every)
    scores_deque.append(0)

    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    n_episodes = 0
    start_time = time.time()                                    # start time for printing
    history = []


    agent_obj = [ddpg_agent.Agent(**agent_instance) for _ in range(num_agents)]  # generate list of agents for each agent instance according to number of agents
    while np.mean(scores_deque) < 0.8:
        n_episodes += 1
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment
        #states = np.reshape(env_info.vector_observations[0], (1, state_size))                 # get the current
        # state (for each agent)
        states = np.reshape(np.concatenate((env_info.vector_observations[0], env_info.vector_observations[1]), axis = 0), (1, state_tensor_size))                 # get the current

        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        for agent in agent_obj:
            agent.reset()

        learn_count = 0

        while True:
            learn_count += 1
            actions = np.array([agent_obj[i].act(states) for i in range(num_agents)]) # select an action (for each agent)

            env_info = env.step(np.reshape(np.concatenate((actions[0], actions[1]), axis = 0), (1, action_tensor_size)))[brain_name]           # send all actions to the environment

            #env_info = env.step(actions)[brain_name]           # send all actions to the environment
            next_states = np.reshape(env_info.vector_observations, (1, state_tensor_size))        # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished

            #(agent_obj[i].step(states[i], actions[i], rewards[i], next_states[i], dones[i], learn_count, update_count, i) for i in range(num_agents))
            (agent_obj[i].step(states, actions[i], rewards[i], next_states, dones[i], learn_count, update_count, i) for i in range(num_agents))

            states = next_states                               # roll over states to next time step

            scores += rewards                         # update the score (for each agent)

            #print("\rEpisode: {}\tAvg. Agent Score: {:.2f}\t{} Episode Rolling Average Score: {:.2f}\tStep: {}".format(n_episodes, np.mean(scores), print_every, np.mean(scores_deque), learn_count), end =" ")

            if np.any(dones):                                  # exit loop if episode finished
                break
        print('\rEpisode {}\tActor 1s score was {: .2f}\t Actor 2s score was {: .2f}'.format(n_episodes, scores[0], scores[1]), end =" ")
        scores_deque.append(np.max(scores))

        history.append(np.max(scores))

        delta_time = str(timedelta(seconds=time.time() - start_time))   #  elapsed time

        for agent in agent_obj:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_%s.pth' % agent)
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_%s.pth' % agent)

        if n_episodes % (print_every) == 0:
            print("\n\rEpisode: {}\tHighest Score {: .2f}\tAverage score for laste {} episodes was {: .2f}\tTime: {:.9}".format(n_episodes, np.max(scores), print_every, np.mean(scores_deque), delta_time))

    print("\n\rLast Episode: {}\tHighest Score {: .2f}\tAverage score for laste {} episodes was {: .2f}\tTime: {:.9}".format(
        n_episodes, np.max(scores), print_every, np.mean(scores_deque), delta_time))
    return history

agent_instance ={"state_size": state_tensor_size, "action_size": action_size, "random_seed": random_seed, "clip_constant": clip_constant}

results = ddpg(agent_instance)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(results)+1), results)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()