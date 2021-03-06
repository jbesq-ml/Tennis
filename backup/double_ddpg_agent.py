import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    actor_local0 = None
    actor_target0 = None
    actor_optimizer0 = None
    critic_local0 = None
    critic_target0 = None
    critic_optimizer0 = None
    memory0 = None
    actor_local1 = None
    actor_target1 = None
    actor_optimizer1 = None
    critic_local1 = None
    critic_target1 = None
    critic_optimizer1 = None
    memory1 = None


    def __init__(self, state_size, action_size, random_seed, clip_constant):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # initialize First Class level Actor Network
        if Agent.actor_local0 is None:
            Agent.actor_local0 = Actor(state_size, action_size, random_seed).to(device)
        if Agent.actor_target0 is None:
            Agent.actor_target0 = Actor(state_size, action_size, random_seed).to(device)
        if Agent.actor_optimizer0 is None:
            Agent.actor_optimizer0 = optim.Adam(Agent.actor_local0.parameters(), lr=LR_ACTOR)
        self.actor_local0 = Agent.actor_local0
        self.actor_target0 = Agent.actor_target0
        self.actor_optimizer0 = Agent.actor_optimizer0

        # Initilise First Class level Critic Network
        if Agent.critic_local0 is None:
            Agent.critic_local0 = Critic(state_size, action_size, random_seed).to(device)
        if Agent.critic_target0 is None:
            Agent.critic_target0 = Critic(state_size, action_size, random_seed).to(device)
        if Agent.critic_optimizer0 is None:
            Agent.critic_optimizer0 = optim.Adam(Agent.critic_local0.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        self.critic_local0 = Agent.critic_local0
        self.critic_target0 = Agent.critic_target0
        self.critic_optimizer0 = Agent.critic_optimizer0

        # initialize Second Class level Actor Network
        if Agent.actor_local1 is None:
            Agent.actor_local1 = Actor(state_size, action_size, random_seed).to(device)
        if Agent.actor_target1 is None:
            Agent.actor_target1 = Actor(state_size, action_size, random_seed).to(device)
        if Agent.actor_optimizer1 is None:
            Agent.actor_optimizer1 = optim.Adam(Agent.actor_local1.parameters(), lr=LR_ACTOR)
        self.actor_local1 = Agent.actor_local1
        self.actor_target1 = Agent.actor_target1
        self.actor_optimizer1 = Agent.actor_optimizer1

        # Initialise Second Class level Critic Network
        if Agent.critic_local1 is None:
            Agent.critic_local1 = Critic(state_size, action_size, random_seed).to(device)
        if Agent.critic_target1 is None:
            Agent.critic_target1 = Critic(state_size, action_size, random_seed).to(device)
        if Agent.critic_optimizer1 is None:
            Agent.critic_optimizer1 = optim.Adam(Agent.critic_local1.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        self.critic_local1 = Agent.critic_local1
        self.critic_target1 = Agent.critic_target1
        self.critic_optimizer1 = Agent.critic_optimizer1

        # Noise process
        self.noise = NormalNoise(action_size, random_seed)

        # Replay memory - only intitialise once per class
        if Agent.memory0 is None:
            Agent.memory0 = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        if Agent.memory1 is None:
            Agent.memory1 = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        self.clip_constant = clip_constant


    def step(self, state, action, reward, next_state, done, learn_count, update_count, player):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward

        if player == 0:
            Agent.memory0.add(state, action, reward, next_state, done)

        elif player == 1:
            Agent.memory1.add(state, action, reward, next_state, done)


        # only learn every n_time_steps
        if learn_count % 20 != 0:
            return

        # Learn, if enough samples are available in memory
        if player == 0:
            if len(Agent.memory0) > BATCH_SIZE:
                for i in range(update_count):
                    experiences = Agent.memory0.sample()
                    self.learn(experiences, GAMMA, player)
        elif player == 1:
            if len(Agent.memory1) > BATCH_SIZE:
                for i in range(update_count):
                    experiences = Agent.memory1.sample()
                    self.learn(experiences, GAMMA, player)

    def act(self, state, player, add_noise=True):
        """Returns actions for given state as per current policy."""
        if player == 0:
            state = torch.from_numpy(state).float().to(device)
            self.actor_local0.eval()
            with torch.no_grad():
                action = self.actor_local0(state).cpu().data.numpy()
            self.actor_local0.train()
            if add_noise:
                action += self.noise.sample()
            return np.clip(action, -1, 1)
        if player == 1:
            state = torch.from_numpy(state).float().to(device)
            self.actor_local1.eval()
            with torch.no_grad():
                action = self.actor_local1(state).cpu().data.numpy()
            self.actor_local1.train()
            if add_noise:
                action += self.noise.sample()
            return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, player):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        if player == 0:
            states, actions, rewards, next_states, dones = experiences

            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models
            actions_next = self.actor_target0(next_states)
            Q_targets_next = self.critic_target0(next_states, actions_next)
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            # Compute critic loss
            Q_expected = self.critic_local0(states, actions)
            critic_loss = F.mse_loss(Q_expected, Q_targets)
            # Minimize the loss
            self.critic_optimizer0.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_local0.parameters(), 1)
            self.critic_optimizer0.step()

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = self.actor_local0(states)
            actor_loss = -self.critic_local0(states, actions_pred).mean()
            # Minimize the loss
            self.actor_optimizer0.zero_grad()
            actor_loss.backward()
            self.actor_optimizer0.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_local0, self.critic_target0, TAU)
            self.soft_update(self.actor_local0, self.actor_target0, TAU)
        elif player == 1:
            states, actions, rewards, next_states, dones = experiences

            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models
            actions_next = self.actor_target1(next_states)
            Q_targets_next = self.critic_target1(next_states, actions_next)
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            # Compute critic loss
            Q_expected = self.critic_local1(states, actions)
            critic_loss = F.mse_loss(Q_expected, Q_targets)
            # Minimize the loss
            self.critic_optimizer1.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_local0.parameters(), 1)
            self.critic_optimizer1.step()

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = self.actor_local1(states)
            actor_loss = -self.critic_local1(states, actions_pred).mean()
            # Minimize the loss
            self.actor_optimizer1.zero_grad()
            actor_loss.backward()
            self.actor_optimizer1.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_local1, self.critic_target0, TAU)
            self.soft_update(self.actor_local1, self.actor_target0, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class NormalNoise:
    """Normal Noise Generation."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        self.epsilon = 0.0001
        x = self.state
        dx = self.epsilon * np.random.normal(self.size)
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
