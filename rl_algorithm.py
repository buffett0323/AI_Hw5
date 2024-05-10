import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils
# Self-added
import torch.optim as optim
from collections import deque

class PacmanActionCNN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PacmanActionCNN, self).__init__()
        "*** YOUR CODE HERE ***"
        # utils.raiseNotDefined()
        # define the network
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(3136, 512), #linear_input_size
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        "*** YOUR CODE HERE ***"
        # utils.raiseNotDefined()
        # forward pass
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layers
        x = self.fc_layers(x)
        
        return x

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.states = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, *action_dim), dtype=np.int64)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.terminated = np.zeros((max_size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.max_size = max_size

    def update(self, state, action, reward, next_state, terminated):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.terminated[self.ptr] = terminated
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.states[ind]),
            torch.FloatTensor(self.actions[ind]),
            torch.FloatTensor(self.rewards[ind]),
            torch.FloatTensor(self.next_states[ind]),
            torch.FloatTensor(self.terminated[ind]), 
        )

class DQN:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-4,
        epsilon=0.9,
        epsilon_min=0.05,
        gamma=0.99,
        batch_size=64,
        warmup_steps=5000,
        buffer_size=int(1e5),
        target_update_interval=10000,
    ):
        """
        DQN agent has four methods.

        - __init__() is the standard initializer.
        - act(), which receives a state as an np.ndarray and outputs actions following the epsilon-greedy policy.
        - process(), which processes a single transition and defines the agent's actions at each step.
        - learn(), which samples a mini-batch from the replay buffer to train the Q-network.
        """
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_interval = target_update_interval

        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

        "*** YOUR CODE HERE ***"
        # utils.raiseNotDefined()
        # define the network, target network, optimizer, buffer, total_steps, epsilon_decay
        self.network = PacmanActionCNN(state_dim, action_dim).to(self.device)
        self.target_network = PacmanActionCNN(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()
        
        self.network.to(self.device)
        self.target_network.to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = ReplayBuffer((4, 84, 84), (1,), buffer_size)  # Assuming state is an 84x84 grayscale image
        self.total_steps = 0

        # Logs for plotting
        self.losses = []
        self.rewards = []
    
    
    @torch.no_grad()
    def act(self, x, training=True):
        self.network.train(training)
        if training and ((np.random.rand() < self.epsilon) or (self.total_steps < self.warmup_steps)):
            "*** YOUR CODE HERE ***"
            # utils.raiseNotDefined()
            # Random action
            action = np.random.randint(self.action_dim)
        else:
            "*** YOUR CODE HERE ***"
            # utils.raiseNotDefined()
            # output actions by following epsilon-greedy policy
            state_tensor = torch.tensor(x, dtype=torch.float32).to(self.device).unsqueeze(0)
            q_values = self.network(state_tensor)
            action = torch.argmax(q_values, dim=1).item()

        return action
    
    
    
    def learn(self):
        "*** YOUR CODE HERE ***"
        # utils.raiseNotDefined()
        # samples a mini-batch from replay buffer and train q-network
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action = action.to(self.device).long().squeeze()
        reward = reward.to(self.device).squeeze()
        done = done.to(self.device).squeeze()
        
        # Forward pass through the current network
        current_q = self.network(state).gather(1, action.unsqueeze(1)).squeeze(1)

        
        # Forward pass through the target network
        next_q_values = self.target_network(next_state).detach()
        max_next_q = next_q_values.max(1)[0]  # A vector of max values
        expected_q = reward + self.gamma * max_next_q * (1 - done)
        expected_q = expected_q.squeeze() # Make sure expected_q is a 1D vector

        # Compute loss
        loss = nn.functional.mse_loss(current_q, expected_q)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())  # Append the loss after each batch update

        return {'loss': loss.item()} # return the information you need for logging
    
    
    
    def process(self, transition):
        "*** YOUR CODE HERE ***"
        # utils.raiseNotDefined()
        self.buffer.update(*transition)
        self.total_steps += 1

        info = {}
        if self.total_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        if self.total_steps > self.warmup_steps and self.buffer.size >= self.batch_size:
            info = self.learn()

        if self.epsilon > 0.05: #self.epsilon_min
            self.epsilon *= 0.995 #self.epsilon_decay

        # Log reward
        self.rewards.append(transition[2])

        return info # return the information you need for logging