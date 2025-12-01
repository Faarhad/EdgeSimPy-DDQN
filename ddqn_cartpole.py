import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================
# 1. Q-network definition (PyTorch)
# ======================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)  # Q-values for each action


# ======================================
# 2. Simple replay buffer
# ======================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ======================================
# 3. Epsilon-greedy action selection
# ======================================
def select_action(q_online, state, epsilon, num_actions, device):
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)
    else:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)  # [1, state_dim]
        with torch.no_grad():
            q_values = q_online(state_t)
        return int(q_values.argmax().item())


# ======================================
# 4. DDQN training step
# ======================================
def train_step(q_online, q_target, optimizer, replay_buffer,
               batch_size, gamma, device):

    if len(replay_buffer) < batch_size:
        return  # not enough samples yet

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    states      = torch.FloatTensor(states).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    actions     = torch.LongTensor(actions).to(device)
    rewards     = torch.FloatTensor(rewards).to(device)
    dones       = torch.FloatTensor(dones).to(device)

    # --- current Q(s,a) ---
    q_values = q_online(states)                                # [B, num_actions]
    q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1) # [B]

    # --- DDQN target ---
    with torch.no_grad():
        # 1) online net chooses best action in next state
        q_next_online = q_online(next_states)                   # [B, num_actions]
        best_actions = q_next_online.argmax(dim=1)              # [B]

        # 2) target net evaluates that action
        q_next_target = q_target(next_states)                   # [B, num_actions]
        q_next_best = q_next_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)

        targets = rewards + gamma * (1.0 - dones) * q_next_best

    loss = nn.MSELoss()(q_sa, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# ======================================
# 5. Main training loop (CartPole)
# ======================================
def main():
    env = gym.make("CartPole-v1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    q_online = QNetwork(state_dim, num_actions).to(device)
    q_target = QNetwork(state_dim, num_actions).to(device)
    q_target.load_state_dict(q_online.state_dict())

    optimizer = torch.optim.Adam(q_online.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(capacity=50_000)

    num_episodes = 300
    batch_size = 64
    gamma = 0.99

    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.995
    epsilon = epsilon_start

    target_update_interval = 10  # episodes

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = select_action(q_online, state, epsilon, num_actions, device)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.add(state, action, reward, next_state, float(done))

            state = next_state
            total_reward += reward

            # one gradient step
            train_step(q_online, q_target, optimizer,
                       replay_buffer, batch_size, gamma, device)

        # epsilon decay
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # update target network
        if (episode + 1) % target_update_interval == 0:
            q_target.load_state_dict(q_online.state_dict())

        print(f"Episode {episode+1}/{num_episodes}, "
              f"Total reward: {total_reward:.1f}, "
              f"Epsilon: {epsilon:.3f}")

    env.close()
    print("Training finished.")

    # ✅ Call evaluation here, AFTER training:
    evaluate_trained_agent(q_online, num_episodes=5, render=True)


# ======================================
# Evaluation function (outside main)
# ======================================
def evaluate_trained_agent(q_online, num_episodes=10, render=True):
    # For evaluation we create a new env; enable rendering if requested
    if render:
        env = gym.make("CartPole-v1", render_mode="human")
    else:
        env = gym.make("CartPole-v1")

    device = next(q_online.parameters()).device
    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    def greedy_action(state):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = q_online(state_t)
        return int(q_values.argmax().item())

    rewards = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = greedy_action(state)  # epsilon = 0 → always best action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

        rewards.append(total_reward)
        print(f"[EVAL] Episode {ep+1}/{num_episodes} - Total reward: {total_reward}")

    avg_reward = sum(rewards) / len(rewards)
    print(f"\nAverage evaluation reward over {num_episodes} episodes: {avg_reward:.1f}")
    env.close()


if __name__ == "__main__":
    main()
