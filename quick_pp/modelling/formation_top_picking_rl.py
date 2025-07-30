# Formation Top Picking with Reinforcement Learning
import gymnasium as gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import warnings
from sklearn.preprocessing import minmax_scale
warnings.filterwarnings('ignore')


# ============================================================================
# FORMATION TOP PICKING ENVIRONMENT
# ============================================================================
class FormationTopPickingEnv:
    """Custom environment for formation top picking using well log data."""

    def __init__(self, well_data: pd.DataFrame, max_depth_window: int = 50,
                 reward_weights: Dict[str, float] = None):
        """
        Initialize the formation top picking environment.

        Args:
            well_data: DataFrame with columns ['DEPTH', 'GR', 'NPHI', 'RHOB', 'RT', 'VSHALE', 'PHIT']
            max_depth_window: Number of depth points to consider in each step
            reward_weights: Weights for different reward components
        """
        self.well_data = well_data.copy()
        self.max_depth_window = max_depth_window
        self.current_depth_idx = 0
        self.picked_tops = []
        self.episode_reward = 0
        self.step_count = 0
        self.max_steps = len(well_data) - max_depth_window

        # Normalize log data
        self._normalize_logs(skip=False)

        # Reward weights
        self.reward_weights = reward_weights or {
            'lithology_change': 2.0,
            'depth_spacing': 3.0,
            'log_response': 1.5,
            'geological_consistency': 1.5,
            'coverage_bonus': 1.0,
            'dense_penalty': 1.0
        }

        # Action space: 0 = no pick, 1 = pick formation top
        self.action_space = gym.spaces.Discrete(2)

        # Observation space: normalized log values + context
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_depth_window * 6 + 6,),  # 6 logs * window + 4 context features
            dtype=np.float32
        )

    def _normalize_logs(self, skip=False):
        """Normalize log data for better learning."""
        log_columns = ['GR', 'NPHI', 'RHOB', 'RT', 'VSHALE', 'PHIT']

        for col in log_columns:
            if col in self.well_data.columns and not skip:
                # # Robust normalization using median and IQR
                # median_val = self.well_data[col].median()
                # q75, q25 = self.well_data[col].quantile([0.75, 0.25])
                # iqr = q75 - q25

                # if iqr > 0:
                #     self.well_data[f'{col}_NORM'] = (self.well_data[col] - median_val) / iqr
                # else:
                #     self.well_data[f'{col}_NORM'] = 0
                self.well_data[f'{col}_NORM'] = minmax_scale(self.well_data[col])
            elif col in self.well_data.columns and skip:
                self.well_data[f'{col}_NORM'] = self.well_data[col]
            else:
                # Fill with zeros if column doesn't exist
                self.well_data[f'{col}_NORM'] = 0

    def reset(self, seed=None):
        """Reset the environment for a new episode."""
        if seed is not None:
            np.random.seed(seed)

        self.current_depth_idx = self.max_depth_window // 2
        self.picked_tops = []
        self.episode_reward = 0
        self.step_count = 0

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Get current observation including log data and context."""
        # Get log data window
        start_idx = max(0, self.current_depth_idx - self.max_depth_window // 2)
        end_idx = min(len(self.well_data), self.current_depth_idx + self.max_depth_window // 2)

        window_data = self.well_data.iloc[start_idx:end_idx]

        # Extract normalized log values
        log_features = []
        for col in ['GR_NORM', 'NPHI_NORM', 'RHOB_NORM', 'RT_NORM', 'VSHALE_NORM', 'PHIT_NORM']:
            log_features.extend(window_data[col].fillna(0).values)

        # Pad if window is smaller than max_depth_window
        while len(log_features) < self.max_depth_window * 6:
            log_features.extend([0] * 6)

        # Add context features
        context_features = [
            self.current_depth_idx / len(self.well_data),  # Depth progress
            len(self.picked_tops) / 50,  # Number of picks (normalized, increased max)
            self.step_count / self.max_steps,  # Episode progress
            self.episode_reward / 200,  # Current reward (normalized)
            self._get_lithology_change_score(),  # Current lithology change intensity
            self._get_log_variance_score()  # Log variance in current window
        ]

        return np.array(log_features + context_features, dtype=np.float32)

    def _get_lithology_change_score(self) -> float:
        """Calculate lithology change intensity around current depth."""
        if self.current_depth_idx < self.max_depth_window // 2 or (
                self.current_depth_idx >= len(self.well_data) - self.max_depth_window // 2):
            return 0.0

        # Look at a window around current depth
        window_start = max(0, self.current_depth_idx - self.max_depth_window)
        window_end = min(len(self.well_data), self.current_depth_idx + self.max_depth_window)
        window_data = self.well_data.iloc[window_start:window_end]

        # Calculate variance in VSHALE (lithology indicator)
        vshale_var = window_data['VSHALE_NORM'].var()
        return min(1.0, vshale_var)  # Normalize to [0, 1]

    def _get_log_variance_score(self) -> float:
        """Calculate log variance in current window."""
        start_idx = max(0, self.current_depth_idx - self.max_depth_window // 2)
        end_idx = min(len(self.well_data), self.current_depth_idx + self.max_depth_window // 2)
        window_data = self.well_data.iloc[start_idx:end_idx]

        # Calculate average variance across all logs
        log_vars = []
        for col in ['GR_NORM', 'NPHI_NORM', 'RHOB_NORM', 'RT_NORM']:
            if col in window_data.columns:
                log_vars.append(window_data[col].var())

        avg_var = np.mean(log_vars) if log_vars else 0.0
        return min(1.0, avg_var)  # Normalize to [0, 1]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take an action and return next state, reward, done, truncated, info."""
        reward = 0
        done = False
        truncated = False

        # Action 0: no pick, Action 1: pick formation top
        if action == 1:
            reward = self._calculate_pick_reward()
            self.picked_tops.append(self.current_depth_idx)

        # Move to next depth point
        self.current_depth_idx += 1
        self.step_count += 1

        if self.current_depth_idx >= len(self.well_data) - self.max_depth_window // 2:
            done = True
            coverage_reward = self._calculate_coverage_bonus()
            reward += coverage_reward

        self.episode_reward += reward

        return self._get_observation(), reward, done, truncated, {
            'picked_tops': self.picked_tops.copy(),
            'total_reward': self.episode_reward
        }

    def _calculate_pick_reward(self) -> float:
        """Calculate reward for picking a formation top."""
        if self.current_depth_idx >= len(self.well_data):
            return 0

        current_data = self.well_data.iloc[self.current_depth_idx]
        reward = 0

        # 1. Lithology change reward
        if self.current_depth_idx > 0:
            prev_data = self.well_data.iloc[self.current_depth_idx - 1]
            lithology_change = abs(current_data['VSHALE_NORM'] - prev_data['VSHALE_NORM'])
            # Use a sigmoid function to reward significant changes more
            lithology_reward = 2.0 / (1 + np.exp(-5 * (lithology_change - 0.3)))
            reward += self.reward_weights['lithology_change'] * lithology_reward

        # 2. Depth spacing reward (penalize too close picks)
        if len(self.picked_tops) > 0:
            min_distance = min(abs(self.current_depth_idx - top) for top in self.picked_tops)
            if min_distance < self.max_depth_window // 2:  # Too close
                reward -= self.reward_weights['depth_spacing'] * (10 - min_distance)
            else:
                reward += self.reward_weights['depth_spacing'] * min(1, min_distance / 50)

        # 3. Log response reward (reward for significant changes in multiple logs)
        log_changes = []
        for col in ['GR_NORM', 'NPHI_NORM', 'RHOB_NORM', 'RT_NORM']:
            if self.current_depth_idx > 0:
                prev_val = self.well_data.iloc[self.current_depth_idx - 1][col]
                curr_val = current_data[col]
                log_changes.append(abs(curr_val - prev_val))

        avg_log_change = np.mean(log_changes) if log_changes else 0
        reward += self.reward_weights['log_response'] * avg_log_change

        # 4. Geological consistency reward
        if len(self.picked_tops) > 1:
            # Check if picks follow a reasonable pattern
            depths = sorted(self.picked_tops)
            intervals = [depths[i+1] - depths[i] for i in range(len(depths)-1)]
            if intervals:
                mean_interval = np.mean(intervals)
                if mean_interval > 10:  # Reasonable spacing
                    reward += self.reward_weights['geological_consistency'] * 0.5

        if len(self.picked_tops) > 50:  # Aim for less than 100 picks
            reward -= self.reward_weights['dense_penalty']

        return reward

    def _calculate_coverage_bonus(self) -> float:
        """Calculate bonus reward for good coverage of the well."""
        if len(self.picked_tops) == 0:
            return 0

        # Calculate how well the picks cover the well
        well_length = len(self.well_data)
        coverage_score = len(self.picked_tops) / (well_length / self.max_depth_window)

        # Bonus for reasonable coverage (not too sparse, not too dense)
        if 0.5 <= coverage_score <= 2.0:
            return self.reward_weights['coverage_bonus'] * coverage_score
        else:
            return 0


class HeuristicFormationAgent:
    """Heuristic agent that uses geological rules for formation top picking."""

    def __init__(self):
        self.name = "Heuristic Formation Agent"

    def choose_action(self, observation):
        # Extract log features from observation
        log_features = observation[:-6]  # Remove context features
        num_logs = 6
        window_size = len(log_features) // num_logs

        # Get current depth point (middle of window)
        current_idx = window_size // 2
        if current_idx >= window_size:
            return 0

        # Extract current log values
        gr_idx = current_idx
        rt_idx = current_idx + 3 * window_size
        vshale_idx = current_idx + 4 * window_size
        phit_idx = current_idx + 5 * window_size

        gr_val = log_features[gr_idx]
        rt_val = log_features[rt_idx]
        vshale_val = log_features[vshale_idx]
        phit_val = log_features[phit_idx]

        # Heuristic rules for formation top picking
        pick_signals = 0

        # Rule 1: Significant shale volume change
        if current_idx > 0:
            prev_vshale = log_features[current_idx - 1 + 4 * window_size]
            if abs(vshale_val - prev_vshale) > 0.3:
                pick_signals += 1

        # Rule 2: Gamma ray spike (shale indicator)
        if gr_val > 0.5:
            pick_signals += 1

        # Rule 3: Porosity change
        if current_idx > 0:
            prev_phit = log_features[current_idx - 1 + 5 * window_size]
            if abs(phit_val - prev_phit) > 0.2:
                pick_signals += 1

        # Rule 4: Resistivity change
        if current_idx > 0:
            prev_rt = log_features[current_idx - 1 + 3 * window_size]
            if abs(rt_val - prev_rt) > 0.4:
                pick_signals += 1

        # Rule 5: Check lithology change score from context
        lithology_change_score = observation[-2]  # Second to last context feature
        if lithology_change_score > 0.3:  # Use the lithology change score
            pick_signals += 1

        # Decide to pick if enough signals (reduced requirement)
        return 1 if pick_signals >= 1 else 0  # Reduced from 2 to 1


class QLearningFormationAgent:
    """Q-Learning agent for formation top picking."""

    def __init__(self, state_size=100, action_size=2, learning_rate=0.1, epsilon=0.1, gamma=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.q_table = {}
        self.name = "Q-Learning Formation Agent"

    def _get_state_key(self, observation):
        # Discretize continuous state for Q-table
        # Use first 100 features and discretize them
        features = observation[:100]
        # Round to 1 decimal place for discretization
        state = tuple(round(f, 1) for f in features)
        return state

    def choose_action(self, observation):
        state = self._get_state_key(observation)

        # Initialize Q-values for new state
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)

        # Epsilon-greedy strategy
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.q_table[state])

    def learn(self, observation, action, reward, next_observation, done):
        state = self._get_state_key(observation)
        next_state = self._get_state_key(next_observation)

        # Initialize Q-values if needed
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_size)

        # Q-learning update
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.gamma * max_next_q * (1 - done) - current_q)
        self.q_table[state][action] = new_q


class DQNFormationAgent:
    """Deep Q-Network agent for formation top picking."""

    def __init__(self, state_size=100, action_size=2, learning_rate=0.001, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=2000)
        self.name = "DQN Formation Agent"

        # Neural Network for formation top picking
        self.model = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def choose_action(self, observation):
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_size - 1)

        state = torch.FloatTensor(observation[:self.state_size]).unsqueeze(0)
        q_values = self.model(state)
        return np.argmax(q_values.detach().numpy())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0][:self.state_size] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3][:self.state_size] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.model(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class PolicyGradientFormationAgent:
    """Policy Gradient agent for formation top picking."""

    def __init__(self, state_size=100, action_size=2, learning_rate=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = "Policy Gradient Formation Agent"

        # Policy network for formation top picking
        self.model = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_states = []

    def choose_action(self, observation):
        state = torch.FloatTensor(observation[:self.state_size]).unsqueeze(0)
        action_probs = self.model(state)
        action = torch.multinomial(action_probs, 1).item()
        return action

    def store_transition(self, state, action, reward):
        self.episode_states.append(state[:self.state_size])
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)

    def update_policy(self):
        if len(self.episode_rewards) == 0:
            return

        # Calculate discounted rewards
        discounted_rewards = []
        R = 0
        for r in reversed(self.episode_rewards):
            R = r + 0.99 * R
            discounted_rewards.insert(0, R)

        # Normalize rewards
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # Calculate loss
        states = torch.FloatTensor(self.episode_states)
        actions = torch.LongTensor(self.episode_actions)
        action_probs = self.model(states)
        selected_action_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze()

        loss = -(torch.log(selected_action_probs) * discounted_rewards).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear episode data
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_states = []


# ============================================================================
# TRAINING AND EVALUATION FUNCTIONS
# ============================================================================
def train_formation_agent(agent, env, episodes=100):
    """Train an agent for formation top picking."""
    print(f"Training {agent.name} for {episodes} episodes...")

    episode_rewards = []
    episode_picks = []

    for episode in range(episodes):
        observation, info = env.reset()
        total_reward = 0
        picks = []

        while True:
            # Choose action
            action = agent.choose_action(observation)

            # Take action
            next_observation, reward, done, truncated, info = env.step(action)

            # Store experience for learning agents
            if hasattr(agent, 'learn'):
                agent.learn(observation, action, reward, next_observation, done)
            elif hasattr(agent, 'remember'):
                agent.remember(observation, action, reward, next_observation, done)
            elif hasattr(agent, 'store_transition'):
                agent.store_transition(observation, action, reward)

            if action == 1:
                picks.append(env.current_depth_idx - 1)

            total_reward += reward
            observation = next_observation

            if done or truncated:
                break

        # Update policy for policy gradient agent
        if hasattr(agent, 'update_policy'):
            agent.update_policy()

        # Replay for DQN agent
        if hasattr(agent, 'replay'):
            agent.replay()

        episode_rewards.append(total_reward)
        episode_picks.append(len(picks))

        if episode % 10 == 0:
            avg_reward = np.nanmean(episode_rewards)
            avg_picks = np.nanmean(episode_picks)
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Picks: {avg_picks:.1f}")

    return episode_rewards, episode_picks


def evaluate_formation_agent(agent, env, num_episodes=10):
    """Evaluate a trained agent."""
    print(f"Evaluating {agent.name}...")

    all_rewards = []
    all_picks = []

    for episode in range(num_episodes):
        observation, info = env.reset()
        total_reward = 0
        picks = []

        while True:
            action = agent.choose_action(observation)
            next_observation, reward, done, truncated, info = env.step(action)

            if action == 1:
                picks.append(env.current_depth_idx - 1)

            total_reward += reward
            observation = next_observation

            if done or truncated:
                break

        all_rewards.append(total_reward)
        all_picks.append(picks)

    avg_reward = np.mean(all_rewards)
    avg_num_picks = np.mean([len(picks) for picks in all_picks])

    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Number of Picks: {avg_num_picks:.1f}")

    return all_rewards, all_picks


def plot_formation_picks(well_data, picks, agent_name):
    """Plot well log data with picked formation tops."""
    _, axes = plt.subplots(1, 4, figsize=(15, 8))

    # Plot each log
    logs = ['GR', 'NPHI', 'RHOB', 'RT']
    colors = ['green', 'blue', 'red', 'orange']

    for i, (log, color) in enumerate(zip(logs, colors)):
        well_data[log] = np.nan if log not in well_data.columns else well_data[log]
        axes[i].plot(well_data[log], well_data['DEPTH'], color=color, linewidth=1)
        axes[i].set_title(f'{log} Log')
        axes[i].set_ylabel('Depth')
        axes[i].grid(True, alpha=0.3)
        axes[i].invert_yaxis()
        if i > 0:  # Share y-axis with first subplot
            axes[i].sharey(axes[0])

        # Mark picked formation tops
        for pick_idx in picks:
            if pick_idx < len(well_data):
                pick_depth = well_data.iloc[pick_idx]['DEPTH']
                axes[i].axhline(y=pick_depth, color='red', linestyle='--', alpha=0.7)

    plt.suptitle(f'Formation Tops Picked by {agent_name}')
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.getcwd())
    import quick_pp.las_handler as las

    # Create synthetic well data
    file_name = r'C:\projects\quick_pp\notebooks\data\01_raw\36_7-3.las'
    with open(file_name, 'rb') as f:
        df, _ = las.read_las_file_welly(f)
        df['RT'] = np.log10(df['RT'])
    well_data = df  # [['DEPTH', 'GR']]

    # Create environment
    env = FormationTopPickingEnv(well_data, max_depth_window=50)

    # Choose which agent to use
    agent_type = "qlearning"  # Options: "heuristic", "qlearning", "dqn", "policy_gradient"

    if agent_type == "heuristic":
        agent = HeuristicFormationAgent()
        print("Using Heuristic Formation Agent")
    elif agent_type == "qlearning":
        agent = QLearningFormationAgent()
        print("Using Q-Learning Formation Agent")
    elif agent_type == "dqn":
        agent = DQNFormationAgent()
        print("Using DQN Formation Agent")
    elif agent_type == "policy_gradient":
        agent = PolicyGradientFormationAgent()
        print("Using Policy Gradient Formation Agent")
    else:
        print("Invalid agent type, using heuristic")
        agent = HeuristicFormationAgent()

    # Train the agent
    if agent_type in ["qlearning", "dqn", "policy_gradient"]:
        rewards, picks = train_formation_agent(agent, env, episodes=50)
        print(f"Training completed. Final average reward: {np.mean(rewards[-10:]):.2f}")

    # Evaluate the agent
    eval_rewards, eval_picks = evaluate_formation_agent(agent, env, num_episodes=5)

    # Plot results
    if eval_picks:
        plot_formation_picks(well_data, eval_picks[0], agent.name)

    print("Formation top picking RL demonstration completed!")
