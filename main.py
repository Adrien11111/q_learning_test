import numpy as np
import tensorflow as tf
from collections import deque
import random

# Environment
grid_rows, grid_cols = 20, 20
n_actions = 4  # Up, Down, Left, Right
goal_state = (19, 19)
obstacles = [(5, 5), (10, 10), (15, 15)]

# Convert (row,col) to state (normalized coordinates)
def get_state(row, col):
    return np.array([row / grid_rows, col / grid_cols])  # Normalized to [0,1]

# DQN Model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(2,)),  # Input: normalized (row,col)
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(n_actions)  # Output: Q-values for each action
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
    return model

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# Hyperparameters
batch_size = 64
gamma = 0.95  # Discount factor
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 1000

# Initialize
model = build_model()
target_model = build_model()  # Target network for stability
target_model.set_weights(model.get_weights())
buffer = ReplayBuffer()

# Training Loop
for episode in range(episodes):
    state = get_state(*np.random.randint(0, grid_rows, size=2))
    total_reward = 0
    done = False
    
    while not done:
        # Epsilon-greedy action
        if np.random.rand() <= epsilon:
            action = np.random.randint(n_actions)
        else:
            q_values = model.predict(state[np.newaxis], verbose=0)[0]
            action = np.argmax(q_values)
        
        # Execute action
        row, col = int(state[0] * grid_rows), int(state[1] * grid_cols)
        new_row, new_col = row, col
        if action == 0: new_row = max(0, row - 1) # Up
        elif action == 1: new_row = min(grid_rows - 1, row + 1) # Down
        elif action == 2: new_col = max(0, col - 1) # Left
        elif action == 3: new_col = min(grid_cols - 1, col + 1) # Right
        
        # Reward and termination
        if (new_row, new_col) in obstacles:
            reward = -1
            new_row, new_col = row, col  # Stay put
        elif (new_row, new_col) == goal_state:
            reward = 10
            done = True
        else:
            reward = -0.01
        
        next_state = get_state(new_row, new_col)
        buffer.add(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        
        # Train on batch
        if len(buffer.buffer) >= batch_size:
            batch = buffer.sample(batch_size)
            states = np.array([x[0] for x in batch])
            actions = np.array([x[1] for x in batch])
            rewards = np.array([x[2] for x in batch])
            next_states = np.array([x[3] for x in batch])
            dones = np.array([x[4] for x in batch])
            
            # Bellman update
            target_q = rewards + gamma * np.max(target_model.predict(next_states, verbose=0), axis=1) * (1 - dones)
            q_values = model.predict(states, verbose=0)
            q_values[np.arange(batch_size), actions] = target_q
            
            model.train_on_batch(states, q_values)
    
    # Update target network and exploration
    if episode % 10 == 0:
        target_model.set_weights(model.get_weights())
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    print(f"Episode {episode}, Total Reward: {total_reward:.1f}, Epsilon: {epsilon:.2f}")

model.save("dqn_gridworld.keras")