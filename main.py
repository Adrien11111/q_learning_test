import numpy as np
from tqdm import tqdm
from scipy.sparse import dok_matrix

# Grid setup
grid_rows, grid_cols = 100, 100
n_actions = 4  # Up, Down, Left, Right
goal_state = (99, 99)
obstacles = [(10, 10), (20, 20)]  # Add your obstacles here

# Convert (row,col) to state index (flattened)
def get_state_index(row, col):
    return row * grid_cols + col

# Initialize Q-table as dictionary (faster than dok_matrix for this use case)
Q_table = {}

# Learning parameters
learning_rate = 0.1
discount_factor = 0.95
initial_exploration = 1.0
min_exploration = 0.01
exploration_decay = 0.999995  # Adjusted for 10M epochs
epochs = 10_000_000

# Training loop
exploration_prob = initial_exploration
for epoch in tqdm(range(epochs), desc="Training"):
    # Decay exploration
    exploration_prob = max(min_exploration, exploration_prob * exploration_decay)
    
    # Random start state (avoid obstacles/goal)
    while True:
        current_row, current_col = np.random.randint(0, grid_rows), np.random.randint(0, grid_cols)
        if (current_row, current_col) not in obstacles and (current_row, current_col) != goal_state:
            break
    
    while (current_row, current_col) != goal_state:
        current_state = get_state_index(current_row, current_col)
        
        # Initialize state if not in Q-table
        if current_state not in Q_table:
            Q_table[current_state] = np.zeros(n_actions)
        
        # Epsilon-greedy action
        if np.random.rand() < exploration_prob:
            action = np.random.randint(0, n_actions)
        else:
            action = np.argmax(Q_table[current_state])
        
        # Calculate new position
        new_row, new_col = current_row, current_col
        if action == 0: new_row = max(0, current_row - 1)      # Up
        elif action == 1: new_row = min(grid_rows - 1, current_row + 1)  # Down
        elif action == 2: new_col = max(0, current_col - 1)    # Left
        elif action == 3: new_col = min(grid_cols - 1, current_col + 1)  # Right
        
        # Reward and transition
        if (new_row, new_col) in obstacles:
            reward = -1
            new_row, new_col = current_row, current_col  # Stay put
        elif (new_row, new_col) == goal_state:
            reward = 10
        else:
            reward = -0.01  # Small penalty per step
        
        next_state = get_state_index(new_row, new_col)
        
        # Initialize next state if new
        if next_state not in Q_table:
            Q_table[next_state] = np.zeros(n_actions)
        
        # Q-learning update
        current_q = Q_table[current_state][action]
        next_max = np.max(Q_table[next_state])
        updated_q = current_q + learning_rate * (reward + discount_factor * next_max - current_q)
        Q_table[current_state][action] = updated_q
        
        current_row, current_col = new_row, new_col

# Save Q-table (sparse format)
np.savez("sparse_qtable.npz", **{str(k): v for k, v in Q_table.items()})