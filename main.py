import numpy as np
import matplotlib.pyplot as plt

# Grid dimensions (4 rows x 7 columns)
grid_rows = 4
grid_cols = 7
n_states = grid_rows * grid_cols
n_actions = 4  # Up, Down, Left, Right

# Special states
goal_state = (3, 6)  # Bottom-right corner
obstacles = [(1, 1), (2, 3), (3, 3)]  # Blocked cells

# Convert (row,col) to state index
def get_state_index(row, col):
    return row * grid_cols + col

# Q-table initialization
Q_table = np.zeros((n_states, n_actions))

# Learning parameters
learning_rate = 0.1
discount_factor = 0.9
exploration_prob = 0.2
epochs = 10000

for epoch in range(epochs):
    # Start at random position (avoid obstacles)
    while True:
        current_row, current_col = np.random.randint(0, grid_rows), np.random.randint(0, grid_cols)
        if (current_row, current_col) not in obstacles and (current_row, current_col) != goal_state:
            break
    
    while (current_row, current_col) != goal_state:
        current_state = get_state_index(current_row, current_col)
        
        # Choose action
        if np.random.rand() < exploration_prob:
            action = np.random.randint(0, n_actions)  # Explore
        else:
            action = np.argmax(Q_table[current_state])  # Exploit
        
        # Calculate new position
        new_row, new_col = current_row, current_col
        if action == 0: new_row = max(0, current_row-1)        # Up
        elif action == 1: new_row = min(grid_rows-1, current_row+1) # Down
        elif action == 2: new_col = max(0, current_col-1)      # Left
        elif action == 3: new_col = min(grid_cols-1, current_col+1) # Right
        
        # Check for obstacles
        if (new_row, new_col) in obstacles:
            new_row, new_col = current_row, current_col  # Stay put
            reward = -1  # Penalty for hitting obstacle
        elif (new_row, new_col) == goal_state:
            reward = 10  # Big reward for reaching goal
        else:
            reward = -0.1  # Small penalty per step to encourage efficiency
        
        next_state = get_state_index(new_row, new_col)
        
        # Q-learning update
        Q_table[current_state, action] += learning_rate * (
            reward + discount_factor * np.max(Q_table[next_state]) - Q_table[current_state, action]
        )
        
        current_row, current_col = new_row, new_col

# Visualization
q_values_grid = np.max(Q_table, axis=1).reshape((grid_rows, grid_cols))

plt.figure(figsize=(10, 6))
plt.imshow(q_values_grid, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Q-value')
plt.title('Learned Q-values for Grid World')

# Mark special cells
for (row, col) in obstacles:
    plt.text(col, row, 'X', ha='center', va='center', color='red', fontsize=20)
plt.text(goal_state[1], goal_state[0], 'G', ha='center', va='center', color='white', fontsize=20)

# Annotate Q-values
for row in range(grid_rows):
    for col in range(grid_cols):
        if (row, col) not in obstacles and (row, col) != goal_state:
            plt.text(col, row, f'{q_values_grid[row, col]:.1f}', 
                    ha='center', va='center', color='white')

plt.show()