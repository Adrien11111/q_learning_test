import numpy as np

# Load the Q-table
q_data = np.load("sparse_qtable.npz", allow_pickle=True)
Q_table = {int(k): v for k, v in q_data.items()}

grid_rows, grid_cols = 100, 100
n_actions = 4  # Up, Down, Left, Right
goal_state = (99, 99)
obstacles = [(10, 10), (20, 20)]  # Add your obstacles here


# Convert (row,col) to state index (flattened)
def get_state_index(row, col):
    return row * grid_cols + col

# For states never visited, default to zeros
def get_q_values(state_index):
    return Q_table.get(state_index, np.zeros(n_actions))

def find_path(start_row, start_col, max_steps=1000):
    path = [(start_row, start_col)]
    current_row, current_col = start_row, start_col
    
    for _ in range(max_steps):
        if (current_row, current_col) == goal_state:
            print("Goal reached!")
            break
        
        state_index = get_state_index(current_row, current_col)
        action = np.argmax(get_q_values(state_index))  # Always choose best action
        
        # Move (same logic as during training)
        if action == 0: current_row = max(0, current_row - 1)          # Up
        elif action == 1: current_row = min(grid_rows - 1, current_row + 1)  # Down
        elif action == 2: current_col = max(0, current_col - 1)        # Left
        elif action == 3: current_col = min(grid_cols - 1, current_col + 1)  # Right
        
        path.append((current_row, current_col))
    
    return path

while True:
    try:
        start_row = int(input("Enter starting row (0 to {}): ".format(grid_rows - 1)))
        start_col = int(input("Enter starting column (0 to {}): ".format(grid_cols - 1)))
        
        if (start_row, start_col) in obstacles or (start_row, start_col) == goal_state:
            print("Invalid starting position. Please choose a valid cell.")
            continue
        
        path = find_path(start_row, start_col)
        print("Path found:", path)
    except ValueError:
        print("Invalid input. Please enter integers for row and column.")
    