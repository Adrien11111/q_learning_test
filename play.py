import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Environment
grid_rows, grid_cols = 20, 20
n_actions = 4  # Up, Down, Left, Right
goal_state = (19, 19)
obstacles = [(5, 5), (10, 10), (15, 15)]

model = tf.keras.models.load_model("dqn_gridworld.keras")

# Convert (row,col) to state (normalized coordinates)
def get_state(row, col):
    return np.array([row / grid_rows, col / grid_cols])  # Normalized to [0,1]

def navigate_with_dqn(start_row, start_col):
    state = get_state(start_row, start_col)
    path = [(start_row, start_col)]
    
    for _ in range(1000):  # Max steps
        q_values = model.predict(state[np.newaxis], verbose=0)[0]
        action = np.argmax(q_values)
        
        row, col = int(state[0] * grid_rows), int(state[1] * grid_cols)
        if action == 0: row = max(0, row - 1)
        elif action == 1: row = min(grid_rows - 1, row + 1)
        elif action == 2: col = max(0, col - 1)
        elif action == 3: col = min(grid_cols - 1, col + 1)
        
        path.append((row, col))
        state = get_state(row, col)
        
        if (row, col) == goal_state:
            print("Goal reached!")
            break
    
    return path

def plot_dqn_policy():
    arrows = ['↑', '↓', '←', '→']
    plt.figure(figsize=(6, 6))
    
    # Create a grid background
    plt.grid(True, zorder=0)
    plt.xlim(-0.5, grid_cols-0.5)
    plt.ylim(-0.5, grid_rows-0.5)

    for row in range(grid_rows):
        for col in range(grid_cols):
            # Draw obstacles first (lowest zorder)
            if (row, col) in obstacles:
                plt.text(col, row, 'X', 
                         ha='center', va='center', 
                         color='red', fontsize=12, 
                         fontweight='bold', zorder=1)
            
            # Then draw goal
            elif (row, col) == goal_state:
                plt.text(col, row, 'G', 
                         ha='center', va='center', 
                         fontsize=12, 
                         bbox=dict(facecolor='yellow', alpha=0.5),
                         zorder=2)
            
            # Finally draw arrows (highest zorder)
            else:
                state = get_state(row, col)
                q_values = model.predict(state[np.newaxis], verbose=0)[0]
                action = np.argmax(q_values)
                plt.text(col, row, arrows[action], 
                         ha='center', va='center', 
                         fontsize=8, zorder=3)
    
    plt.title("DQN Learned Policy")
    plt.show()

def print_console_policy():
    arrows = ['↑', '↓', '←', '→']
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'X', 'Y', 'Z']
    
    # Print column numbers
    print("    " + " ".join(letters[i] for i in range(letters.index('A'), letters.index('A') + grid_cols)))
    print("  +" + "--" * grid_cols + "+")
    
    for row in range(grid_rows):
        print(f"{row:2}|", end="")
        for col in range(grid_cols):
            if (row, col) == goal_state:
                print(" G", end="")
            elif (row, col) in obstacles:
                print(" X", end="")
            else:
                state = get_state(row, col)
                q_values = model.predict(state[np.newaxis], verbose=0)[0]
                action = np.argmax(q_values)
                print(f" {arrows[action]}", end="")
        print("|")
    
    print("  +" + "--" * grid_cols + "+")
    print("Legend: G=Goal, X=Obstacle, Arrows=Optimal Action")

# Example usage
# Test the DQN navigation
optimal_path = navigate_with_dqn(0, 0)
print("Path:", optimal_path)

print("Goal state:", goal_state)  # Should be (19,19)
print("Obstacles:", obstacles)  # Should match what you're plotting

# plot_dqn_policy()
print_console_policy()