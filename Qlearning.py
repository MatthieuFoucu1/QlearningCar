# car driving trainer

#---imports---
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tracks import Track

# replace with velocity and py game enviroment to make it more driveable
# define all the nessecary components - maybe make it editable
track = Track.getExpert()

num_states = track.size # number of positions in the track, aka the size
states = ["start/end", "wall", "path"]
num_actions = 4
actions = ["up", "down", "left", "right"]
goal = 1000
penalties = [-1, -10]
# 2d grid where at each cell it contains information on each action
Qtable = np.zeros((track.shape[0], track.shape[1], num_actions))

#hyperperameters to fine tune
alpha = 0.8 # this is the learning rate to be used in the Temporal Difference update ranges form .0001 to .1
epsilon = 1.0 # exploration rate
gamma = 0.9  # Discount factor for the TD equation, which type or rewards to focus on
epsilon_decay = 0.9995 # decay for the epsilon greedy policy
epsilon_min = 0.1
episodes = 10000 # -  also known as epochs

# functions
def showTrack(grid):
    # makes an image that will map the track
    # i,j and j,i are flipped between matrix and image
    img = Image.new('RGB', (track.shape[1], track.shape[0]), "white")
    pixels = img.load()
    # itterate through the track double array
    for i in range(track.shape[0]):
        for j in range(track.shape[1]):
            if grid[i, j] == 1:
                pixels[j, i] = (0,0,0) # black walls
            elif grid[i, j] == 0:
                pixels[j, i] = (255,255,255) # white path
            elif grid[i, j] == 2:
                pixels[j, i] = (29,184,94) # player
            elif grid[i, j] == 3:
                pixels[j, i] = (0,255,255)  # cyan for the agent's path
            else:
                pixels[j, i] = (255,0,0) # end goal

    # scales the image to be bigger than literally 42 pixels - easier for the user to see
    img = img.resize((800,800), Image.Resampling.NEAREST)
    img.show()


# Modify move function to return the new state directly
def move(action, state):

    # makes sure it cant go off the screen
    x, y = state
    if action == 'up':
        return (max(x - 1, 0), y)
    elif action == 'down':
        return (min(x + 1, track.shape[0] - 1), y)
    elif action == 'left':
        return (x, max(y - 1, 0))
    elif action == 'right':
        return (x, min(y + 1, track.shape[1] - 1))
    

def getReward(state):
    #rewards/penalty values can be changed
    x, y = state
    if track[x,y] == 1:
        return penalties[1] #penalty crashing into walls  - end episode here in training loop
    elif track[x,y] == -1:
        return goal #finished the track - end episode by wining
    else:
        return penalties[0] #just moving along path
    

# Modify chooseAction to work with the new state representation
def chooseAction(state):
    if np.random.rand() < epsilon:
        return np.random.choice(actions)
    else:
        return actions[np.argmax(Qtable[state])]
    
#update all the values in the q table with the temporal differce equation
def updateQ(state, action, reward, next_state):
    # use the temporal differnce equation to update the table
    best_next_action = np.argmax(Qtable[next_state])
    # Q(S,A)←Q(S,A)+α(R+γQ(S’,A’)–Q(S,A))   
    Qtable[state][action_to_index[action]] += alpha * (reward + gamma * Qtable[next_state][best_next_action] - Qtable[state][action_to_index[action]])

def train():
    # uses the epsilon saves it between calls?
    global epsilon
    #tracks rewards for each episode - aka how good it does
    performance = []

    #start the loop for the number of defined episodes
    for episode in range(episodes):
        # specify the staring coordinates
        state = (1, 1)
        totalReward = 0
        done = False # flag to end episode

        while not done:
            action = chooseAction(state)
            next_state = move(action, state)
            reward = getReward(next_state)
            
            updateQ(state, action, reward, next_state)
            
            state = next_state
            totalReward += reward
             
            # Updated termination condition
            if reward == goal or reward == -10:  # Reached goal or hit wall
                done = True

        # after that run, apply the hyperparameters
        epsilon = max(epsilon_min, epsilon * epsilon_decay)    
        # track performances    
        performance.append(totalReward)
        
        #if episode % 100 == 0:
            #print(f"Episode {episode}, Total Reward: {totalReward}, Epsilon: {epsilon:.2f}")
            #print(f"Final state: {state}, Reward: {reward}")

    return performance



def plot_performance(performance):
    plt.plot(performance)
    plt.title('Performance over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

def visualize_path():
    state = (1, 1) # starting grid
    path = [state]
    while track[state] != -1:
        # should be picking the best option here from the q table that has been updated after 3000 tests
        action = actions[np.argmax(Qtable[state])]
        state = move(action, state)
        path.append(state)
        if len(path) > 100:  # Prevent infinite loops
            break
    
    # Visualize the path
    track_copy = track.copy()
    for x, y in path:
        if track_copy[x, y] == 0:
            track_copy[x, y] = 3  # Mark path
    showTrack(track_copy)



action_to_index = {action: i for i, action in enumerate(actions)} # uses dictionary to map each action to an index

# main/start
def main():
    print("Initial Q-table")
    #print(Qtable)
    showTrack(track)

    performance = train()
    plot_performance(performance)
    
    print("\nFinal Q-table")
    #print(Qtable)

    # maybe save the q table? or something l=to save the training per run
    
    print("\nVisualizing the trained agent's path:")
    visualize_path()

if __name__=="__main__":
    main()