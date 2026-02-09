import gymnasium as gym
import numpy as np
import time
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# --- Suppress the Pygame/pkg_resources UserWarning ---
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

# ===================================================================
# --- 1. The Q-Learning Agent Class ---
# ===================================================================
class QLearningAgent:
    """
    A professional Q-Learning Agent that encapsulates the Q-Table 
    and the update logic, matching the style of the DQNAgent.
    """
    def __init__(self, state_size, action_size, learning_rate, gamma, 
                 epsilon_start, epsilon_min, decay_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate
        self.total_steps = 0 
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state, use_epsilon=True):
        if use_epsilon and np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.action_size) # EXPLORE
        return np.argmax(self.q_table[state, :]) # EXPLOIT

    def learn(self, state, action, reward, next_state, done):
        target = reward + (self.gamma * np.max(self.q_table[next_state, :]) * (1 - done))
        self.q_table[state, action] += self.lr * (target - self.q_table[state, action])
        self.total_steps += 1

    def update_epsilon(self, episode):
        self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * \
                       np.exp(-self.decay_rate * episode)

    def save_model(self, filepath):
        np.save(filepath, self.q_table)

    def load_model(self, filepath):
        self.q_table = np.load(filepath)

# ===================================================================
# --- 2. Visualization & Helper Functions ---
# ===================================================================

def visualize_q_table(q_table, filename="results/q_table_heatmap_q_learning.png"):
    """
    Generates a heatmap of the Q-table to visualize the learned values.
    Rows = States (0-15), Columns = Actions (Left, Down, Right, Up).
    """
    plt.figure(figsize=(10, 8))
    actions = ["Left", "Down", "Right", "Up"]
    sns.heatmap(q_table, annot=True, fmt=".3f", cmap="YlGnBu", 
                xticklabels=actions, yticklabels=range(q_table.shape[0]))
    plt.title("Learned Q-Table Values (Q-Learning)")
    plt.xlabel("Actions")
    plt.ylabel("States")
    plt.savefig(filename)
    print(f"Q-Table heatmap saved to {filename}")

def plot_results(rewards, window=500, filename="results/q_learning_results.png"):
    """Fixes the 'solid block' issue by plotting a moving average."""
    rewards_arr = np.array(rewards)
    if len(rewards_arr) < window: return
    
    moving_avg = np.convolve(rewards_arr, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(10, 5))
    plt.plot(moving_avg)
    plt.title(f"Success Rate (Moving Avg {window} Episodes)")
    plt.ylabel("Success Rate")
    plt.xlabel("Episode")
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    print(f"Training plot saved to {filename}")

# ===================================================================
# --- 3. Separate Training & Deployment Methods ---
# ===================================================================

def train(env, agent, num_episodes, start_episode, all_rewards, print_every):
    """
    Handles the main training loop. 
    """
    print(f"Starting training from episode {start_episode}...")
    successes_in_batch = 0

    for episode in range(start_episode, num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        
        agent.update_epsilon(episode)
        all_rewards.append(total_reward)
        if total_reward == 1: successes_in_batch += 1

        if (episode + 1) % print_every == 0:
            success_rate = (successes_in_batch / print_every) * 100
            print(f"Episode {episode+1}/{num_episodes} | Success: {success_rate:.1f}% | Epsilon: {agent.epsilon:.4f}")
            successes_in_batch = 0
            
    return all_rewards

def deploy(env_name, agent, num_trials=5):
    """
    Deploys the agent with visualization.
    """
    print("\nDeploying trained agent (Human Render Mode)...")
    env = gym.make(env_name, is_slippery=True, render_mode="human")
    
    for trial in range(num_trials):
        state, _ = env.reset()
        done = False
        print(f"--- Trial {trial + 1} ---")
        
        while not done:
            action = agent.choose_action(state, use_epsilon=False)
            state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            time.sleep(0.2)
            
            if done:
                print("Goal Reached! 🏆" if reward == 1 else "Fell in Hole 🕳️")
    env.close()

# ===================================================================
# --- 4. The Main Program Execution ---
# ===================================================================
def main():
    # Ensure results directory exists
    RESULTS_DIR = "results"
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    print(f"--- Starting Q-Learning Frozen Lake (DQN Style) ---")
    print(f"Using Gymnasium version: {gym.__version__}")

    # --- 1. Set Device (Hardware Parity) ---
    # Q-learning is CPU-bound (NumPy), but we keep the structure for Mac/Intel/GPU checks.
    device = "cpu"
    print(f"Using device: {device}")

    # --- 2. Hyperparameters ---
    ENV_NAME = "FrozenLake-v1"
    NUM_EPISODES = 50000
    LEARNING_RATE = 0.05
    GAMMA = 0.95
    
    EPSILON_START = 1.0
    EPSILON_MIN = 0.01
    EPSILON_DECAY_RATE = 0.0005
    
    MODEL_FILEPATH = os.path.join(RESULTS_DIR, "q_table_frozenlake.npy")
    HISTORY_FILEPATH = os.path.join(RESULTS_DIR, "frozenlake_history.npz")
    PRINT_EVERY = 1000

    # --- 3. Initialization ---
    env = gym.make(ENV_NAME, is_slippery=True)
    agent = QLearningAgent(
        state_size=env.observation_space.n,
        action_size=env.action_space.n,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        decay_rate=EPSILON_DECAY_RATE
    )

    # Mode Selection: 'new' (scratch), 'resume' (continue), or 'deploy' (watch)
    MODE = 'new'  

    episode_rewards = []
    start_episode = 0

    # --- 4. Mode Selection & Loading ---
    if MODE in ['resume', 'deploy']:
        if os.path.exists(MODEL_FILEPATH):
            print(f"Loading model from {MODEL_FILEPATH}...")
            agent.load_model(MODEL_FILEPATH)
            
            if os.path.exists(HISTORY_FILEPATH):
                print(f"Loading history from {HISTORY_FILEPATH}...")
                data = np.load(HISTORY_FILEPATH)
                episode_rewards = data['rewards'].tolist()
                start_episode = len(episode_rewards)
                print(f"Resuming from Episode {start_episode}")
        else:
            print("No model found. Starting from scratch.")
            MODE = 'new'

    # --- 5. Training Phase ---
    if MODE in ['new', 'resume']:
        try:
            # Call train function 
            episode_rewards = train(
                env=env, 
                agent=agent, 
                num_episodes=NUM_EPISODES, 
                start_episode=start_episode, 
                all_rewards=episode_rewards, 
                print_every=PRINT_EVERY
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        finally:
            print("Saving model and history...")
            agent.save_model(MODEL_FILEPATH)
            np.savez(HISTORY_FILEPATH, rewards=np.array(episode_rewards))
            plot_results(episode_rewards)
            visualize_q_table(agent.q_table)

    # --- 6. Deployment Phase ---
    if MODE == 'deploy':
        # Call deploy function
        deploy(ENV_NAME, agent, num_trials=5)
    
    env.close()
    print("\n--- Program Finished ---")

if __name__ == "__main__":
    main()