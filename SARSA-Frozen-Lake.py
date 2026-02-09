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
# --- 1. The SARSA Agent Class ---
# ===================================================================
class SARSAAgent:
    """
    A professional SARSA Agent. Unlike Q-Learning, SARSA is 'On-Policy',
    meaning it updates its Q-values based on the actual action taken 
    in the next state.
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
        """Standard epsilon-greedy action selection."""
        if use_epsilon and np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.action_size) 
        return np.argmax(self.q_table[state, :]) 

    def learn(self, state, action, reward, next_state, next_action, done):
        """
        The SARSA update rule:
        Q(s,a) = Q(s,a) + alpha * [R + gamma * Q(s', a') - Q(s,a)]
        """
        # Target uses the Q-value of the ACTUAL next action (On-Policy)
        target = reward + (self.gamma * self.q_table[next_state, next_action] * (1 - done))
        
        # Bellman equation update
        self.q_table[state, action] += self.lr * (target - self.q_table[state, action])
        self.total_steps += 1

    def update_epsilon(self, episode):
        """Exponential decay for epsilon."""
        self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * \
                       np.exp(-self.decay_rate * episode)

    def save_model(self, filepath):
        """Saves the Q-table to a .npy file."""
        np.save(filepath, self.q_table)

    def load_model(self, filepath):
        """Loads a Q-table from a .npy file."""
        self.q_table = np.load(filepath)

# ===================================================================
# --- 2. Visualization & Helper Functions ---
# ===================================================================

def visualize_q_table(q_table, filename="results/sarsa_q_table_heatmap.png"):
    """
    Generates a heatmap of the Q-table to visualize the learned values.
    Rows = States (0-15), Columns = Actions (Left, Down, Right, Up).
    """
    plt.figure(figsize=(10, 8))
    actions = ["Left", "Down", "Right", "Up"]
    
    sns.heatmap(q_table, annot=True, fmt=".3f", cmap="YlGnBu", 
                xticklabels=actions, yticklabels=range(q_table.shape[0]))
    
    plt.title("Learned Q-Table Values (SARSA)")
    plt.xlabel("Actions")
    plt.ylabel("States")
    plt.savefig(filename)
    print(f"SARSA Q-Table heatmap saved to {filename}")

def plot_results(rewards, window=500, filename="results/sarsa_success_rate.png"):
    """Plots a moving average of rewards to show the success rate over time."""
    rewards_arr = np.array(rewards)
    if len(rewards_arr) < window: return
    
    moving_avg = np.convolve(rewards_arr, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(10, 5))
    plt.plot(moving_avg)
    plt.title(f"SARSA Success Rate (Moving Avg {window} Episodes)")
    plt.ylabel("Success Rate")
    plt.xlabel("Episode")
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    print(f"SARSA Training plot saved to {filename}")

# ===================================================================
# --- 3. Separate Training & Deployment Methods ---
# ===================================================================

def train(env, agent, num_episodes, start_episode, all_rewards, print_every):
    """
    Handles the main SARSA training loop.
    SARSA is On-Policy: we must pick the NEXT action before updating the current Q-value.
    """
    print(f"Starting SARSA training from episode {start_episode}...")
    successes_in_batch = 0

    for episode in range(start_episode, num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        # Step A: Choose the initial action
        action = agent.choose_action(state)
        
        while not done:
            # Step B: Take action A, observe R, S'
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            
            # Step C: Choose the NEXT action A' from S' (Crucial for SARSA)
            next_action = agent.choose_action(next_state)
            
            # Step D: Update Q(S,A) using R and Q(S', A')
            agent.learn(state, action, reward, next_state, next_action, done)
            
            # Step E: Move to next state and next action
            state = next_state
            action = next_action
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
    print("\nDeploying trained SARSA agent (Human Render Mode)...")
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

    print(f"--- Starting SARSA Frozen Lake (DQN Style) ---")
    print(f"Using Gymnasium version: {gym.__version__}")

    # --- 1. Set Device (Hardware Parity) ---
    # SARSA is CPU-bound (NumPy)
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
    
    MODEL_FILEPATH = os.path.join(RESULTS_DIR, "sarsa_table_frozenlake.npy")
    HISTORY_FILEPATH = os.path.join(RESULTS_DIR, "sarsa_history_frozenlake.npz")
    PRINT_EVERY = 1000

    # --- 3. Initialization ---
    env = gym.make(ENV_NAME, is_slippery=True)
    agent = SARSAAgent(
        state_size=env.observation_space.n,
        action_size=env.action_space.n,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        decay_rate=EPSILON_DECAY_RATE
    )

    # Mode Selection: 'new' (scratch), 'resume' (continue), or 'deploy' (watch)
    MODE = 'deploy'  

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
        deploy(ENV_NAME, agent, num_trials=5)
    
    env.close()
    print("\n--- Program Finished ---")

if __name__ == "__main__":
    main()