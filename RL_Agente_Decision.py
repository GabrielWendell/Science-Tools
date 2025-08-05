# Step 1: RL agent decision-making in conversation context
# This is a basic RL example using Q-learning to decide the best action based on conversation state.

import numpy as np
import random
from collections import defaultdict

# Define possible states (simplified representations of conversation situations)
states = ["greeting", "question", "complaint", "feedback", "farewell"]

# Define possible actions the agent can take in response to each state
actions = ["say_hello", "answer_question", "apologize", "thank_user", "say_goodbye"]

# Initialize Q-table with zeros (state-action values)
Q = defaultdict(lambda: np.zeros(len(actions)))

# Hyperparameters
alpha = 0.1        # learning rate
gamma = 0.9        # discount factor
epsilon = 0.2      # exploration rate

# Simulated environment: rewards for taking actions in states (toy example)
reward_table = {
    ("greeting", "say_hello"): 1.0,
    ("question", "answer_question"): 1.0,
    ("complaint", "apologize"): 1.0,
    ("feedback", "thank_user"): 1.0,
    ("farewell", "say_goodbye"): 1.0
}

# Function to simulate environment response and return reward
def get_reward(state, action):
    return reward_table.get((state, action), -1.0)  # -1 penalty for inappropriate responses

# Epsilon-greedy action selection
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(len(actions)))
    else:
        return np.argmax(Q[state])

# Training loop
def train_agent(episodes=500):
    for episode in range(episodes):
        state = random.choice(states)  # Random initial state
        action_idx = choose_action(state)
        action = actions[action_idx]
        reward = get_reward(state, action)

        # Q-learning update
        best_next_action = np.max(Q[state])
        Q[state][action_idx] += alpha * (reward + gamma * best_next_action - Q[state][action_idx])

    print("Training completed.")

# Evaluation function to show best action for each state
def evaluate_policy():
    for state in states:
        best_action = actions[np.argmax(Q[state])]
        print(f"Best action for state '{state}': {best_action}")

if __name__ == "__main__":
    train_agent()
    evaluate_policy()
