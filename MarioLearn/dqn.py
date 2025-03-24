import gym
import numpy as np
import cv2
import tensorflow as tf
from collections import deque
import random
import time
import matplotlib.pyplot as plt
from gym_super_mario_bros import make

# Disable GPU (if you don't need it).
tf.config.set_visible_devices([], 'GPU')

# Hyperparameters
gamma = 0.99           # Discount factor
epsilon = 1.0          # Exploration rate
epsilon_min = 0.01     # Minimum exploration rate
epsilon_decay = 0.995  # Decay factor for epsilon
learning_rate = 0.00025
batch_size = 16
buffer_size = 100000   # Replay buffer size
target_update_freq = 10  # Update target network every N episodes

# Create environment
env = make("SuperMarioBros-v3")

# Define Q-Network (CNN)
class DQN(tf.keras.Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(env.action_space.n, activation='linear')  # Adjust the output size

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

# Initialize the Q-Network and Target Network
q_network = DQN()
target_network = DQN()
target_network.set_weights(q_network.get_weights())

# Experience Replay buffer
replay_buffer = deque(maxlen=buffer_size)

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Preprocess state: Convert to grayscale, then normalize to [0, 1]
def preprocess_state(state):
    # Convert to grayscale
    state_gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)

    # Resize to (28, 28) to make it smaller for faster processing and reduce memory usage
    state_resized = cv2.resize(state_gray, (28, 28))  # Resize to 28x28 or smaller

    # Normalize the pixel values to [0, 1]
    state_normalized = np.array(state_resized, dtype=np.float32) / 255.0

    # Add the channel dimension (greyscale has 1 channel)
    return np.expand_dims(state_normalized, axis=-1)

# Epsilon-greedy policy
def epsilon_greedy_policy(state, epsilon):
    if np.random.rand() <= epsilon:
        return np.random.choice(env.action_space.n)  # Random action
    q_values = q_network(state[np.newaxis])  # Predict Q-values
    return np.argmax(q_values[0])  # Select the action with max Q-value

# Train the Q-Network
@tf.function
def train_step(batch):
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert lists of states, next_states, etc., to TensorFlow tensors
    states = tf.convert_to_tensor(states, dtype=tf.float32)
    next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    dones = tf.convert_to_tensor(dones, dtype=tf.float32)  # Convert dones to float32 for arithmetic

    # Q-targets for the current states
    q_values_next = target_network(next_states)
    q_values_next_max = tf.reduce_max(q_values_next, axis=1)  # Get max Q-values for next states
    q_targets = rewards + (gamma * q_values_next_max * (1 - dones))  # Calculate the Q-targets

    with tf.GradientTape() as tape:
        q_values = q_network(states)
        q_values = tf.reduce_sum(q_values * tf.one_hot(actions, env.action_space.n), axis=1)
        loss = tf.reduce_mean(tf.square(q_values - q_targets))  # Mean squared error loss

    grads = tape.gradient(loss, q_network.trainable_variables)
    optimizer.apply_gradients(zip(grads, q_network.trainable_variables))

    return loss

# Main Training Loop
rewards = []
def train():
    global epsilon
    for episode in range(5000):  # Number of episodes
        state = env.reset()  # Reset environment
        state = preprocess_state(state)  # Preprocess state

        # Ensure state has the correct dimensions
        if len(state.shape) == 3:  # If it's (height, width, channels)
            state = np.expand_dims(state, axis=0)  # Add batch dimension

        episode_reward = 0

        while True:
            action = epsilon_greedy_policy(state, epsilon)

            next_state, reward, done, _ = env.step(action)
            next_state = preprocess_state(next_state)  # Preprocess next state

            # Ensure next_state has the correct dimensions
            if len(next_state.shape) == 3:
                next_state = np.expand_dims(next_state, axis=0)  # Add batch dimension

            # Store experience in replay buffer
            replay_buffer.append((state, action, reward, next_state, done))
            episode_reward += reward

            # Sample a batch from the replay buffer and train the Q-network
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                loss = train_step(batch)  # Train the Q-network

            state = next_state

            # Render the environment to visualize the gameplay
            env.render()
            time.sleep(0.05)  # Slow down the rendering slightly

            if done:
                break

        rewards.append(episode_reward)

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Update target network periodically
        if episode % target_update_freq == 0:
            target_network.set_weights(q_network.get_weights())

        print(f"Episode {episode+1}, Reward: {episode_reward}, Epsilon: {epsilon:.3f}")

        # Save the model periodically (every 100 episodes)
        if episode % 100 == 0:
            q_network.save(f"model_episode_{episode}.h5")
            print(f"Model saved at episode {episode+1}")

    # After training, visualize the learning progress
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Agent Learning Progress')
    plt.show()

# Run training
train()

