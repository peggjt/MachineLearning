import gym
import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from nes_py.wrappers import JoypadSpace
import numpy as np

# Define a simple action space
ACTIONS = [["right"], ["right", "A"], ["right", "B"], ["right", "A", "B"]]

# Create the environment
env = gym.make("SuperMarioBros-v3")  # Use latest version
env = JoypadSpace(env, ACTIONS)  # Restrict to a smaller action space
env = ResizeObservation(env, (84, 84))  # Resize frames
env = GrayScaleObservation(env, keep_dim=True)  # Convert to grayscale
env = FrameStack(env, 4)  # Stack 4 frames

# ✅ Handle different Gym versions for `reset()`
try:
    reset_result = env.reset()
    if isinstance(reset_result, tuple):  # Gym >= 0.26
        obs = reset_result[0]  # Extract observation
    else:  # Gym < 0.26
        obs = reset_result
except ValueError:  # Catch unpacking errors
    obs = env.reset()  # Use single-value return

# Run the environment
done = False
while not done:
    env.render()
    action = env.action_space.sample()  # Choose a random action
    step_result = env.step(action)

    # ✅ Handle `step()` returning different numbers of values
    if len(step_result) == 5:  # Gym >= 0.26
        obs, reward, done, truncated, info = step_result
        done = done or truncated  # Combine done & truncated
    else:  # Older Gym API
        obs, reward, done, info = step_result

env.close()

