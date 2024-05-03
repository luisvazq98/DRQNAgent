import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam

# Define constants
FRAME_WIDTH = 84
FRAME_HEIGHT = 84
NUM_FRAMES = 4
NUM_ACTIONS = 6
GAMMA = 0.99
LEARNING_RATE = 0.0001
MEMORY_SIZE = 10000
BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 1000
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.1
NUM_EPISODES = 10000
MAX_STEPS_PER_EPISODE = 10000

# Preprocessing function
def preprocess(frame):
    # Convert to grayscale
    frame = frame.mean(axis=2)
    # Resize
    frame = frame[35:195]
    frame = frame[::2, ::2]
    return frame.astype(np.uint8)


# Build DRQN model
def build_model(input_shape, num_actions):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape))
    model.add(Dense(num_actions, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))
    return model


# Replay buffer for experience replay
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in idx]


# Agent class
class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.state_shape = (NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH)
        self.model = build_model(self.state_shape, NUM_ACTIONS)
        self.target_model = build_model(self.state_shape, NUM_ACTIONS)
        self.update_target_model()
        self.memory = ReplayBuffer(max_size=MEMORY_SIZE)
        self.epsilon = 1.0

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        q_values = self.model.predict(np.expand_dims(state, axis=0))
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory.buffer) < batch_size:
            return
        minibatch = self.memory.sample(batch_size)
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + GAMMA * np.amax(self.target_model.predict(np.expand_dims(next_state, axis=0))[0])
            target_full = self.model.predict(np.expand_dims(state, axis=0))
            target_full[0][action] = target
            states.append(state)
            targets.append(target_full)
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def decay_epsilon(self):
        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)


# Main training loop
def train(game):
    env = gym.make(game)
    agent = DQNAgent(env)
    episode_rewards = []

    for episode in range(NUM_EPISODES):
        state = preprocess(env.reset())
        state = np.stack((state, state, state, state), axis=0)
        total_reward = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = preprocess(next_state)
            next_state = np.reshape(next_state, (*next_state.shape, 1))
            next_state = np.append(state[1:, :, :], next_state, axis=0)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay(BATCH_SIZE)

            total_reward += reward
            if done:
                break

        episode_rewards.append(total_reward)
        agent.update_target_model()
        agent.decay_epsilon()

        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_model()

        if episode % 10 == 0:
            print(f"Episode: {episode}, Epsilon: {agent.epsilon}, Reward: {total_reward}")

    env.close()
    return episode_rewards


if __name__ == "__main__":
    # Set game to train
    game = 'ALE/Blackjack-v5'
    episode_rewards = train(game)
    print("Training Complete")