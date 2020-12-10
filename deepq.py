from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from collections import deque
import numpy as np
import tensorflow as tf
import time
import random
from tqdm import tqdm
import blackjack


class DQNAgent:
    MODEL_NAME = "4x10x2"
    DISCOUNT = 1
    REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
    MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
    MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
    UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
    EPSILON_DECAY = 0.99975
    MIN_EPSILON = 0.001
    AGGREGATE_STATS_EVERY = 50

    def __init__(self):
        # main model  # gets trained every step
        self.model = self.create_model()

        # Target model this is what we .predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)
        # self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{self.MODEL_NAME}-{int(time.time())}")

        self.target_update_counter = 0

        self.env = blackjack.BlackjackEnv(natural=False)
        self.ep_rewards = []
        self.epsilon = 1

    def create_model(self):
        model = Sequential()
        model.add(Dense(3, activation='relu', input_dim=3))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train_network(self, terminal_state):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_state_list = np.array([self.normalize_state(transition[0]) for transition in minibatch]).reshape(
            self.MINIBATCH_SIZE, 3)
        current_qs_list = self.model.predict(current_state_list)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        future_state_list = np.array([self.normalize_state(transition[3]) for transition in minibatch]).reshape(
            self.MINIBATCH_SIZE, 3)
        future_qs_list = self.target_model.predict(future_state_list)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(self.normalize_state(current_state))
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X).reshape(self.MINIBATCH_SIZE, 3), np.array(y), batch_size=self.MINIBATCH_SIZE,
                       verbose=0, shuffle=False)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        state = self.normalize_state(state)
        return self.model.predict(state)[0]

    def normalize_state(self, state):
        return np.array([(state[0] - 11) / 10, state[1] / 10, int(state[2])]).reshape(1, 3)

    def TrainAgent(self, n_episodes=50_000):
        self.epsilon = 1
        for episode in tqdm(range(1, n_episodes + 1), ascii=True, unit='episodes'):

            # Start game
            current_state = self.env.reset()
            done = False
            while not done:
                if np.random.random() > self.epsilon:
                    # Get action from Q table
                    action = np.argmax(self.get_qs(current_state))
                else:
                    # Get random action
                    action = np.random.randint(0, 1)

                new_state, reward, done, _ = self.env.step(action)

                # Every step we update replay memory and train main network
                self.update_replay_memory((current_state, action, reward, new_state, done))
                self.train_network(done)

                current_state = new_state

                # Append episode reward to a list and log stats (every given number of episodes)
            self.ep_rewards.append(reward)

            # Decay epsilon
            if self.epsilon > self.MIN_EPSILON:
                self.epsilon *= self.EPSILON_DECAY
                self.epsilon = max(self.MIN_EPSILON, self.epsilon)


def main():
    agent = DQNAgent()

    agent.TrainAgent(n_episodes=10_000)


if __name__ == '__main__':
    main()
