import blackjack
from itertools import product
import random
import time
import visualization
import matplotlib.pyplot as plt
import numpy as np


class QLearningAgent:
    def __init__(self, explore_policy='constant_epson', eps=0.05, step_size=0.1, discount_rate=1, natural=False,
                 eps_decay=0.9999, min_eps=0.001, show_every=10000, evaluate_iter=1000, temp_decay=0.9999, init_temp=20):
        self.lr = step_size
        self.discount_rate = discount_rate
        self.Q = self.initializeQ()
        self.winrates = []
        self.natural = natural
        self.show_every = show_every
        self.evaluate_iter = evaluate_iter
        self.env = blackjack.BlackjackEnv(natural=self.natural)
        self.n_sub_optimals = []
        self.min_eps = min_eps
        if explore_policy == 'constant_epson':
            self.explore_policy = self.e_greedy
            self.eps = eps
            self.eps_decay = 1
            self.temp = init_temp
            self.temp_decay = 1
        elif explore_policy == 'decay_epson':
            self.explore_policy = self.e_greedy
            self.eps = 1
            self.eps_decay = eps_decay
            self.temp = init_temp
            self.temp_decay = 1
        elif explore_policy == 'boltzmann_exploration':
            self.explore_policy = self.boltzmann_exploration
            self.eps = 1
            self.eps_decay = 1
            self.temp = init_temp
            self.temp_decay = temp_decay

    def e_greedy(self, state_values):
        best_action = state_values.index(max(state_values))
        roll = random.random()
        if roll < self.eps:
            action = random.randint(0, 1)
        else:
            action = best_action
        return action

    def boltzmann_exploration(self, state_values):
        state_values = np.array(state_values)
        exp_values = np.exp(state_values) / self.temp
        probs = exp_values / np.sum(exp_values)
        roll = random.random()
        if roll < probs[0]:
            action = 0
        else:
            action = 1
        return action

    def initializeQ(self):
        return {state: [0, 0] for state in
                product(range(12, 22), range(1, 11), [True, False])}

    def train(self, niter=10_000):
        self.Q = self.initializeQ()
        self.winrates = []
        env = 1
        self.n_sub_optimals = []
        for i in range(niter + 1):
            state = self.env.reset()
            done = False

            while not done:
                if state[0] < 12:
                    new_state, reward, done, _ = self.env.step(1)
                else:
                    state_values = self.Q[state]
                    action = self.explore_policy(state_values)
                    new_state, reward, done, _ = self.env.step(action)
                    if not done:
                        self.Q[state][action] = self.Q[state][action] + self.lr * (
                                reward + self.discount_rate * max(self.Q[new_state]) - self.Q[state][action])
                    else:
                        self.Q[state][action] = self.Q[state][action] + self.lr * (
                                reward + self.discount_rate * 0 - self.Q[state][action])

                state = new_state

            # decay eps
            if self.eps_decay != 1 and self.eps > self.min_eps:
                self.eps *= self.eps_decay
                self.eps = max(self.eps, self.min_eps)

            # temp decay
            if self.temp_decay != 1:
                self.temp *= self.temp_decay

            if i % self.show_every == 0:
                print('Iteration ', i)
                print('Eps: {}\nTemp: {}'.format(self.eps, self.temp))
                self.evaluate_policy()

        return self.Q, self.winrates, self.n_sub_optimals

    def get_best_policy(self):
        return {state: int(values[1] > values[0]) for state, values in self.Q.items()}

    def evaluate_policy(self):
        results = {-1: 0, 0: 0, 1: 0, 1.5: 0}
        policy = self.get_best_policy()
        game = blackjack.BlackjackEnv(natural=self.natural)
        for i in range(self.evaluate_iter):
            state = game.reset()
            done = False

            while not done:
                if state[0] < 12:
                    new_state, reward, done, _ = game.step(1)
                else:
                    action = policy[state]
                    new_state, reward, done, _ = game.step(action)
                state = new_state
            results[reward] += 1

        winrate = (results[1] + results[1.5]) / self.evaluate_iter * 100
        print('\nWin Rate: {:.2f} % ({} games)'.format(winrate, self.evaluate_iter))
        n_sub_optimal = visualization.compare2Optimal(policy)
        print('Suboptimal Actions: {}/200\n'.format(n_sub_optimal))
        self.winrates.append(winrate)
        self.n_sub_optimals.append(n_sub_optimal)


def TestParameters(niter=100000, natural=False):
    epss = [0.025 * i for i in range(1, 8)]
    step_sizes = [0.1 * i for i in range(1, 10)]
    results = {}
    for eps, step_size in product(epss, step_sizes):
        print('\n\nStarting Training for eps={:.3f} step_size={:.3f}'.format(eps, step_size))
        Q, winrates, n_sub_optimals = QLearning(eps=eps, step_size=step_size, niter=niter, natural=natural)
        results[(eps, step_size)] = {'Q': Q, 'winrates': winrates, 'n_sub_optimals': n_sub_optimals}

    print(results)


def main():
    tic = time.time()
    # Q, winrates, n_sub_optimals = QLearning(eps=0.05, step_size=0.1, niter=100000, natural=False)
    agent = QLearningAgent(explore_policy='boltzmann_exploration')
    print(agent.explore_policy)
    Q, winrates, n_sub_optimals = agent.train(niter=1_000_000)
    toc = time.time()
    print('Elapsed time: {:.4f} s'.format(toc - tic))
    policy = agent.get_best_policy()
    with plt.style.context('grayscale'):
        fig_policy = visualization.showPolicy(Q, policy)

    with plt.style.context('ggplot'):
        fig_learn = visualization.LearningProgess(winrates, n_sub_optimals)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
    # TestParameters()
