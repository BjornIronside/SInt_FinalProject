import blackjack
from itertools import product
import random
import matplotlib.pyplot as plt
import numpy as np
import time


def QLearning(eps=0.1, step_size=0.1, niter=10000, discount_rate=1, natural=False):
    game = blackjack.BlackjackEnv(natural=natural)
    game_results = {-1: 0, 0: 0, 1: 0, 1.5: 0}
    winrates = []
    Q = InitializeQ()
    for i in range(niter):
        state = game.reset()
        done = False

        while not done:
            if state[0] < 12:
                new_state, reward, done, _ = game.step(1)
            else:
                state_values = Q[state]
                best_action = state_values.index(max(state_values))
                roll = random.random()
                if roll < eps:
                    action = random.choice([0, 1])
                else:
                    action = best_action
                new_state, reward, done, _ = game.step(action)
                if not done:
                    Q[state][action] = Q[state][action] + step_size * (
                            reward + discount_rate * max(Q[new_state]) - Q[state][action])
                else:
                    Q[state][action] = Q[state][action] + step_size * (
                            reward + discount_rate * 0 - Q[state][action])

            state = new_state
        game_results[reward] += 1
        if i % 1000 == 0:
            print('Iteration ', i)
            last_winrate = (game_results[1] + game_results[1.5]) / 10
            winrates.append(last_winrate)
            print('Win rate of the last 1000 games: {:.2f} %\n'.format(last_winrate))
            game_results = {-1: 0, 0: 0, 1: 0, 1.5: 0}

    return Q, winrates


def getOptimalPolicy(Q):
    return {state: int(values[1] > values[0]) for state, values in Q.items()}


def InitializeQ():
    # Initialize Q matrix as [0, 0] for each possible state
    return {state: [0, 0] for state in
            product(range(12, 22), range(1, 11), [True, False])}


def showPolicy(Q):
    policy = getOptimalPolicy(Q)
    usable_ace = np.zeros([10, 10])
    values_usable_ace = np.zeros([10, 10])
    values_no_usable_ace = np.zeros([10, 10])
    no_usable_ace = np.zeros([10, 10])
    for state, action in policy.items():
        # Usable Ace
        if state[2]:
            usable_ace[-(state[0] - 11)][state[1] - 1] = action
            values_usable_ace[-(state[0] - 11)][state[1] - 1] = Q[state][action]
        # No Usable Ace
        else:
            values_no_usable_ace[-(state[0] - 11)][state[1] - 1] = Q[state][action]
            no_usable_ace[-(state[0] - 11)][state[1] - 1] = action

    y_tick_labels = [str(i + 12) for i in range(10)]
    x_tick_labels = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    y_tick_labels.reverse()

    plt.style.use('grayscale')

    fig, axs = plt.subplots(nrows=2, ncols=2)

    # Policy (Usable Ace)
    ax = axs[0, 0]
    ax.imshow(usable_ace)
    ax.set_xticks(range(10))
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticks(range(10))
    ax.set_yticklabels(y_tick_labels)
    ax.set_title('Usable Ace')
    ax.set_ylabel('Player Sum')
    ax.set_xlabel('Dealer Showing')

    # Policy (No Usable Ace)
    ax = axs[1, 0]
    ax.imshow(no_usable_ace)
    ax.set_xticks(range(10))
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticks(range(10))
    ax.set_yticklabels(y_tick_labels)
    ax.set_title('No Usable Ace')
    ax.set_ylabel('Player Sum')
    ax.set_xlabel('Dealer Showing')

    # Value Function (Usable Ace)
    ax = axs[0, 1]
    im = ax.imshow(values_usable_ace)
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(10))
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticks(range(10))
    ax.set_yticklabels(y_tick_labels)
    ax.set_title('Value Function\nUsable Ace')
    ax.set_ylabel('Player Sum')
    ax.set_xlabel('Dealer Showing')

    # Value Function (Usable Ace)
    ax = axs[1, 1]
    im = ax.imshow(values_no_usable_ace)
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(10))
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticks(range(10))
    ax.set_yticklabels(y_tick_labels)
    ax.set_title('Value Function\nNo Usable Ace')
    ax.set_ylabel('Player Sum')
    ax.set_xlabel('Dealer Showing')

    plt.tight_layout()
    plt.show()


def EvaluatePolicy(policy, niter=1000, natural=False):
    results = {-1: 0, 0: 0, 1: 0, 1.5: 0}
    game = blackjack.BlackjackEnv(natural=natural)
    for i in range(niter):
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

    print('Win Rate: {:.2f} %'.format((results[1] + results[1.5]) / niter * 100))


def LearningProgess(winrates):
    winrates = np.array(winrates)
    x_iter = np.array([(i+1)*1000 for i in range(len(winrates))])

    plt.style.use('ggplot')
    plt.plot(x_iter, winrates)
    plt.title('Win Rate of the Last 1000 Games')
    plt.xlabel('Iteration')
    plt.ylabel('Win Rate %')
    plt.tight_layout()
    plt.show()


def main():
    tic = time.time()
    Q, winrates = QLearning(eps=0.05, step_size=0.1, niter=1000000, natural=False)
    toc = time.time()
    print('Elapsed time: ', toc - tic, ' s')
    policy = getOptimalPolicy(Q)
    EvaluatePolicy(policy)
    showPolicy(Q)
    LearningProgess(winrates)


if __name__ == '__main__':
    main()
