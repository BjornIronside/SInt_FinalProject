from random import randint
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import blackjack


def monteCarloControl(ep=0.025, numIterations=100000000):
    game = blackjack.BlackjackEnv()
    Q = InitializeQ()
    agentSumSpace = [i for i in range(12, 22)]
    dealerShowCardSpace = [i + 1 for i in range(10)]
    agentAceSpace = [False, True]
    actionSpace = [0, 1]  # stick or hit

    numberOfVisits = {}
    policy = {}

    # Initialize Q returns and policy
    for total in agentSumSpace:
        for card in dealerShowCardSpace:
            for ace in agentAceSpace:
                policy[(total, card, ace)] = np.random.choice(actionSpace)
                for action in actionSpace:
                    numberOfVisits[((total, card, ace), action)] = 0

    gamma = 1

    for i in range(numIterations):

        state = game.reset()
        memory = []
        if i % 1000 == 0:
            print('starting episode', i)

        gameEnd = False

        while not gameEnd:
            if state[0] < 12:
                game.step(1)
                newState, reward, gameEnd, _ = game.step(action)
            else:
                rng = np.random.rand()
                if rng < ep:
                    if policy[state] == 1:
                        action = 0
                    elif policy[state] == 0:
                        action = 1
                else:
                    action = policy[state]
                newState, reward, gameEnd, _ = game.step(action)
                memory.append((state, action, reward))
            state = newState

        G = 0

        for state, action, reward in reversed(memory):
            G = gamma * G + reward
            numberOfVisits[state, action] += 1
            Q[state][action] = Q[state][action] + G / numberOfVisits[state, action]
            if Q[state][1] >= Q[state][0]:
                policy[state] = 1
            else:
                policy[state] = 0
    return (Q, policy)


def showPolicy(Q, policy):
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


def main():
    (Q, policy) = monteCarloControl()
    showPolicy(Q, policy)

def InitializeQ():
    # Initialize Q matrix as [0, 0] for each possible state
    return {state: [0, 0] for state in
            product(range(12, 22), range(1, 11), [True, False])}

if __name__ == '__main__':
    main()
