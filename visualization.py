import numpy as np
import matplotlib.pyplot as plt


def getOptimalPolicy(Q):
    return {state: int(values[1] > values[0]) for state, values in Q.items()}


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


def LearningProgess(winrates):
    winrates = np.array(winrates)
    x_iter = np.array([(i + 1) * 1000 for i in range(len(winrates))])

    plt.style.use('ggplot')
    plt.plot(x_iter, winrates)
    plt.title('Win Rate of the Last 1000 Games')
    plt.xlabel('Iteration')
    plt.ylabel('Win Rate %')
    plt.tight_layout()
    plt.show()
