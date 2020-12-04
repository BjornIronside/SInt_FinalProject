from itertools import product
import numpy as np
import blackjack
import visualization


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
    return Q, policy




def main():
    (Q, policy) = monteCarloControl()
    visualization.showPolicy(Q, policy)


def InitializeQ():
    # Initialize Q matrix as [0, 0] for each possible state
    return {state: [0, 0] for state in
            product(range(12, 22), range(1, 11), [True, False])}


if __name__ == '__main__':
    main()
