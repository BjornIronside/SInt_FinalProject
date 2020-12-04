import blackjack
from itertools import product
import random
import time
import visualization


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


def InitializeQ():
    # Initialize Q matrix as [0, 0] for each possible state
    return {state: [0, 0] for state in
            product(range(12, 22), range(1, 11), [True, False])}


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


def main():
    tic = time.time()
    Q, winrates = QLearning(eps=0.05, step_size=0.1, niter=1000000, natural=False)
    toc = time.time()
    print('Elapsed time: ', toc - tic, ' s')
    policy = visualization.getOptimalPolicy(Q)
    EvaluatePolicy(policy)
    visualization.showPolicy(Q)
    visualization.LearningProgess(winrates)


if __name__ == '__main__':
    main()
