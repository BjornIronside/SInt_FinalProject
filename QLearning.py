import blackjack
from itertools import product
import random
import time
import visualization
import matplotlib.pyplot as plt


def QLearning(eps=0.1, step_size=0.1, niter=100000, discount_rate=1, natural=False):
    game = blackjack.BlackjackEnv(natural=natural)
    game_results = {-1: 0, 0: 0, 1: 0, 1.5: 0}
    winrates = []
    n_sub_optimals = []
    Q = InitializeQ()
    for i in range(niter + 1):
        state = game.reset()
        done = False

        while not done:
            if state[0] < 12:
                new_state, reward, done, _ = game.step(1)
            else:
                state_values = Q[state]
                action = eGreedy(state_values, eps)
                new_state, reward, done, _ = game.step(action)
                if not done:
                    Q[state][action] = Q[state][action] + step_size * (
                            reward + discount_rate * max(Q[new_state]) - Q[state][action])
                else:
                    Q[state][action] = Q[state][action] + step_size * (
                            reward + discount_rate * 0 - Q[state][action])

            state = new_state
        game_results[reward] += 1
        if i % 100000 == 0:
            print('Iteration ', i)
            policy = getOptimalPolicy(Q)
            last_winrate, last_n_sub_optimal = EvaluatePolicy(policy, natural=natural)
            winrates.append(last_winrate)
            n_sub_optimals.append(last_n_sub_optimal)

    return Q, winrates, n_sub_optimals


def eGreedy(state_values, eps):
    best_action = state_values.index(max(state_values))
    roll = random.random()
    if roll < eps:
        action = random.choice([0, 1])
    else:
        action = best_action
    return action


def InitializeQ():
    # Initialize Q matrix as [0, 0] for each possible state
    return {state: [0, 0] for state in
            product(range(12, 22), range(1, 11), [True, False])}


def EvaluatePolicy(policy, niter=10000, natural=False):
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

    winrate = (results[1] + results[1.5]) / niter * 100
    print('Win Rate: {:.2f} % ({} games)'.format(winrate, niter))
    n_sub_optimal = visualization.compare2Optimal(policy)
    print('Suboptimal Actions: {}/200\n'.format(n_sub_optimal))
    return winrate, n_sub_optimal


def getOptimalPolicy(Q):
    return {state: int(values[1] > values[0]) for state, values in Q.items()}


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
    Q, winrates, n_sub_optimals = QLearning(eps=0.05, step_size=0.1, niter=100000, natural=False)
    toc = time.time()
    print('Elapsed time: {:.4f} s'.format(toc - tic))
    policy = getOptimalPolicy(Q)
    EvaluatePolicy(policy)
    with plt.style.context('grayscale'):
        fig_policy = visualization.showPolicy(Q, policy)

    with plt.style.context('ggplot'):
        fig_learn = visualization.LearningProgess(winrates, n_sub_optimals)
    # plt.style.use('seaborn')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
    # TestParameters()
