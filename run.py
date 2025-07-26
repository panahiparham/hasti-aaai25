import math
import pickle

import cvxpy as cp
import numpy as np
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm

def integral_first(x, original_opt, r, c):
    sum_term = 0
    for i in range(1, r + 1):
        sum_term += math.log(1 / x) ** i / math.factorial(i)

    nom = (1 / c - 1) * original_opt
    denom = x * sum_term - original_opt
    return nom / denom

def calc_original_opt(r):
    a_star = math.pow(
        math.factorial(r),
        1 / r,
    )

    original_opt = 0
    for i in range(1, r + 1):
        original_opt = original_opt + (a_star) ** i / math.factorial(i)

    return original_opt * math.pow(math.e, -1.0 * a_star)

def calc_end_points(x, k, r):
    term = 0
    for i in range(1, r + 1):
        term = term + (np.log(1 / x) ** i / math.factorial(i))
    return x * term - k

def calc_consistency_robustness(r, c, rand):
    original_opt = calc_original_opt(r)

    robustness = (1 / c) * original_opt
    points = (1 / c) * original_opt

    end_points = scipy.optimize.fsolve(calc_end_points, [0.001, 0.999], args=(points, r))
    L = end_points[1] - end_points[0]
    integral_result = scipy.integrate.quad(integral_first, 0, end_points[0], args=(original_opt, r, c))
    before_I_start = [integral_result[0] , 0]
    if rand:
        before_I_start = integral_result[0]
    else:
        before_I_start = 0
    consistency = before_I_start + L + 1 / c * original_opt

    return consistency, robustness

def get_consistency_robustness(randomized: bool):
    x = []
    y = []
    for c in np.arange(1, 5, 0.2):
        consistency, robustness = calc_consistency_robustness(1, c, randomized)
        x.append(consistency)
        y.append(robustness)

    return x, y

def get_continuous_relaxation(candidate_n: int):
    x_vals = np.linspace(1 / candidate_n, 1, candidate_n)
    dx = 1 / candidate_n

    avg_x = []
    avg_y = []

    progress = tqdm(np.arange(1.2, 5, 0.2))
    for lam in progress:
        progress.set_postfix({'lambda': lam})

        win_probs = []
        for idx, t in enumerate(x_vals):
            p_x = cp.Variable(candidate_n)

            objective = cp.Maximize(t * p_x[idx])

            constraints = [
                cp.sum(cp.multiply(x_vals, p_x)) * dx >= 1 / (lam * np.e)
            ]
            for i in range(candidate_n):
                cumulative_sum = cp.sum(p_x[:i+1]) * dx
                constraints.append(x_vals[i] * p_x[i] <= 1 - cumulative_sum)
            constraints.append(p_x >= 0)

            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.SCS, verbose=False)

            if p_x.value is not None:
                win_probs.append(t * p_x.value[idx])
            else:
                win_probs.append(0)

        avg_x.append(np.mean(win_probs))
        avg_y.append(1 / (lam * np.e))

    return avg_x, avg_y


if __name__ == '__main__':
    candidate_ns = [10, 50, 100]

    # Run the experiment
    results = {}

    print('Computing consistency/robustness solutions...')
    results['randomized'] = get_consistency_robustness(randomized = True)
    results['deterministic'] = get_consistency_robustness(randomized = False)

    for candidate_n in candidate_ns:
        print(f'Computing continuous relaxation for n = {candidate_n}')
        results[f'cont_relax_{candidate_n}'] = get_continuous_relaxation(candidate_n)

    # Save the results
    with open('results.pickle', 'wb') as f:
        pickle.dump(results, f)

    # Plotting
    plt.plot(results['randomized'][0], results['randomized'][1], label = "R-WnA", linestyle='-', markersize=4)
    plt.plot(results['deterministic'][0], results['deterministic'][1], label = "D-WnA", linestyle='--', markersize=4)

    for candidate_n in candidate_ns:
        plt.plot(
            results[f'cont_relax_{candidate_n}'][0],
            results[f'cont_relax_{candidate_n}'][1],
            label = "Cont. Relaxation",
            linestyle=':',
            linewidth=2,
        )

    plt.title("Robustness Consistency Curves Deterministic/Randomized")
    plt.xlabel("Consistency", size = 15)
    plt.ylabel("Robustness", size = 15)
    plt.legend(fontsize = 10)
    plt.show()
