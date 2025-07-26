
import pickle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    candidate_ns = [10, 50, 100]

    # Load the results
    with open('results.pickle', 'rb') as f:
        results = pickle.load(f)

    # Plotting

    fig, ax = plt.subplots()
    ax.plot(results['randomized'][0], results['randomized'][1], label = "R-WnA", linestyle='-', markersize=4)
    ax.plot(results['deterministic'][0], results['deterministic'][1], label = "D-WnA", linestyle='--', markersize=4)

    for candidate_n in candidate_ns:
        ax.plot(
            results[f'cont_relax_{candidate_n}'][0],
            results[f'cont_relax_{candidate_n}'][1],
            label = f"Cont. Relaxation n={candidate_n}",
            linestyle=':',
            linewidth=2,
        )


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Consistency", size = 15)
    ax.set_ylabel("Robustness", size = 15)
    ax.legend(loc='lower left')

    fig.savefig('plot.png')
