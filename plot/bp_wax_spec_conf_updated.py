import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms

# Define the folder where the data files are located
folder_path = 'data/wax_spectrum/'

# Define the beta values and approaches
betas = [0.275, 0.375, 0.475, 0.575, 0.675, 0.775, 0.875, 0.975]
#approaches = ['sp', 'kx0', 'kx1', 'k10', 'k50', 'k50x0', 'k50x1']
approaches = ['sp', 'k50x0', 'k50x1', 'k50']

# Generate file paths for each approach and beta
files = {approach: {beta: f'{folder_path}{approach}_{beta}.csv' for beta in betas} for approach in approaches}

# Label definitions for each approach
labels = {
    'sp': 'Shortest-path',
    #'kx0': r'$k_{10}x_{0}$ path-selection',
    #'kx1': r'$k_{10}x_{1}$ path-selection',
    #'k10': r'$k_{10}$ shortest-path',
    'k50': r'$k_{50}$ shortest-path',
    'k50x0': r'$k_{50}x_{0}$ path-selection',
    'k50x1': r'$k_{50}x_{1}$ path-selection'
}

def calculate_blocking_probabilities_and_ci(files_for_approach):
    betas = []
    blocking_probabilities = []
    confidence_intervals = []
    for beta, file_path in files_for_approach.items():
        data = pd.read_csv(file_path)
        data['blocked'] = pd.isna(data['fidelity'])
        total_rows = len(data)
        blocked_count = data['blocked'].sum()
        blocking_probability = blocked_count / total_rows
        blocking_probs_per_seed = data.groupby('eff_seed')['blocked'].mean()

        stat = sms.DescrStatsW(blocking_probs_per_seed)
        ci = stat.tconfint_mean(alpha=0.05)
        ci_rounded = (round(ci[0], 4), round(ci[1], 4))

        betas.append(beta)
        blocking_probabilities.append(blocking_probability)
        confidence_intervals.append(ci_rounded)
    return betas, blocking_probabilities, confidence_intervals

# Plotting setup
plt.figure(figsize=(6.4, 4.8))
colors = ['b', 'r', 'g', 'm']#, 'c', 'k', 'lime'] 
markers = ['o', 'x', '^', 's']#, 'd', 'p', '*'] 
styles = ['-', '--', '-.', ':']#, '-', '--', '-.'] 

# Calculate and plot for each approach
for idx, approach in enumerate(approaches):
    betas, blocking_probabilities, confidence_intervals = calculate_blocking_probabilities_and_ci(files[approach])
    plt.plot(betas, blocking_probabilities, marker=markers[idx], linestyle=styles[idx], color=colors[idx], label=labels[approach])
    plt.fill_between(betas, [ci[0] for ci in confidence_intervals], [ci[1] for ci in confidence_intervals], color=colors[idx], alpha=0.1)

# Finalizing the plot
plt.xlabel(r'$\beta$')
plt.ylabel('Blocking Probability')
#plt.title(r'b)')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('doc/bp_wax_spec_conf_updated.pdf')
