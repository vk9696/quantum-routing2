import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.stats.api as sms

# Total number of nodes in the network
total_nodes = 25

# Read all CSV files
data_sp_775 = pd.read_csv('data/wax/beta_0.775/sp_nsd_5.csv')
data_kx0_775 = pd.read_csv('data/wax/beta_0.775/kx0_nsd_5.csv')
data_sp_375 = pd.read_csv('data/wax/beta_0.375/sp_nsd_5.csv')
data_kx0_375 = pd.read_csv('data/wax/beta_0.375/kx0_nsd_5.csv')
data_sp_275 = pd.read_csv('data/wax/beta_0.275/sp_nsd_5_8_10_12.csv')
data_kx0_275 = pd.read_csv('data/wax/beta_0.275/kx0_nsd_5_8_10_12.csv')

data_sp_275 = data_sp_275[data_sp_275['n_sd'] == 5]
data_kx0_275 = data_kx0_275[data_kx0_275['n_sd'] == 5]

total_edges_275 = 55
total_edges_375 = 69
total_edges_775 = 133

# Define a helper function to compute probabilities and confidence intervals
def compute_probabilities_pe_and_ci(filtered_df, no_of_edges):
    filtered_df['blocked'] = pd.isna(filtered_df['fidelity'])
    blocking_prob = filtered_df.groupby('HQ_seed')['blocked'].mean()  # Mean per seed
    blocking_prob_per_edge = blocking_prob/no_of_edges
    stat = sms.DescrStatsW(blocking_prob_per_edge)
    ci = stat.tconfint_mean(alpha=0.05)
    ci_rounded = (round(ci[0], 4), round(ci[1], 4))
    return blocking_prob_per_edge.mean(), ci_rounded

# Lists to store blocking probabilities and confidence intervals for different cases
xi_values = [i / 100.0 for i in range(48, 101, 4)]
blocking_probs = {'sp_275': [], 'kx0_275': [], 'sp_375': [], 'kx0_375': [], 'sp_775': [], 'kx0_775': []}
cis = {'sp_275': [], 'kx0_275': [], 'sp_375': [], 'kx0_375': [], 'sp_775': [], 'kx0_775': []}

# Calculate probabilities and confidence intervals for each scenario
for hq_percent in range(48, 101, 4):
    prob_pe_sp_275, ci_sp_275_val = compute_probabilities_pe_and_ci(data_sp_275[data_sp_275['HQ_percent'] == hq_percent], total_edges_275)
    prob_pe_kx0_275, ci_kx0_275_val = compute_probabilities_pe_and_ci(data_kx0_275[data_kx0_275['HQ_percent'] == hq_percent], total_edges_275)
    prob_pe_sp_375, ci_sp_375_val = compute_probabilities_pe_and_ci(data_sp_375[data_sp_375['HQ_percent'] == hq_percent], total_edges_375)
    prob_pe_kx0_375, ci_kx0_375_val = compute_probabilities_pe_and_ci(data_kx0_375[data_kx0_375['HQ_percent'] == hq_percent], total_edges_375)
    prob_pe_sp_775, ci_sp_775_val = compute_probabilities_pe_and_ci(data_sp_775[data_sp_775['HQ_percent'] == hq_percent], total_edges_775)
    prob_pe_kx0_775, ci_kx0_775_val = compute_probabilities_pe_and_ci(data_kx0_775[data_kx0_775['HQ_percent'] == hq_percent], total_edges_775)

    blocking_probs['sp_275'].append(prob_pe_sp_275)
    blocking_probs['kx0_275'].append(prob_pe_kx0_275)
    blocking_probs['sp_375'].append(prob_pe_sp_375)
    blocking_probs['kx0_375'].append(prob_pe_kx0_375)
    blocking_probs['sp_775'].append(prob_pe_sp_775)
    blocking_probs['kx0_775'].append(prob_pe_kx0_775)

    cis['sp_275'].append(ci_sp_275_val)
    cis['kx0_275'].append(ci_kx0_275_val)
    cis['sp_375'].append(ci_sp_375_val)
    cis['kx0_375'].append(ci_kx0_375_val)
    cis['sp_775'].append(ci_sp_775_val)
    cis['kx0_775'].append(ci_kx0_775_val)

# Plotting with confidence intervals
fig, ax = plt.subplots(figsize=(6.4, 4.8))
linestyles = ['-', '--', '-.', ':']
markers = ['o', '^', 's', 'p', '*', 'D']
for idx, (key, color) in enumerate(zip(blocking_probs.keys(), plt.cm.tab10.colors)):
    prob = blocking_probs[key]
    ci = cis[key]
    beta = key.split('_')[1]
    method = 'Shortest-path' if 'sp' in key else '$kx_{0}$ path-selection'
    label = f"$\\beta$ = 0.{beta}  ({method})"
    ax.plot(xi_values, prob, label=label, linestyle=linestyles[idx % len(linestyles)], marker=markers[idx % len(markers)])
    ax.fill_between(xi_values, [c[0] for c in ci], [c[1] for c in ci], color=color, alpha=0.1)

ax.set_xlabel(r'$\xi$')
ax.set_ylabel('Blocking Probability per Edge')
ax.set_title(r'Random Topology (Waxman: $\alpha = 0.85$)')
ax.legend(loc='best')
ax.grid(True)
ax.set_xlim(0.5, 1)
plt.tight_layout()
plt.show()
plt.savefig('doc/bpe_wax_conf.pdf')
