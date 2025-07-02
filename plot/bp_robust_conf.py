import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.stats.api as sms

# Total number of nodes in the network
total_nodes = 25

# Read all CSV files
data_sp = pd.read_csv('data/wax/beta_0.275/sp_nsd_5_8_10_12.csv')
data_ka = pd.read_csv('data/wax/beta_0.275/kx0_nsd_5_8_10_12.csv')
data_kx0 = pd.read_csv('data/wax/beta_0.275/kx0_lam1_nsd_5.csv')
data_kx1 = pd.read_csv('data/wax/beta_0.275/kx0_lam8_nsd_5.csv')
data_k10 = pd.read_csv('data/wax/beta_0.275/kx0_lam16_nsd_5.csv')

# List of HQ_percent values to plot
hq_percent_values = [i for i in range(48, 101, 4)]

xi_values = []
ci_sp, ci_ka, ci_kx0, ci_kx1, ci_k10 = [], [], [], [], []

# Lists to store blocking probabilities and confidence intervals for different cases
blocking_probabilities_sp = []
blocking_probabilities_ka = []
blocking_probabilities_kx0 = []
blocking_probabilities_kx1 = []
blocking_probabilities_k10 = []

for hq_percent in hq_percent_values:
    xi = (hq_percent / 100.0)
    xi_values.append(xi)
    
    # Define a helper function to compute probabilities and confidence intervals
    def compute_probabilities_and_ci(filtered_df):
        filtered_df = filtered_df[filtered_df['n_sd'] == 5]
        # Identifying blocked paths by checking for NaN values in fidelity
        filtered_df.loc[:, 'blocked'] = pd.isna(filtered_df['fidelity'])
        total_paths = len(filtered_df)
        blocked_paths = filtered_df['blocked'].sum()
        blocking_prob = blocked_paths / total_paths if total_paths > 0 else 0

        blocking_probs = filtered_df.groupby('HQ_seed')['blocked'].mean()  # Mean per seed
        stat = sms.DescrStatsW(blocking_probs)
        ci = stat.tconfint_mean(alpha=0.05)
        ci_rounded = (round(ci[0], 4), round(ci[1], 4))

        return blocking_prob, ci_rounded

    # Calculate probabilities and confidence intervals for each scenario
    prob_sp, ci_sp_val = compute_probabilities_and_ci(data_sp[data_sp['HQ_percent'] == hq_percent])
    prob_ka, ci_ka_val = compute_probabilities_and_ci(data_ka[data_ka['HQ_percent'] == hq_percent])
    prob_kx0, ci_kx0_val = compute_probabilities_and_ci(data_kx0[data_kx0['HQ_percent'] == hq_percent])
    prob_kx1, ci_kx1_val = compute_probabilities_and_ci(data_kx1[data_kx1['HQ_percent'] == hq_percent])
    prob_k10, ci_k10_val = compute_probabilities_and_ci(data_k10[data_k10['HQ_percent'] == hq_percent])

    blocking_probabilities_sp.append(prob_sp)
    blocking_probabilities_ka.append(prob_ka)
    blocking_probabilities_kx0.append(prob_kx0)
    blocking_probabilities_kx1.append(prob_kx1)
    blocking_probabilities_k10.append(prob_k10)

    ci_sp.append(ci_sp_val)
    ci_ka.append(ci_ka_val)
    ci_kx0.append(ci_kx0_val)
    ci_kx1.append(ci_kx1_val)
    ci_k10.append(ci_k10_val)

# Plotting with confidence intervals
def plot_with_confidence(ax, xi, probs, ci, label, linestyle='-'):
    ax.plot(xi, probs, label=label, linestyle=linestyle)
    ax.fill_between(xi, [c[0] for c in ci], [c[1] for c in ci], alpha=0.1)

fig, ax = plt.subplots()
plot_with_confidence(ax, xi_values, blocking_probabilities_sp, ci_sp, 'Shortest-path')
plot_with_confidence(ax, xi_values, blocking_probabilities_ka, ci_ka, '$kx_{0}$ path-selection')
plot_with_confidence(ax, xi_values, blocking_probabilities_kx0, ci_kx0, '$kx_{0}$ path-selection (High inaccuracy, $\lambda = 1$)', linestyle='--')
plot_with_confidence(ax, xi_values, blocking_probabilities_kx1, ci_kx1, '$kx_{0}$ path-selection (Moderate inaccuracy, $\lambda = 8$)', linestyle='--')
plot_with_confidence(ax, xi_values, blocking_probabilities_k10, ci_k10, '$kx_{0}$ path-selection (Low inaccuracy, $\lambda = 16$)', linestyle='--')

ax.set_yscale('log')
ax.set_xlabel(r'$ \xi $')
ax.set_ylabel('$Blocking \ Probability$')
#ax.set_title('a)')
ax.legend()
ax.grid(True)
ax.set_xlim(0.5, 1)
ax.set_ylim(bottom=0.01, top=1)
plt.show()
plt.savefig('doc/bp_robust_conf_nsd5.pdf')
