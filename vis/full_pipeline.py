import os
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 24})

from vis_util import read_exp, PLOT_DATA_ANNOTATIONS
os.makedirs('assets', exist_ok=True)

policies = ['eva_vos', 'oracle_oracle_3clicks_mask', 'rand_rand_3clicks_mask', 'rand_type_3clicks', 'rand_mask']
fig, ax = plt.subplots(figsize=(30,15), tight_layout=True)
ax.axhline(y=0.85, xmin=0, xmax=300, color='#000075', linestyle = ':', linewidth=4)
ax.text(12, 0.86, 'J & F = 0.85', color='#000075')
for policy in policies:
    annotation_time, metric = read_exp(f'./Experiments/MOSE/{policy}.csv')
    color, linestyle, label = PLOT_DATA_ANNOTATIONS[policy]
    ax.plot(annotation_time, metric, linestyle=linestyle, color=color, linewidth=5, label=label)

ax.set_ylabel('J & F', fontsize=24)
ax.set_xlabel('time (hours)', fontsize=24)  
ax.set_title('Full Pipeline', fontsize=28)  
ax.set_xscale('log')
fig.legend(loc='lower right', ncol=2, bbox_to_anchor=(0.992, 0.07))
fig.savefig('assets/2.full_pipeline.png', bbox_inches='tight', dpi=500)
plt.close(fig)