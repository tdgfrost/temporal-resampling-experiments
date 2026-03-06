import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerBase


# --- Custom Handler for Section Headers in Legend ---
class LegendTitleHandler(HandlerBase):
    def __init__(self, fontsize=14):
        super().__init__()
        self.fontsize = fontsize

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # We manually create the text artist.
        # We cannot rely on legend.get_title() here as it causes an AttributeError
        # because the title is not yet fully initialized when this runs.

        # xdescent is the left edge of the handle area.
        txt = plt.Text(xdescent, ydescent + height / 2, orig_handle.get_label(),
                       fontsize=self.fontsize, ha='left', va='center', color='black')
        return [txt]


# 1. Load Data
df = pl.read_csv('../logs/iql_minigrid_logs/final_results.csv')
df = df.with_columns(pl.col('algo').str.to_uppercase())

# 2. Extract PPO Baseline Values
ppo_stats = df.filter((pl.col('algo') == 'PPO') & (pl.col('decoy_interval') == 0))
ppo_irr_lower = ppo_stats.select('lavagap_1_3_eval_2.5%').item()
ppo_irr_upper = ppo_stats.select('lavagap_1_3_eval_97.5%').item()
ppo_reg_lower = ppo_stats.select('lavagap_1_1_eval_2.5%').item()
ppo_reg_upper = ppo_stats.select('lavagap_1_1_eval_97.5%').item()

# 3. Process Main Data
target_algos = ['BC', 'IQL', 'CQL']
df_clean = (
    df
    .filter(pl.col('algo').is_in(target_algos))
    .with_columns(
        pl.col('decoy_interval')
        .cast(pl.Utf8)
        .replace({'0': 'Unprocessed', '1': 'Interpolated', '2': 'Binned'})
        .alias('Training Dataset')
    )
)

# Prepare Environments
df_irr = df_clean.select([
    pl.col('algo'), pl.col('Training Dataset'),
    pl.lit('LavaGap Irregular').alias('Evaluation Environment'),
    pl.col('lavagap_1_3_eval_iqm').alias('Average Return'),
    pl.col('lavagap_1_3_eval_2.5%').alias('Lower'),
    pl.col('lavagap_1_3_eval_97.5%').alias('Upper'),
])

df_reg = df_clean.select([
    pl.col('algo'), pl.col('Training Dataset'),
    pl.lit('LavaGap Regular').alias('Evaluation Environment'),
    pl.col('lavagap_1_1_eval_iqm').alias('Average Return'),
    pl.col('lavagap_1_1_eval_2.5%').alias('Lower'),
    pl.col('lavagap_1_1_eval_97.5%').alias('Upper'),
])

df_plot = pl.concat([df_irr, df_reg]).to_pandas()

# 4. Plotting
sns.set_theme(style="whitegrid")
algo_order = ['BC', 'IQL', 'CQL']
dataset_order = ['Unprocessed', 'Interpolated', 'Binned']
env_order = ['LavaGap Irregular', 'LavaGap Regular']

g = sns.catplot(
    data=df_plot,
    x='algo',
    y='Average Return',
    hue='Training Dataset',
    col='Evaluation Environment',
    kind='bar',
    height=3.5,
    aspect=1.5,
    order=algo_order,
    hue_order=dataset_order,
    col_order=env_order,
    palette='viridis',
    alpha=0.9,
)

g.set_axis_labels("Algorithm", "Average Return", fontsize=14)
g.set_titles("{col_name}", size=16)
g.tick_params(labelsize=12)

# 5. Add PPO Bands and Error Bars
for ax_idx, ax in enumerate(g.axes.flat):
    current_env = env_order[ax_idx]

    # Bands
    if current_env == 'LavaGap Irregular':
        ax.axhspan(ppo_irr_lower, ppo_irr_upper, color='red', alpha=0.2)
    elif current_env == 'LavaGap Regular':
        ax.axhspan(ppo_reg_lower, ppo_reg_upper, color='red', alpha=0.2)

    # Error Bars
    for hue_idx, dataset in enumerate(dataset_order):
        if hue_idx >= len(ax.containers): continue
        container = ax.containers[hue_idx]
        subset = df_plot[(df_plot['Evaluation Environment'] == current_env) & (df_plot['Training Dataset'] == dataset)]
        subset = subset.set_index('algo').reindex(algo_order).reset_index()
        yerr = [subset['Average Return'] - subset['Lower'], subset['Upper'] - subset['Average Return']]
        x_coords = [bar.get_x() + bar.get_width() / 2 for bar in container]
        y_coords = [bar.get_height() for bar in container]
        ax.errorbar(x_coords, y_coords, yerr=yerr, fmt='none', c='black', capsize=4, elinewidth=1.5)

# 6. Consolidated Legend Logic
plt.subplots_adjust(right=0.8)

# A. Extract handles and labels from the automatic Seaborn legend
handles = g.legend.legend_handles
labels = [t.get_text() for t in g.legend.texts]

# B. Remove the default split-out legend
g.legend.remove()

# C. Add a spacer
handles.append(mpatches.Patch(alpha=0))
labels.append("")  # Empty line for spacing

# D. Add the "Reference" Header
# We create a special handle that carries the label "Reference"
# We pass an empty string to the main labels list so the text column is empty
ref_handle = mpatches.Patch(color='none', label='Reference')
handles.append(ref_handle)
labels.append("")

# E. Add the PPO Patch
red_patch = mpatches.Patch(color='red', alpha=0.2)
handles.append(red_patch)
labels.append("PPO 95% CI")

# F. Create the single combined legend with the Custom Handler
g.fig.legend(
    handles=handles,
    labels=labels,
    # Pass the desired fontsize (14) directly to the handler
    handler_map={ref_handle: LegendTitleHandler(fontsize=14)},
    loc='upper left',
    bbox_to_anchor=(0.82, 0.75),
    title="Training Dataset",
    frameon=True,
    fontsize=12,
    title_fontsize=14
)

plt.savefig('../logs/iql_minigrid_logs/final_plot.png', dpi=1200)#, bbox_inches='tight')
plt.show()
