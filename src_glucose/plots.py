import glob
import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

paths = sorted(glob.glob("../logs/iql_minigrid_logs/log_expectile=*_decoy=*.csv"))

# Read with polars for speed, then convert to pandas for seaborn
dfs = []
for p in paths:
    df_pl = pl.read_csv(p)
    # Extract metadata (expectile, decoy) from filename if not in columns already
    # If those columns already exist in the CSV (as per your example), this is not needed.
    # Below is safe: if present in file, it will simply overwrite with the same values.
    # Example file name pattern: log_expectile=0.5_decoy=0.csv
    import re
    m = re.search(r"expectile=([0-9.]+)_decoy=([0-9]+)", p)
    if m:
        exp_from_name = float(m.group(1))
        decoy_from_name = int(m.group(2))
        df_pl = df_pl.with_columns([
            pl.lit(exp_from_name).alias("expectile"),
            pl.lit(decoy_from_name).alias("decoy")
        ])
    dfs.append(df_pl)

df_pl = pl.concat(dfs, how="vertical")
cols = ["expectile", "decoy", "dataset_reward", "lavagap_1_3_eval", "lavagap_1_1_eval"]
df_pl = df_pl.select(cols)
decoy_map = {"0": "Real 1-3", "1": "Artificial 1-1", "2": "Artificial 2-2"}
df_pl = df_pl.with_columns(pl.col("decoy").cast(pl.Utf8).replace(decoy_map).alias("dataset"))

df_long_pl = df_pl.unpivot(
    index=["expectile", "decoy", "dataset", "dataset_reward"],
    on=["lavagap_1_3_eval", "lavagap_1_1_eval"],
    variable_name="task",
    value_name="score",
)
df_long_pl = df_long_pl.with_columns([pl.col('expectile').cast(pl.Utf8).replace({"0.5": "Cloning"})])
df = df_long_pl.to_pandas()

task_name_map = {
    "lavagap_1_3_eval": "LavaGap 1-3",
    "lavagap_1_1_eval":  "LavaGap 1-1",
}
df["task"] = df["task"].map(task_name_map)
# df["expectile"] = df["expectile"].astype(float)

# Fixed ordering
expectile_order = ["Cloning", "0.6", "0.7", "0.8", "0.9"]
dataset_order   = ['Real 1-3', 'Artificial 2-2', 'Artificial 1-1']
task_order      = ["LavaGap 1-3", "LavaGap 1-1"]

df["expectile"] = pd.Categorical(df["expectile"], categories=expectile_order, ordered=True)
df["dataset"]   = pd.Categorical(df["dataset"],   categories=dataset_order,   ordered=True)
df["task"]      = pd.Categorical(df["task"],      categories=task_order,      ordered=True)

# -----------------------------------------------------------
# 2) Summarize: mean and standard error across 10 repeats
# -----------------------------------------------------------
summary = (
    df.groupby(["expectile", "dataset", "task"], observed=True)
      .agg(mean_score=("score", "mean"),
           sem_score =("score", "sem"))     # standard error of the mean
      .reset_index()
)

# Optionally use 95% CI instead of SEM:
use_ci = False
if use_ci:
    # approximate normal 95% CI from SEM with n≈10 (1.96 * SEM)
    summary["yerr"] = 1.96 * summary["sem_score"]
else:
    summary["yerr"] = summary["sem_score"]

# -----------------------------------------------------------
# 3) Plot: nested bars (dataset outer grouping, task inner)
# -----------------------------------------------------------
sns.set_theme(context="talk", style="whitegrid")
fig, ax = plt.subplots(figsize=(12, 5.2))

# Geometry
x_positions = np.arange(len(expectile_order))  # one base position per expectile
group_width = 0.8                              # total width per expectile cluster
D = len(dataset_order)
T = len(task_order)
subgroup_spacing = 0.20  # 15% of each dataset slot reserved as empty gap
dataset_slot_width = group_width / D
bar_width = dataset_slot_width * (1 - subgroup_spacing) / T
offset_within_slot = dataset_slot_width * (1 - subgroup_spacing)

# Aesthetics: colors by task, hatches by dataset
task_palette = ['#1f78b4', '#b2df8a']
dataset_hatches = ["", "x", "."]  # length must match number of datasets

# Baseline reference
baseline = 0.75
ax.axhline(baseline, ls="--", lw=3.0, alpha=0.9, zorder=0, color='#f33')

# Draw bars
handles_task = []
handles_dataset = []
task_handles_once = {t: None for t in task_order}
dataset_handles_once = {d: None for d in dataset_order}

for d_idx, dataset in enumerate(dataset_order):
    for t_idx, task in enumerate(task_order):
        # positions for this (dataset, task) across expectiles
        xpos = (
            x_positions
            - group_width/2
            + d_idx * dataset_slot_width
            + t_idx * bar_width
            + bar_width/2
        )

        # Fetch y and yerr for this slice in expectile order
        sub = summary[(summary["dataset"] == dataset) & (summary["task"] == task)]
        sub = sub.set_index("expectile").reindex(expectile_order)  # ensure correct order
        y = sub["mean_score"].to_numpy()
        yerr = sub["yerr"].to_numpy()

        bars = ax.bar(
            xpos, y, yerr=yerr, width=bar_width,
            label=f"{task} | {dataset}",  # full label not used in final legends
            color=task_palette[t_idx],
            edgecolor="black",
            linewidth=0.8,
            hatch=dataset_hatches[d_idx],
            zorder=3,
            capsize=3
        )

        # Capture representative handles for separate legends
        if task_handles_once[task] is None:
            task_handles_once[task] = bars[0]
        if dataset_handles_once[dataset] is None:
            # Make a proxy artist for dataset hatch without relying on specific height
            proxy = plt.Rectangle((0,0),1,1,
                                  facecolor="white",
                                  edgecolor="black",
                                  hatch=dataset_hatches[d_idx],
                                  linewidth=0.8)
            dataset_handles_once[dataset] = proxy

# Axes cosmetics
ax.set_xticks(x_positions)
ax.set_xticklabels([str(e) for e in expectile_order])
ax.set_xlabel("IQL Expectile")
ax.set_ylabel("Performance (0–1)")
ax.set_ylim(0, 1)

# Build two legends: tasks (colors) and datasets (hatches)
task_legend = ax.legend(
    [task_handles_once[t] for t in task_order],
    task_order, title="LavaGap Environment", title_fontproperties={"weight": "bold"},
    loc="upper left", bbox_to_anchor=(1.0, 0.40), frameon=False
)
dataset_legend = ax.legend(
    [dataset_handles_once[d] for d in dataset_order],
    dataset_order, title="Dataset", title_fontproperties={"weight": "bold"},
    loc="upper left", bbox_to_anchor=(1.04, 0.95), frameon=False
)
ax.add_artist(task_legend)  # keep both legends

plt.ylim(0.35, 1.0)
ax.set_title("Live Returns with IQL after Time Binning (All Expectiles)", fontsize=20, fontweight="bold", pad=15)
os.makedirs('../figures', exist_ok=True)
fig.subplots_adjust(right=0.75, left=0.1, bottom=0.15)
plt.savefig('../figures/Time Binning All Expectiles.png', dpi=1500)
plt.show()


#### Flip so that the x-axis is the evaluation task nd the group is the dataset type

# If your summary expectile is categorical/string, match it as "0.7".
# If it's numeric, use == 0.7 instead.
mask_07 = summary["expectile"].astype(str) == "0.7"
sum07 = summary.loc[mask_07].copy()

# Ensure the desired plotting order
sum07["dataset"] = pd.Categorical(sum07["dataset"], categories=dataset_order, ordered=True)
sum07["task"] = pd.Categorical(sum07["task"], categories=task_order, ordered=True)

# Colors: replace these with your task hex codes
task_palette = ["#1f78b4", "#b2df8a"]

sns.set_theme(context="talk", style="whitegrid")
fig, ax = plt.subplots(figsize=(9, 4.8))

x = np.arange(len(task_order))
bar_width = 0.25
offsets = np.linspace(-bar_width, bar_width, len(dataset_order))  # three datasets

# Baseline: use dataset_reward for this expectile if available; otherwise default to 0.75
if "dataset_reward" in summary.columns:
    # This assumes dataset_reward is constant within expectile; otherwise it averages.
    baseline = float(df.loc[df["expectile"] == "0.7", "dataset_reward"].mean()) if not df.empty else 0.75
else:
    baseline = 0.75

# Horizontal baseline
ax.axhline(baseline, ls="--", lw=1.5, alpha=0.9, zorder=0, color='#f33')

ax.text(0.52, baseline + 0.01, f"Avg Dataset Return = {baseline:.2f}",
        ha="center", va="bottom", fontsize=12, color="red",
        transform=ax.get_yaxis_transform(),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

# Draw bars by task
handles = []
for d_idx, dataset in enumerate(dataset_order):
    sub = (sum07[sum07["dataset"] == dataset]
           .set_index("task")
           .reindex(task_order))  # align to tasks

    y = sub["mean_score"].to_numpy()
    yerr = sub["yerr"].to_numpy()

    bars = ax.bar(
        x + offsets[d_idx], y, yerr=yerr,
        width=bar_width,
        label=dataset,
        color=sns.color_palette("Set2")[d_idx],
        edgecolor="black",
        linewidth=0.8,
        capsize=3,
        zorder=3,
    )
    handles.append(bars[0])

# Axes cosmetics
ax.set_xticks(x)
ax.set_xticklabels(task_order)
ax.set_xlabel("LavaGap Environment", labelpad=15)
ax.set_ylabel("Average Return")
ax.set_ylim(0.0, 1.0)

ax.legend(handles, dataset_order, title="Dataset", frameon=True, fontsize=12)

for tick in ax.get_xticklabels():
    tick.set_fontstyle("italic")
    tick.set_fontsize(13)

for tick in ax.get_yticklabels():
    tick.set_fontsize(12)

# Title (bold)
ax.set_title("Live Returns with IQL after Time Binning",
             fontsize=18, fontweight="bold", pad=12)

# Legend (bold title)
leg = ax.legend(
    handles, dataset_order,
    title="Dataset",
    frameon=True,
    fontsize=15,
    loc=(1.05, 0.3)
)
leg.set_title("Dataset", prop={"weight": "bold", "size": 15})
for t in leg.get_texts(): t.set_fontsize(13)

plt.ylim(0.4, 1.0)
plt.grid(False)
ax.grid(True, axis="y")
plt.tight_layout()
plt.savefig('../figures/Time Binning.png', dpi=1500, bbox_inches='tight')
plt.show()