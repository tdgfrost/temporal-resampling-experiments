import matplotlib as mpl
from matplotlib.legend_handler import HandlerBase

mpl.rcParams['figure.dpi'] = 300


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


if __name__ == "__main__":
    plot_patient_example = True
    plot_padova_results = False
    plot_fqe_results = False

    if plot_patient_example:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import matplotlib.ticker as mticker
        from simglucose.simulation.env import bg_in_range_magni
        from scipy.optimize import minimize_scalar
        from gym_wrappers import *
        from utils import *

        # Load the dataset
        replay_buffer_env = RecurrentReplayBufferEnv(make_glucose_env(), buffer_size=int(5e7))
        replay_buffer_env.load('./replay_buffer_train')


        # --- 0. Helper Functions (Assumes replay_buffer_env and bg_in_range_magni exist) ---

        def get_patient_data(dataset_number, patient_idx_in_batch=0):
            """Extracts and normalizes data for a specific dataset number."""
            obs = np.array(replay_buffer_env.observations[dataset_number])
            acts = np.array(replay_buffer_env.actions[dataset_number])
            dones = np.array(replay_buffer_env.dones[dataset_number])

            # De-normalize
            blood_glucose = obs[:, 0] * 590 + 10
            # insulin_acts = obs[:, 1] * 1
            insulin_acts = acts * 1
            cho = obs[:, 2] * 300
            hour_float = obs[:, 3] * 48

            # Slice specific patient episode
            # Note: Using logic from your snippet
            current_patient_idx = np.where(dones)[0][patient_idx_in_batch] + 1
            next_patient_idx = np.roll(np.where(dones)[0], -1)[patient_idx_in_batch] + 1

            if next_patient_idx < current_patient_idx:
                next_patient_idx = -1

            return {
                'time': hour_float[current_patient_idx: next_patient_idx],
                'bg': blood_glucose[current_patient_idx: next_patient_idx],
                'insulin': insulin_acts[current_patient_idx: next_patient_idx],
                'cho': cho[current_patient_idx: next_patient_idx],
                'dataset_num': dataset_number
            }


        def draw_panel(ax, ax_ins, data, y_max, show_legend=True, is_bottom_plot=True,
                       explicit_xlim=None, explicit_xticks=None, show_gradient=True, plot_type='insulin'):
            """
            Draws a single plot panel with optional forced x-limits/ticks.
            """
            pt_time = data['time']
            pt_bg = data['bg']
            pt_cho = data['cho']
            pt_insulin = data['insulin']

            # Use explicit limits if provided, otherwise calculate from data
            if explicit_xlim:
                x_min, x_max = explicit_xlim
            else:
                x_min, x_max = pt_time.min(), pt_time.max()

            y_min = 10

            # --- A. Gradient Background (remains the same) ---
            y_vals_for_gradient = np.linspace(y_min, y_max, 500)

            # ... [Insert gradient calculation logic from previous code here] ...
            # (Re-using the logic for brevity, ensure you copy the gradient block here)
            def f(x):
                return bg_in_range_magni([x])

            vectorized_reward_func = np.vectorize(f)
            rewards = np.array(list(map(vectorized_reward_func, y_vals_for_gradient.tolist())))
            v_min, v_max = rewards.min(), rewards.max()
            cmap = plt.get_cmap('RdYlGn')
            norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=v_min, vmax=v_max)
            colors_rgba = cmap(norm(rewards))
            max_alpha = 0.5
            alpha_vals = np.zeros_like(rewards)
            pos_mask = rewards > 0
            neg_mask = rewards < 0
            alpha_vals[pos_mask] = np.exp(rewards[pos_mask] / v_max) ** 0.1 * max_alpha
            alpha_vals[neg_mask] = np.exp(rewards[neg_mask] / v_min) ** 0.3 * max_alpha
            colors_rgba[:, 3] = alpha_vals
            gradient_image = colors_rgba.reshape(len(y_vals_for_gradient), 1, 4)
            gradient_image = np.tile(gradient_image, (1, 10, 1))

            if show_gradient:
                # Plot Image
                ax.imshow(
                    gradient_image, origin='lower', extent=[x_min, x_max, y_min, y_max],
                    aspect='auto', zorder=1
                )

            # --- B. Plot Lines and Targets (remains the same) ---
            LOWER_TARGET, HIGHER_TARGET = 70, 180
            PEAK_TARGET = minimize_scalar(lambda x: -bg_in_range_magni([x]), bounds=(70, 180),
                                          method='bounded').x.item()

            if y_max == 700:
                ax.axhline(LOWER_TARGET, color='gray', linestyle='--', linewidth=1, zorder=5, label='Target glucose range (70–180 mg/dL)')
                ax.axhline(HIGHER_TARGET, color='gray', linestyle='--', linewidth=1, zorder=5)
                ax.axhline(PEAK_TARGET, color='green', linestyle=':', linewidth=1, zorder=5, label='Peak reward')

            if data['dataset_num'] == 3:
                ax.step(pt_time, pt_bg, color='black', linewidth=1.5, zorder=10, label='Blood glucose', where='post')
            else:
                ax.plot(pt_time, pt_bg, color='black', linewidth=1.5, zorder=10, label='Blood glucose')

            if y_max == 250:
                cho_mask = pt_cho > 0
                cho_vals = pt_cho[cho_mask]
                if data['dataset_num'] == 3: cho_vals *= 12
                markerline, stemlines, baseline = ax.stem(
                    pt_time[cho_mask], cho_vals, linefmt='purple', markerfmt='D', basefmt=' ', label='Carbohydrates (grams)'
                )
                plt.setp(markerline, markersize=5, color='purple', zorder=9)
                plt.setp(stemlines, linewidth=1.5, color='purple', zorder=9)

            ax2 = ax_ins
            if ax2 is not None:
                if plot_type == 'insulin':
                    ax2.step(pt_time, pt_insulin, color='blue', linestyle='-', linewidth=1.5, label='Insulin rate', zorder=8,
                             where='post')
                    ax2.set_ylabel('Insulin\n(U/min)', color='blue', fontsize=12)
                    ax2.tick_params(axis='y', labelcolor='blue', labelsize=10)
                    ins_max = pt_insulin.max() if len(pt_insulin) > 0 else 0.1
                    ax2.set_ylim(0, max(0.1, ins_max * 1.2))
                elif plot_type == 'reward':
                    rewards_array = vectorized_reward_func(pt_bg)
                    cum_rewards = np.cumsum(rewards_array)
                    ax2.plot(pt_time, cum_rewards, color='black', linestyle='-', linewidth=1.5, label='Cumulative reward', zorder=8)
                    ax2.set_ylabel('Cumulative\nreward', color='black', fontsize=12)
                    ax2.tick_params(axis='y', labelcolor='black', labelsize=10)

                ax2.grid(True, which='both', linestyle=':', alpha=0.3)

            # --- C. Formatting & TICKS ---
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(x_min, x_max)
            ax.grid(True, which='both', linestyle=':', alpha=0.3)

            # Handle X-Ticks
            def hour_formatter(x, pos):
                hour = int(x % 24)
                day = int((x - x_min) // 24) + 1
                return f'{hour:02d}:00\nDay {day}'

            ax.xaxis.set_major_formatter(mticker.FuncFormatter(hour_formatter))

            if explicit_xticks is not None:
                ax.set_xticks(explicit_xticks)
            else:
                ax.set_xticks(np.arange(np.ceil(x_min), x_max, 3))

            if ax2 is not None:
                ax2.xaxis.set_major_formatter(mticker.FuncFormatter(hour_formatter))
                if explicit_xticks is not None:
                    ax2.set_xticks(explicit_xticks)
                else:
                    ax2.set_xticks(np.arange(np.ceil(x_min), x_max, 3))
                ax2.set_xlim(x_min, x_max)

            ax.set_ylabel('Glucose\n(mg/dL)', fontsize=12)
            ax.tick_params(labelbottom=False)

            if ax2 is not None:
                if is_bottom_plot:
                    ax2.set_xlabel('Time', fontsize=12)
                else:
                    ax2.tick_params(labelbottom=False)
            else:
                if is_bottom_plot:
                    ax.set_xlabel('Time', fontsize=12)
                    ax.tick_params(labelbottom=True)

            # Legend Logic
            if show_legend:
                h1, l1 = ax.get_legend_handles_labels()
                ax.legend(h1, l1, loc='upper left', fontsize=9, framealpha=0.9)
                if ax2 is not None:
                    h2, l2 = ax2.get_legend_handles_labels()
                    ax2.legend(h2, l2, loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=9, framealpha=0.9)

            return norm, cmap, v_min, v_max


        # ==============================================================================
        # EXECUTION
        # ==============================================================================

        # 1. TASK ONE: Standalone Plot (Dataset 0, Y_MAX=700)
        # ---------------------------------------------------
        import matplotlib.gridspec as gridspec
        print("Generating Standalone Plot (Dataset 0, Y=700)...")
        data_0 = get_patient_data(0)

        fig1 = plt.figure(figsize=(12, 6))
        gs1 = fig1.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.0)
        ax1 = fig1.add_subplot(gs1[0])
        ax1_ins = fig1.add_subplot(gs1[1], sharex=ax1)
        norm, cmap, vmin, vmax = draw_panel(ax1, ax1_ins, data_0, y_max=700, plot_type='reward')

        ax1.set_title('Simulated Patient Trajectory (Glucose/Rewards Only)', fontsize=16)

        # Add Colorbar
        neg_ticks = list(np.arange(np.floor(vmin / 20) * 20, 1, 20))
        final_ticks = sorted(list(set(neg_ticks + list(np.arange(0, vmax, 10)))))

        cax1 = fig1.add_axes([0.08, 0.15, 0.02, 0.7])
        mappable1 = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar1 = fig1.colorbar(mappable1, cax=cax1, orientation='vertical', ticks=final_ticks)
        cbar1.ax.yaxis.set_ticks_position('left')
        cbar1.ax.yaxis.set_label_position('left')
        cbar1.set_label('Glucose Reward Value', fontsize=12)
        fig1.subplots_adjust(left=0.18)

        plt.savefig(f'../logs_glucose/patient_simulator_example_glucose_only.png', dpi=1200)
        plt.show()

        # 2. TASK TWO: Combined Vertical Plot (Dataset 0 & 3, Y_MAX=250)
        # ---------------------------------------------------
        print("Generating Combined Plot (Dataset 0 & 3, Y=250)...")
        data_3 = get_patient_data(3)

        # --- 1. Calculate Common Axes Limits ---
        # We take the earliest start time and the latest end time across BOTH datasets
        common_min = min(data_0['time'].min(), data_3['time'].min())
        common_max = max(data_0['time'].max(), data_3['time'].max())

        # --- 2. Create Common Ticks ---
        # This ensures the grid lines line up perfectly vertically.
        # We start at the ceiling of the min to ensure we hit an integer hour.
        common_ticks = np.arange(np.ceil(common_min), common_max, 3)

        # --- 3. Plotting ---
        fig2 = plt.figure(figsize=(14, 10))
        outer_gs = fig2.add_gridspec(2, 1, hspace=0.3)

        top_gs = outer_gs[0].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.0)
        ax_top = fig2.add_subplot(top_gs[0])
        ax_top_ins = fig2.add_subplot(top_gs[1], sharex=ax_top)

        bot_gs = outer_gs[1].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.0)
        ax_bot = fig2.add_subplot(bot_gs[0], sharex=ax_top)
        ax_bot_ins = fig2.add_subplot(bot_gs[1], sharex=ax_top)

        # Plot Top (Dataset 0)
        norm, cmap, vmin, vmax = draw_panel(
            ax_top, ax_top_ins,
            data_0,
            y_max=250,
            is_bottom_plot=False,
            explicit_xlim=(common_min, common_max),
            explicit_xticks=common_ticks,
            show_gradient=False
        )

        # Override Top Legends to Upper Right
        ax_top.get_legend().remove()
        ax_top_ins.get_legend().remove()
        h1, l1 = ax_top.get_legend_handles_labels()
        ax_top.legend(h1, l1, loc='upper right', fontsize=9, framealpha=0.9)
        h2, l2 = ax_top_ins.get_legend_handles_labels()
        ax_top_ins.legend(h2, l2, loc='upper right', bbox_to_anchor=(0.95, 1.0), fontsize=9, framealpha=0.9)

        ax_top.set_title(f'Unprocessed Dataset', fontsize=14, loc='left')

        # Plot Bottom (Dataset 3)
        draw_panel(
            ax_bot, ax_bot_ins,
            data_3,
            y_max=250,
            is_bottom_plot=True,
            show_legend=False,
            explicit_xlim=(common_min, common_max),
            explicit_xticks=common_ticks,
            show_gradient=False
        )
        ax_bot.set_title(f'Binned (2hr) Dataset', fontsize=14, loc='left')
        ax_bot_ins.set_ylim(0, 0.12)
        ax_top_ins.set_ylim(0, 0.12)

        plt.suptitle('Simulated Patient Trajectory', fontsize=20, y=0.93)

        plt.savefig('../logs_glucose/patient_simulator_example'
                    '.png', dpi=1200)
        plt.show()

    if plot_padova_results:
        import matplotlib.pyplot as plt
        import polars as pl
        import seaborn as sns
        import matplotlib.patches as mpatches

        # 1. Load Data
        df = pl.read_csv('../logs_glucose/iql_logs/final_results_normalised.csv')
        df = df.with_columns(pl.col('algo').str.to_uppercase())

        # 2. Extract PPO Baseline Values
        ppo_stats = df.filter((pl.col('algo') == 'PPO') & (pl.col('decoy_interval') == 0))
        ppo_irr_lower = ppo_stats.select('online_irregular_2.5%').item()
        ppo_irr_upper = ppo_stats.select('online_irregular_97.5%').item()
        ppo_reg_lower = ppo_stats.select('online_interpolated_2.5%').item()
        ppo_reg_upper = ppo_stats.select('online_interpolated_97.5%').item()

        # 3. Process Main Data
        target_algos = ['BC', 'IQL', 'CQL']
        df_clean = (
            df
            .filter(pl.col('algo').is_in(target_algos))
            .with_columns(
                pl.col('decoy_interval')
                .cast(pl.Utf8)
                .replace({'0': 'Unprocessed', '1': 'Interpolated', '2': 'Binned (4hr)', '3': 'Binned (2hr)'})
                .alias('Training Dataset')
            )
        )

        # Prepare Environments
        df_irr = df_clean.select([
            pl.col('algo'), pl.col('Training Dataset'),
            pl.lit('UVA/Padova Irregular').alias('Evaluation Environment'),
            pl.col('online_irregular_iqm').alias('Average Return'),
            pl.col('online_irregular_2.5%').alias('Lower'),
            pl.col('online_irregular_97.5%').alias('Upper'),
        ])

        df_reg = df_clean.select([
            pl.col('algo'), pl.col('Training Dataset'),
            pl.lit('UVA/Padova Regular').alias('Evaluation Environment'),
            pl.col('online_interpolated_iqm').alias('Average Return'),
            pl.col('online_interpolated_2.5%').alias('Lower'),
            pl.col('online_interpolated_97.5%').alias('Upper'),
        ])

        df_plot = pl.concat([df_irr, df_reg]).to_pandas()

        # 4. Plotting
        sns.set_theme(style="whitegrid")
        algo_order = ['BC', 'IQL', 'CQL']
        dataset_order = ['Unprocessed', 'Interpolated', 'Binned (2hr)', 'Binned (4hr)']
        env_order = ['UVA/Padova Irregular', 'UVA/Padova Regular']

        g = sns.catplot(
            data=df_plot,
            x='algo',
            y='Average Return',
            hue='Training Dataset',
            col='Evaluation Environment',
            kind='bar',
            height=3.5,
            aspect=1.4,
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
            if current_env == 'UVA/Padova Irregular':
                ax.axhspan(ppo_irr_lower, ppo_irr_upper, color='red', alpha=0.2)
            elif current_env == 'UVA/Padova Regular':
                ax.axhspan(ppo_reg_lower, ppo_reg_upper, color='red', alpha=0.2)

            # Error Bars
            for hue_idx, dataset in enumerate(dataset_order):
                if hue_idx >= len(ax.containers): continue
                container = ax.containers[hue_idx]
                subset = df_plot[
                    (df_plot['Evaluation Environment'] == current_env) & (df_plot['Training Dataset'] == dataset)]
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

        plt.savefig('../logs_glucose/iql_logs/final_plot.png', dpi=2000)
        plt.show()

    if plot_fqe_results:
        import matplotlib.pyplot as plt
        import polars as pl
        import seaborn as sns
        from matplotlib.lines import Line2D

        # 1. Load Data
        df = pl.read_csv('../logs_glucose/fqe_logs/fqe_results_normalised.csv')
        df = df.with_columns(pl.col('algo').str.to_uppercase())

        # Get error bars
        df = df.with_columns([
            (pl.col('ratio_iqm') - pl.col('ratio_2.5%')).alias('xerr_low'),
            (pl.col('ratio_97.5%') - pl.col('ratio_iqm')).alias('xerr_high'),
            (pl.col('true_ratio') - pl.col('true_ratio_2.5%')).alias('yerr_low'),
            (pl.col('true_ratio_97.5%') - pl.col('true_ratio')).alias('yerr_high'),
        ])

        # Remove random and dataset
        df = df.filter(~pl.col('algo').is_in(['RANDOM', 'DATASET']))

        # Map dataset names
        dataset_mapping = {
            0: "Unprocessed",
            1: "Interpolated",
            2: "Binned (4hr)",
            3: "Binned (2hr)"
        }

        df = df.with_columns(
            pl.col('decoy_interval')
            .cast(pl.Utf8)
            .replace({'0': 'Unprocessed', '1': 'Interpolated', '2': 'Binned (4hr)', '3': 'Binned (2hr)'})
            .alias('Training Dataset')
        )

        dataset_order = ['Unprocessed', 'Interpolated', 'Binned (2hr)', 'Binned (4hr)']
        algos = ['BC', 'IQL', 'CQL']

        # Get colour palette
        dataset_colors = sns.color_palette("deep", len(dataset_order))

        # Define Markers for Algorithms
        algo_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
        if len(algos) > len(algo_markers):
            algo_markers = algo_markers * (len(algos) // len(algo_markers) + 1)

        # Start plotting
        plt.figure(figsize=(10, 8))

        for d_idx, dataset in enumerate(dataset_order):
            for a_idx, algo in enumerate(algos):
                sub_df = df.filter(pl.col('Training Dataset') == dataset).filter(pl.col('algo') == algo).to_pandas()

                xerr = [sub_df['xerr_low'], sub_df['xerr_high']]
                yerr = [sub_df['yerr_low'], sub_df['yerr_high']]

                plt.errorbar(
                    x=sub_df['ratio_iqm'],
                    y=sub_df['true_ratio'],
                    xerr=xerr,
                    yerr=yerr,
                    fmt='none',
                    ecolor=dataset_colors[d_idx],
                    alpha=0.5,
                    capsize=3
                )

                plt.scatter(
                    sub_df['ratio_iqm'],
                    sub_df['true_ratio'],
                    color=dataset_colors[d_idx],
                    marker=algo_markers[a_idx],
                    s=120,
                    edgecolor='k',
                    zorder=10
                )

        # Create custom legend
        # Legend 1: Colors (Mapped Dataset Names with Custom Order)
        color_legend_elements = []
        for d_idx, dataset in enumerate(dataset_order):
            color = dataset_colors[d_idx]
            color_legend_elements.append(
                Line2D([0], [0], marker='o', color='w', label=dataset,
                       markerfacecolor=color, markersize=10, markeredgecolor='k')
            )

        # Legend 2: Markers (Algorithms)
        marker_legend_elements = [
            Line2D([0], [0], marker=marker, color='w', label=algo.upper(),
                   markerfacecolor='gray', markersize=10, markeredgecolor='k')
            for algo, marker in zip(algos, algo_markers[:len(algos)])
        ]

        # Add Identity Line
        min_val = min(df.select('ratio_2.5%').min().item(), df.select('true_ratio_2.5%').min().item())
        max_val = max(df.select('ratio_97.5%').max().item(), df.select('true_ratio_97.5%').max().item())
        padding = (max_val - min_val) * 0.1
        limit_min = min_val - padding
        limit_max = max_val + padding

        plt.plot([limit_min, limit_max], [limit_min, limit_max], 'k--', alpha=0.5, label='Perfect Calibration')

        plt.xlabel('Predicted Performance Ratio (relative to dataset policy)', fontsize=14)
        plt.ylabel('True Performance Ratio (relative to dataset policy)', fontsize=14)
        plt.title('Retrospective Evaluation vs True Performance', fontsize=16)

        # Add the text annotations
        text_offset_ratio = 0.25
        # Upper Left (Underestimated: y > x)
        plt.text(
            0.7,
            2.0,
            'Underestimated \n performance',
            fontsize=14,
            color='gray',
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5')
        )

        # Bottom Right (Overestimated: y < x)
        plt.text(
            3.3,
            2.0,
            'Overestimated \n performance',
            fontsize=14,
            color='gray',
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5')
        )

        # Add legends
        legend1 = plt.legend(
            handles=color_legend_elements,
            title="Training Dataset",
            loc='upper left',
            frameon=True,
            fontsize=12,
            title_fontsize=14
        )
        plt.gca().add_artist(legend1)
        plt.legend(
            handles=marker_legend_elements,
            title="Algorithm",
            loc='lower right',
            frameon=True,
            fontsize=12,
            title_fontsize=14
        )

        plt.grid(True, alpha=0.3)
        plt.xlim(limit_min, limit_max)
        plt.ylim(limit_min, limit_max)

        plt.tight_layout()
        plt.savefig('../logs_glucose/fqe_calibration_plot.png')
        plt.show()
