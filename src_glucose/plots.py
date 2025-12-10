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
    plot_patient_example = False
    plot_padova_results = False

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
            next_patient_idx = np.roll(np.where(dones)[0], -1)[patient_idx_in_batch]
            if next_patient_idx < current_patient_idx:
                next_patient_idx = -1

            return {
                'time': hour_float[current_patient_idx: next_patient_idx],
                'bg': blood_glucose[current_patient_idx: next_patient_idx],
                'insulin': insulin_acts[current_patient_idx: next_patient_idx],
                'cho': cho[current_patient_idx: next_patient_idx],
                'dataset_num': dataset_number
            }


        def draw_panel(ax, data, y_max, show_legend=True, is_bottom_plot=True,
                       explicit_xlim=None, explicit_xticks=None):
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

            # Plot Image
            ax.imshow(
                gradient_image, origin='lower', extent=[x_min, x_max, y_min, y_max],
                aspect='auto', zorder=1
            )

            # --- B. Plot Lines and Targets (remains the same) ---
            LOWER_TARGET, HIGHER_TARGET = 70, 180
            PEAK_TARGET = minimize_scalar(lambda x: -bg_in_range_magni([x]), bounds=(70, 180),
                                          method='bounded').x.item()

            ax.axhline(LOWER_TARGET, color='gray', linestyle='--', linewidth=1, zorder=5, label='Target range (70-180)')
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
                    pt_time[cho_mask], cho_vals, linefmt='purple', markerfmt='D', basefmt=' ', label='Carbohydrates (g)'
                )
                plt.setp(markerline, markersize=5, color='purple', zorder=9)
                plt.setp(stemlines, linewidth=1.5, color='purple', zorder=9)

            ax2 = None
            if y_max == 250:
                ax2 = ax.twinx()
                ax2.step(pt_time, pt_insulin, color='blue', linestyle=':', linewidth=2, label='Insulin rate', zorder=8,
                         where='post')
                ax2.set_ylabel('Insulin (U/min)', color='blue', fontsize=10)
                ax2.tick_params(axis='y', labelcolor='blue', labelsize=9)
                ax2.set_ylim(0, 0.3)
                ax2.grid(False)

            # --- C. Formatting & TICKS ---
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(x_min, x_max)
            ax.grid(True, which='both', linestyle=':', alpha=0.3)

            # Handle X-Ticks
            def hour_formatter(x, pos):
                return f'{int(x % 24)}'

            ax.xaxis.set_major_formatter(mticker.FuncFormatter(hour_formatter))

            if explicit_xticks is not None:
                ax.set_xticks(explicit_xticks)
            else:
                ax.set_xticks(np.arange(np.ceil(x_min), x_max, 3))

            if is_bottom_plot:
                ax.set_xlabel('Time (Hour of Day)', fontsize=12)
            else:
                # Hide x-labels for the top plot to clean up the look
                ax.tick_params(labelbottom=False)

            ax.set_ylabel('Glucose (mg/dL)', fontsize=12)

            # Legend Logic
            if show_legend:
                h1, l1 = ax.get_legend_handles_labels()
                if ax2:
                    h2, l2 = ax2.get_legend_handles_labels()
                    ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=9, framealpha=0.9)
                else:
                    ax.legend(h1, l1, loc='upper left', fontsize=9)

            return norm, cmap, v_min, v_max


        # ==============================================================================
        # EXECUTION
        # ==============================================================================

        # 1. TASK ONE: Standalone Plot (Dataset 0, Y_MAX=700)
        # ---------------------------------------------------
        print("Generating Standalone Plot (Dataset 0, Y=700)...")
        data_0 = get_patient_data(0)

        fig1, ax1 = plt.subplots(figsize=(12, 6))
        norm, cmap, vmin, vmax = draw_panel(ax1, data_0, y_max=700)

        ax1.set_title('Simulated Patient Trajectory (Glucose Only)', fontsize=16)

        # Add Colorbar
        neg_ticks = list(np.arange(np.floor(vmin / 20) * 20, 1, 20))
        final_ticks = sorted(list(set(neg_ticks + list(np.arange(0, vmax, 10)))))

        cax1 = fig1.add_axes([0.92, 0.15, 0.02, 0.7])
        mappable1 = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig1.colorbar(mappable1, cax=cax1, orientation='vertical', ticks=final_ticks).set_label('Reward Value',
                                                                                                fontsize=12)
        fig1.subplots_adjust(right=0.90)

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
        fig2, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        # sharex=True is helpful, but since we are forcing ticks manually,
        # passing explicit_xlim is the safest way to ensure the image background fills correctly.

        # Plot Top (Dataset 0)
        norm, cmap, vmin, vmax = draw_panel(
            ax_top,
            data_0,
            y_max=250,
            is_bottom_plot=False,
            explicit_xlim=(common_min, common_max),
            explicit_xticks=common_ticks
        )
        ax_top.set_title(f'Unprocessed Dataset', fontsize=14, loc='left')

        # Plot Bottom (Dataset 3)
        draw_panel(
            ax_bot,
            data_3,
            y_max=250,
            is_bottom_plot=True,
            show_legend=False,
            explicit_xlim=(common_min, common_max),
            explicit_xticks=common_ticks
        )
        ax_bot.set_title(f'Binned (2hr) Dataset', fontsize=14, loc='left')

        # --- 4. Shared Colorbar (Same as before) ---
        fig2.subplots_adjust(right=0.83, hspace=0.1)  # Reduced hspace since x-axes match now
        cax2 = fig2.add_axes([0.88, 0.15, 0.02, 0.7])

        neg_ticks = list(np.arange(np.floor(vmin / 20) * 20, 1, 20))
        final_ticks = sorted(list(set(neg_ticks + list(np.arange(0, vmax, 10)))))

        mappable2 = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig2.colorbar(mappable2, cax=cax2, orientation='vertical', ticks=final_ticks)
        cbar.set_ticklabels([f'{t:.0f}' for t in final_ticks])
        cbar.set_label('Reward Value', fontsize=12)
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
