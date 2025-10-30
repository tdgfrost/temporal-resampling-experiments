import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib as mpl
import numpy as np
from simglucose.simulation.env import bg_in_range_magni

mpl.rcParams['figure.dpi'] = 300

from gym_wrappers import *
from utils import *

if __name__ == "__main__":
    plot_patient_example = True

    if plot_patient_example:
        # Load the dataset
        replay_buffer_env = RecurrentReplayBufferEnv(make_glucose_env(), buffer_size=int(5e7))
        replay_buffer_env.load('./replay_buffer')

        obs = np.array(replay_buffer_env.observations[0])
        dones = np.array(replay_buffer_env.dones[0])
        visibles = np.array(replay_buffer_env.visible_states[0])

        blood_glucose = obs[:, 0] * 590 + 10  # Scale back to mg/dL
        insulin_acts = obs[:, 1] * 30
        cho = obs[:, 2] * 300
        hour_float = obs[:, 3] * 48
        patient_idx = np.where(dones)[0][0]

        # --- 1. Create Mock Blood Glucose Data ---
        pt_time = hour_float[:patient_idx]
        pt_blood_glucose = blood_glucose[:patient_idx]
        pt_insulin_acts = insulin_acts[1:patient_idx]
        pt_insulin_acts_time = hour_float[0:patient_idx-1] # shift in line with CHO
        pt_cho = cho[:patient_idx]

        # --- 2. Define Reward Function Parameters ---
        LOWER_TARGET = 64.637
        HIGHER_TARGET = 194.807
        PEAK_TARGET = 113.0  # Skewed peak
        MAX_REWARD = 7.0  # The max positive reward at the peak

        # --- 3. Set Up Plot Boundaries --- # Set Y-axis limits
        # --- 3. Set Up Plot Boundaries --- # Set Y-axis limits
        for y_max in [250]:
            fig, ax1 = plt.subplots(figsize=(14, 7))
            x_min, x_max = pt_time.min(), pt_time.max()
            y_min = 10

            # --- 4. Create the Gradient Image ---
            # Create an array of 500 y-values from the bottom to the top of the plot
            # We will calculate the reward for each y-value
            y_vals_for_gradient = np.linspace(y_min, y_max, 500)


            # Vectorize the reward function so we can apply it to the whole array
            def f(x): return bg_in_range_magni([x])


            vectorized_reward_func = np.vectorize(f)
            rewards = np.array(list(map(vectorized_reward_func, y_vals_for_gradient.tolist())))

            # Find the min (max penalty) and max (max reward) for normalization
            v_min = rewards.min()
            v_max = rewards.max()

            # Create a diverging colormap centered at 0
            # vcenter=0 ensures 0 is yellow, negatives are red, positives are green
            cmap = plt.get_cmap('RdYlGn')
            norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=v_min, vmax=v_max)

            # Get the (R, G, B, A) color for each reward value
            # This is an array of shape (500, 4)
            colors_rgba = cmap(norm(rewards))

            # --- 5. Create the Alpha (Transparency) Gradient ---
            # We want alpha=0 at reward=0, and max_alpha at v_min and v_max
            max_alpha = 0.5  # Set max opacity (e.g., 60%)
            alpha_gamma = 0.5
            alpha_vals = np.zeros_like(rewards)

            # Get masks for positive and negative rewards
            pos_mask = rewards > 0
            neg_mask = rewards < 0

            # Calculate alpha for positive rewards (scales from 0 to max_alpha)
            alpha_vals[pos_mask] = np.exp(rewards[pos_mask] / v_max) ** 0.1 * max_alpha
            # Calculate alpha for negative rewards (scales from 0 to max_alpha)
            # (v_min is negative, so division results in a positive value)
            alpha_vals[neg_mask] = np.exp(rewards[neg_mask] / v_min) ** 0.3 * max_alpha

            # Apply this new alpha channel to our colors
            colors_rgba[:, 3] = alpha_vals

            # --- 6. Reshape Colors into a Plot-able Image ---
            # Reshape (500, 4) -> (500, 1, 4)
            # This is a 1-pixel-wide, 500-pixel-high image
            gradient_image = colors_rgba.reshape(len(y_vals_for_gradient), 1, 4)
            # Tile it horizontally to make a 10-pixel-wide image.
            # This is more efficient for rendering than a full-width image.
            gradient_image = np.tile(gradient_image, (1, 10, 1))

            # --- 7. Plot the Gradient Image and Data ---
            # Plot the gradient image on the background (zorder=1)
            # `extent` maps the image pixels to the data coordinates [x_min, x_max, y_min, y_max]
            # `origin='lower'` means the 0-index of the array is at the bottom (y_min)
            # `aspect='auto'` stretches the image to fill the axes
            ax1.imshow(
                gradient_image,
                origin='lower',
                extent=[x_min, x_max, y_min, y_max],
                aspect='auto',
                zorder=1
            )

            # Plot the target lines
            ax1.axhline(LOWER_TARGET, color='gray', linestyle='--', linewidth=1, zorder=5,
                        label='Target Range (65-195)')
            ax1.axhline(HIGHER_TARGET, color='gray', linestyle='--', linewidth=1, zorder=5)
            ax1.axhline(PEAK_TARGET, color='green', linestyle=':', linewidth=1, zorder=5,
                        label=f'Peak Reward ({int(PEAK_TARGET)})')

            # Plot CHO as a stem plot
            cho_mask = pt_cho > 0
            cho_times = pt_time[cho_mask]
            cho_values = pt_cho[cho_mask]
            markerline, stemlines, baseline = ax1.stem(
                cho_times,
                cho_values,
                linefmt='purple',
                markerfmt='D',
                basefmt=' ',
                label='Carbohydrates (g)',
            )
            plt.setp(markerline, markersize=5, color='purple', zorder=9)
            plt.setp(stemlines, linewidth=1.5, color='purple', zorder=9)

            # Plot the actual blood glucose data on top (zorder=10)
            ax1.plot(pt_time, pt_blood_glucose, color='black', linewidth=1.5, zorder=10, label='Blood Glucose')

            # Create twin axis to plot insulin
            ax2 = ax1.twinx()
            ax2.plot(pt_insulin_acts_time, pt_insulin_acts, color='blue', linestyle=':',
                     linewidth=2, label='Insulin Rate', zorder=8)
            ax2.set_ylabel('Insulin Rate (U/hr)', color='blue', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='blue')
            ax2.set_ylim(bottom=0)
            ax2.grid(True, which='both', linestyle=':', alpha=0.3)

            # --- 8. Add a Colorbar to Show Reward Mapping ---
            # Create a new axes for the colorbar
            # [left, bottom, width, height] in figure-relative coordinates
            cax = fig.add_axes([0.92, 0.11, 0.02, 0.77])
            mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

            neg_ticks = list(np.arange(np.floor(v_min / 20) * 20, 1, 20))
            final_ticks = sorted(list(set(neg_ticks + [1, 2, 3, 4, 5, 6, v_max])))

            cbar = fig.colorbar(mappable, cax=cax, orientation='vertical', ticks=final_ticks)
            cbar.set_ticklabels([f'{t:.1f}' for t in final_ticks])
            cbar.set_label('Reward Value', fontsize=12)
            # Note: The colorbar doesn't show the custom alpha, but it
            # correctly shows the color-to-value mapping.

            # --- 9. Final Plot Styling ---
            ax1.set_xlim(x_min, x_max)
            ax1.set_ylim(y_min, y_max)
            ax1.set_title('Patient Simulator with CHO and Insulin', fontsize=16)
            ax1.set_xlabel('Time (Hour of Day)', fontsize=12)
            ax1.set_ylabel('Blood Glucose (mg/dL)', fontsize=12)
            # Removed zorder=20 from the legend call to fix the TypeError
            ax1.legend(loc='upper left')
            ax1.grid(True, which='both', linestyle=':', alpha=0.3)

            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc='upper left')

            # Adjust x-tick formatting
            def hour_formatter(x, pos):
                """Formats a continuous hour (e.g., 25.5) as a 24-hour string (e.g., '1.5')"""
                hour = x % 24
                return f'{int(hour)}'

            ax1.xaxis.set_major_formatter(mticker.FuncFormatter(hour_formatter))
            start_tick = np.ceil(x_min)
            tick_locations = np.arange(start_tick, x_max, 3)
            ax1.set_xticks(tick_locations)

            # Adjust main plot to make room for colorbar
            fig.subplots_adjust(right=0.9)
            plt.show()
