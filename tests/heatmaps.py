import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import re
from collections import defaultdict

# === CONFIGURATION ===
base_dir = "/Users/giuliopalcic/Desktop/Albatross - results"
output_dir = "/Users/giuliopalcic/Desktop/heatmaps"
os.makedirs(output_dir, exist_ok=True)

target_basin = "ZRB"  # Used in file naming
metric = "r2"  # Choose from: "r2", "corr", "mape"

# === METRIC SETUP ===
if metric=="r2":
    results_selected = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    clim_label = "R² Score"
    file_suffix = "r2"
    cmap = "magma_r"
    vmin, vmax = None, None
    is_higher_better = True
elif metric=="corr":
    results_selected = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    clim_label = "Pearson Correlation"
    file_suffix = "corr"
    cmap = "magma_r"
    vmin, vmax = None, None
    is_higher_better = True
elif metric=="mape":
    results_selected = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    clim_label = "MAPE"
    file_suffix = "mape"
    cmap = "magma_r"
    vmin, vmax = None, None
    is_higher_better = False
else:
    raise ValueError(f"Unsupported metric: {metric}")

# === PATTERNS AND CONSTANTS ===
folder_pattern = re.compile(
    r"(?P<target>[\w\-]+)_(?P<basin_name>[\w\-]+)_(?P<target_variable>[\w\-]+)_(?P<glo_idx>[\w\-]+)_(?P<glo_var>[\w\-]+)_(?P<n_phases>\d+)_(?P<period>[\d,]+)_(?P<start>\d{4})_(?P<end>\d{4})"
)

month_map = {1: 'J', 2: 'F', 3: 'M', 4: 'A', 5: 'M', 6: 'J',
             7: 'J', 8: 'A', 9: 'S', 10: 'O', 11: 'N', 12: 'D'}

if target_basin == "ZRB":
    seasons = ["NDJ", "FMA","MJJ", "ASO"]  # or whatever specific to ZRB
else:
    seasons = ["DJF", "MAM", "JJA", "SON"]  # standard seasons

def period_str_from_numbers(period):
    nums = [ int(m) for m in period.split(",") ]
    return "".join(month_map [ n ] for n in nums)

# === DATA LOADING ===
print(f"Starting data loading from: {base_dir}")
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    print(f"Searching in parent folder: {folder_path}")

    for subfolder in os.listdir(folder_path):
        match = folder_pattern.match(subfolder)
        if not match:
            print(f"  - Skipping '{subfolder}': Does not match folder pattern.")
            continue

        info = match.groupdict()

        target_var = info [ 'target_variable' ]
        glo_idx = info [ 'glo_idx' ]
        glo_var = info [ 'glo_var' ]
        phases = info [ 'n_phases' ]
        season = period_str_from_numbers(info [ 'period' ])

        if season not in seasons:
            print(f"  - Skipping '{subfolder}': Season '{season}' is not a valid standard season ({seasons}).")
            continue

        full_path = os.path.join(folder_path, subfolder)
        print(f"  - Processing subfolder: {full_path}")
        csvs = [ f for f in os.listdir(full_path) if f.endswith("_timeseries.csv") ]

        if not csvs:
            print(f"    ⚠️ No CSV files found in {full_path}")

        for csv_file in csvs:
            try:
                df = pd.read_csv(os.path.join(full_path, csv_file))
                if 'observed' in df.columns and 'hindcast' in df.columns:
                    if df [ 'observed' ].std()==0 or df [ 'hindcast' ].std()==0:
                        print(f"    - Skipping {csv_file}: Zero standard deviation in data (observed or hindcast).")
                        continue

                    if metric=="r2":
                        value = r2_score(df [ 'observed' ], df [ 'hindcast' ])
                    elif metric=="corr":
                        value, _ = pearsonr(df [ 'observed' ], df [ 'hindcast' ])
                    elif metric=="mape":
                        value = mean_absolute_percentage_error(df [ 'observed' ], df [ 'hindcast' ])

                    results_selected [ target_var ] [ glo_var ] [ glo_idx ] [ phases ] [ season ] = value
                    print(f"    - Processed '{csv_file}': {metric}={value:.2f}")
                else:
                    print(f"    - Skipping '{csv_file}': Missing 'observed' or 'hindcast' columns.")
            except Exception as e:
                print(f"    ⚠️ Error reading {csv_file}: {e}")
                continue

print("\nData loading complete.")
if not results_selected:
    print("No data was successfully loaded. No plots will be generated.")
else:
    print("Data loaded. Proceeding to best model selection and plotting.")

# === BEST MODEL SELECTION ===
best_models_per_phase = defaultdict(lambda: defaultdict(dict))

print("\nStarting best model selection per phase...")
for target_var, target_data in results_selected.items():
    for source, source_data in target_data.items():
        all_phases = sorted(list(set(p for glo_idx_data in source_data.values() for p in glo_idx_data.keys())))

        for phase in all_phases:
            best_avg_score = -np.inf if is_higher_better else np.inf
            best_glo_idx_for_phase = None
            best_scores_for_phase = None

            for glo_idx, phases_data in source_data.items():
                if phase in phases_data:
                    current_model_scores = phases_data [ phase ]

                    valid_scores = [ v for s, v in current_model_scores.items() if
                                     s in seasons and isinstance(v, (int, float)) ]

                    if not valid_scores:
                        continue

                    avg_score = np.mean(valid_scores)

                    if (is_higher_better and avg_score > best_avg_score) or \
                        (not is_higher_better and avg_score < best_avg_score):
                        best_avg_score = avg_score
                        best_glo_idx_for_phase = glo_idx
                        best_scores_for_phase = current_model_scores

            if best_glo_idx_for_phase:
                best_models_per_phase [ target_var ] [ source ] [ phase ] = {
                    'glo_idx': best_glo_idx_for_phase,
                    'avg_score': best_avg_score,
                    'scores_per_season': best_scores_for_phase
                }
                print(
                    f"  - Best model for {target_var} - {source} - Phase {phase}: {best_glo_idx_for_phase} (Avg {metric}: {best_avg_score:.2f})")
            else:
                print(f"  - No best model found for {target_var} - {source} - Phase {phase}.")

print("\nBest model selection complete.")

# === PLOTTING INDIVIDUAL HEATMAPS AND RADAR PLOTS ===
print("\nGenerating individual heatmaps and radar plots...")
for target_var in results_selected:
    for source in [ 'slp', 'sst' ]:
        source_data = results_selected [ target_var ].get(source, {})
        if not source_data:
            continue

        all_phases = sorted(list(set(p for glo_idx in source_data for p in source_data [ glo_idx ])))

        for phase in all_phases:
            phase_data = {
                glo_idx: scores [ phase ]
                for glo_idx, scores in source_data.items()
                if phase in scores
            }

            if not phase_data:
                continue

            df = pd.DataFrame(phase_data).T
            df = df.reindex(columns=seasons)

            # --- HEATMAP ---
            plt.figure(figsize=(10, max(4, len(df) * 0.6)))
            sns.heatmap(df, annot=True, cmap=cmap, vmin=vmin, vmax=vmax, cbar_kws={"label": clim_label})
            plt.title(f"{clim_label} – {source.upper()} – {target_var} – Phase {phase}")
            plt.xlabel("Season")
            plt.ylabel("Climate Index")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir,
                                     f"{target_basin}_heatmap_{file_suffix}_{target_var}_{source}_phase{phase}.png"))
            plt.close()

            # --- RADAR ---
            angles = np.linspace(0, 2 * np.pi, len(seasons), endpoint=False).tolist() + [ 0 ]
            fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

            for idx, row in df.iterrows():
                values = [ row.get(s, 0) for s in seasons ] + [ row.get(seasons [ 0 ], 0) ]
                ax.plot(angles, values, label=idx)
                ax.fill(angles, values, alpha=0.1)

            ax.set_thetagrids(np.degrees(angles [ :-1 ]), seasons)
            ax.set_ylim(vmin if vmin is not None else 0, vmax if vmax is not None else 1)
            ax.set_title(f"{clim_label} Radar – {source.upper()} – {target_var} – Phase {phase}", y=1.1)
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"{target_basin}_radar_{file_suffix}_{target_var}_{source}_phase{phase}.png"))
            plt.close()

    # === Combined Radar: SLP vs SST for each phase ===
    slp_data = results_selected [ target_var ].get('slp', {})
    sst_data = results_selected [ target_var ].get('sst', {})

    if slp_data and sst_data:
        all_phases_combined = sorted(list(set(p for glo_idx in slp_data for p in slp_data [ glo_idx ]) |
                                          set(p for glo_idx in sst_data for p in sst_data [ glo_idx ])))

        for phase in all_phases_combined:
            df_slp = pd.DataFrame({
                glo_idx: scores.get(phase, {})
                for glo_idx, scores in slp_data.items()
            }).T.reindex(columns=seasons)

            df_sst = pd.DataFrame({
                glo_idx: scores.get(phase, {})
                for glo_idx, scores in sst_data.items()
            }).T.reindex(columns=seasons)

            if df_slp.empty and df_sst.empty:
                continue

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            angles = np.linspace(0, 2 * np.pi, len(seasons), endpoint=False).tolist() + [ 0 ]

            for idx, row in df_slp.iterrows():
                values = [ row.get(s, 0) for s in seasons ] + [ row.get(seasons [ 0 ], 0) ]
                ax.plot(angles, values, label=f"{idx}-SLP", linestyle='-')
                ax.fill(angles, values, alpha=0.1)

            for idx, row in df_sst.iterrows():
                values = [ row.get(s, 0) for s in seasons ] + [ row.get(seasons [ 0 ], 0) ]
                ax.plot(angles, values, label=f"{idx}-SST", linestyle='--')
                ax.fill(angles, values, alpha=0.1)

            ax.set_thetagrids(np.degrees(angles [ :-1 ]), seasons)
            ax.set_ylim(vmin if vmin is not None else 0, vmax if vmax is not None else 1)
            ax.set_title(f"{clim_label} Radar – SLP vs SST – {target_var} – Phase {phase}", y=1.1)
            ax.grid(True)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir,
                                     f"{target_basin}_radar_{file_suffix}_{target_var}_slp_vs_sst_phase{phase}.png"))
            plt.close()

# === NEW PLOTTING SECTION: BEST MODELS ACROSS PHASES ===
print("\nGenerating radar plots for best models across phases...")
for target_var in best_models_per_phase:
    for source in [ 'slp', 'sst' ]:
        best_source_models = best_models_per_phase [ target_var ].get(source, {})
        if not best_source_models:
            continue

        plot_data_for_radar = {}
        for phase, model_info in best_source_models.items():
            label = f"Phase {phase} ({model_info [ 'glo_idx' ]})"
            plot_data_for_radar [ label ] = model_info [ 'scores_per_season' ]

        df_best_models_radar = pd.DataFrame(plot_data_for_radar).T.reindex(columns=seasons)

        angles = np.linspace(0, 2 * np.pi, len(seasons), endpoint=False).tolist() + [ 0 ]
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for idx_label, row in df_best_models_radar.iterrows():
            values = [ row.get(s, 0) for s in seasons ] + [ row.get(seasons [ 0 ], 0) ]
            ax.plot(angles, values, label=idx_label)
            ax.fill(angles, values, alpha=0.1)

        ax.set_thetagrids(np.degrees(angles [ :-1 ]), seasons)
        ax.set_ylim(vmin if vmin is not None else 0, vmax if vmax is not None else 1)
        ax.set_title(f"{clim_label} Radar – Best {source.upper()} Models Across Phases – {target_var}", y=1.1)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
                                 f"{target_basin}_radar_{file_suffix}_{target_var}_{source}_best_models_across_phases.png"))
        plt.close()

# === FINAL NEW PLOTTING SECTION: BEST SLP vs SST ACROSS PHASES ===
print("\nGenerating combined radar plot for best SLP vs SST across phases...")
for target_var in best_models_per_phase:
    best_slp_models = best_models_per_phase [ target_var ].get('slp', {})
    best_sst_models = best_models_per_phase [ target_var ].get('sst', {})

    if not best_slp_models and not best_sst_models:
        continue

    # Get all unique phases available in either best SLP or best SST models
    all_best_phases = sorted(list(set(best_slp_models.keys()) | set(best_sst_models.keys())))

    if not all_best_phases:
        continue

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(seasons), endpoint=False).tolist() + [ 0 ]

    # Use a categorical colormap like 'tab10' for distinct colors per phase
    colors = plt.cm.tab10.colors
    linestyles = [ '-', '--' ]  # Solid for SLP, dashed for SST

    for i, phase in enumerate(all_best_phases):
        color = colors [ i ]

        # Plot best SLP model for this phase
        if phase in best_slp_models:
            model_info = best_slp_models [ phase ]
            values = [ model_info [ 'scores_per_season' ].get(s, 0) for s in seasons ] + [
                model_info [ 'scores_per_season' ].get(seasons [ 0 ], 0) ]
            label = f"Phase {phase} - SLP ({model_info [ 'glo_idx' ]})"
            ax.plot(angles, values, label=label, linestyle=linestyles [ 0 ], color=color)
            ax.fill(angles, values, alpha=0.1, color=color)

        # Plot best SST model for this phase
        if phase in best_sst_models:
            model_info = best_sst_models [ phase ]
            values = [ model_info [ 'scores_per_season' ].get(s, 0) for s in seasons ] + [
                model_info [ 'scores_per_season' ].get(seasons [ 0 ], 0) ]
            label = f"Phase {phase} - SST ({model_info [ 'glo_idx' ]})"
            ax.plot(angles, values, label=label, linestyle=linestyles [ 1 ], color=color)
            ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_thetagrids(np.degrees(angles [ :-1 ]), seasons)
    ax.set_ylim(vmin if vmin is not None else 0, vmax if vmax is not None else 1)
    ax.set_title(f"{clim_label} Radar – Best SLP vs SST Models Across Phases – {target_var}", y=1.1)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{target_basin}_radar_{file_suffix}_{target_var}_best_slp_vs_sst_across_phases.png"))
    plt.close()

print("\nScript execution complete.")
