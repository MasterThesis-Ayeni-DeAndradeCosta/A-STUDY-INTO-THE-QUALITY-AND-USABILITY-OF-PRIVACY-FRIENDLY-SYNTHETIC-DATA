import os
import pandas as pd
import matplotlib.pyplot as plt

# ========== CONFIGURATION ==========

output_root = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\thesisgraphs"
os.makedirs(output_root, exist_ok=True)

dataset_configs = {
    "Loan": {
        "path": r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\loan\loan_combined_results.xlsx",
        "synth": "GaussianCopula",
        "model": "LogisticRegression",
        "l_diversity": 15,
        "suppression_limit": 0.1
    },
    "StudentPerformance": {
        "path": r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\studentPerformance\studentPerformance_combined_results.xlsx",
        "synth": "TVAE",
        "model": "RandomForest",
        "l_diversity": 1,
        "suppression_limit": 0.5
    },
    "BankMarketing": {
        "path": r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\bankMarketing\bankMarketing_combined_results.xlsx",
        "synth": "CTGAN",
        "model": "RandomForest",
        "l_diversity": 2,
        "suppression_limit": 1
    }
}

metric_column = "MCC"
x_axis = "k_anonymity"
target_epochs = 300

plt.figure(figsize=(10, 6))
plotted_any = False

# ========== PROCESS EACH DATASET ==========

for dataset_name, config in dataset_configs.items():
    df = pd.read_excel(config["path"])
    synth = config["synth"]
    model = config["model"]
    hybrid_label = f"{synth}_HYBRID"

    # === SYNTHETIC BASELINE ===
    baseline_df = df[
        (df["Model"] == model) &
        (df["Dataset"] == synth) &
        (df["epochs"] == target_epochs)
    ]
    if baseline_df.empty:
        print(f"❌ No synthetic baseline (epochs=300) for {dataset_name} – {synth}")
        continue
    baseline_mcc = baseline_df[metric_column].max()

    # === HYBRID DATA FILTER ===
    hybrid_filtered = df[
        (df["Model"] == model) &
        (df["Dataset"] == hybrid_label) &
        (df["epochs"] == target_epochs) &
        (df["l_diversity"] == config["l_diversity"]) &
        (df["suppression_limit"] == config["suppression_limit"])
    ].copy()

    if hybrid_filtered.empty:
        print(f"⚠️ No hybrid data (epochs=300, l={config['l_diversity']}, supp={config['suppression_limit']}) for {dataset_name}")
        continue

    # Compute delta from synthetic
    hybrid_filtered["Δ MCC (%) from Synthetic"] = ((hybrid_filtered[metric_column] - baseline_mcc) / baseline_mcc) * 100
    hybrid_filtered = hybrid_filtered.sort_values(by=x_axis)

    # Keep only relevant columns
    export_cols = [x_axis, metric_column, "l_diversity", "suppression_limit", "row_multiplier", "epochs", "Δ MCC (%) from Synthetic"]
    export_df = hybrid_filtered[export_cols]

    # Save Excel
    excel_path = os.path.join(output_root, f"{dataset_name}_hybrid_vs_synth_fixed_l_supp.xlsx")
    export_df.to_excel(excel_path, index=False)

    # Plot
    grouped = export_df.groupby(x_axis)["Δ MCC (%) from Synthetic"].mean().sort_index()
    plt.plot(grouped.index, grouped.values, marker='o', label=f"{dataset_name} – {synth}")
    plotted_any = True

    print(f"✅ Processed: {dataset_name} — Exported Excel to {excel_path}")

# ========== FINALIZE PLOT ==========

if plotted_any:
    plt.axhline(0, color='black', linestyle='--', linewidth=1, label="Synthetic Baseline")
    plt.title("Δ MCC of Hybrid vs Synthetic (Epochs = 300) vs $k$-anonymity")
    plt.xlabel("$k$-anonymity level")
    plt.ylabel("Δ MCC (%) from Synthetic")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(output_root, "hybrid_vs_synthetic_delta_vs_k_fixed_params.png")
    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(f"📊 Plot saved to: {plot_path}")
else:
    print("❌ Nothing plotted — check data filters or paths.")
