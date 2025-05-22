import pandas as pd
import matplotlib.pyplot as plt
import os

# === CONFIGURATION ===
file_path = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\nice graphs\anonymous\MCC_vs_k_anonymity_tables.xlsx"
output_path = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\nice graphs\anonymous"
x_axis = "k_anonymity"
metric_to_plot = "MCC"

# === LOAD AND PLOT ALL SHEETS ===
xls = pd.ExcelFile(file_path)
plt.figure(figsize=(10, 6))

for sheet in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet)
    df = df.sort_values(by=x_axis)
    if x_axis in df.columns and metric_to_plot in df.columns:
        plt.plot(df[x_axis], df[metric_to_plot], marker='o', label=sheet)

plt.title("MCC vs k_anonymity across Datasets", fontsize=14)
plt.xlabel("k_anonymity", fontsize=12)
plt.ylabel("MCC", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.ylim(None, 1)
plt.legend()
plt.tight_layout()

# === SAVE FIGURE ===
output_file = os.path.join(output_path, "mcc_vs_k_all_datasets.png")
plt.savefig(output_file, dpi=300)
plt.show()
print(f"✅ Plot saved to: {output_file}")
