import os
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import numpy as np

def val_to_str(v):
    """Helper to convert float -> '0p2' etc."""
    return str(v).replace(".", "p")

def generate_static_plots(df, variation_info_path, output_dir):
    """
    Creates plots with logic:
    1) For anonymization xparams in [k_anonymity, l_diversity, suppression_limit],
       filter to Dataset=Anonymous, test_size=0.2, then we do a "nested" approach:
         - For each xparam, we produce multiple PNGs. Specifically:
             -> for each unique 'suppression_limit' or 'l_diversity' (the ones not being swept),
                we produce lines for the other param, x-axis is xparam.
    2) For synthetic xparams in [epochs, custom_generated_rows],
       filter to Dataset in [CTGAN, TVAE, etc.] and test_size=0.2. Produce one chart
       with x= xparam, lines by Dataset.
    3) For test_size, produce one chart with x= test_size, lines by Dataset= [Original,Anonymous,CTGAN,TVAE,...],
       but we fix k=3, l=2, sup=0.1, epochs=50, custom_rows=1000, etc.
    """
    if not os.path.exists(variation_info_path):
        print("⚠️ No variation_info.yaml found. Skipping plots.")
        return
    with open(variation_info_path,"r") as f:
        var_info = yaml.safe_load(f)

    xparams = var_info.get("varied_parameters", [])
    metrics = ["Accuracy","Precision","Recall","F1","AUC-ROC"]
    metrics = [m for m in metrics if m in df.columns]

    # Make sure output_dir exists
    os.makedirs(output_dir, exist_ok=True)

    # We'll define a helper for anonymized "nested approach"
    def plot_anonym_param(df, xparam, metric):
        """
        For xparam in [k_anonymity, l_diversity, suppression_limit].
        Filter to test_size=0.2, Dataset=Anonymous
        Then produce multiple PNGs, each one for a unique combination of the other 2 anonym. params.
        Within each PNG, we have lines for the second param, x-axis is xparam.
        """
        # filter
        dfa = df.copy()
        dfa = dfa[(dfa["Dataset"]=="Anonymous") & (dfa["test_size"]==0.2)]
        dfa = dfa.dropna(subset=[xparam, metric])
        if dfa.empty:
            print(f"⚠️ No data for {metric} vs {xparam} (Anonymous, test_size=0.2).")
            return

        # The other anonym params are {k_anonymity, l_diversity, suppression_limit} - {xparam}
        anonym_params = ["k_anonymity","l_diversity","suppression_limit"]
        other_params = [p for p in anonym_params if p != xparam]

        # We'll produce a separate chart for each combination of the other 2 param values
        # Then inside that chart, we do lines for the first of the other param, x-axis is xparam
        # For example, if xparam=k, then other_params=[l_diversity, suppression_limit].
        # We'll fix "suppression_limit" as separate charts, lines by l_diversity, etc.
        # Or choose your approach. We'll fix the first param in "other_params" as separate charts,
        # and lines for the second param.
        if len(other_params)!=2:
            return

        fix_param = other_params[0]  # e.g. "suppression_limit"
        line_param = other_params[1] # e.g. "l_diversity"

        unique_fix_values = sorted(dfa[fix_param].dropna().unique())
        unique_line_values = sorted(dfa[line_param].dropna().unique())

        for fix_val in unique_fix_values:
            df_chart = dfa[dfa[fix_param]==fix_val]
            if df_chart.empty:
                continue

            # create the plot
            plt.figure(figsize=(8,5))
            has_data = False
            for lv in unique_line_values:
                sub = df_chart[df_chart[line_param]==lv]
                if sub.empty:
                    continue
                # sort by xparam so lines connect left->right
                sub = sub.sort_values(by=xparam)
                plt.plot(sub[xparam], sub[metric], label=f"{line_param}={lv}", marker="o")
                has_data = True

            if not has_data:
                plt.close()
                continue

            plt.xlabel(xparam)
            plt.ylabel(metric)
            title = f"{metric} vs {xparam}\n(Dataset=Anonymous, test_size=0.2, {fix_param}={fix_val})"
            plt.title(title)
            plt.legend()
            plt.grid(True)

            # filename
            fix_val_str = val_to_str(fix_val)
            out_name = f"{metric}_vs_{xparam}_fix_{fix_param}={fix_val_str}.png"
            out_path = os.path.join(output_dir, out_name)
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
            print(f"✅ Saved {out_name}")

    # Synthetic
    def plot_synth_param(df, xparam, metric):
        """
        For xparam in [epochs, custom_generated_rows].
        Only keep rows where Dataset in [CTGAN, TVAE, GaussianCopula, Hybrid],
        test_size=0.2. Then produce ONE chart with lines for each dataset
        on x= xparam.
        """
        dfa = df.copy()
        synth_datasets = ["CTGAN","TVAE","GaussianCopula","Hybrid"]
        dfa = dfa[dfa["Dataset"].isin(synth_datasets)]
        dfa = dfa[dfa["test_size"]==0.2]
        dfa = dfa.dropna(subset=[xparam, metric])
        if dfa.empty:
            print(f"⚠️ No data for {metric} vs {xparam} in synthetic sets.")
            return

        plt.figure(figsize=(8,5))
        has_data = False
        for ds in dfa["Dataset"].unique():
            sub = dfa[dfa["Dataset"]==ds]
            sub = sub.sort_values(by=xparam)
            if sub.empty:
                continue
            plt.plot(sub[xparam], sub[metric], label=ds, marker="o")
            has_data = True

        if not has_data:
            plt.close()
            return
        plt.xlabel(xparam)
        plt.ylabel(metric)
        plt.title(f"{metric} vs {xparam} (test_size=0.2, Synthetic Only)")
        plt.legend()
        plt.grid(True)

        out_name = f"{metric}_vs_{xparam}_synthOnly.png"
        out_path = os.path.join(output_dir, out_name)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"✅ Saved {out_name}")

    # test_size
    def plot_test_size(df, metric):
        """
        x= test_size, lines = each dataset type (Original, Anonymous, CTGAN,...).
        Fix k=3, l=2, sup=0.1, epochs=50, custom_rows=1000, etc.
        """
        dfa = df.copy()
        # fix anonymization
        # if row is Anonymous, we only keep those with k=3, l=2, sup=0.1
        # if row is synthetic, we only keep those with epochs=50, custom=1000
        # if row is Original, no param needed
        # We'll do that filter:
        # let's define a function that checks the row
        def keep_row(row):
            ds = row["Dataset"]
            if ds=="Anonymous":
                return (row.get("k_anonymity",None)==3 and
                        row.get("l_diversity",None)==2 and
                        abs(row.get("suppression_limit",0)-0.1)<1e-9)
            elif ds in ["CTGAN","TVAE","GaussianCopula","Hybrid"]:
                return (row.get("epochs",None)==50 and
                        row.get("custom_generated_rows",None)==1000)
            elif ds=="Original":
                return True
            else:
                return False

        dfa = dfa[dfa.apply(keep_row, axis=1)]
        dfa = dfa.dropna(subset=["test_size", metric])
        if dfa.empty:
            print(f"⚠️ No data for test_size vs {metric} after param fixing.")
            return

        plt.figure(figsize=(8,5))
        has_data=False
        for ds in dfa["Dataset"].unique():
            sub = dfa[dfa["Dataset"]==ds].copy()
            sub = sub.sort_values(by="test_size")
            if sub.empty:
                continue
            plt.plot(sub["test_size"], sub[metric], label=ds, marker="o")
            has_data=True

        if not has_data:
            plt.close()
            return
        plt.xlabel("test_size")
        plt.ylabel(metric)
        plt.title(f"{metric} vs test_size (k=3, l=2, sup=0.1, epochs=50, custom=1000)")
        plt.legend()
        plt.grid(True)

        out_name = f"{metric}_vs_test_size_fixedParams.png"
        out_path = os.path.join(output_dir, out_name)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"✅ Saved {out_name}")

    # Now the main logic
    for metric in metrics:
        for xp in xparams:
            if xp not in df.columns:
                print(f"❌ Skipping {metric} vs {xp}: no such column.")
                continue

            # Decide which function to call
            if xp in ["k_anonymity","l_diversity","suppression_limit"]:
                plot_anonym_param(df, xp, metric)

            elif xp in ["epochs","custom_generated_rows"]:
                plot_synth_param(df, xp, metric)

            elif xp=="test_size":
                # single chart for test_size
                plot_test_size(df, metric)

            else:
                print(f"⚠️ xparam {xp} not recognized, skipping.")
                continue

    print(f"📊 Static plots generated in: {output_dir}")
