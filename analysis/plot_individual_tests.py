import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

def plot_individual_test_risks():
    results_dir = "results"
    files = [f for f in os.listdir(results_dir) if f.endswith(".csv")]
    files.sort(reverse=True)

    if not files:
        print("❌ No CSV files found.")
        return

    latest_file = os.path.join(results_dir, files[0])
    df = pd.read_csv(latest_file)

    # Normalize risk_score column
    df['risk_score'] = df['risk_score'].apply(lambda x: 1 if str(x).lower().strip() == 'true' else 0)

    # Pivot for side-by-side comparison
    pivot_df = df.pivot_table(index="prompt", columns="model", values="risk_score", aggfunc="first").fillna(0)

    # Sort by max risk
    pivot_df = pivot_df.sort_values(by=list(pivot_df.columns), ascending=False)

    # Plot grouped bars
    ax = pivot_df.plot(kind="barh", figsize=(12, 8), color=["orange", "skyblue"])
    plt.xlabel("Risk Score (1 = High Risk)")
    plt.title("Privacy Risk Comparison per Prompt by Model")
    plt.legend(title="Model")
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"results/individual_test_risks_{timestamp}.png"
    plt.savefig(path)
    plt.show()
    print(f"✅ Saved updated individual test comparison chart to {path}")

if __name__ == "__main__":
    plot_individual_test_risks()
