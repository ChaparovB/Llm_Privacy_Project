import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Load the most recent result file from the results folder
def find_latest_csv(results_dir="results"):
    files = [f for f in os.listdir(results_dir) if f.endswith(".csv")]
    files.sort(reverse=True)
    return os.path.join(results_dir, files[0]) if files else None

def plot_results(csv_path):
    df = pd.read_csv(csv_path)
    df['risk_score'] = df['risk_score'].apply(lambda x: 1 if x == True or x == 'True' else 0)

    # Bar chart: Average risk score per model and category
    pivot = df.pivot_table(index="category", columns="model", values="risk_score", aggfunc="mean")
    ax = pivot.plot(kind="bar", figsize=(8, 5), title="Average Risk Score by Prompt Category")
    plt.ylabel("Average Risk Score (0-1)")
    plt.tight_layout()

    # Export chart as PNG
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/privacy_plot_{timestamp}.png"
    plt.savefig(output_path)
    print(f"📸 Chart saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    latest_csv = find_latest_csv()
    if latest_csv:
        print(f"📊 Plotting results from: {latest_csv}")
        plot_results(latest_csv)
    else:
        print("❌ No results file found in 'results' folder.")
