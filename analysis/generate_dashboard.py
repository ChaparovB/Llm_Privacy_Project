import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np

def find_latest_csv(results_dir="results/analysis"):
    files = [f for f in os.listdir(results_dir) if f.endswith(".csv")]
    files.sort(reverse=True)
    return os.path.join(results_dir, files[0]) if files else None

def plot_risk_score_bar_chart(df, timestamp):
    df['risk_score'] = df['risk_score'].apply(lambda x: 1 if str(x).strip().lower() == "true" else 0)
    pivot = df.pivot_table(index="category", columns="model", values="risk_score", aggfunc="mean")
    ax = pivot.plot(kind="bar", figsize=(10, 6), title="Average Privacy Risk Score by Prompt Category")
    ax.set_xlabel("Prompt Category")
    ax.set_ylabel("Average Risk Score (0 = Safe, 1 = Risky)")
    ax.legend(title="Model", loc="upper right")
    plt.xticks(rotation=0)
    plt.tight_layout()
    bar_path = f"results/bar_risk_score_{timestamp}.png"
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Risk score bar chart saved to {bar_path}")

def plot_comparison_bar_chart(timestamp):
    labels = ['Data Control', 'Auditability', 'Inference Cost', 'Response Time', 'Privacy Risk']
    hosted = [2, 2, 3, 3, 3]
    local = [5, 5, 1, 2, 1]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, hosted, width, label='OpenAI GPT-4o', color='gold')
    ax.bar(x + width / 2, local, width, label='Local Agent', color='orangered')

    ax.set_ylabel('Score (1 = Best, 5 = Worst)')
    ax.set_title('Hosted vs Local Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout()
    bar_path = f"results/model_comparison_bar_{timestamp}.png"
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Model comparison bar chart saved to {bar_path}")

def plot_cost_vs_tokens(timestamp):
    tokens = [1000, 5000, 10000, 20000, 50000]
    cost_per_token = 0.00000002  # Example OpenAI cost
    costs = [t * cost_per_token for t in tokens]

    plt.figure(figsize=(8, 5))
    plt.plot(tokens, costs, marker='o', linestyle='-', color='green')
    plt.xlabel("Token Count")
    plt.ylabel("Cost in USD")
    plt.title("Estimated Embedding Cost vs Token Count")
    plt.grid(True)
    plt.tight_layout()
    cost_path = f"results/embedding_cost_{timestamp}.png"
    plt.savefig(cost_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📈 Cost plot saved to {cost_path}")

def generate_dashboard():
    print("📊 generate_dashboard() has started...")
    os.makedirs("results/analysis", exist_ok=True)

    latest_csv = find_latest_csv()
    if latest_csv:
        print(f"📊 Plotting results from: {latest_csv}")
        df = pd.read_csv(latest_csv)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("results", exist_ok=True)
        plot_risk_score_bar_chart(df, timestamp)
        plot_comparison_bar_chart(timestamp)
        plot_cost_vs_tokens(timestamp)
    else:
        print("❌ No results file found in 'results/analysis' folder.")

if __name__ == "__main__":
    generate_dashboard()
