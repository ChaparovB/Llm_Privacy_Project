import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Define mapping: keyword in prompt → category label
CATEGORY_MAP = {
    "firestore": "Response logging via Firestore",
    "wikipedia": "Wikipedia tool query",
    "chromadb": "Retrieval from ChromaDB",
    "credit card": "Prompt injection with fake credit card",
}

def infer_category(prompt):
    for keyword, label in CATEGORY_MAP.items():
        if keyword.lower() in prompt.lower():
            return label
    return None  # Unmatched prompts won't be included

def plot_privacy_risk_categories(csv_path):
    df = pd.read_csv(csv_path)
    df['risk_score'] = df['risk_score'].apply(lambda x: 1 if str(x).lower() == "true" else 0)
    df['test_case'] = df['prompt'].apply(infer_category)
    df = df.dropna(subset=['test_case'])

    summary = df.groupby('test_case')['risk_score'].sum().sort_values()

    plt.figure(figsize=(10, 6))
    summary.plot(kind="barh", color="skyblue")
    plt.title("Privacy Risk Evaluation of Test Cases")
    plt.xlabel("Risk Score (1 = Low, 3 = High)")
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("results", f"risk_categories_{timestamp}.png")
    plt.savefig(output_path)
    print(f"✅ Saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    results_dir = "results"
    latest_csv = sorted(
        [f for f in os.listdir(results_dir) if f.endswith(".csv")],
        reverse=True
    )[0]
    plot_privacy_risk_categories(os.path.join(results_dir, latest_csv))
