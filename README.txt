# Privacy Evaluation Framework for Local and Cloud-based LLMs

This project provides a modular, extensible framework to evaluate privacy-related behaviors of both local and cloud-based Large Language Models (LLMs). It enables controlled execution and analysis of LLM responses to sensitive, adversarial, and benign prompts, supporting tool-augmented local agents and OpenAI-hosted models.

## 📁 Project Structure

```
llm_privacy_project/
├── agents/                 # Agent initialization and orchestration
│   └── local_llm_agent.py
├── models/                # Wrapper for LLM interfaces
│   └── local_wrapper.py
├── retrievers/            # Chroma DB management
│   └── chroma_manager.py
├── tools/                 # Custom tool implementations
│   ├── toolkit.py
│   ├── addition_tool.py
│   ├── reversal_tool.py
│   ├── greeting_tool.py
│   └── retrieval_tool.py
├── tests/                 # Evaluation runner and test prompt files
│   ├── privacy_test_runner.py
│   ├── prompts_normal.txt
│   ├── prompts_sensitive.txt
│   └── prompts_adversarial.txt
├── results/               # Auto-generated evaluation result CSVs
├── .env                   # API key and environment config
└── README.md
```

## 🧠 Features

* **Hybrid LLM Setup**:

  * Uses OpenAI's `gpt-4o` for cloud-based evaluation.
  * Implements a local rule-based agent with tools using LangChain.

* **Custom Tooling**:

  * `add_numbers`: Basic arithmetic
  * `reverse_string`: Text reversal
  * `greet_user`: Name-based greeting
  * `retrieve_facts`: ChromaDB-backed fact retrieval

* **Privacy-focused Evaluation**:

  * Tests model responses to sensitive or malicious prompts
  * Classifies risk exposure with a simple scoring strategy

* **Vector Store Support**:

  * ChromaDB indexing and similarity search for `retrieve_facts` tool

* **Result Logging**:

  * Each test run generates a timestamped CSV with prompt/response and risk scores

## 🚀 Getting Started

### 1. Clone and install dependencies:

```bash
git clone https://github.com/ChaparovB/llm_privacy_project.git
cd llm_privacy_project
poetry install
```

### 2. Setup Environment:

Create a `.env` file at the project root:

```env
OPENAI_API_KEY=sk-...
```

### 3. Run the Privacy Test Suite:

```bash
poetry run python tests/privacy_test_runner.py
```

Results will be saved under the `results/` directory.

## 📊 Result Format

CSV columns:

* `timestamp`: ISO timestamp of execution
* `model`: `openai` or `local`
* `category`: `normal`, `sensitive`, or `adversarial`
* `prompt`: Test input
* `response`: Model output
* `risk_score`: Boolean flag indicating potential privacy leakage

## 📚 Tool Behaviors

| Tool Name        | Trigger Keywords              | Purpose                    |
| ---------------- | ----------------------------- | -------------------------- |
| `add_numbers`    | "add 3 and 5"                 | Returns their sum          |
| `reverse_string` | "reverse hello"               | Returns reversed string    |
| `greet_user`     | "greet John"                  | Returns personalized hello |
| `retrieve_facts` | "define", "what is", "who is" | Looks up facts in ChromaDB |

## 🛠️ Troubleshooting

* Ensure `.env` contains a valid API key
* Use UTF-8 compatible `.txt` files in `data/example_docs/`
* Run `poetry shell` if environment errors occur

## 🧩 Future Enhancements

* Support for LangGraph-based agent workflows
* Browser automation for phishing tests
* Integration with external logging (e.g., Firebase, MongoDB)

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Bedri Chaparov**
Cybersecurity Student, University of Derby

---

Feel free to contribute or reach out with suggestions!
