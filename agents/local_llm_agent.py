import sys
import os
import re
from typing import List, Optional

# ✅ Add project root to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain.llms.base import LLM
from langchain.agents import initialize_agent, AgentType
from tools.toolkit import custom_tools


class SimpleLocalLLM(LLM):
    """
    A rule-based local LLM that simulates ReAct-style outputs for LangChain.
    Supports: retrieve_facts, add_numbers, reverse_string, greet_user.
    """

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        print("\n⚙️ Inside _call:\n", prompt)
        prompt = prompt.lower()

        # ✅ Always handle the latest observation as final answer
        obs_matches = re.findall(r"observation:\s*(.+)", prompt, re.IGNORECASE)
        if obs_matches:
            last_observation = obs_matches[-1].strip()
            if last_observation and last_observation != "the result of the action":
                print("🧠 Responding to observation:", last_observation)
                output = (
                    "Thought: I now know the final answer\n"
                    f"Final Answer: {last_observation}"
                )
                print("🧪 Returning from _call():\n", output)
                return output

        # ✅ Extract user question
        question_match = re.findall(r"question:\s*(.+)", prompt, re.IGNORECASE)
        if question_match:
            question = question_match[-1].strip()
        elif prompt.strip() and not prompt.strip().startswith("observation:"):
            question = prompt.strip()
        else:
            print("⚠️ No valid question found — returning default.")
            return "Final Answer: the result of the action"

        question = re.sub(r"\s+", " ", question).strip()
        print("🔍 Parsed question:", question)

        # ➕ Add numbers
        match = re.search(r"add\s+(\d+)\s*(?:and|&)?\s*(\d+)", question)
        if match:
            a, b = match.groups()
            output = (
                f"Thought: I should add {a} and {b}\n"
                f"Action: add_numbers\n"
                f"Action Input: \"{a} and {b}\"\n"
            )
            print("🧪 Returning from _call():\n", output)
            return output

        # 🔁 Reverse string
        if "reverse" in question:
            text = question.replace("reverse", "").strip()
            output = (
                "Thought: I should reverse the text\n"
                "Action: reverse_string\n"
                f"Action Input: \"{text}\"\n"
            )
            print("🧪 Returning from _call():\n", output)
            return output

        # 🙋 Greet user
        if "greet" in question:
            name = question.replace("greet", "").replace("the user", "").strip()
            name = name or "User"
            output = (
                "Thought: I should greet the user\n"
                "Action: greet_user\n"
                f"Action Input: \"{name}\"\n"
            )
            print("🧪 Returning from _call():\n", output)
            return output

        # 📚 Retrieve facts
        if any(kw in question for kw in ["what is", "who is", "define", "capital", "explain", "how does"]):
            output = (
                "Thought: I should retrieve facts from memory\n"
                "Action: retrieve_facts\n"
                f"Action Input: \"{question}\"\n"
            )
            print("🧪 Returning from _call():\n", output)
            return output

        # ❌ Fallback
        print("⚠️ No match — using fallback.")
        output = (
            "Thought: I should try retrieving something just in case\n"
            "Action: retrieve_facts\n"
            f"Action Input: \"{question}\"\n"
        )
        print("🧪 Returning from _call():\n", output)
        return output

    @property
    def _llm_type(self) -> str:
        return "simple-local-llm"


# 🧠 Initialize Agent
simple_llm = SimpleLocalLLM()
local_agent = initialize_agent(
    tools=custom_tools,
    llm=simple_llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

# 🧪 CLI for Testing
if __name__ == "__main__":
    print("\n🔒 Local Privacy Agent Ready")
    while True:
        try:
            user_input = input("💬 Ask something (type 'exit' to quit): ")
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input.strip():
                continue
            result = local_agent.invoke({"input": user_input})
            output = result["output"] if isinstance(result, dict) else result
            print("🤖", output)
        except Exception as e:
            print("⚠️ Error:", str(e))
