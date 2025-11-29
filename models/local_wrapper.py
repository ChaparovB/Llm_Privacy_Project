from agents.local_llm_agent import local_agent

class LocalAgentWrapper:
    def query(self, prompt: str) -> str:
        try:
            result = local_agent.invoke({"input": prompt})
            return result["output"] if isinstance(result, dict) else str(result)
        except Exception as e:
            return f"[Local Agent Error] {str(e)}"
