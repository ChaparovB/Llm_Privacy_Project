# tools/toolkit.py

from langchain.agents import Tool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.tools import tool
import os

# ✅ Define your tools

@tool
def add_numbers(input: str) -> str:
    """Add two numbers from a string like '4 and 6'."""
    try:
        parts = [int(x.strip()) for x in input.lower().split("and")]
        return str(sum(parts))
    except Exception as e:
        return f"Error: {e}"

@tool
def reverse_string(text: str) -> str:
    """Reverse the given string."""
    return text[::-1]

@tool
def greet_user(name: str) -> str:
    """Greet the user by name."""
    return f"Hello, {name.strip().capitalize()}! 👋"

# ✅ Retrieval logic
@tool
def retrieve_facts(query: str) -> str:
    """Retrieve facts using a vectorstore backed by ChromaDB."""
    try:
        PERSIST_DIR = "chroma_store"
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=OpenAIEmbeddings()
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return "No relevant facts found."
        raw_chunk = docs[0].page_content
        print("🧠 Raw chunk:\n", raw_chunk)

        # Best match line
        lines = raw_chunk.strip().split("\n")
        best_line = next((line for line in lines if query.lower() in line.lower()), lines[0])
        print("✅ Matched best line:", best_line)
        return best_line.strip()
    except Exception as e:
        return f"Retrieval error: {str(e)}"

# ✅ Bundle into a list
custom_tools = [add_numbers, reverse_string, greet_user, retrieve_facts]
from langchain.tools import tool
from retrievers.chroma_manager import retrieve_top_fact

@tool
def greet_user(name: str) -> str:
    """Greets the user by name."""
    return f"Hello, {name}! How can I help you today?"

@tool
def reverse_string(text: str) -> str:
    """Reverses the given string."""
    return text[::-1]

@tool
def add_numbers(input: str) -> str:
    """Adds two numbers from a string like '4 and 6'."""
    try:
        numbers = [int(n.strip()) for n in input.split("and")]
        return str(sum(numbers))
    except Exception:
        return "Sorry, I couldn't parse the numbers."

@tool
def retrieve_facts(query: str) -> str:
    """
    Retrieves the most relevant fact related to the query from the local Chroma vectorstore.
    Returns the best matched sentence.
    """
    return retrieve_top_fact(query)

# ✅ This must be at the bottom
custom_tools = [greet_user, reverse_string, add_numbers, retrieve_facts]
