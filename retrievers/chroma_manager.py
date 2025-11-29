import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Define where the knowledge text is
DATA_DIR = "data/example_docs"
PERSIST_DIR = "chroma_store"

# Load and chunk documents
documents = []
for file in os.listdir(DATA_DIR):
    if file.endswith(".txt"):
        loader = TextLoader(os.path.join(DATA_DIR, file), encoding="utf-8")
        documents.extend(loader.load())

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Build or load the vectorstore
if os.path.exists(PERSIST_DIR):
    print("📁 Loading existing Chroma vector store...")
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=OpenAIEmbeddings())
else:
    print("🔧 Building new Chroma vector store...")
    vectorstore = Chroma.from_documents(
        docs, OpenAIEmbeddings(), persist_directory=PERSIST_DIR
    )

# ✅ Export this so tools can use it
retriever = vectorstore.as_retriever()
def retrieve_top_fact(query: str, k: int = 1) -> str:
    """
    Retrieve the top-matching fact from the Chroma vector store.
    """
    docs = retriever.get_relevant_documents(query)
    return docs[0].page_content if docs else "No match found."
