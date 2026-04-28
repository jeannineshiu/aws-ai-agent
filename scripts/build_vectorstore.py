# scripts/build_vectorstore.py
import json
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

CHUNKS_PATH = "data/processed/chunks.json"
CHROMA_DIR = "data/processed/chroma"


def build_vectorstore():
    # Load chunks
    print("Loading chunks...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")

    # Convert to LangChain Document objects
    documents = [
        Document(
            page_content=chunk["text"],
            metadata=chunk["metadata"],
        )
        for chunk in chunks
    ]

    # Build vectorstore
    # text-embedding-3-small: cheap, fast, good enough for retrieval
    print("\nBuilding vectorstore (this will call OpenAI embeddings API)...")
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")

    # Remove old vectorstore if exists
    if os.path.exists(CHROMA_DIR):
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print("Removed old vectorstore")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=CHROMA_DIR,
        collection_name="aws_docs",
    )

    print(f"\n=== Vectorstore Summary ===")
    print(f"Documents stored: {vectorstore._collection.count()}")
    print(f"Saved to: {CHROMA_DIR}")
    return vectorstore


def test_retrieval(vectorstore):
    """Run a few test queries to verify retrieval works."""
    print("\n=== Retrieval Tests ===")

    test_queries = [
        "How do I deploy a model on SageMaker?",
        "What models are available in Bedrock?",
        "How does SageMaker Model Monitor work?",
    ]

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    for query in test_queries:
        print(f"\nQuery: {query}")
        docs = retriever.invoke(query)
        for i, doc in enumerate(docs):
            print(f"  [{i+1}] {doc.metadata['title'][:50]} | {doc.metadata['service']} | {len(doc.page_content)} chars")


def main():
    vectorstore = build_vectorstore()
    test_retrieval(vectorstore)


if __name__ == "__main__":
    main()