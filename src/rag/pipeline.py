# src/rag/pipeline.py
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_DIR = os.path.join(_PROJECT_ROOT, "data", "processed", "chroma")

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an expert AWS AI/ML assistant.
Answer the question using ONLY the provided context.
If the context does not contain enough information, say so clearly.
Always cite the source page title and service name.

Context:
{context}

Question: {question}

Answer with:
1. A clear explanation
2. Source citations in format: [Source: <title> | <service>]
""")


class RAGPipeline:
    def __init__(self, k: int = 4):
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-small", timeout=30)
        self.vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=self.embedding,
            collection_name="aws_docs",
        )
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=60)

    def retrieve(self, query: str) -> list[Document]:
        """Retrieve relevant chunks for a query."""
        return self.retriever.invoke(query)

    def format_context(self, docs: list[Document]) -> str:
        """Format retrieved docs into a context string."""
        sections = []
        for doc in docs:
            title = doc.metadata.get("title", "Unknown")
            service = doc.metadata.get("service", "AWS")
            source = doc.metadata.get("source", "")
            sections.append(
                f"[{title} | {service}]\n{doc.page_content}\nURL: {source}"
            )
        return "\n\n---\n\n".join(sections)

    def run(self, query: str) -> dict:
        """Run full RAG pipeline: retrieve → format → generate."""
        # Retrieve
        docs = self.retrieve(query)
        context = self.format_context(docs)

        # Generate
        prompt = RAG_PROMPT.format_messages(
            context=context,
            question=query,
        )
        response = self.llm.invoke(prompt)

        # Build citations list
        citations = [
            {
                "title": doc.metadata.get("title", ""),
                "service": doc.metadata.get("service", ""),
                "url": doc.metadata.get("source", ""),
            }
            for doc in docs
        ]

        return {
            "query": query,
            "answer": response.content,
            "citations": citations,
            "retrieved_texts": [doc.page_content for doc in docs],
            "source_count": len(docs),
        }