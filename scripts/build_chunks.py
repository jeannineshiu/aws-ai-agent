# scripts/build_chunks.py
import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_PATH = "data/raw/aws_docs/docs.json"
OUTPUT_PATH = "data/processed/chunks.json"

# Chunking strategy:
# - chunk_size=800: large enough to contain a complete concept,
#   small enough for precise retrieval
# - chunk_overlap=150: prevents losing context at chunk boundaries
# - separators: respect document structure (headers, paragraphs, sentences)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def build_chunks(docs: list[dict]) -> list[dict]:
    """Split documents into chunks with metadata."""
    all_chunks = []

    for doc in docs:
        # Determine service from URL
        url = doc["url"]
        if "sagemaker" in url:
            service = "SageMaker"
        elif "bedrock" in url:
            service = "Bedrock"
        elif "rekognition" in url:
            service = "Rekognition"
        elif "comprehend" in url:
            service = "Comprehend"
        elif "lambda" in url:
            service = "Lambda"
        else:
            service = "AWS"

        chunks = splitter.split_text(doc["text"])

        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{doc['url'].split('/')[-1]}_{i}",
                "text": chunk_text,
                "metadata": {
                    "source": doc["url"],
                    "title": doc["title"],
                    "service": service,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
            })

        print(f"  {doc['title'][:55]:<55} → {len(chunks)} chunks")

    return all_chunks


def main():
    print("Loading docs...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)

    print(f"Loaded {len(docs)} pages\n")
    print("Splitting into chunks...")
    chunks = build_chunks(docs)

    # Save chunks
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    # Summary
    from collections import Counter
    service_counts = Counter(c["metadata"]["service"] for c in chunks)

    print(f"\n=== Chunking Summary ===")
    print(f"Total chunks: {len(chunks)}")
    print(f"\nChunks per service:")
    for service, count in sorted(service_counts.items(), key=lambda x: -x[1]):
        print(f"  {service:<15} {count}")
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()