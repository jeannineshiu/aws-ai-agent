# scripts/fetch_aws_docs.py
import os
import json
import time
import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = "data/raw/aws_docs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# AWS documentation pages to scrape
DOC_URLS = [
    # SageMaker
    "https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html",
    "https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html",
    "https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-deployment.html",
    "https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-automate-model-development.html",
    "https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html",
    "https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html",
    "https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html",
    "https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-fairness-and-explainability.html",
    # Bedrock
    "https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html",
    "https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html",
    "https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html",
    "https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html",
    "https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html",
    "https://docs.aws.amazon.com/bedrock/latest/userguide/fine-tuning.html",
    # Rekognition
    "https://docs.aws.amazon.com/rekognition/latest/dg/what-is.html",
    "https://docs.aws.amazon.com/rekognition/latest/dg/labels.html",
    "https://docs.aws.amazon.com/rekognition/latest/dg/faces.html",
    # Comprehend
    "https://docs.aws.amazon.com/comprehend/latest/dg/what-is.html",
    "https://docs.aws.amazon.com/comprehend/latest/dg/how-it-works.html",
    # Lambda (for ML inference use cases)
    "https://docs.aws.amazon.com/lambda/latest/dg/welcome.html",
    "https://docs.aws.amazon.com/lambda/latest/dg/lambda-ml.html",
]


def scrape_page(url: str) -> dict | None:
    """Scrape a single AWS documentation page."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (educational research bot)"}
        response = httpx.get(url, headers=headers, timeout=15, follow_redirects=True)

        if response.status_code != 200:
            print(f"  HTTP {response.status_code}: {url}")
            return None

        soup = BeautifulSoup(response.text, "lxml")

        # Extract main content (AWS docs use #main-content or .awsdocs-container)
        main = (
            soup.find("div", {"id": "main-content"})
            or soup.find("div", {"class": "awsdocs-container"})
            or soup.find("main")
            or soup.find("article")
        )

        if not main:
            print(f"  No main content found: {url}")
            return None

        # Remove navigation, breadcrumbs, feedback widgets
        for tag in main.find_all(["nav", "footer", "script", "style"]):
            tag.decompose()
        for tag in main.find_all(class_=["feedback", "prev-next", "awsdocs-filter-selector"]):
            tag.decompose()

        # Extract title
        title = ""
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)

        # Extract clean text
        text = main.get_text(separator="\n", strip=True)

        # Remove excessive blank lines
        lines = [l for l in text.splitlines() if l.strip()]
        clean_text = "\n".join(lines)

        return {
            "url": url,
            "title": title,
            "text": clean_text,
            "char_count": len(clean_text),
        }

    except Exception as e:
        print(f"  Error scraping {url}: {e}")
        return None


def main():
    results = []
    failed = []

    for i, url in enumerate(DOC_URLS):
        print(f"[{i+1}/{len(DOC_URLS)}] Scraping: {url.split('/')[-1]}")
        doc = scrape_page(url)

        if doc:
            results.append(doc)
            print(f"  OK — {doc['char_count']:,} chars | {doc['title'][:60]}")
        else:
            failed.append(url)

        time.sleep(1)

    # Save to JSON
    output_path = os.path.join(OUTPUT_DIR, "docs.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n=== Summary ===")
    print(f"Success: {len(results)} pages")
    print(f"Failed:  {len(failed)} pages")
    print(f"Total chars: {sum(d['char_count'] for d in results):,}")
    print(f"Saved to: {output_path}")

    if failed:
        print(f"\nFailed URLs:")
        for url in failed:
            print(f"  {url}")


if __name__ == "__main__":
    main()