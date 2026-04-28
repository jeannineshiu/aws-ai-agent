# scripts/fetch_github_issues.py
import os
import json
import time
import sqlite3
import httpx
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}

REPOS = [
    "aws/amazon-sagemaker-examples",
    "aws/sagemaker-python-sdk",
    "awslabs/gluonts",
    "autogluon/autogluon",
    "aws-neuron/aws-neuron-sdk",
    "awslabs/sagemaker-debugger",
]

def fetch_issues(repo: str, max_pages: int = 10) -> list[dict]:
    """Fetch issues from a GitHub repo (both open and closed)."""
    issues = []
    for state in ["open", "closed"]:
        for page in range(1, max_pages + 1):
            url = f"https://api.github.com/repos/{repo}/issues"
            params = {
                "state": state,
                "per_page": 100,
                "page": page,
            }
            response = httpx.get(url, headers=HEADERS, params=params)

            if response.status_code == 404:
                print(f"  Repo not found: {repo}")
                return issues

            if response.status_code == 403:
                print(f"  Rate limited, waiting 60s...")
                time.sleep(60)
                continue

            if response.status_code != 200:
                print(f"  Error {response.status_code} on {repo} page {page}")
                break

            batch = response.json()
            if not batch:
                break

            # Filter out pull requests
            real_issues = [i for i in batch if "pull_request" not in i]
            issues.extend(real_issues)
            print(f"  {repo} | {state} | page {page} | {len(real_issues)} issues")

            # Respect GitHub rate limit
            time.sleep(0.8)

    return issues


def save_to_sqlite(all_issues: list[dict], db_path: str = "data/processed/issues.db"):
    """Save issues to SQLite database."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    if os.path.exists(db_path):
        os.remove(db_path)
        print("Removed old database")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS issues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo TEXT,
            issue_number INTEGER,
            title TEXT,
            body TEXT,
            state TEXT,
            labels TEXT,
            comments INTEGER,
            created_at TEXT,
            closed_at TEXT,
            url TEXT,
            UNIQUE(repo, issue_number)
        )
    """)

    inserted = 0
    for issue in all_issues:
        labels = json.dumps([l["name"] for l in issue.get("labels", [])])
        repo_name = "/".join(issue["repository_url"].split("/")[-2:])
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO issues
                (repo, issue_number, title, body, state, labels,
                 comments, created_at, closed_at, url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                repo_name,
                issue["number"],
                issue.get("title", ""),
                issue.get("body", "") or "",
                issue.get("state", ""),
                labels,
                issue.get("comments", 0),
                issue.get("created_at", ""),
                issue.get("closed_at"),
                issue.get("html_url", ""),
            ))
            inserted += 1
        except Exception as e:
            print(f"  Failed to insert issue {issue.get('number')}: {e}")

    conn.commit()
    conn.close()
    print(f"\nSaved {inserted} issues to {db_path}")


def verify(db_path: str = "data/processed/issues.db"):
    """Print summary of saved data."""
    import pandas as pd
    conn = sqlite3.connect(db_path)

    print("\n=== Database Summary ===")
    df = pd.read_sql("""
        SELECT repo, state, COUNT(*) as count
        FROM issues
        GROUP BY repo, state
        ORDER BY repo, state
    """, conn)
    print(df.to_string(index=False))

    print("\n=== Totals per Repo ===")
    df2 = pd.read_sql("""
        SELECT repo,
               COUNT(*) as total,
               SUM(CASE WHEN state='open' THEN 1 ELSE 0 END) as open,
               SUM(CASE WHEN state='closed' THEN 1 ELSE 0 END) as closed,
               ROUND(AVG(comments), 1) as avg_comments
        FROM issues
        GROUP BY repo
        ORDER BY total DESC
    """, conn)
    print(df2.to_string(index=False))

    conn.close()


def main():
    all_issues = []
    for repo in REPOS:
        print(f"\nFetching: {repo}")
        issues = fetch_issues(repo, max_pages=10)
        all_issues.extend(issues)
        print(f"  Subtotal from {repo}: {len(issues)}")

    print(f"\nTotal issues fetched: {len(all_issues)}")
    save_to_sqlite(all_issues)
    verify()


if __name__ == "__main__":
    main()