# scripts/import_stackoverflow.py
import os
import sqlite3
import pandas as pd

CSV_PATH = "data/raw/stackoverflow_posts.csv"
DB_PATH = "data/processed/issues.db"


def import_stackoverflow(csv_path: str, db_path: str):
    """Import Stack Overflow posts into the existing SQLite database."""
    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create stackoverflow table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stackoverflow (
            id INTEGER PRIMARY KEY,
            title TEXT,
            body TEXT,
            tags TEXT,
            score INTEGER,
            view_count INTEGER,
            answer_count INTEGER,
            comment_count INTEGER,
            created_at TEXT,
            closed_at TEXT
        )
    """)

    inserted = 0
    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO stackoverflow
                (id, title, body, tags, score, view_count,
                 answer_count, comment_count, created_at, closed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row.get("Id"),
                row.get("Title", ""),
                row.get("Body", "") or "",
                row.get("Tags", "") or "",
                row.get("Score", 0),
                row.get("ViewCount", 0),
                row.get("AnswerCount", 0),
                row.get("CommentCount", 0),
                row.get("CreationDate", ""),
                row.get("ClosedDate") if pd.notna(row.get("ClosedDate")) else None,
            ))
            inserted += 1
        except Exception as e:
            print(f"  Failed to insert row {row.get('Id')}: {e}")

    conn.commit()
    conn.close()
    print(f"Inserted {inserted} rows into stackoverflow table")


def verify(db_path: str):
    """Print summary of all tables."""
    conn = sqlite3.connect(db_path)

    print("\n=== issues table ===")
    df1 = pd.read_sql("""
        SELECT repo, COUNT(*) as count
        FROM issues
        GROUP BY repo
        ORDER BY count DESC
    """, conn)
    print(df1.to_string(index=False))

    print("\n=== stackoverflow table ===")
    df2 = pd.read_sql("""
        SELECT
            COUNT(*) as total,
            AVG(score) as avg_score,
            AVG(answer_count) as avg_answers,
            MIN(created_at) as earliest,
            MAX(created_at) as latest
        FROM stackoverflow
    """, conn)
    print(df2.to_string(index=False))

    print("\n=== Top tags in stackoverflow ===")
    df3 = pd.read_sql("""
        SELECT tags, COUNT(*) as count
        FROM stackoverflow
        GROUP BY tags
        ORDER BY count DESC
        LIMIT 10
    """, conn)
    print(df3.to_string(index=False))

    conn.close()


def main():
    if not os.path.exists(CSV_PATH):
        print(f"CSV not found at {CSV_PATH}")
        print("Please download from Stack Exchange Data Explorer first")
        return

    import_stackoverflow(CSV_PATH, DB_PATH)
    verify(DB_PATH)


if __name__ == "__main__":
    main()