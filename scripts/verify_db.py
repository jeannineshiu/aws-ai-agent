# scripts/verify_db.py
import sqlite3
import pandas as pd

conn = sqlite3.connect("data/processed/issues.db")

# Basic stats
print("=== Database Summary ===")
df = pd.read_sql("SELECT repo, state, COUNT(*) as count FROM issues GROUP BY repo, state", conn)
print(df.to_string(index=False))

# Sample questions we can answer with SQL
print("\n=== Sample SQL Query ===")
df2 = pd.read_sql("""
    SELECT repo, COUNT(*) as total_issues, AVG(comments) as avg_comments
    FROM issues
    GROUP BY repo
    ORDER BY total_issues DESC
""", conn)
print(df2.to_string(index=False))

conn.close()