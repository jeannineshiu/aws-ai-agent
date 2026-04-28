# scripts/test_sql_queries.py
import sqlite3
import pandas as pd

DB_PATH = "data/processed/issues.db"
conn = sqlite3.connect(DB_PATH)

queries = {
    "Most common AWS services by question volume": """
        SELECT
            CASE
                WHEN tags LIKE '%amazon-sagemaker%' THEN 'SageMaker'
                WHEN tags LIKE '%amazon-bedrock%' THEN 'Bedrock'
                WHEN tags LIKE '%aws-lambda%' THEN 'Lambda'
                WHEN tags LIKE '%amazon-rekognition%' THEN 'Rekognition'
                WHEN tags LIKE '%amazon-comprehend%' THEN 'Comprehend'
                ELSE 'Other'
            END as service,
            COUNT(*) as questions,
            ROUND(AVG(score), 2) as avg_score,
            ROUND(AVG(answer_count), 2) as avg_answers
        FROM stackoverflow
        GROUP BY service
        ORDER BY questions DESC
    """,

    "Unanswered questions per service (pain points)": """
        SELECT
            CASE
                WHEN tags LIKE '%amazon-sagemaker%' THEN 'SageMaker'
                WHEN tags LIKE '%amazon-bedrock%' THEN 'Bedrock'
                WHEN tags LIKE '%aws-lambda%' THEN 'Lambda'
                ELSE 'Other'
            END as service,
            COUNT(*) as unanswered,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) as pct
        FROM stackoverflow
        WHERE answer_count = 0
        GROUP BY service
        ORDER BY unanswered DESC
    """,

    "Question volume trend by year": """
        SELECT
            SUBSTR(created_at, 1, 4) as year,
            COUNT(*) as questions
        FROM stackoverflow
        GROUP BY year
        ORDER BY year
    """,

    "Top GitHub repos by open issues": """
        SELECT repo, COUNT(*) as open_issues
        FROM issues
        WHERE state = 'open'
        GROUP BY repo
        ORDER BY open_issues DESC
    """,
}

for title, sql in queries.items():
    print(f"\n=== {title} ===")
    df = pd.read_sql(sql, conn)
    print(df.to_string(index=False))

conn.close()