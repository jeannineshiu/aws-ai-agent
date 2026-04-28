# scripts/test_setup.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load .env file
load_dotenv()

# Check keys are loaded
api_key = os.getenv("OPENAI_API_KEY")
langsmith_key = os.getenv("LANGSMITH_API_KEY")

print(f"OpenAI key loaded: {'YES' if api_key else 'NO'}")
print(f"LangSmith key loaded: {'YES' if langsmith_key else 'NO'}")

# Test LLM connection
print("\nTesting LLM connection...")
llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke("Say 'setup OK' and nothing else.")
print(f"LLM response: {response.content}")