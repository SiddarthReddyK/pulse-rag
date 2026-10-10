import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GROQ_API_KEY", "").strip()
print(f"Loaded KEY: {api_key[:10]}...")

if not api_key:
    print("NO API KEY LOADED!")
    exit(1)

from langchain_groq import ChatGroq

try:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, api_key=api_key)
    response = llm.invoke("Testing 1 2 3. Reply 'Test OK'")
    print("SUCCESS. Response:", response.content)
except Exception as e:
    print("API ERROR:", e)
