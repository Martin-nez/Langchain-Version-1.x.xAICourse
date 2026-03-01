
# response = [{"name": "addidas", "usage": "sportswear"}], [{"name": "nike", "usage": "sportswearPro"}]
# for i in response:
#     print(i)

from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

response = llm.invoke("Explain AI in simple terms")
print(response.content)