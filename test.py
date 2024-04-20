from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate   

llm = Ollama(model="llama3:8b")

ans = llm.invoke("Summarize Quantum Computing in 10 bullet points.")
print(ans)