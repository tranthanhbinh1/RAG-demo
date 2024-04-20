from langchain_community.llms.ollama import Ollama
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser

# What the fuck

llm = Ollama(model="llama3:8b")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are world class technical documentation writer."),
        ("user", "{input}"),
    ]
)

loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")

docs = loader.load()

embeddings = OllamaEmbeddings()

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

ans = chain.invoke({"input": "how can langsmith help with testing?"})

print(ans)