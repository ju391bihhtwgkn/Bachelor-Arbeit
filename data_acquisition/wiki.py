import re
import os
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss


os.environ["OPENAI_API_KEY"] = 'sk-proj-lnvXunAC7Xxl7vIpJF0jT3BlbkFJn9AyJpHH8ZyoUSM5kkvF'
os.environ["USER_AGENT"] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

headers = {
    os.environ["USER_AGENT"]
}


loader = WebBaseLoader(
    web_paths=(
    "https://en.wikipedia.org/wiki/Python_(programming_language)",
    "https://de.wikipedia.org/wiki/Java_(Programmiersprache)",
    "https://en.wikipedia.org/wiki/C_Sharp_(programming_language)",
    "https://en.wikipedia.org/wiki/C_(programming_language)",
    "https://en.wikipedia.org/wiki/JavaScript",
    "https://en.wikipedia.org/wiki/C%2B%2B",
    "https://en.wikipedia.org/wiki/Go_(programming_language)",
    "https://en.wikipedia.org/wiki/Visual_Basic_(.NET)",
    "https://en.wikipedia.org/wiki/Fortran",
    "https://en.wikipedia.org/wiki/SQL",
    "https://en.wikipedia.org/wiki/Delphi_(software)",
    "https://en.wikipedia.org/wiki/Rust_(programming_language)",
    "https://en.wikipedia.org/wiki/Ruby_(programming_language)",
    "https://en.wikipedia.org/wiki/Swift_(programming_language)",
    "https://en.wikipedia.org/wiki/PHP"
    )
)

docs = loader.load()

def remove_empty_lines(text):
    return re.sub(r'\n\s*\n', '\n', text)

for doc in docs:
    doc.page_content = remove_empty_lines(doc.page_content)



text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

#print(f"Number of documents: {len(docs)}")
#print(f"Number of splits: {len(splits)}")

with open("documents.txt", "w", encoding="utf-8") as f:
    for doc in splits:
        f.write(doc.page_content + "\n\n")

print(f"Successfully saved {len(splits)} document chunks")