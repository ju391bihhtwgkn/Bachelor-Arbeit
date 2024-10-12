import tkinter as tk
from tkinter import scrolledtext
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import os
import configparser

# Konfiguriere API-Schlüssel
config = configparser.ConfigParser()
config.read('../config.ini')

os.environ["OPENAI_API_KEY"] = config['openai']['openai_key']

# Initialisiere Modell und Vektordatenbank
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

documents = []
with open("../data_acquisition/documents.txt", "r", encoding="utf-8") as f:
    docs_content = f.read().split("\n\n")
    documents = [Document(page_content=doc) for doc in docs_content if doc.strip()]

vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=OpenAIEmbeddings(),
)

faiss_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 10})

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful AI bot. You only answer questions about programming languages."
     "Answer the questions in maximum 6 sentences and keep the answer as concise as possible."
     "Use this context to answer the question {context}"),
    ("human:", "{question}")
])

def ask_question():
    question = question_entry.get()

    rag_chain = (
        {"context": faiss_retriever | format_docs, "question": RunnablePassthrough()}
        | chat_template
        | llm
        | StrOutputParser()
    )
    response = "".join([chunk for chunk in rag_chain.stream(question)])
    display_response(response)


def display_response(response):
    response_area.configure(state='normal')
    response_area.insert(tk.END, f"Frage: {question_entry.get()}\nAntwort: {response}\n\n")
    response_area.configure(state='disabled')
    question_entry.delete(0, tk.END)

# GUI
app = tk.Tk()
app.title("RAG-basiertes Chatbot")

question_label = tk.Label(app, text="Geben Sie Ihre Frage ein:")
question_label.pack()

question_entry = tk.Entry(app, width=80)
question_entry.pack(pady=5)

ask_button = tk.Button(app, text="Frage stellen", command=ask_question)
ask_button.pack(pady=5)

response_area = scrolledtext.ScrolledText(app, width=80, height=20, state='disabled', wrap=tk.WORD)
response_area.pack(pady=10)

app.mainloop()
