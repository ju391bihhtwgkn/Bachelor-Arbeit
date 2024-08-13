import os
import configparser

# Outputparser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Vektorstore
from langchain_community.vectorstores import FAISS

# Open Ai
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# Retrievers
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from langchain_community.docstore.document import Document

# Prompt_Template
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# setup API keys

config = configparser.ConfigParser()
config.read('../config.ini')

os.environ["OPENAI_API_KEY"] = config['openai']['openai_key']
os.environ["LANGCHAIN_API_KEY"] = config['langchain']['langchain_key']
#os.environ["LANGCHAIN_TRACING_V2"] = "true"


# initialize model

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4)

documents = []
with open("../data_acquisition/documents.txt", "r", encoding="utf-8") as f:
    docs_content = f.read().split("\n\n")  
    documents = [Document(page_content=doc) for doc in docs_content if doc.strip()]


# initialize FAISS-Vectorstore

vectorstore = FAISS.from_documents(
    documents = documents,
    embedding = OpenAIEmbeddings(),
)

# setup vectorstore as retriever

faiss_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 10})

bm25_retriever = BM25Retriever.from_documents(documents)

hybrid_retriever = EnsembleRetriever(
    retrievers=[faiss_retriever, bm25_retriever],
    weight=[0.5, 0.5],
)

retriever = hybrid_retriever

# costum prompt template

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use as many sentences as you need and keep the answer as concise as possible. Only answer questions
about Programming languages. If you have to list things, list as many as you can, don't write some 
of them and then only list a few.

{context}

Question: {question}

Helpful Answer:"""

costum_prompt = PromptTemplate.from_template(template)

# ChatPromptTemplate

chat_template = ChatPromptTemplate([
                ("system", "You are a helpful AI bot. You only anser questions about programming languages."
                "Answer the questions in maximum 6 sentences and keep the answer as concise as possible"
                "use this context to answer the question {context}"),
                ("human", "{question}")
])

chat_template2 = """You are having a conversation with a user. Use the following pieces of context
to respond to the user's question appropriately. If you don't know the answer, just say that you don't know.
Ensure your response is relevant to programming languages and based on the context provided.

{context}

User: {question}
Response:"""

chat_prompt = ChatPromptTemplate.from_template(chat_template2)


chat_prompt = chat_template

# doc formatting

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG-chain

rag_chain = (
    {"context": retriever | format_docs , "question": RunnablePassthrough()}
    | chat_prompt
    | llm
    | StrOutputParser()
)

questions = [
    # Yes/No Questions

    "Which arithmetic operations does Python have?",
    "Is JavaScript case-sensitive?",
    #"Does C++ support operator overloading?",
    #"Does Rust support farbage collection?",
    #"Can Java be used as a frontend technology?",

    # Comparative Questions

    #"Is C faster than Python?",
    #"Does Java have a stricter type checking than JavaScript",
    #"Is Ruby more similar to Perl or Python?",
    #"Is Python more beginner-friendly than C++?",
    #"is RUst safer in memory management compared to C?",

    # Generic Questions

    #"What does the public keyword mean in Java?",
    #"What is a goroutine in Go?",
    #"How do you define a function in Ruby?",
    #"How do you declare a variable in Javascript?",
    #"What is the purpose of the cargo tool in Rust?",

    # Intersection Questions

    #"Which versions of Java introduced both generic and annotiations?",
    #"Which versions of Python support both f-strings and type hints?",
    #"Which versions of C++ support both smart pointers and lambda expressions?",
    #"Which versions of Swift support both protocol-oriented programming and error handling?",
    #"Which versions of Rust include both async/await and the borrow checker?",

    # Ordinal Questions

    #"What was the first version of Python to introduce the with statement?",
    #"In which iteration of the ECMAScript standard was let introduced in JavaScript?",
    #"What was the first official release of Rust?",
    #"What version of Java introduced lambdas?",
    #"Which version of Go introduced modules for dependency management?",

    # Count Questions

    #"How many primitive data types does Java have, name them?",
    #"What are the built-in data types in Ruby, name please?",
    #"Which programming languages do have garbage collection?",
    #"Which arithmetic operations does Python have?",
    #"What are the primitive data types in Java?",

    # Difference Questions

    #"What is the difference between Java and C++ regarding memory management?",
    #"How does Python's range function differ in Python 2 and Python 3?",
    #"What is the difference between let and var in JavaScript?",
    #"How does error handling differ in Go compared to traditional try-catch in languages like Java?",
    #"What is the difference between match expressions in Rust and switch statements in C#?",

    # Superlative Questions

    #"What is the most commonly used Python library for data analysis?"
    #"What is the most popular IDE for Java development?"
    #"What is the most widely used JavaScript framework for front-end development?"
    #"What is the most popular web framework in Ruby?"
    #"What is the most commonly used build tool for C# applications?"

    # Multihop Questions

    #"In which versions of Python were both asyncio introduced and type annotations standardized?"
    #"What tools and languages do you need to build a full-stack web application using JavaScript?"
    #"How do you set up a continuous integration pipeline for a Go project using GitHub Actions?"
    #"In what order do you initialize a Rust project, create a basic function, and handle an error?"
    #"How would you migrate a Java application from JDK 8 to JDK 11 and replace the use of PermGen with Metaspace?"



]

# generation

for question in questions:
    print(f"Question: {question}")
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)
    print("\n")