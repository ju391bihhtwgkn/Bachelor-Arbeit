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

from google.cloud import aiplatform

# Prompt_Template
from langchain_core.prompts import ChatPromptTemplate

# Bert
from bert_score import score

# Bleurt
from bleurt import score as bleurt_score

# setup API keys

config = configparser.ConfigParser()
config.read('../config.ini')

os.environ["OPENAI_API_KEY"] = config['openai']['openai_key']
os.environ["LANGCHAIN_API_KEY"] = config['langchain']['langchain_key']
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#os.environ["LANGCHAIN_TRACING_V2"] = "true"

PROJECT_ID = "bachelorarbeit-433202"
REGION = "europe-west8"
RANKING_LOCATION_ID = "global"


aiplatform.init(project=PROJECT_ID, location=REGION)

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

retriever = faiss_retriever

# ChatPromptTemplate

chat_template = """You are having a conversation with a user. Use the following pieces of context
to respond to the user's question appropriately. If you don't know the answer, just say that you don't know.
Use as many sentences as you need and keep the answer as concise as possible.
Ensure your response is relevant to programming languages and based on the context provided.  If you have to list things, 
list as many as you can, don't write some of them and then only list a few. Only answer questions about Programming languages.

{context}

User: {question}
Response:"""

chat_prompt = ChatPromptTemplate.from_template(chat_template)

# doc formatting

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# RAG-chain

rag_chain = (
    {"context": hybrid_retriever | format_docs , "question": RunnablePassthrough()}
    | chat_prompt
    | llm
    | StrOutputParser()
)

golden_answers = [
    "addition (+), subtraction (-), multiplication (*), division (/), floor division (//), modulo operation (%), ** operator for exponentiation.",
    "Yes, Javascript is case-sensitive",
    "Yes, C++ does support operator overloading",
    #"No, Rust does not support garbage collection",
    #"Yes, it can be used, but it is not the best solution",

    #"Yes, C is generally faster than Python.",
    #"Yes, Java has a stricter type checking than JavaScript",
    #"Ruby is more similar to Perl in terms of syntax and some language features.",

    #"The public keyword in Java is an access modifier that indicates that the member (such as class, method or variable) is accessible from any other class in any package",
    #"In Ruby, a function is defined using the def keyword followed by the function name, optional parameters and a block of code. The function is ended with the end keyword.",
    #"A goroutine in Go is a lightweight thread managed by the Go runtime.",
    
    #"Java introduced both generics and annotations in Java 5 (JDK 1.5)",
    #"Python introduced f-strings and supported type hints in Version 3.6",
    #"C++ introduced smart pointers und lambda expressions in Version C++11",

    #"The with statement was introduced in Python 2.5.",
    #"The let keyword was introduced in ECMAScript 6 (ECMAScript 2015, ES6)",
    #"The first offical stable release of Rust was version 1.0.0,",

    #"Java has 8 primtive data types",
    #"Python has 7 arithemtic data types",
    #"Ruby does have 11 built-in data types",

    #"Java uses automatic garbage collection to manage memory, whereas C++ relies on manual memory management.",
    #"In Python 2, the range() function returns a list of numbers, which is immediately generated and stored in memory."
    #"In Python 3, range() behaves like xrange() in Python 2, returning a immutable sequence type(which behaves like a iterator)",
    
    #"""
    #var is function-scoped. let is block-scoped. var is recognized at the start of the scope, but initialized with undefined
    #let is hoisted but not initialized, leading to temporal dead zone, where variable cannot be accessed before its declaration line.
    #var allows re-declaration of the same variable within the same scope. let does not allow re-declaration""",


    #"The most commonly used Python library for data analysis is Pandas. Pandas provides data structures like DataFrames and Series, which are essential for data manipulation, cleaning, and analysis in Python.",
    #"The most popular Integrated Development Environment (IDE) for Java development is IntelliJ IDEA by JetBrains. It is highly regarded for its powerful features, intelligent code completion, and seamless integration with version control systems.",
    #"The most widely used JavaScript framework for front-end development is React. Developed by Facebook, React has gained widespread adoption due to its component-based architecture, virtual DOM, and efficient rendering.",

    #"Python 3.5 and later versions support both asyncio and type annotations.",
    #"""
    #Frontend: HTML/CSS for structure and styling, JavaScript for interaction, JavaScript framework or library. Backend: Node.js as runtime environment,
    #Express.js as a web framework. Database: MongoDB or MySQL/PostgresSQL for data storage. Version Control: Github. Deployment: Docker, Heroku or AWS for cloud deployment,
    #""",
    
    #"""
    #Create a .github/workflows directory in repo. Create YAML file in .github/workflows directory. Define the pipeline in YAML file.
    #Commit and push the github/workflows/ci.yml file to repo.
    #Github actions will automatically run the CL pipeline on each push to the main branch of when a pull request is opened. The pipeline checks out the code,
    #sets up the Go environment, installs depencencies and runs the tests.
    #""",


    #"Python has the usual symbols for arithmetic operators such as addition (+), subtraction (-), multiplication (*), and division (/). It also includes floor division (//) and the modulo operation (%) for obtaining the remainder. Additionally, Python uses the ** operator for exponentiation.",
    #"Yes, JavaScript is case-sensitive. This means that variables, functions, and other identifiers in JavaScript must be typed with the exact capitalization each time they are used.",
    #"Yes, C++ does support operator overloading. It allows almost all operators to be overloaded for user-defined types, with a few exceptions like member access and the conditional operator. Overloading operators in C++ is a key aspect of advanced programming techniques and making user-defined types behave like built-in types.",
    #"No, Rust does not support garbage collection. Memory and resources are managed through the ownership system and reference counting, without the need for a garbage collector.",
    #"Yes, Java can be used as a frontend technology for web applications. Java can be used to develop the client-side of Ajax-based applications, as well as for developing Java-Webanwendungen that run in a web browser. Additionally, Java can be used to create Java-Desktop-Anwendungen with graphical user interfaces."

]


questions = [
    # Yes/No Questions

    "Which arithmetic operations does Python have?",
    "Is JavaScript case-sensitive?",
    "Does C++ support operator overloading?",
    #"Does Rust support garbage collection?",
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

    #"How many primitive data types does Java have?",
    #"How many built-in data types are in Ruby?",
    #"Which programming languages do have garbage collection?",
    #"How many arithmetic operations does Python have?",
    #"Which arithmetic operations does Python have?",
    #"What are the primitive data types in Java?",

    # Difference Questions

    #"What is the difference between Java and C++ regarding memory management?",
    #"How does Python's range function differ in Python 2 and Python 3?",
    #"What is the difference between let and var in JavaScript?",
    #"How does error handling differ in Go compared to traditional try-catch in languages like Java?",
    #"What is the difference between match expressions in Rust and switch statements in C#?",

    # Superlative Questions

    #"What is the most commonly used Python library for data analysis?",
    #"What is the most popular IDE for Java development?",
    #"What is the most widely used JavaScript framework for front-end development?",
    #"What is the most popular web framework in Ruby?",
    #"What is the most commonly used build tool for C# applications?",

    # Multihop Questions

    #"In which versions of Python were both asyncio introduced and type annotations standardized?",
    #"What tools and languages do you need to build a full-stack web application using JavaScript?",
    #"How do you set up a continuous integration pipeline for a Go project using GitHub Actions?",
    #"In what order do you initialize a Rust project, create a basic function, and handle an error?",
    #"How would you migrate a Java application from JDK 8 to JDK 11 and replace the use of PermGen with Metaspace?",
]





def gpt_without_rag(question):
    response = llm(question)
    if isinstance(response, dict):
        return response.get('content', '')
    elif hasattr(response, 'content'):
        return response.content
    else:
        return str(response)

baseline_responses = {}
#for question in questions:
#    response = gpt_without_rag(question)
#    baseline_responses[question] = response
#    print(f"Baseline Response for '{question}': {response}")
#    print("\n")

# generation

rag_responses = {}
for question in questions:
    response = "".join([chunk for chunk in rag_chain.stream(question)])
    rag_responses[question] = response
    print(f"RAG Response for '{question}': {response}")
    print("\n")


def llm_as_judge(question, generated_answer, golden_answer):
    prompt = f"""You are an expert in computer science.
    You are a judge for evaluating AI-generated answers. The user asked: '{question}'.
    The AI provided the following answer:
    {generated_answer}

    The correct or ideal answer is:
    {golden_answer}

    On a scale of 0 to 1, how close is the AI-generated answer to the ideal answer in terms of accuracy, relevance, and completeness?
    Give a score for each of those aspects.
    Provide a rating and a short justification for your score."""
    
    response = llm(prompt)
    return response.content

llm_judgements = {}
for question, generated_answer, golden_answer in zip(questions, rag_responses.values(), golden_answers):
    llm_judgements[question] = llm_as_judge(question, generated_answer, golden_answer)
    print(f"LLM Judgement for '{question}': {llm_judgements[question]}")
# Bleurt

#baseline_scores = evaluate_with_bert(list(baseline_responses.values()), golden_answers)
#rag_scores = evaluate_with_bert(list(rag_responses.values()), golden_answers)

#baseline_bleurt_scores = evaluate_with_bleurt(list(baseline_responses.values()), golden_answers)
#rag_bleurt_scores = evaluate_with_bleurt(list(rag_responses.values()), golden_answers)

#print("Baseline GPT-only Scores:", baseline_scores)
#print("RAG Model Scores:", rag_scores)
#print("Baseline Bleurt Scores:", baseline_bleurt_scores)
#print("RAG Bleurt Scores:", rag_bleurt_scores)