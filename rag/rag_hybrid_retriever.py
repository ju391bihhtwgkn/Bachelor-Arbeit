import os, json, csv, re
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

# Rerank
import pandas as pd
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_google_community.vertex_rank import VertexAIRank

from google.cloud import aiplatform

# Prompt_Template
from langchain_core.prompts import ChatPromptTemplate

# Bert
from bert_score import score

# Bleurt
from bleurt import score as bleurt_score

# RAGAS

from langchain.smith import RunEvalConfig
from ragas.integrations.langchain import EvaluatorChain
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

# setup API keys

config = configparser.ConfigParser()
config.read('../config.ini')

os.environ["OPENAI_API_KEY"] = config['openai']['openai_key']
os.environ["LANGCHAIN_API_KEY"] = config['langchain']['langchain_key']
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#os.environ["LANGCHAIN_TRACING_V2"] = "true"

# initialize model

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4)

documents = []
with open("../data_acquisition/hybrid_documents.txt", "r", encoding="utf-8") as f:
    docs_content = f.read().split("\n\n")  
    documents = [Document(page_content=doc) for doc in docs_content if doc.strip()]


# initialize FAISS-Vectorstore

vectorstore = FAISS.from_documents(
    documents = documents,
    embedding = OpenAIEmbeddings(),
)

# setup vectorstore as retriever

faiss_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 5})

bm25_retriever = BM25Retriever.from_documents(documents)

hybrid_retriever = EnsembleRetriever(
    retrievers=[faiss_retriever, bm25_retriever],
    weight=[0.5, 0.5],
)

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
    {"context": hybrid_retriever 
    | format_docs , "question": RunnablePassthrough()}
    | chat_prompt
    | llm
    | StrOutputParser()
)

with open("../evaluation_data/easy_dataset_for_evaluation.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

qa_pairs = qa_data["qa_pairs"]

questions = [item["question"] for item in qa_pairs]
golden_answers = [item["golden_answer"] for item in qa_pairs]

rag_responses = {}
for question in questions:
    response = "".join([chunk for chunk in rag_chain.stream(question)])
    rag_responses[question] = response


rag_responses_list = list(rag_responses.values())

def evaluate_with_bert(responses, golden_answers):
    P, R, F1 = score(responses, golden_answers, lang="en", rescale_with_baseline=True)
    return {"Precision": P.mean().item(), "Recall": R.mean().item(), "F1": F1.mean().item()}

# Bleurt

checkpoint = "../bleurt/bleurt/BLEURT-20"
bleurt_scorer = bleurt_score.BleurtScorer(checkpoint)

def evaluate_with_bleurt(responses, golden_answers):
    score = bleurt_scorer.score(references=golden_answers, candidates=responses)
    #average_score = sum(scores) / len(scores)
    return {"BLEURT-Score": score}

#baseline_scores = evaluate_with_bert(list(baseline_responses.values()), golden_answers)
rag_scores = evaluate_with_bert(rag_responses_list, golden_answers)

#baseline_bleurt_scores = evaluate_with_bleurt(list(baseline_responses.values()), golden_answers)
rag_bleurt_scores = evaluate_with_bleurt(rag_responses_list, golden_answers)

# LLm as judge

def llm_as_judge(question, generated_answer, golden_answer):
    prompt = f"""You are an expert in computer science.
    You are a judge for evaluating AI-generated answers. The user asked: '{question}'.
    The AI provided the following answer:
    {generated_answer}

    The correct or ideal answer is:
    {golden_answer}

    On a scale of 0 to 1, how close is the AI-generated answer to the ideal answer in terms of accuracy, relevance, and completeness?
    Give a score in float for each of those aspects in folgendem Format:
    Accuracy: "Your score"
    Relevance: "Your score"
    Completeness: "Your score"
    Overall: "Your score"
    """
    
    response = llm(prompt)
    return response.content

llm_judgements = {}
for question, generated_answer, golden_answer in zip(questions, rag_responses_list, golden_answers):
    llm_judgements[question] = llm_as_judge(question, generated_answer, golden_answer)
    print(f"LLM Judgement for '{question}': {llm_judgements[question]}")


# RAGAS

examples = [
    {
        "question": questions[i],
        "answer": rag_responses_list[i],
        "ground_truth": golden_answers[i],
        "contexts": [doc.page_content for doc in hybrid_retriever.invoke(questions[i])]
    }
    for i in range(len(questions))
]

evaluators = [
    EvaluatorChain(metric)
    for metric in [
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    ]
]
eval_config = RunEvalConfig(custom_evaluators=evaluators)

score_summaries = {
    'answer_correctness': [],
    'answer_relevancy': [],
    'context_precision': [],
    'context_recall': [],
    'faithfulness': []
}

def calculate_average(scores):
    return sum(scores) / len(scores) if scores else 0

for evaluator in evaluators:
    for example in examples:
        eval_results = evaluator.invoke(example)

        score_summaries[evaluator.metric.name].append(eval_results[f"{evaluator.metric.name}"])

average_scores = {metric: calculate_average(scores) for metric, scores in score_summaries.items()}


for metric, avg_score in average_scores.items():
    print(f"Average score for {metric}: {avg_score}")


#print("Baseline GPT-only Scores:", baseline_scores)
print("RAG Model BERTScores:", rag_scores)
#print("Baseline Bleurt Scores:", baseline_bleurt_scores)
print("RAG Bleurt Scores:", rag_bleurt_scores)

csv_filename = "../evaluation_results/rag_evaluation_results_hybrid_50_50_easy.csv"

csv_headers = [
    "typ", "question", "response", "gold_answers", 
    "BERT_precision", "BERT_recall", "BERT_F1", "BLEURT-Score", 
    "ragas_answer_correctness", "ragas_answer_relevancy", 
    "ragas_context_precision", "ragas_context_recall", "ragas_faithfulness", 
    "llm_accuracy", "llm_relevance", "llm_completeness", "llm_overall"
]

def extract_first_float(text):
    match = re.search(r"\d+\.\d+", text)
    if match:
        return float(match.group())
    return None

with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=csv_headers)
    writer.writeheader()
    
    for i, question in enumerate(questions):
        response = rag_responses_list[i]
        golden_answer = golden_answers[i]
        context = "\n\n".join(examples[i]["contexts"])

        llm_scores = llm_judgements[question]

        llm_scores_dict = {}
        

        for line in llm_scores.splitlines():
            if "Accuracy:" in line:
                llm_scores_dict["llm_accuracy"] = extract_first_float(line)
            elif "Relevance:" in line:
                llm_scores_dict["llm_relevance"] = extract_first_float(line)
            elif "Completeness:" in line:
                llm_scores_dict["llm_completeness"] = extract_first_float(line)
            elif "Overall:" in line:
                llm_scores_dict["llm_overall"] = extract_first_float(line)
        

        writer.writerow({
            "typ": "hybrid_50_50",
            "question": question,
            "response": response,
            "gold_answers": golden_answer,
            "BERT_precision": rag_scores["Precision"],
            "BERT_recall": rag_scores["Recall"],
            "BERT_F1": rag_scores["F1"],
            "BLEURT-Score": rag_bleurt_scores["BLEURT-Score"][i],
            "ragas_answer_correctness": score_summaries['answer_correctness'][i],
            "ragas_answer_relevancy": score_summaries['answer_relevancy'][i],
            "ragas_context_precision": score_summaries['context_precision'][i],
            "ragas_context_recall": score_summaries['context_recall'][i],
            "ragas_faithfulness": score_summaries['faithfulness'][i],
            "llm_accuracy": llm_scores_dict.get("llm_accuracy", ""),
            "llm_relevance": llm_scores_dict.get("llm_relevance", ""),
            "llm_completeness": llm_scores_dict.get("llm_completeness", ""),
            "llm_overall": llm_scores_dict.get("llm_overall", "")
        })