import os, json, configparser
from langchain_openai import ChatOpenAI
from langchain_community.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# API-Key Konfiguration
config = configparser.ConfigParser()
config.read('../config.ini')
os.environ["OPENAI_API_KEY"] = config['openai']['openai_key']

# Initialisiere LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4)

languages = [
    {"name": "Python", "url": "https://en.wikipedia.org/wiki/Python_(programming_language)"},
    {"name": "Java", "url": "https://de.wikipedia.org/wiki/Java_(Programmiersprache)"},
    {"name": "C#", "url": "https://en.wikipedia.org/wiki/C_Sharp_(programming_language)"},
    {"name": "C", "url": "https://en.wikipedia.org/wiki/C_(programming_language)"},
    {"name": "JavaScript", "url": "https://en.wikipedia.org/wiki/JavaScript"},
    {"name": "C++", "url": "https://en.wikipedia.org/wiki/C%2B%2B"},
    {"name": "Go", "url": "https://en.wikipedia.org/wiki/Go_(programming_language)"},
    {"name": "Visual Basic (.NET)", "url": "https://en.wikipedia.org/wiki/Visual_Basic_(.NET)"},
    {"name": "Fortran", "url": "https://en.wikipedia.org/wiki/Fortran"},
    {"name": "SQL", "url": "https://en.wikipedia.org/wiki/SQL"},
    {"name": "Delphi", "url": "https://en.wikipedia.org/wiki/Delphi_(software)"},
    {"name": "Rust", "url": "https://en.wikipedia.org/wiki/Rust_(programming_language)"},
    {"name": "Ruby", "url": "https://en.wikipedia.org/wiki/Ruby_(programming_language)"},
    {"name": "Swift", "url": "https://en.wikipedia.org/wiki/Swift_(programming_language)"},
    {"name": "PHP", "url": "https://en.wikipedia.org/wiki/PHP"}
]

def generate_questions_for_language(language_name, url):
    prompt = f"""
    You are an AI whose purpose it is to generate question and answer, difficulty pairs for the programming language {language_name}.
    The context is from the Wikipedia page: {url}.
    
    These are the difficulties you need to find a question-answer pair for:
    1. Yes/No Question (Answer is a Yes or No)
    2. Comparative (Compare 2 items by an attribute)

    
    Generate for the language {language_name} 2 questions on each difficulty.
    Return the question, answer, and difficulty pairs in a JSON format.
    """
    
    response = llm(prompt)
    

    try:
        qa_pairs = json.loads(response.content)
    except json.JSONDecodeError:
        print(f"Error decoding JSON for language: {language_name}")
        return None
    
    return qa_pairs

qa_pairs_all = {"qa_pairs": []} 
for language in languages:
    language_name = language["name"]
    language_url = language["url"]
    
    qa_pairs = generate_questions_for_language(language_name, language_url)

    if qa_pairs and "questions" in qa_pairs:
        for question in qa_pairs["questions"]:
            
            qa_pairs_all["qa_pairs"].append({
                "question": question["question"],
                "golden_answer": question["answer"],
                "difficulty": question["difficulty"]
            })
with open("easy_dataset_for_evaluation.json", "w", encoding="utf-8") as f:
    json.dump(qa_pairs_all, f, indent=4)