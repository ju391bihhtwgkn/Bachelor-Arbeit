import json

golden_answers = [
    "addition (+), subtraction (-), multiplication (*), division (/), floor division (//), modulo operation (%), ** operator for exponentiation.",
    "Yes, Javascript is case-sensitive",
    "Yes, C++ does support operator overloading",


    "Yes, C is generally faster than Python.",
    "Yes, Java has a stricter type checking than JavaScript",
    "Ruby is more similar to Perl in terms of syntax and some language features.",

    "The public keyword in Java is an access modifier that indicates that the member (such as class, method or variable) is accessible from any other class in any package",
    "A goroutine in Go is a lightweight thread managed by the Go runtime.",
    "In Ruby, a function is defined using the def keyword followed by the function name, optional parameters and a block of code. The function is ended with the end keyword.",
    
    "Java introduced both generics and annotations in Java 5 (JDK 1.5)",
    "Python introduced f-strings and supported type hints in Version 3.6",
    "C++ introduced smart pointers und lambda expressions in Version C++11",

    "The with statement was introduced in Python 2.5.",
    "The let keyword was introduced in ECMAScript 6 (ECMAScript 2015, ES6)",
    "The first offical stable release of Rust was version 1.0.0,",

    "Java has 8 primtive data types",
    "Python has 7 arithemtic data types",
    "Ruby does have 11 built-in data types",

    "Java uses automatic garbage collection to manage memory, whereas C++ relies on manual memory management.",
    "In Python 2, the range() function returns a list of numbers, which is immediately generated and stored in memory. In Python 3, range() behaves like xrange() in Python 2, returning a immutable sequence type(which behaves like a iterator)",
    """
    var is function-scoped. let is block-scoped. var is recognized at the start of the scope, but initialized with undefined
    let is hoisted but not initialized, leading to temporal dead zone, where variable cannot be accessed before its declaration line.
    var allows re-declaration of the same variable within the same scope. let does not allow re-declaration""",


    "The most commonly used Python library for data analysis is Pandas. Pandas provides data structures like DataFrames and Series, which are essential for data manipulation, cleaning, and analysis in Python.",
    "The most popular Integrated Development Environment (IDE) for Java development is IntelliJ IDEA by JetBrains. It is highly regarded for its powerful features, intelligent code completion, and seamless integration with version control systems.",
    "The most widely used JavaScript framework for front-end development is React. Developed by Facebook, React has gained widespread adoption due to its component-based architecture, virtual DOM, and efficient rendering.",

    "Python 3.5 and later versions support both asyncio and type annotations.",
    """
    Frontend: HTML/CSS for structure and styling, JavaScript for interaction, JavaScript framework or library. Backend: Node.js as runtime environment,
    Express.js as a web framework. Database: MongoDB or MySQL/PostgresSQL for data storage. Version Control: Github. Deployment: Docker, Heroku or AWS for cloud deployment,
    """,
    
    """
    Create a .github/workflows directory in repo. Create YAML file in .github/workflows directory. Define the pipeline in YAML file.
    Commit and push the github/workflows/ci.yml file to repo.
    Github actions will automatically run the CL pipeline on each push to the main branch of when a pull request is opened. The pipeline checks out the code,
    sets up the Go environment, installs depencencies and runs the tests.
    """,
]

questions = [

    "Which arithmetic operations does Python have?",
    "Is JavaScript case-sensitive?",
    "Does C++ support operator overloading?",

    "Is C faster than Python?",
    "Does Java have a stricter type checking than JavaScript",
    "Is Ruby more similar to Perl or Python?",
 
    "What does the public keyword mean in Java?",
    "What is a goroutine in Go?",
    "How do you define a function in Ruby?",

    "Which versions of Java introduced both generic and annotiations?",
    "Which versions of Python support both f-strings and type hints?",
    "Which versions of C++ support both smart pointers and lambda expressions?",

    "What was the first version of Python to introduce the with statement?",
    "In which iteration of the ECMAScript standard was let introduced in JavaScript?",
    "What was the first official release of Rust?",

    "How many primitive data types does Java have?",
    "How many built-in data types are in Ruby?",
    "How many arithmetic operations does Python have?",

    "What is the difference between Java and C++ regarding memory management?",
    "How does Python's range function differ in Python 2 and Python 3?",
    "What is the difference between let and var in JavaScript?",


    "What is the most commonly used Python library for data analysis?",
    "What is the most popular IDE for Java development?",
    "What is the most widely used JavaScript framework for front-end development?",


    "In which versions of Python were both asyncio introduced and type annotations standardized?",
    "What tools and languages do you need to build a full-stack web application using JavaScript?",
    "How do you set up a continuous integration pipeline for a Go project using GitHub Actions?",
]

qa_pairs = [
    {
        "question": questions[i],
        "golden_answer": golden_answers[i],
    }
    for i in range(len(questions))
]

with open("golden_dataset.json", "w", encoding="utf-8") as f:
    json.dump({"qa_pairs": qa_pairs}, f, ensure_ascii=False, indent=4)