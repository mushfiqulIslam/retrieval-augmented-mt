"""Test questions about Finnish culture for RAG-QA evaluation."""

QA_TEST_QUESTIONS = [
    {
        "question": "What are the official languages of Finland?",
        "answer_contains": ["Finnish", "Swedish"],
        "doc_id": "corp_001",
    },
    {
        "question": "When did Finland declare independence and from whom?",
        "answer_contains": ["1917", "Russia"],
        "doc_id": "corp_014",
    },
    {
        "question": "What is the capital of Finland?",
        "answer_contains": ["Helsinki"],
        "doc_id": "corp_001",
    },
    {
        "question": "What is the Finnish national epic called?",
        "answer_contains": ["Kalevala"],
        "doc_id": "corp_017",
    },
    {
        "question": "What natural phenomenon can be seen in Lapland during winter?",
        "answer_contains": ["northern lights", "aurora borealis"],
        "doc_id": "corp_005",
    },
    {
        "question": "What are some traditional Finnish foods?",
        "answer_contains": ["rye bread", "fish", "potatoes", "berries", "salmon soup"],
        "doc_id": "corp_004",
    },
    {
        "question": "What kind of wildlife lives in Finland?",
        "answer_contains": ["wolves", "bears", "lynx", "reindeer"],
        "doc_id": "corp_013",
    },
    {
        "question": "Who is Finland's most famous composer?",
        "answer_contains": ["Jean Sibelius", "Sibelius"],
        "doc_id": "corp_017",
    },
    {
        "question": "What is Finland called due to its many lakes?",
        "answer_contains": ["Land of a Thousand Lakes"],
        "doc_id": "corp_005",
    },
    {
        "question": "How does Finland's education system rank internationally?",
        "answer_contains": ["best", "highly", "top"],
        "doc_id": "corp_001",
    },
    {
        "question": "What is the most popular sport in Finland?",
        "answer_contains": ["ice hockey"],
        "doc_id": "corp_010",
    },
    {
        "question": "When did Finland join the European Union?",
        "answer_contains": ["1995"],
        "doc_id": "corp_014",
    },
    {
        "question": "What is a traditional Finnish pastry?",
        "answer_contains": ["Karjalanpiirakka", "rice porridge"],
        "doc_id": "corp_004",
    },
    {
        "question": "What kind of economy does Finland have?",
        "answer_contains": ["mixed", "manufacturing", "services", "technology"],
        "doc_id": "corp_012",
    },
    {
        "question": "What is the Saimaa ringed seal?",
        "answer_contains": ["endangered", "seal", "Finland"],
        "doc_id": "corp_013",
    },
]
