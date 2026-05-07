"""20 Finnish culture questions for RAG-QA evaluation — covers all corpus topics."""

QA_TEST_QUESTIONS = [
    # History
    {
        "question": "When did Finland declare independence and from whom?",
        "answer_contains": ["1917", "Russia"],
        "doc_id": "corp_014",
    },
    # Festivals & Traditions
    {
        "question": "What is Juhannus and how do Finns celebrate it?",
        "answer_contains": ["midsummer", "bonfire", "sauna", "lake"],
        "doc_id": "corp_015",
    },
    # Language & Culture
    {
        "question": "What are the official languages of Finland?",
        "answer_contains": ["Finnish", "Swedish"],
        "doc_id": "corp_001",
    },
    {
        "question": "What is the Finnish national epic called?",
        "answer_contains": ["Kalevala"],
        "doc_id": "corp_017",
    },
    {
        "question": "Who is Finland's most famous composer?",
        "answer_contains": ["Jean Sibelius", "Sibelius"],
        "doc_id": "corp_017",
    },
    # Geography & Nature
    {
        "question": "What is the capital of Finland?",
        "answer_contains": ["Helsinki"],
        "doc_id": "corp_001",
    },
    {
        "question": "What is Finland called due to its many lakes?",
        "answer_contains": ["Land of a Thousand Lakes"],
        "doc_id": "corp_005",
    },
    {
        "question": "What natural phenomenon can be seen in Lapland during winter?",
        "answer_contains": ["northern lights", "aurora borealis"],
        "doc_id": "corp_005",
    },
    # Wildlife
    {
        "question": "What kind of wildlife lives in Finland?",
        "answer_contains": ["wolves", "bears", "lynx", "reindeer"],
        "doc_id": "corp_013",
    },
    {
        "question": "What is the Saimaa ringed seal?",
        "answer_contains": ["endangered", "seal", "Finland"],
        "doc_id": "corp_013",
    },
    # Food & Daily Life
    {
        "question": "What are some traditional Finnish foods?",
        "answer_contains": ["rye bread", "fish", "salmon soup", "Karjalanpiirakka"],
        "doc_id": "corp_004",
    },
    {
        "question": "What role does sauna play in Finnish culture?",
        "answer_contains": ["sauna", "tradition", "health", "relaxation"],
        "doc_id": "corp_015",
    },
    # Education & Society
    {
        "question": "How does Finland's education system rank internationally?",
        "answer_contains": ["best", "highly", "top"],
        "doc_id": "corp_008",
    },
    {
        "question": "What is Finland's approach to gender equality?",
        "answer_contains": ["equality", "parental", "women"],
        "doc_id": "corp_020",
    },
    # Arts & Design
    {
        "question": "What is Finland known for in art and design?",
        "answer_contains": ["design", "architecture", "Aalto", "arts"],
        "doc_id": "corp_017",
    },
    # Technology
    {
        "question": "What famous technology company originated in Finland?",
        "answer_contains": ["Nokia"],
        "doc_id": "corp_009",
    },
    # Sports & Climate
    {
        "question": "What is the most popular sport in Finland?",
        "answer_contains": ["ice hockey"],
        "doc_id": "corp_010",
    },
    {
        "question": "What are the four seasons like in Finland?",
        "answer_contains": ["winter", "summer", "snow", "midnight sun"],
        "doc_id": "corp_016",
    },
    # Healthcare & Housing
    {
        "question": "How is healthcare organized in Finland?",
        "answer_contains": ["public", "universal", "free", "government"],
        "doc_id": "corp_007",
    },
    {
        "question": "What is a typical Finnish summer cottage?",
        "answer_contains": ["cottage", "lake", "sauna", "nature"],
        "doc_id": "corp_018",
    },
]
