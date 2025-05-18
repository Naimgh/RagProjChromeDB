from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
#from vector import retriever
from vectorelocalmemory import retriever

model = OllamaLLM(model="llama3.2")

#template = """
#You are an exeprt in answering questions about a pizza restaurant

#Here are some relevant reviews: {reviews}

#Here is the question to answer: {question}
#"""
#prompt = ChatPromptTemplate.from_template(template)

template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
2. If you find the answer, write the answer in a concise way with five sentences maximum.

{context}

Question: {question}

Helpful Answer:
"""
prompt = ChatPromptTemplate.from_template(template)



chain = prompt | model

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    context = retriever.invoke(question)
    result = chain.invoke({"context": context, "question": question})
    print(result)
