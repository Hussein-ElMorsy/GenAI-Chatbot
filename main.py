from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

_ = load_dotenv(find_dotenv())

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    temperature=0.1
)

parser = StrOutputParser()

def chat():
    chat_history = [
        ('system', 'You are helpful chatbot. Be concise and accurate.')
    ]

    print("Langchain Chatbot. Type 'exit' to quit\n")

    while True:
        user_input = input('You: ').strip()

        if user_input.lower() == 'exit':
            break

        chat_history.append(('user', user_input))

        prompt = ChatPromptTemplate.from_messages(chat_history)

        chain = prompt | llm | parser

        response = chain.invoke({})

        print(f"Bot: {response}\n")

        chat_history.append(('assistant', response))

        print('-'*60,'\n')

chat()