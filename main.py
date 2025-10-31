from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

# Load environment variables
_ = load_dotenv(find_dotenv())

# Configure Streamlit page
st.set_page_config(
    page_title='Chatbot',
    page_icon='🤖',
    layout='centered'
)

st.title("💬 GenAI Chatbot")

# Initialize LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [] 

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input field
if prompt := st.chat_input("Type your message here..."):
    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Combine chat history into a single string
    history = "\n".join(
        [f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages]
    )

    # Create contextual prompt
    chat_prompt = ChatPromptTemplate.from_template(
        """You are a helpful AI assistant. Continue the conversation based on the history below.

        Conversation history:
        {history}

        User: {input}
        Assistant:"""
    )

    # Run LLM
    chain = chat_prompt | llm
    response = chain.invoke({"history": history, "input": prompt})

    # Extract reply
    reply = response.content

    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(reply)

    # Save assistant reply to history
    st.session_state.messages.append({"role": "assistant", "content": reply})
