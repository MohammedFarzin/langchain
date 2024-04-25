from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import langchain
from dotenv import load_dotenv
import streamlit as st
import os

langchain.verbose = False
load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful expert assistant. Please response to the user queries."),
        ("user", "Question: {question}")
    ]
)


# Streamlit framework

st.title("Chatbot")
input_text = st.text_input("Enter your question here:")

llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))