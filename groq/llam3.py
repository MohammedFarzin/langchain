import streamlit as st
from langchain.chains import create_stuff_documents_chain, create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.llms import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()

# load the GROQ and OPENAI API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

st.title("GROQ")

llm = ChatGroq(groq_api_key=groq_api_key)
