import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import time
import warnings

warnings.filterwarnings("ignore")

load_dotenv()

# Load the GROQ and OPENAI API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

st.title("GROQ")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """Answer the question based on the following context:
    Always recheck the answer to make sure it is correct. Context is delimited by ````.
    ````{context}````
    Question: {input}"""
)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./groq/research_paper")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.write("Vector Embedding Done")

prompt1 = st.text_input("Enter your question from documents")

if st.button("Document Embedding"):
    vector_embedding()

if prompt1:
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({"input": prompt1})
        st.write(f"Response time: {time.process_time() - start}")
        st.write(response['answer'])
    else:
        st.error("Please perform document embedding first by clicking the 'Document Embedding' button.")
