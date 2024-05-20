# libraries
import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
import time
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if "vectors" not in st.session_state:
    st.session_state.embeddings = OpenAIEmbeddings()
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    

st.title("GROQ")
llm = ChatGroq(groq_api_key=groq_api_key)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the following context:
    Always recheck the answer to make sure it is correct. Context is delimited by ```.
    ```
    {context}
    ```

    Question: {input}
    """
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()

retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt_input = st.text_input("Enter your question")

if prompt_input:
    start = time.process_time()
    response = retrieval_chain.invoke({"input" : prompt_input})
    print("Response time : ", time.process_time() - start)
    st.write(response["answer"])



