import streamlit as st
from langchain.chains import create_stuff_documents_chain, create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.llms import ChatGroq
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import time
load_dotenv()

# load the GROQ and OPENAI API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

st.title("GROQ")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
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

def vector_embedding():
    st.session_state.embeddings = OpenAIEmbeddings()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.documents = PyPDFDirectoryLoader("./research_paper")
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.documents)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


if  st.button("Document Embedding"):
    vector_embedding()
    st.write("Vector Embedding Done")



prompt = st.text_input("Enter your questsion from documents")

if prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({"input" : prompt})
    print("Response time : ", time.process_time() - start)
    st.write(response['answer'])


