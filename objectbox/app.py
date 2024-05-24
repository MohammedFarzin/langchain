import os
import streamlit as st
import time
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_objectbox import ObjectBox
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model="mixtral-8x7b-32768")
prompt = ChatPromptTemplate.from_template("""
    Answer the question based on the following context. Use the context as the basis and don't try to make new reason way to answer the question.
    Always recheck the answer to make sure it is correct. Context is delimited by ````.
    ````{context}````
    Question: {input}""")


def vector_embedings():
    try:
        if "vectors" not in st.session_state:
            st.session_state.embeddings = OpenAIEmbeddings()
            st.session_state.loader = PyPDFDirectoryLoader("./objectbox/pdfs")
            st.session_state.document = st.session_state.loader.load()
            print(len(st.session_state.document))
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.document)
            st.session_state.vectors = ObjectBox.from_documents(st.session_state.final_documents, st.session_state.embeddings, embedding_dimensions=1024)
    except Exception as e:
        print(e)

st.title("GROQ")

if st.button("Document Generator"):
    vector_embedings()
    st.write("Vectors created")

input_prompt = st.text_input("Enter your question")

if input_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({"input" : input_prompt})
    print("Response time : ", time.process_time() - start)
    st.write(response["answer"])




