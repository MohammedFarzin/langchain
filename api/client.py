import requests
import streamlit as st
from aiohttp import ClientSession
import asyncio


def get_response_india(input_text):
    url = "http://localhost:11434/india/invoke"
    data = {
        "input": {"law":input_text}
    }
    response = requests.post(url, json=data)
    return response.json()["output"]["content"]

def get_response_uae(input_text):
    url = "http://localhost:11434/uae/invoke"
    data = {
        "input": {"law":input_text}
    }
    response = requests.post(url, json=data)
    print(response)
    return response.json()




st.title("Langchain Chatbot Using FastAPI")
text_input1 = st.text_input("Write an law in India")
text_input2 = st.text_input("Write an law in UAE")

if text_input1:
    response = get_response_india(text_input1)
    st.write(response)

if text_input2:
    response = get_response_uae(text_input2)
    st.write(response)