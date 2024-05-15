import requests
import streamlit as st

def get_openai_response(dish):
    response = requests.post(
        url="http://localhost:8000/openai-dish/invoke",
        json = {"input":{"dish":dish}}
    )
    
    return response.json()['output']['content']


def get_ollama_response(dish):
    response = requests.post(
        url="http://localhost:8000/locallama-ingredients/invoke",
        json = {"input":{"dish":dish}}
    )

    return response.json()["output"]


def get_anthropic_response(dish):
    response = requests.post(
        url="http://localhost:8000/anthropic-dish/invoke",
        json = {"input":{"dish":dish}}
    )
    return response.json()['output']['content']



st.title("Dish Chatbot")    
dish_1 = st.text_input("Enter a dish1:")
if dish_1:
    st.write(get_openai_response(dish_1))

dish_2 = st.text_input("Enter a dish3:")
if dish_2:
    st.write(get_anthropic_response(dish_3))

dish_3 = st.text_input("Enter a dish2:")
if dish_3:
    st.write(get_ollama_response(dish_2))









    
