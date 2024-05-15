from fastapi import FastAPI, Request
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langserve import add_routes
from dotenv import load_dotenv
import uvicorn
import os
from langchain_community.llms import Ollama

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
print(os.environ["OPENAI_API_KEY"], 'anthropic', os.environ["ANTHROPIC_API_KEY"])
app = FastAPI(
    title="Chatbot",
    description="Chatbot API using OpenAI",
    version="0.1.0"
)

add_routes(
    app,
    ChatOpenAI(),
    path="/openai"

)

add_routes(
    app, 
    ChatAnthropic(model_name="claude-3-opus-20240229"),
    path="/anthropic"
)

add_routes(
    app,
    Ollama(model="llama3"),
    path="/locallama"
)

# models
model_anthropic = ChatAnthropic(model_name="claude-3-opus-20240229")
model_openai = ChatOpenAI()
model_locallama = Ollama(model="llama3")

# prompts
prompt = ChatPromptTemplate.from_template("Tell me how to make {dish}?")
prompt2 = ChatPromptTemplate.from_template("What are the ingredients for {dish}?")
prompt3 = ChatPromptTemplate.from_template("Can you provide a step-by-step recipe for {dish}?")
prompt4 = ChatPromptTemplate.from_template("Tell me a fun fact about {dish}.")

# add routes for prompt and models
add_routes(
    app,
    prompt | model_anthropic,
    path="/anthropic-dish"
)

add_routes(
    app,
    prompt2 | model_openai,
    path="/openai-dish"
)

add_routes(
    app,
    prompt2 | model_locallama,
    path="/locallama-ingredients"
)
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
