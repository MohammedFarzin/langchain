 # import libraries
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

print(os.getenv("OPENAI_API_KEY"))
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# create FastAPI instance
app = FastAPI(
    title="Langchain API",
    description="Langchain API for AI models",
    version="0.1.0",
)

add_routes(
    app,
    ChatOpenAI(),
    path="/openai"
)

# create a prompt
prompt1 = ChatPromptTemplate.from_messages(["what are the charges imposed on {law} in india?"])
prompt2 = ChatPromptTemplate.from_messages(["what are the charges imposed on {law} in uae?"])

# create a chat model
llm1 = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))
llm2 = Ollama(model="llama2")

# add routes
add_routes(app, prompt1|llm1, path="/india")
add_routes(app, prompt2|llm2, path="/uae")



if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=11434)