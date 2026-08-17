import os
import requests
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_tavily import TavilySearch

from langchain.agents import create_agent


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

search_tool = TavilySearch(
    max_results=5,
)

# print(search_tool.invoke("Who is the current president of the United States?"))

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=GROQ_API_KEY
)

# result = llm.invoke("Hey there! Can you tell me a joke?")
# print(result.content)

agent = create_agent(
    model = llm,
    system_prompt = "You are a helpful assistant that can answer questions using the Tavily search tool.",
    tools = [search_tool]
)