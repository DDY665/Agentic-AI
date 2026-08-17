import os
from langchain.agents import create_agent
from pydantic import BaseModel
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

class Answer(BaseModel):
    summary : str
    confidence : float

@tool
def search(query: str) -> str:
    """search for information"""
    return f"results for : {query}"



agent = create_agent(
    model = "google_genai:gemini-3.6-flash",
    tools = [search],
    system_prompt = "You are an helpful assistant. Be consise and accurate",
    response_format = Answer
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is the time now?"
        }
    ]
})

print(result["structured_response"])