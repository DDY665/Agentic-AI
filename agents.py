import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from pydantic import BaseModel

load_dotenv()


class Answer(BaseModel):
    summary: str
    confidence: float


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"results for: {query}"


# Initialize the Groq client natively instead of routing via OpenAI
model = ChatGroq(
    model="openai/gpt-oss-120b",  # Highly recommended model for tool/structure accuracy
    temperature=0.0,  # Lower temperature keeps structural output accurate
)

# create_agent automatically resolves how to handle the structured response schema
agent = create_agent(
    model=model,
    tools=[],
    system_prompt="You are a helpful assistant. Be concise and accurate.",
    response_format=Answer,
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Explain what LangChain is."
        }
    ]
})

# print(result["structured_response"])

# print("------------------------------------------------")

# print(result)