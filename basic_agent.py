import os
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()


def get_weather(city: str) -> str:
    """Get the weather of the city."""
    return f"Its sunny in {city}!"

agent = create_agent(
    model = "google_genai:gemini-2.5-flash-lite",
    tools = [get_weather],
    system_prompt = "You are a helpful assistant"
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Whats the weather in San Francisco?"    
            }
        ]
    }
)

print(result["messages"][-1].content_blocks)
