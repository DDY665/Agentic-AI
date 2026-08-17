import os
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv


load_dotenv()

model = init_chat_model(
    model = "google_genai:gemini-3.6-flash",
    max_retries=10,
    timeout=120,
)

# response = model.invoke("What is the synopsis of Avengers: Doomsday?")

# print(response.content)

#batch

conversation = [
    {"role": "system", "content": "You are a helpful assistant that translates English to French."},
    {"role": "user", "content": "Translate: I love programming."},
    {"role": "assistant", "content": "J'adore la programmation."},
    {"role": "user", "content": "Translate: I love building applications."}
]

response2 = model.invoke(conversation)

conversation = [
    SystemMessage("You are a helpful assistant that translates English to French."),
    HumanMessage("Translate: I love programming."),
    AIMessage("J'adore la programmation."),
    HumanMessage("Translate: I love building applications.")
]

response = model.invoke(conversation)
# print(response)

# print(response2.content)

#Streaming

# full = None
# for chunk in model.stream("what happened to Dr. Dooms mother in the comics?"):
#     print(chunk.text, end="|", flush=True)


# Batch

# responses = model.batch([
#     "Why do parrots have colorful feathers?",
#     "How do airplanes fly?",
#     "What is quantum computing?"
# ])

# for response in responses:
#     print(response)

for response in model.batch_as_completed([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
]):
    print(response)