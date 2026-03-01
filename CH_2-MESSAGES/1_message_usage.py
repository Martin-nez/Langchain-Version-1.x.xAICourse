# MESSAGES
# Messages are the fundamental unit of context for models in LangChain. 
# They represent the input and output of models, carrying both the content and metadata 
# Needed to represent the state of a conversation when interacting with an LLM.

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage 
from dotenv import load_dotenv

#load environments variables from .env file
load_dotenv()

llm = init_chat_model(model="gpt-5-nano", temperature=0)

system_msg = SystemMessage(content="You are a helpful assistant.")
human_msg = HumanMessage(content="How are you?")

# use with chat models
messages = [system_msg, human_msg]
response = llm.invoke(messages)
print(response.content)
