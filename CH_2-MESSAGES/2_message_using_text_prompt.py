
# Text prompts
# text prompts are strings - used to generate straightforward tasks
# where you don't need to retain conversation history

# Use text prompts when:
# - you have a simple task that doesn't require context from previous intertions
# - you don't need conversation history
# - you want minimal code complexity

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv() 

llm = init_chat_model(model="gpt-5-nano", temperature=0)

response = llm.invoke("write a summary note about langchain in 50 words.")
print(response.content)

