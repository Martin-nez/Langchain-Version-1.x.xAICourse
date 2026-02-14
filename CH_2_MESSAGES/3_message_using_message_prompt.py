
# Message promps
# Alternatively, we can pass in a list of messages instead of a string prompt to the model by
# providing a list of message objects.

# Use message prompts when:
# - you want to retain conversation history i.e managing multi-turn conversations
# - working with multimodal content(images, audio, files)
# - you want to leverage system instructions to guide the model's behavior

from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model ="gpt-5-nano", temperature=0)

# message must always be in a list 

# using message objects
message_1 = [
    SystemMessage(content = "You are a music expert."),
    HumanMessage(content = "write short note on 'chord progression across different music genres' in 50 words"),
    AIMessage(content= "chord I, IV, V.......") # this is optional
]


# using dictionaries
message_2 = [
    {"role": "system", "content":"You are a music expert."},
    {"role": "human", "content": "write short note on 'chord progression across different music genres' in 50 words"},
    # it's either you use "human" or "user" for human message.
    {"role": "assistant", "content": "chord I, IV, V......."} # this is optional

]

response = llm.invoke(message_1)
print(response.content)

response = llm.invoke(message_2)
print(response.content)


# Notes

## System message
# A SystemMessage represent an initial set of instructions that primes the model’s behavior. 
# You can use a system message to set the tone, define the model’s role, and establish guidelines for responses.

# Human message
# A HumanMessage represents user input and interactions. 
# They can contain text, images, audio, files, and any other amount of multimodal content.

# AI message
# An AIMessage represents the output of a model invocation. 
# They can include multimodal data, tool calls, and provider-specific metadata that you can later access.
