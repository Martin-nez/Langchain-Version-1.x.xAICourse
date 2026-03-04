from dotenv import load_dotenv
from langchain.agents import create_agent

from langchain.chat_models import init_chat_model

# Normally an agent would need access to tools call, but for this simple example, we will create an agent without any tools to demonstrate the basic setup
load_dotenv()

model = init_chat_model(
    model=  "llama-3.3-70b-versatile",
    model_provider="groq",
    temperature=0
)


agent = create_agent(
    model= model,
    tools = [],
    system_prompt= " You are a helpful assistant. Answer the user's question to the best of your ability."
)

response = agent.invoke({
    "messages": [{
        "role" : "user", "content" : "What is the capital of Nigeria?"
    }]
})

print(response["messages"][-1].content) # The response is a list of messages,and we print the content of the last message which is the agent's response
