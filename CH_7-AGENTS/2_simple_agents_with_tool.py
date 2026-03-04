from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()

model_used = init_chat_model(
    model = "llama-3.3-70b-versatile",
    model_provider= "groq",
    temperature = 0
)

# Defining a basic tool using @tool decorator
@tool
def addition_calculator (a: int, b: int) -> str:
    """Adds two numbers and returns the result as a string. Use this tool only when you need to perform addition
    
    Args:
       a: The first number.
       b: The second number.
    
    """
    return f"The sum of {a} and {b} is {a+b}."

@tool
def multiplication_calculator(a: int, b: int) -> str:
    """Multiplies two numbers and returns the result as a string. Use this tool only when you need to perform multiplication
    
    Args:
       a: The first number.
       b: The second number.

    """
    return f"The product of {a} and {b} is {a*b}."

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city. Use this tool only when you need to get the weather for a city.
    
    Args:
        city: The name of the city to get the weather for.

    """
    return f"The weather in {city} is shady"



agent = create_agent(
    model = model_used,
    tools = [addition_calculator, multiplication_calculator, get_weather],
    system_prompt= """You're a helpful assistant. Answer the user's questions to the best of your ability.
    Use tools where necessary to provide accurate and complete answers. If the user asks for a calculation, use the appropriate calculation tool.
    If the user asks for a weather information, use the get_weather tool.""",
)

response = agent.invoke(
    {"messages": [
        {"role": "user", "content": "what is the sum of 7 and 8?"}
    ]
    })

print(response["messages"]) # This will print all the messages in the response, HumanMessage, AIMessage, ToolMessage, etc.
print("\n", "=" *110, "\n" )
print(response["messages"][-1]) # This will print the last message in the response List
print("\n", "=" *110, "\n")
print(response["messages"][-1].content) # This will print the content of the last message in the response List, which is the agent's final answer to the user's question