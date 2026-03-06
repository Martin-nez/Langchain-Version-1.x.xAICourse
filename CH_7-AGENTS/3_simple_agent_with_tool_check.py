from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

model = init_chat_model(
    model = "llama-3.3-70b-versatile",
    model_provider= "groq",
    temperature = 0
)

# defining a basic tool using @tool decorator
@tool
def addition_calculator(a: int, b: int) -> str:
    """"Adds two numbers and returns the results as a string. Use this tool only when you need to perform addition.
    
    Args:
       a: The first number.
       b: The second number.
    """
    return f"The sum of {a} and {b} is {a+b}."


@tool
def multiplication_calculator(a: int, b: int) -> str:
    """Multiplies two numbers and returns the results as a string. Use this tool only when you need to perform multiplication.
    
    Args:
       a: The first number.
       b: The second number.
    """
    return f"The product of {a} and {b} is {a*b}."


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city. Use this tool only when you need to get the weather for a city.
    
    Args: 
       city: Onitsha.
    """
    return f"The weather in {city} is shady"

agent = create_agent(
    model = model,
    tools= [addition_calculator, multiplication_calculator, get_weather],
    system_prompt= """You are a helpful assistant. Answer the user's questions to the best of your ability. Use tools where necessary to provide accurate and complete answers."""
)

response = agent.invoke(
    {"messages": [
        {"role": "user", "content": "what is the weather in Onitsha? and what is the product of 5 and 10?"}
    ]}
)

print("\n === All messages === \n")                    # this print a header "All messages" on the console
for msg in response["messages"]:                       # this loops through every message in the conversation
    print(f"\n {type(msg).__name__}: {msg.content}")   # this prints the message type and content on the console 
    if hasattr(msg, 'tool_calls') and msg.tool_calls:  # this checks if a tool was called in the message
        print(f" → Tool called: {msg.tool_calls}")     # this prints the tool name and attributes

print("\n === Final Answer === \n")
print(response["messages"][-1].content)
