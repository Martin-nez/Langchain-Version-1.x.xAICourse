# Basic Tool definition
# A Tool is a function that an LLM can call to perform specific tasks.
# The simplest way to create a tool is with @tool decorator.
# By default, the functions docstring becomes the tool's description that helps the model understand when to use it.


from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

load_dotenv()

llm = ChatOpenAI(model ="gpt-5-nano", temperature = 0)

#defining a basic tool using the @tool decorator
@tool
def addition_calculator(a : int, b : int) -> str:
    """Adds two numbers and returns the result as string. Use this tool only when you need to perform addition.
    Args:
    a : The first number.
    b : The second number.
    """
    return f"The sum of {a} and {b} is {a+b}."

def multiplication_calculator(a : int, b : int) -> str:
    """Multiplies two number and returns the result as string. Use this tool only when you need to perform multiplication.
    
    Args:
    a : The first number.
    b : The second number.
    """
    return f"The product of {a} and {b} is {a * b}."

tools = [addition_calculator, multiplication_calculator]

llm_with_tools = llm.bind_tools(tools = tools)

response = llm_with_tools.invoke("what is the product of 8 and 5")

#print(response) # if you print only response, you will the whole details,
# It will include the content as well the tool usage information.
# Sometimes, the content will be an empty another time the tool_calls will be empty.
print("-"*100)
print(response.content)
print("-"*100)
print(response.tool_calls)


    

