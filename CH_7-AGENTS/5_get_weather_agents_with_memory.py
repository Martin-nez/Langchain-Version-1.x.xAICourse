from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
import requests
from dataclasses import dataclass

load_dotenv()

@dataclass
class Context:
    user_id: str

@dataclass
class ResponseFormat:
    summary: str
    temperature: str
    temperature_fahrenheit: float
    humidity: float

@tool("get_weather", description="Return weather information for a given city.", return_direct= False)
def get_weather(city: str) -> str:
    """Get the weather for a city.
    
    Args:
       city: The name of the city.
    """
    response = requests.get(f"http://wttr.in/{city}?format=j1") # This is a free weather endpoint that returns weather information in json format. You can replace it with any other weather API that you prefer.
    if response.status_code == 200:
        return response.json()
    else:
        return f"Could not retrieve weather information for {city}."
    
@tool("locate_user", description= "Look up user's location based on the context")
def locate_user(runtime: ToolRuntime[Context]):
    match runtime.context.user_id:
        case "user_123":
            return "Asaba, Nigeria"
        case "user_456":
            return  "Ajah, Lagos, Nigeria"
        case _:
            return "Unknown Location"
        

model = init_chat_model(
    model = "llama-3.3-70b-versatile",
    model_provider= "groq",
    temperature = 0
)

tool = [get_weather, locate_user]

checkpointer = InMemorySaver()

agent = create_agent(
    model = model,
    tools = tool,
    system_prompt= """You are a helpful weather assistant who always cracks jokes and is humurous while remaining helpful.""",
    context_schema= Context,
    response_format= ResponseFormat,
    checkpointer = checkpointer
)


config ={ "configurable": {"thread_id": 1}}

ai_response = agent.invoke({
    "messages": [
        {"role": "user", "content": "what is the weather in my location?"}
    ],
},
    config = config,
    context= Context(user_id = "user_456") # Change the user_id to "user_456" or  "user_123" to get weather information for Beijing or Asaba respectively. For any other user_id, it returns "Unknown location".
    )

print("\n=== All Messages ===\n")
print(ai_response["structured_response"].summary)