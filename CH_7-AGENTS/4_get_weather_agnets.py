from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.agents import create_agent
import requests

load_dotenv()

model = init_chat_model(
    model = "llama-3.3-70b-versatile",
    model_provider= "groq",
    temperature = 0
)

@tool("get_weather", description= "Return weather information for a given city.", return_direct= False)
def get_weather(city: str) -> str:
    """Get the weather for a city.
    """
    response = requests.get(f"http://wttr.in/{city}?format=j1") # This is a free weather API that returns weather information in Json format. You can replace this with any other weather API of your choice.
    if response.status_code == 200:
        data= response.json()
        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]['value']
        temp_C = current_condition['temp_C']
        temp_F = current_condition['temp_F']
        return f"The weather in {city} is {weather_desc} with a temperature of {temp_C}°C ({temp_F}°F)"
    else:
        return f"Sorry, I couldn't get the weather information for {city} at the moment."
    

agent = create_agent(
    model = model,
    tools = [get_weather],
    system_prompt= """You are a helpful weather assistant who always cracks jokes and is humurous while remaining helpful."""
)

ai_response = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is the weather like in Asaba, Nigeria?"}
    ]}
)

print(ai_response)
print("\n", "=" *110, "\n")
print(ai_response["messages"][-1].content)