
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

load_dotenv()

llm = ChatOpenAI (model = "gpt-5-nano", temperature =0)

# define a reusable prompt template
# - "system" sets the behavior of the assistant
# - "human" is the input from the user
# - {topic} is a variable that will be filled in at run time
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Give me the most rescent answers. Make it short and concise."),
    ("human", "Tell me a joke about {topic}.")
])

formated_prompt = prompt_template.invoke({"topic" : "artificial intelligence"})
response = llm.invoke(formated_prompt)
print(response.content)