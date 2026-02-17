
# Prompt Templates

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

prompt_template = PromptTemplate.from_template(
    "Tell me 6 different jokes about {topic} tailored to dating and aging."
)

formated_prompt = prompt_template.invoke({"topic" : "life"})

response = llm.invoke(formated_prompt)
print(response.content)
