
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
llm = ChatOpenAI(model ="gpt-5-nano", temperature = 0)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Your answer should be concise in one sentence."),
    ("human", "Tell me a joke about {topic}.")
])

chain = prompt_template | llm


response = chain.invoke({"topic" : "web 3"})
print(response.content)