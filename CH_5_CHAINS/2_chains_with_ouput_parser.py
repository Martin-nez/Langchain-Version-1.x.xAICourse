
# A parser forces the LLM to return output in a specific format.
# LLMs are unstable and can return different ouput for the same prompt, which can crash your code.

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model = "gpt-5-nano", temperature = 0)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You're a helpful assistant. Your answer should be concise in one sentence."),
    ("human", "Tell me a funny joke about {topic}.")
])

parser = StrOutputParser()  # with this, there's no need of adding 'content' while printing the response

chain = prompt_template| llm | parser

response = chain.invoke({"topic": "feminism"})
print(response)