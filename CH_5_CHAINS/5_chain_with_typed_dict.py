from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableLambda
from typing import TypedDict
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class DictionaryOutput(TypedDict):
    text : str


llm = ChatOpenAI(model= "gpt-5-nano", temperature = 0)
llm_with_typed_dict = llm.with_structured_output(DictionaryOutput) # LLM with TypedDict output

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Make your answers short and concise."),
    ("human", "{input}.")
])

parser = StrOutputParser()

word = "Softwares"

# Using TypedDict with structured output - no need for custom_dict_maker fuction.
chain = RunnableSequence(prompt_template, llm_with_typed_dict)

instagram_post = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful Instagram post generator. just one."),
    ("human", "Create short and engaging Instagram posts on the word: {text}.")
    ])

final_chain = RunnableSequence(
    prompt_template,
    llm_with_typed_dict,
    instagram_post,
    llm,
    parser
)


response = final_chain.stream({"input": word})

for chunk in response:
    print(chunk, end = "", flush = True)
