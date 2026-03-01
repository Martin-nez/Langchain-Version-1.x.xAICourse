from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from typing import TypedDict
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# Structured output schema
class DictionaryOutput(TypedDict):
    text: str


# LLM
llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

# LLM with TypedDict structured output
llm_with_typed_dict = llm.with_structured_output(DictionaryOutput)


# First prompt (generate structured text)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Return a short word explanation."),
    ("human", "Explain this word briefly: {input}")
])


# Instagram post prompt
instagram_post = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful Instagram post generator. Create just one short post."),
    ("human", "Create a short and engaging Instagram post about: {text}")
])


parser = StrOutputParser()

word = "Softwares"


# Correct chain
final_chain = RunnableSequence(
    prompt_template,
    llm_with_typed_dict,   # returns {"text": "..."}
    instagram_post,        # uses {text}
    llm,
    parser
)


response = final_chain.stream({"input": word})

for chunk in response:
    print(chunk, end="", flush=True)