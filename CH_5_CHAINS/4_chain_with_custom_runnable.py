from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm = ChatOpenAI(model = "gpt-5-nano", temperature = 0)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistant. Your answer should be precise and in one sentence."),
    ("human", "{input}.")
])

parser = StrOutputParser()

# Custom dictionary maker. The function just helps to convert the output text to dictionary with key 'text'
def dictionary_maker(text: str) -> dict:
    """This function converts text to dictionary with key 'text.""" # Docstring is important to desceribe the funcion
    return {"text": text}

custom_dict_maker = RunnableLambda(dictionary_maker)
word = "money" # you can change this word to test with different words.

# This line is optional. It just shows the converted dictionary output. {'text': '...'}
chain = RunnableSequence([prompt_template, llm, parser, custom_dict_maker])
print(chain.invoke({"input": word}))


instagram_post = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful instagram post generator. Make it short and engaging."),
    ("human", "Create short and engaging instagram post on the word {input}.")
])

final_chain = RunnableSequence(
    prompt_template,
    llm,
    parser,
    custom_dict_maker,
    instagram_post,
    llm,
    parser

)


response = final_chain.stream({"input" : word})  # streaming response generates the output in chunks

for chunk in response:
    print(chunk, end="", flush=True)