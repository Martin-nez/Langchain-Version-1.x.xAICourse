from langchain_openai import ChatOpenAI
import os 
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnableParallel

load_dotenv()


if os.environ.get("OPENAI_API_KEY") is None:
    raise ValueError("OPENAI_API_KEY is not set.")


llm = ChatOpenAI(model = "gpt-5-nano", temperature=0)

# This prompt is used to search about any topic in the internet and summarize the result in brief
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Make a search about {input} on the internet and make a summary in brief."),
    ("human", "{input}.")
])

parser = StrOutputParser()

def dictionary_maker(text: str) -> dict:
    """This fuction converts text to dictionary with key 'text'.""" 
    return {"text": text}


custom_dict_maker = RunnableLambda(dictionary_maker)

instagram_post= ChatPromptTemplate.from_messages([
    ("system", "You are a helpful instagram post generator. Just one"),
    ("human", "{text}.")
])


instagram_chain = RunnableSequence(
    instagram_post,
    llm,
    parser
)

def twitter_post_maker (text: dict):
    """This function is to make twitter post from 'text'."""
    twitter_text = text['text']
    twitter_post = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful twitter post generator. Just one."),
        ("human", "Create short and engaging twitter post on the word: {text}")
    ])

    twitter_chain = RunnableSequence(
        twitter_post,
        llm,
        parser
    )
    result = twitter_chain.invoke({"text" : twitter_text})
    return result

twitter_post_maker_runnable = RunnableLambda(twitter_post_maker)

parellel_chain = RunnableParallel({"instagram_post": instagram_chain, 
                                   "twitter_post" : twitter_post_maker_runnable})

final_chain = RunnableSequence(prompt_template, llm, parser, custom_dict_maker, parellel_chain)

user_input = "Star" # you can change this to test with other topics.

response =final_chain.invoke({"input": user_input})

print("Instagram Post:")
print(response["instagram_post"])
print("\nTwitter Post:")
print(response["twitter_post"])