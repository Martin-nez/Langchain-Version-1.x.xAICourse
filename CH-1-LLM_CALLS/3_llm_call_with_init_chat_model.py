# init_chat_model is a universal factory  function in Langchain.  
# It's primary purpose is to simplify how you initialize different AI models using a single consistent interface.
# It abstracts away the complexities of setting up various models, allowing you to create instances of difference models with a single function and a few parameters.


from langchain.chat_models import init_chat_model

from dotenv import load_dotenv
load_dotenv()

llm = init_chat_model(model = "gpt-5-nano", temperature =0)
response = llm.stream("Tell me a short history about US in 50 words.")

for chunk in response:
    print(chunk.content, end="", flush=True)


