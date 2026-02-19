
# A parser forces the LLM to return output in a specific format.
# LLMs are unstable and can return different ouput for the same prompt, which can crash your code.

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
