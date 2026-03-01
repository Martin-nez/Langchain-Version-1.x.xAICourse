
# Multimodal
# Multimodlity is the ability to work with data comes in different formats, such as texts, images, audio and video.
# Langchain includes standard types for these data that can be used across providers and tools.


from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

# from url
message_1 = [
{
    "role": "user", 
    "content":[
        {"type": "text", "text": "Describe the content of this image."},
        #{"type": "image", "url": "https://res.cloudinary.com/dw8j1umff/image/upload/v1769710064/orange_qegkzg.jpg"}
        {"type": "image", "url": "https://tse4.mm.bing.net/th/id/OIP.VSLx_x-QzKgWChMZwRY0zwHaEw?rs=1&pid=ImgDetMain&o=7&rm=3"}
    ]
}
]

# from base64 data
message_2 = [
    {
        "role" : "user", 
        "content": [
            {"type": "text", "text": "Describe the content of this image."},
            {"type": "image_url","image_url":{"url": "data:image/jpeg;base64,AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2..." }}
        ]
    }
]

# from provider-managed ID
message_3 = [
    {
        "role": "user", "content": [
            {"type": "text", "text": "Describe the content of this image."},
            {"type": "image_url", "image_url": {"url": "file-abc123"}}
        ]
    }
]

response = llm.invoke(message_1)
print(response.content)