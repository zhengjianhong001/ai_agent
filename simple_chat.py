from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()


base_params = {
    "model": os.getenv("OPENAI_MODEL", ""),
    "temperature": 0.1,
    "streaming": True,
    "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
    "openai_api_base": os.getenv("OPENAI_API_BASE", ""),
}
chat_model = ChatOpenAI(**base_params)
response = chat_model.invoke([HumanMessage(content="你好，我是小明，今天天气怎么样？")])
print(response.content)



