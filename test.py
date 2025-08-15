from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 准备创建 LLM 实例的参数
llm_params = {
    "model": os.getenv("OPENAI_MODEL", ""),
    "temperature": 0.1,  # 默认值
    "streaming": True,   # 默认值
    "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
    "openai_api_base": os.getenv("OPENAI_API_BASE", ""),
}

# 创建聊天模型
chat_model = ChatOpenAI(**llm_params)

# 方法1：使用 PromptTemplate 生成提示，然后发送给模型
print("=== 方法1：使用 PromptTemplate ===")
prompt_template = PromptTemplate.from_template("Tell me a joke about {topic}")
formatted_prompt = prompt_template.invoke({"topic": "cats"})
print(f"生成的提示: {formatted_prompt}")
print(f"提示类型: {type(formatted_prompt)}")

# 提取文本内容并发送给模型
prompt_text = formatted_prompt.to_string()
print(f"提取的文本: {prompt_text}")

response1 = chat_model.invoke([HumanMessage(content=prompt_text)])
print(f"AI回复: {response1.content}")
print()

# 方法2：直接使用字符串模板
print("=== 方法2：直接使用字符串 ===")
response2 = chat_model.invoke([HumanMessage(content="推荐一本学习LangChain的书")])
print(f"AI回复: {response2.content}")
print()

# 方法3：使用 PromptTemplate 的 format 方法
print("=== 方法3：使用 format 方法 ===")
joke_prompt = prompt_template.format(topic="programming")
print(f"格式化的提示: {joke_prompt}")
response3 = chat_model.invoke([HumanMessage(content=joke_prompt)])
print(f"AI回复: {response3.content}")
print()

# 方法4：使用 PromptTemplate 的 format_prompt 方法
print("=== 方法4：使用 format_prompt 方法 ===")
prompt_value = prompt_template.format_prompt(topic="dogs")
print(f"PromptValue: {prompt_value}")
prompt_text_4 = prompt_value.to_string()
response4 = chat_model.invoke([HumanMessage(content=prompt_text_4)])
print(f"AI回复: {response4.content}")