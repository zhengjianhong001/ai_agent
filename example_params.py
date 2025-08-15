from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()


# 基础配置
base_params = {
    "model": os.getenv("OPENAI_MODEL", ""),
    "temperature": 0.1,
    "streaming": True,
    "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
    "openai_api_base": os.getenv("OPENAI_API_BASE", ""),
}

# 方法1：直接使用基础配置
print("=== 方法1：直接使用基础配置 ===")
chat_model1 = ChatOpenAI(**base_params)

# 方法2：动态修改参数
print("\n=== 方法2：动态修改参数 ===")
creative_params = base_params.copy()
creative_params["temperature"] = 0.8  # 提高创造性
chat_model2 = ChatOpenAI(**creative_params)

# 方法3：条件性参数构建
print("\n=== 方法3：条件性参数构建 ===")
def create_model_params(use_streaming=True, temperature=0.1):
    params = {
        "model": os.getenv("OPENAI_MODEL", ""),
        "temperature": temperature,
        "streaming": use_streaming,
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "openai_api_base": os.getenv("OPENAI_API_BASE", ""),
    }
    return params

# 创建不同配置的模型
streaming_model = ChatOpenAI(**create_model_params(use_streaming=True, temperature=0.1))
non_streaming_model = ChatOpenAI(**create_model_params(use_streaming=False, temperature=0.8))

print("参数解包示例完成！") 