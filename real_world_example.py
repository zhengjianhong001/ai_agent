from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import json
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()


# 创建聊天模型
llm_params = {
    "model": os.getenv("OPENAI_MODEL", ""),
    "temperature": 0.1,
    "streaming": True,
    "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
    "openai_api_base": os.getenv("OPENAI_API_BASE", ""),
}
chat_model = ChatOpenAI(**llm_params)

print("=== PromptTemplate 实际应用价值演示 ===\n")

# 场景1：客服机器人系统
print("场景1：客服机器人系统")
class CustomerServiceResponse(BaseModel):
    intent: str = Field(description="用户意图")
    confidence: float = Field(description="置信度")
    response: str = Field(description="回复内容")
    action: str = Field(description="需要执行的动作")

# 客服模板
customer_service_template = PromptTemplate.from_template(
    """你是一个专业的客服机器人。

用户消息：{user_message}
用户历史：{chat_history}
产品信息：{product_info}

请分析用户意图并提供合适的回复。返回JSON格式：
{{
    "intent": "用户意图（咨询/投诉/购买/其他）",
    "confidence": 0.95,
    "response": "回复内容",
    "action": "需要执行的动作（转人工/提供信息/记录问题）"
}}"""
)

# 模拟客服场景
user_message = "我想了解一下你们的产品价格"
chat_history = "用户之前询问过产品功能"
product_info = "我们的产品价格从99元起，支持免费试用"

# 创建输出解析器
parser = JsonOutputParser(pydantic_object=CustomerServiceResponse)

# 组合模板、模型和解析器
customer_service_chain = customer_service_template | chat_model | parser

# 执行
result = customer_service_chain.invoke({
    "user_message": user_message,
    "chat_history": chat_history,
    "product_info": product_info
})

print(f"客服机器人分析结果: {result}")
print()

# 场景2：代码审查助手
print("场景2：代码审查助手")
code_review_template = PromptTemplate.from_template(
    """你是一个资深的代码审查专家。

代码语言：{language}
代码内容：
```{language}
{code}
```

请进行代码审查，重点关注：
1. 代码质量
2. 潜在问题
3. 性能优化
4. 安全风险

请以JSON格式返回：
{{
    "score": 85,
    "issues": ["问题1", "问题2"],
    "suggestions": ["建议1", "建议2"],
    "security_risks": ["风险1"],
    "overall_comment": "总体评价"
}}"""
)

# 模拟代码审查
python_code = """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

result = calculate_fibonacci(100)
print(result)
"""

# 执行代码审查
code_review_chain = code_review_template | chat_model | StrOutputParser()
review_result = code_review_chain.invoke({
    "language": "python",
    "code": python_code
})

print("代码审查结果:")
print(review_result)
print()

# 场景3：多语言翻译系统
print("场景3：多语言翻译系统")
def create_translation_prompt(source_lang, target_lang, context=""):
    """根据语言对和上下文创建翻译模板"""
    
    if context:
        template = f"""你是一个专业的翻译专家，精通{source_lang}和{target_lang}。

原文（{source_lang}）：{{text}}
上下文：{context}

请将原文翻译成{target_lang}，注意：
- 保持原文的语气和风格
- 考虑上下文语境
- 确保翻译准确自然"""
    else:
        template = f"""请将以下{source_lang}文本翻译成{target_lang}：

{{text}}"""
    
    return PromptTemplate.from_template(template)

# 测试不同场景的翻译
formal_translation = create_translation_prompt("中文", "英文", "商务邮件")
casual_translation = create_translation_prompt("中文", "英文", "社交媒体")

# 执行翻译
formal_chain = formal_translation | chat_model | StrOutputParser()
casual_chain = casual_translation | chat_model | StrOutputParser()

text = "我们很高兴地通知您，您的订单已经发货了。"

formal_result = formal_chain.invoke({"text": text})
casual_result = casual_chain.invoke({"text": text})

print("正式翻译（商务邮件）:")
print(formal_result)
print("\n非正式翻译（社交媒体）:")
print(casual_result)
print()

# 场景4：智能文档生成
print("场景4：智能文档生成")
document_template = PromptTemplate.from_template(
    """你是一个专业的文档撰写专家。

项目信息：
- 项目名称：{project_name}
- 技术栈：{tech_stack}
- 功能描述：{features}
- 目标用户：{target_users}

请生成一份{doc_type}文档，包含：
{requirements}

文档风格：{style}
文档长度：{length}"""
)

# 生成API文档
api_doc = document_template.format(
    project_name="用户管理系统",
    tech_stack="Python + FastAPI + PostgreSQL",
    features="用户注册、登录、权限管理、数据统计",
    target_users="开发者和系统管理员",
    doc_type="API文档",
    requirements="- 接口说明\n- 请求参数\n- 响应格式\n- 错误码",
    style="技术文档",
    length="详细"
)

print("生成的API文档模板:")
print(api_doc)
print()

print("=== 总结 ===")
print("PromptTemplate 的实际价值：")
print("🔧 构建可复用的提示工程组件")
print("🔗 与LangChain生态系统无缝集成")
print("🎯 实现复杂的业务逻辑和条件判断")
print("📊 支持结构化输出和数据解析")
print("🌍 支持多语言和多场景适配")
print("⚡ 提高开发效率和代码可维护性") 