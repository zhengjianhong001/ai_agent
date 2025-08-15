from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
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

print("=== PromptTemplate 高级功能演示 ===\n")

# 1. 结构化输出解析
print("1. 结构化输出解析")
class BookRecommendation(BaseModel):
    title: str = Field(description="书名")
    author: str = Field(description="作者")
    reason: str = Field(description="推荐理由")
    difficulty: str = Field(description="难度级别")

# 创建带输出解析器的模板
structured_template = PromptTemplate.from_template(
    """你是一个专业的图书推荐专家。
请为学习{topic}的{level}学习者推荐一本书。

要求：
- 书名要准确
- 推荐理由要具体
- 难度要适合学习者水平

请以JSON格式返回：
{{
    "title": "书名",
    "author": "作者", 
    "reason": "推荐理由",
    "difficulty": "难度级别"
}}"""
)

# 创建输出解析器
output_parser = JsonOutputParser(pydantic_object=BookRecommendation)

# 组合模板和解析器
chain = structured_template | chat_model | output_parser

# 执行链式调用
result = chain.invoke({"topic": "Python编程", "level": "初学者"})
print(f"结构化输出: {result}")
print()

# 2. 多轮对话模板
print("2. 多轮对话模板")
chat_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的{role}，请用{style}的方式回答问题。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# 模拟对话历史
chat_history = [
    HumanMessage(content="什么是Python？"),
    HumanMessage(content="Python是一种编程语言，特点是语法简洁...")
]

# 生成多轮对话提示
messages = chat_template.format_messages(
    role="编程导师",
    style="简洁明了",
    chat_history=chat_history,
    question="Python有哪些主要应用领域？"
)

print("多轮对话模板:")
for msg in messages:
    print(f"- {msg.type}: {msg.content[:50]}...")
print()

# 3. 条件模板
print("3. 条件模板")
def create_conditional_prompt(user_level, topic):
    if user_level == "初学者":
        template = """你是一个耐心的编程导师。
请用最简单易懂的方式解释{topic}。
要求：
- 使用生活中的比喻
- 提供具体的例子
- 避免使用专业术语"""
    elif user_level == "进阶":
        template = """你是一个资深的编程专家。
请深入讲解{topic}的高级概念。
要求：
- 分析底层原理
- 提供最佳实践
- 讨论性能优化"""
    else:
        template = """请简单介绍一下{topic}。"""
    
    return PromptTemplate.from_template(template)

# 测试不同级别的模板
beginner_prompt = create_conditional_prompt("初学者", "函数")
advanced_prompt = create_conditional_prompt("进阶", "函数")

print("初学者模板:")
print(beginner_prompt.template)
print("\n进阶模板:")
print(advanced_prompt.template)
print()

# 4. 模板组合和复用
print("4. 模板组合和复用")
# 基础模板
base_template = PromptTemplate.from_template(
    "你是一个{role}，专门负责{domain}领域的问题。"
)

# 任务模板
task_template = PromptTemplate.from_template(
    "请完成以下任务：{task_description}"
)

# 格式模板
format_template = PromptTemplate.from_template(
    "请按照{format_style}的格式输出结果。"
)

# 组合模板
combined_template = PromptTemplate.from_template(
    "{base_context}\n\n{task_info}\n\n{format_requirement}"
)

# 使用组合模板
result = combined_template.format(
    base_context=base_template.format(role="数据分析师", domain="金融"),
    task_info=task_template.format(task_description="分析股票价格趋势"),
    format_requirement=format_template.format(format_style="图表+文字说明")
)

print("组合模板结果:")
print(result)
print()

# 5. 动态内容注入
print("5. 动态内容注入")
def create_context_aware_prompt(context_info, user_query):
    # 根据上下文信息动态构建提示
    if "error" in context_info.lower():
        template = """检测到错误信息：{context}
请分析这个错误并提供解决方案。
用户问题：{query}"""
    elif "performance" in context_info.lower():
        template = """性能相关信息：{context}
请提供性能优化建议。
用户问题：{query}"""
    else:
        template = """相关信息：{context}
请回答用户问题：{query}"""
    
    return PromptTemplate.from_template(template)

# 测试不同上下文
error_context = "TypeError: 'NoneType' object is not callable"
perf_context = "函数执行时间超过5秒"
normal_context = "用户正在学习Python"

error_prompt = create_context_aware_prompt(error_context, "如何解决这个问题？")
perf_prompt = create_context_aware_prompt(perf_context, "如何优化性能？")

print("错误上下文模板:")
print(error_prompt.template)
print("\n性能上下文模板:")
print(perf_prompt.template)
print()

print("=== 总结 ===")
print("PromptTemplate 不仅仅是变量替换，它还提供：")
print("✅ 结构化输出解析")
print("✅ 多轮对话支持") 
print("✅ 条件模板生成")
print("✅ 模板组合和复用")
print("✅ 动态内容注入")
print("✅ 与LangChain生态系统的深度集成") 