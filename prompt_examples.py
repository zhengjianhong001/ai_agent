from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
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

print("=== PromptTemplate 使用示例 ===\n")

# 示例1：基础模板
print("1. 基础模板")
basic_template = PromptTemplate.from_template("你好，{name}！今天天气怎么样？")
result1 = basic_template.format(name="小明")
print(f"模板: {basic_template.template}")
print(f"结果: {result1}")
print()

# 示例2：多变量模板
print("2. 多变量模板")
multi_template = PromptTemplate.from_template(
    "请用{language}语言写一个关于{topic}的{style}故事，长度大约{length}字。"
)
result2 = multi_template.format(
    language="中文",
    topic="人工智能",
    style="科幻",
    length="500"
)
print(f"模板: {multi_template.template}")
print(f"结果: {result2}")
print()

# 示例3：带默认值的模板
print("3. 带默认值的模板")
default_template = PromptTemplate.from_template(
    "请推荐{category}的{count}本书，难度级别为{difficulty}。"
)
# 使用部分参数
result3 = default_template.format(
    category="Python编程",
    count="3"
    # difficulty 没有提供，会报错
)
print(f"模板: {default_template.template}")
print(f"结果: {result3}")
print()

# 示例4：与AI模型结合使用
print("4. 与AI模型结合使用")
ai_template = PromptTemplate.from_template(
    "你是一个专业的{role}。请用{style}的方式回答以下问题：{question}"
)

# 使用模板生成提示并发送给AI
prompt_text = ai_template.format(
    role="编程导师",
    style="简洁明了",
    question="什么是Python的装饰器？"
)
print(f"生成的提示: {prompt_text}")

# 发送给AI模型
response = chat_model.invoke([HumanMessage(content=prompt_text)])
print(f"AI回复: {response.content[:200]}...")  # 只显示前200个字符
print()

# 示例5：模板验证
print("5. 模板验证")
try:
    # 缺少必需参数会报错
    invalid_result = basic_template.format()
    print(invalid_result)
except Exception as e:
    print(f"错误: {e}")
print()

# 示例6：动态模板构建
print("6. 动态模板构建")
def create_prompt_template(role, task, format_style="详细"):
    template = f"你是一个{role}。请以{format_style}的格式完成以下任务：{{task_description}}"
    return PromptTemplate.from_template(template)

# 动态创建模板
dynamic_template = create_prompt_template("数据分析师", "分析数据", "简洁")
result6 = dynamic_template.format(task_description="分析销售数据趋势")
print(f"动态模板: {dynamic_template.template}")
print(f"结果: {result6}")
print()

print("=== PromptTemplate 使用总结 ===")
print("✅ 基础用法: PromptTemplate.from_template()")
print("✅ 格式化: template.format() 或 template.invoke()")
print("✅ 多变量: 使用 {variable_name} 语法")
print("✅ 与AI结合: 生成提示后发送给模型")
print("✅ 动态构建: 根据条件创建不同模板") 