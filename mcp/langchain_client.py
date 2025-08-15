# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
import asyncio
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from dotenv import load_dotenv
import os
# 加载环境变量
load_dotenv()

# 最朴素的方式，把工具列表发送给 LLM，根据 LLM 返回需要调用的工具，然后手动调用工具，并把工具结果发送给模型，模型返回最终结果

base_params = {
    "model": os.getenv("OPENAI_MODEL", ""),
    "temperature": 0.1,
    "streaming": True,
    "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
    "openai_api_base": os.getenv("OPENAI_API_BASE", ""),
}
model = ChatOpenAI(**base_params)


server_params = StdioServerParameters(
    command="uvx",
    # Make sure to update to the full absolute path to your math_server.py file
    args=["mcp-server-tapd"],
    env={
        "TAPD_ACCESS_TOKEN": os.getenv("TAPD_ACCESS_TOKEN", ""),
        "TAPD_API_BASE_URL": os.getenv("TAPD_API_BASE_URL", ""),
        "TAPD_BASE_URL": os.getenv("TAPD_BASE_URL", ""),
        "CURRENT_USER_NICK": os.getenv("CURRENT_USER_NICK", ""),
        "BOT_URL": os.getenv("BOT_URL", "")
    }
)

async def run_agent():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)
            # 直接绑定工具到模型
            model_with_tools = model.bind_tools(tools)
            
              # 第一轮：调用模型
            messages = [HumanMessage(content="获取我参与的项目，只返回项目名称列表，不要返回其他内容")]
            result = await model_with_tools.ainvoke(messages)
            
            # 检查是否有工具调用
            if result.tool_calls:
                print("检测到工具调用，正在执行...")
                
                # 执行工具调用
                tool_results = []
                for tool_call in result.tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    
                    # 找到对应的工具
                    for tool in tools:
                        if tool.name == tool_name:
                            print(f"执行工具: {tool_name}")
                            tool_result = await tool.ainvoke(tool_args)
                            tool_results.append(tool_result)
                            break
                
                # 第二轮：将工具结果发送给模型
                if tool_results:
                    # 添加工具结果到消息列表
                    new_messages = messages + [result] + [
                        ToolMessage(
                            content=str(tool_results[0]),  # 假设只有一个工具调用
                            tool_call_id=result.tool_calls[0]['id']
                        )
                    ]
                    print("将工具结果发送给模型...")
                    final_result = await model_with_tools.ainvoke(new_messages)
                    return final_result
            
            return result

# Run the async function
if __name__ == "__main__":
    result = asyncio.run(run_agent())
    # 最简单的方法：直接获取最后一个有内容的 AI 消息
    print("AI 最终内容：", result.content)
    # print(result)