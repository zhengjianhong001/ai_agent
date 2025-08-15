# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import asyncio
from dotenv import load_dotenv
import os
# 加载环境变量
load_dotenv()

# 使用 langgraph 的 react agent，组合 tapd 的 mcp server

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
            # print(tools)
            # Create and run the agent
            agent = create_react_agent(model, tools)
            agent_response = await agent.ainvoke({"messages": "获取我参与的项目，只返回项目名称列表，不要返回其他内容"})
            return agent_response

# Run the async function
if __name__ == "__main__":
    result = asyncio.run(run_agent())
    # 最简单的方法：直接获取最后一个有内容的 AI 消息
    ai_final_content = result['messages'][-1].content if result and 'messages' in result else None
    print("AI 最终内容：", ai_final_content)
    # print(result)