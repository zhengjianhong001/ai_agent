# LangChain MCP 项目

这是一个基于 LangChain 和 MCP (Model Context Protocol) 的智能对话项目，集成了 TAPD 项目管理工具，提供智能客服、代码审查、多语言翻译等功能。

## 功能特性

- **智能对话系统**：基于 OpenAI 的聊天机器人
- **MCP 集成**：支持 Model Context Protocol 工具调用
- **TAPD 集成**：连接 TAPD 项目管理平台
- **多场景应用**：
  - 客服机器人系统
  - 代码审查助手
  - 多语言翻译系统
  - 智能文档生成

## 系统要求

- Python 3.8+
- OpenAI或deepseek或元宝 API 密钥
- TAPD 访问令牌（可选，用于 MCP 功能）

## 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/zhengjianhong001/ai_agent_test
cd ai_agent_test
```

### 2. 创建虚拟环境

```bash
uv venv && source .venv/bin/activate
```

### 3. 安装依赖

```bash
uv pip install -r requirements.txt
```

### 4. 配置环境变量

创建 `.env` 文件并配置以下环境变量：

```bash
# OpenAI 配置
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_API_BASE=https://api.openai.com/v1

# TAPD 配置（可选，用于 MCP 功能）
TAPD_ACCESS_TOKEN=your_tapd_access_token
TAPD_API_BASE_URL=https://api.tapd.cn
TAPD_BASE_URL=https://www.tapd.cn
CURRENT_USER_NICK=your_nickname
BOT_URL=your_bot_url
```

## 快速开始

### 基础聊天功能

运行简单的聊天示例：

```bash
python simple_chat.py
```

### 高级提示工程示例

```bash
python advanced_prompts.py
```

### 实际应用场景演示

```bash
python real_world_example.py
```

### 测试不同方法

```bash
python test.py
```

## 🔧 MCP 功能使用

### 使用 LangChain MCP 客户端

```bash
python mcp/langchain_client.py
```

### 使用 LangGraph MCP 客户端

```bash
python mcp/langgraph_client.py
```
