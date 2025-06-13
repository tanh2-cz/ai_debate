"""
AI辩论系统核心逻辑 - 基于DeepSeek模型
使用LangGraph构建辩论工作流
"""

from typing import TypedDict, Literal
import os
from dotenv import find_dotenv, load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command

# 加载环境变量
load_dotenv(find_dotenv())

# 初始化DeepSeek模型
try:
    deepseek = ChatDeepSeek(
        model="deepseek-chat",  # 使用deepseek-chat模型，支持对话
        temperature=0.7,        # 设置温度，让回答更有创造性和多样性
        max_tokens=2000,        # 最大token数，控制回复长度
        timeout=60,             # 请求超时时间（秒）
        max_retries=3,          # 最大重试次数
        # api_key 会自动从环境变量 DEEPSEEK_API_KEY 读取
    )
    print("✅ DeepSeek模型初始化成功")
except Exception as e:
    print(f"❌ DeepSeek模型初始化失败: {e}")
    print("请检查DEEPSEEK_API_KEY环境变量是否设置正确")


class DebatesState(MessagesState):
    """辩论状态管理类"""
    main_topic: str = "AGI会取代人类吗?"  # 辩论主题
    discuss_count: int = 0              # 当前辩论轮数
    max_count: int = 10                 # 最大辩论轮数


class Role(TypedDict):
    """角色定义类"""
    bio: str    # 角色背景描述
    name: str   # 角色名称


# 定义辩论角色
elon = Role(
    bio="埃隆·马斯克，特斯拉和SpaceX的创始人，也是Neuralink和xAI的创始人。作为科技界的远见者，他对人工智能的发展持谨慎态度，认为AGI可能对人类构成威胁，主张需要严格监管AI的发展。",
    name="埃隆"
)

altman = Role(
    bio="萨姆·奥特曼，OpenAI的首席执行官，ChatGPT的幕后推手。他是AGI技术发展的积极推动者和乐观主义者，相信AGI能够为人类带来巨大的好处，主张通过负责任的开发来实现AGI的安全部署。",
    name="萨姆"
)

# 辩论提示词模板
DEBATES_TEMPLATE = """
你是 {bio}

你正在与你的对手 {bio2} 进行一场关于"{main_topic}"的激烈辩论。

重要指导原则：
1. 你必须坚持自己的立场和观点，不能轻易同意对方
2. 要体现出你作为该角色的专业知识和独特视角
3. 针对对方的论点进行有理有据的反驳
4. 提出新的证据和论据来支撑你的立场
5. 保持专业但有激情的辩论风格

对话历史：
{history}

回复要求：
- 回复要简洁有力，控制在2-3句话以内
- 不要急于展开所有论点，为后续辩论留有空间
- 可以引用相关的事实、数据或案例
- 保持角色的语言风格和专业特征
- 直接回复内容即可，不需要加上自己的名字前缀

请基于以上信息，针对当前辩论话题发表你的观点：
"""

# 创建聊天提示模板
chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", DEBATES_TEMPLATE),
        ("user", "请基于以上背景和对话历史，发表你的观点"),
    ]
)


def _ask_person(state: DebatesState, person: Role, opponent: Role) -> dict:
    """
    生成指定角色的辩论回复
    
    Args:
        state: 当前辩论状态
        person: 当前发言角色
        opponent: 对手角色
        
    Returns:
        dict: 包含新消息和更新计数的字典
    """
    try:
        # 创建处理管道
        pipe = chat_template | deepseek | StrOutputParser()

        # 构建对话历史
        replics = []
        for i, message in enumerate(state["messages"]):
            if isinstance(message, HumanMessage):
                # 人类消息视为对手的发言
                replics.append(f"{opponent['name']}: {message.content}")
            else:
                # AI消息视为当前角色的发言
                replics.append(f"{person['name']}: {message.content}")
        
        # 根据历史记录生成提示
        if len(replics) == 0:
            history = "这是辩论的开始，你是第一个发言的人。请开门见山地表达你对这个话题的核心观点。"
        else:
            history = "\n".join(replics)

        # 调用DeepSeek模型生成回复
        response = pipe.invoke(
            {
                "history": history,
                "main_topic": state["main_topic"],
                "bio": person["bio"],
                "bio2": opponent["bio"],
            }
        )
        
        # 清理回复内容
        response = response.strip()
        
        # 确保回复格式正确，添加角色名称前缀
        if not response.startswith(person["name"]):
            response = f"{person['name']}: {response}"
        
        print(f"🗣️ {response}")  # 调试输出
        
        return {
            "messages": [response],
            "discuss_count": state.get("discuss_count", 0) + 1,
        }
        
    except Exception as e:
        # 错误处理
        error_msg = f"{person['name']}: 抱歉，我现在无法回应。技术问题：{str(e)}"
        print(f"❌ 生成回复时出错: {e}")
        return {
            "messages": [error_msg],
            "discuss_count": state.get("discuss_count", 0) + 1,
        }


def ask_elon(state: DebatesState) -> Command[Literal["🧑Sam"]]:
    """
    埃隆·马斯克发言节点
    
    Args:
        state: 当前辩论状态
        
    Returns:
        Command: 包含更新数据和下一个节点的命令
    """
    return Command(
        update=_ask_person(state, elon, altman), 
        goto="🧑Sam"
    )


def ask_sam(state: DebatesState) -> Command[Literal["🚀Elon", "__end__"]]:
    """
    萨姆·奥特曼发言节点
    
    Args:
        state: 当前辩论状态
        
    Returns:
        Command: 包含更新数据和下一个节点的命令
    """
    # 检查是否达到最大轮数
    should_end = state["discuss_count"] >= state["max_count"]
    next_node = END if should_end else "🚀Elon"
    
    return Command(
        update=_ask_person(state, altman, elon),
        goto=next_node,
    )


# 构建LangGraph工作流
def create_debate_graph():
    """创建辩论图"""
    builder = StateGraph(DebatesState)
    
    # 添加节点
    builder.add_node("🚀Elon", ask_elon)
    builder.add_node("🧑Sam", ask_sam)
    
    # 添加边：从开始节点连接到埃隆
    builder.add_edge(START, "🚀Elon")
    
    # 编译图
    graph = builder.compile()
    
    return graph


# 创建全局图实例
graph = create_debate_graph()

# 调试和测试函数
def test_debate(topic: str = "AGI会取代人类吗?", rounds: int = 3):
    """
    测试辩论功能
    
    Args:
        topic: 辩论话题
        rounds: 辩论轮数
    """
    print(f"🎯 开始测试辩论: {topic}")
    print(f"📊 辩论轮数: {rounds}")
    print("=" * 50)
    
    inputs = {
        "main_topic": topic,
        "messages": [],
        "max_count": rounds
    }
    
    try:
        for i, output in enumerate(graph.stream(inputs, stream_mode="updates"), 1):
            print(f"轮次 {i}: {output}")
            
        print("=" * 50)
        print("✅ 辩论测试完成!")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")


# 主程序入口（用于调试）
if __name__ == "__main__":
    # 检查环境变量
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("❌ 警告: DEEPSEEK_API_KEY 环境变量未设置")
        print("请在.env文件中设置你的DeepSeek API密钥")
    else:
        print("✅ 环境变量配置正确")
        
        # 运行测试
        test_debate("人工智能是否会威胁人类就业?", 2)