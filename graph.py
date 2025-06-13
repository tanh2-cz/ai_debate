"""
多角色AI辩论系统核心逻辑 - 增强版本集成动态RAG
支持3-6个不同角色的智能辩论，基于真实学术资料
优化：第一轮为每个专家检索论文，后续轮次使用缓存
"""

from typing import TypedDict, Literal, List, Dict, Any
import os
from dotenv import find_dotenv, load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command

# 导入RAG模块
from rag_module import initialize_rag_module, get_rag_module, DynamicRAGModule

# 加载环境变量
load_dotenv(find_dotenv())

# 全局变量
deepseek = None
rag_module = None

# 初始化DeepSeek模型和RAG模块
try:
    deepseek = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0.8,        # 稍微提高温度增加观点多样性
        max_tokens=2000,        # 增加token限制以容纳RAG内容
        timeout=60,
        max_retries=3,
    )
    print("✅ DeepSeek模型初始化成功")
    
    # 初始化RAG模块
    rag_module = initialize_rag_module(deepseek)
    print("✅ RAG模块初始化成功")
    
except Exception as e:
    print(f"❌ 模型初始化失败: {e}")
    deepseek = None
    rag_module = None


class MultiAgentDebateState(MessagesState):
    """多角色辩论状态管理"""
    main_topic: str = "人工智能的发展前景"
    current_round: int = 0              # 当前轮次
    max_rounds: int = 3                 # 最大轮次
    active_agents: List[str] = []       # 活跃的Agent列表
    current_agent_index: int = 0        # 当前发言Agent索引
    total_messages: int = 0             # 总消息数
    rag_enabled: bool = True            # RAG功能开关
    rag_sources: List[str] = ["arxiv", "crossref"]  # RAG数据源
    collected_references: List[Dict] = [] # 收集的参考文献
    # 新增：每个专家的论文缓存
    agent_paper_cache: Dict[str, str] = {}  # 格式: {agent_key: rag_context}
    first_round_rag_completed: List[str] = []  # 已完成第一轮RAG检索的专家列表


# 定义所有可用的角色（保持原有定义）
AVAILABLE_ROLES = {
    "environmentalist": {
        "name": "环保主义者",
        "role": "环境保护倡导者",
        "icon": "🌱",
        "color": "#4CAF50",
        "focus": "生态平衡与可持续发展",
        "perspective": "任何决策都应考虑对环境的长远影响",
        "bio": "专业的环境保护主义者，拥有环境科学博士学位。长期关注气候变化、生物多样性保护和可持续发展。坚信经济发展必须与环境保护相协调，主张采用清洁技术和循环经济模式。",
        "speaking_style": "理性分析环境数据，引用科学研究，强调长期后果",
        "rag_keywords": "environment climate sustainability ecology conservation"
    },
    
    "economist": {
        "name": "经济学家", 
        "role": "市场经济分析专家",
        "icon": "📊",
        "color": "#FF9800",
        "focus": "成本效益与市场机制",
        "perspective": "追求经济效率和市场最优解决方案",
        "bio": "资深经济学教授，专攻宏观经济学和政策分析。擅长成本效益分析、市场失灵研究和经济政策评估。相信市场机制的力量，但也认识到政府干预的必要性。",
        "speaking_style": "用数据说话，分析成本收益，关注市场效率和经济可行性",
        "rag_keywords": "economic cost benefit market analysis finance"
    },
    
    "policy_maker": {
        "name": "政策制定者",
        "role": "公共政策专家", 
        "icon": "🏛️",
        "color": "#3F51B5",
        "focus": "政策可行性与社会治理",
        "perspective": "平衡各方利益，制定可执行的政策",
        "bio": "资深公务员和政策分析师，拥有公共管理硕士学位。在政府部门工作多年，熟悉政策制定流程、法律法规和实施挑战。善于协调各方利益，寻求平衡解决方案。",
        "speaking_style": "考虑实施难度，关注法律框架，寻求各方共识",
        "rag_keywords": "policy governance regulation implementation law"
    },
    
    "tech_expert": {
        "name": "技术专家",
        "role": "前沿科技研究者",
        "icon": "💻", 
        "color": "#9C27B0",
        "focus": "技术创新与实现路径",
        "perspective": "技术进步是解决问题的关键驱动力",
        "bio": "计算机科学博士，在科技公司担任首席技术官。专注于人工智能、机器学习和新兴技术研发。相信技术创新能够解决人类面临的重大挑战，但也关注技术伦理问题。",
        "speaking_style": "分析技术可行性，讨论创新解决方案，关注实现路径",
        "rag_keywords": "technology innovation technical feasibility AI"
    },
    
    "sociologist": {
        "name": "社会学家",
        "role": "社会影响研究专家", 
        "icon": "👥",
        "color": "#E91E63",
        "focus": "社会影响与人文关怀",
        "perspective": "关注对不同社会群体的影响和社会公平",
        "bio": "社会学教授，专注于社会变迁、不平等研究和社会政策分析。长期关注技术变革对社会结构的影响，特别是对弱势群体的影响。主张包容性发展和社会公正。",
        "speaking_style": "关注社会公平，分析对不同群体的影响，强调人文关怀",
        "rag_keywords": "social impact society community effects inequality"
    },
    
    "ethicist": {
        "name": "伦理学家",
        "role": "道德哲学研究者",
        "icon": "⚖️", 
        "color": "#607D8B",
        "focus": "伦理道德与价值判断",
        "perspective": "坚持道德原则和伦理标准",
        "bio": "哲学博士，专攻应用伦理学和技术伦理。在大学教授道德哲学，并为政府和企业提供伦理咨询。关注新技术带来的伦理挑战，主张在发展中坚持道德底线。",
        "speaking_style": "引用伦理原则，分析道德后果，坚持价值标准",
        "rag_keywords": "ethics moral responsibility values philosophy"
    }
}


# 增强版多角色辩论提示词模板（集成RAG）
ENHANCED_MULTI_AGENT_DEBATE_TEMPLATE = """
你是一位{role} - {name}。

【角色背景】
{bio}

【你的专业视角】
- 关注重点：{focus}
- 核心观点：{perspective}
- 表达风格：{speaking_style}

【当前辩论情况】
辩论主题：{main_topic}
当前轮次：第 {current_round} 轮
你的发言顺序：第 {agent_position} 位

【其他参与者】
{other_participants}

【学术参考资料】
{rag_context}

【对话历史】
{history}

【发言要求】
1. 基于你的专业背景和角色定位发表观点
2. 适当引用上述学术资料支撑你的论点（简要提及即可）
3. 针对前面发言者的观点进行回应或补充
4. 保持你的角色特色和专业立场
5. 回复控制在2-4句话，言简意赅但有说服力
6. 可以同意其他角色的合理观点，但要提出自己独特的见解
7. 直接表达观点，不需要加名字前缀
8. 如果引用研究，请简洁地说明（如"根据最新研究表明..."）

请从你的专业角度，结合学术资料发表观点：
"""


def create_enhanced_chat_template():
    """创建增强版聊天模板"""
    return ChatPromptTemplate.from_messages([
        ("system", ENHANCED_MULTI_AGENT_DEBATE_TEMPLATE),
        ("user", "请基于以上背景和学术资料发表你的专业观点"),
    ])


def format_agent_history(messages: List, active_agents: List[str]) -> str:
    """格式化对话历史"""
    if not messages:
        return "这是辩论的开始，你是本轮第一个发言的人。"
    
    formatted_history = []
    for i, message in enumerate(messages):
        # 确定发言者
        agent_index = i % len(active_agents)
        agent_key = active_agents[agent_index]
        agent_name = AVAILABLE_ROLES[agent_key]["name"]
        
        # 获取消息内容
        if hasattr(message, 'content'):
            message_content = message.content
        elif isinstance(message, str):
            message_content = message
        else:
            message_content = str(message)
        
        # 清理消息内容
        clean_message = message_content.replace(f"{agent_name}:", "").strip()
        formatted_history.append(f"{agent_name}: {clean_message}")
    
    return "\n".join(formatted_history)


def get_other_participants(active_agents: List[str], current_agent: str) -> str:
    """获取其他参与者信息"""
    others = []
    for agent_key in active_agents:
        if agent_key != current_agent:
            agent_info = AVAILABLE_ROLES[agent_key]
            others.append(f"- {agent_info['name']}({agent_info['role']})")
    return "\n".join(others)


def get_rag_context_for_agent(agent_key: str, debate_topic: str, state: MultiAgentDebateState) -> str:
    """
    为Agent获取RAG上下文（优化版）
    第一轮：检索并缓存论文
    后续轮次：使用缓存的论文
    """
    
    # 检查RAG是否启用
    if not state.get("rag_enabled", True) or not rag_module:
        return "当前未启用学术资料检索功能。"
    
    # 检查当前轮次
    current_round = state.get("current_round", 1)
    agent_paper_cache = state.get("agent_paper_cache", {})
    first_round_rag_completed = state.get("first_round_rag_completed", [])
    
    try:
        # 如果是第一轮且该专家还未检索过，进行检索并缓存
        if current_round == 1 and agent_key not in first_round_rag_completed:
            print(f"🔍 第一轮：为{AVAILABLE_ROLES[agent_key]['name']}检索专属学术资料...")
            
            # 基于角色和主题获取上下文
            context = rag_module.get_rag_context_for_agent(
                agent_role=agent_key,
                debate_topic=debate_topic,
                max_sources=3  # 第一轮获取更多资料
            )
            
            # 将结果缓存到状态中
            if context and context.strip() != "暂无相关学术资料。":
                agent_paper_cache[agent_key] = context
                first_round_rag_completed.append(agent_key)
                print(f"✅ 已为{AVAILABLE_ROLES[agent_key]['name']}缓存专属学术资料")
                return context
            else:
                return "暂未找到直接相关的最新学术研究，请基于你的专业知识发表观点。"
        
        # 如果不是第一轮或该专家已检索过，使用缓存
        elif agent_key in agent_paper_cache:
            print(f"📚 使用缓存：为{AVAILABLE_ROLES[agent_key]['name']}提供专属学术资料")
            return agent_paper_cache[agent_key]
        
        # 兜底情况
        else:
            return "暂未找到直接相关的最新学术研究，请基于你的专业知识发表观点。"
        
    except Exception as e:
        print(f"❌ 获取{agent_key}的RAG上下文失败: {e}")
        return "学术资料检索遇到技术问题，请基于你的专业知识发表观点。"


def _generate_agent_response(state: MultiAgentDebateState, agent_key: str) -> Dict[str, Any]:
    """
    生成指定Agent的回复（增强版，集成RAG，优化缓存机制）
    
    Args:
        state: 当前辩论状态
        agent_key: Agent标识符
        
    Returns:
        dict: 包含新消息和状态更新的字典
    """
    if deepseek is None:
        error_msg = f"{AVAILABLE_ROLES[agent_key]['name']}: 抱歉，AI模型未正确初始化。"
        return {
            "messages": [AIMessage(content=error_msg)],
            "total_messages": state.get("total_messages", 0) + 1,
            "current_agent_index": state.get("current_agent_index", 0) + 1,
        }
    
    try:
        agent_info = AVAILABLE_ROLES[agent_key]
        chat_template = create_enhanced_chat_template()
        pipe = chat_template | deepseek | StrOutputParser()
        
        # 格式化对话历史
        history = format_agent_history(state["messages"], state["active_agents"])
        
        # 获取其他参与者信息
        other_participants = get_other_participants(state["active_agents"], agent_key)
        
        # 确定当前Agent在本轮的位置
        current_agent_index = state.get("current_agent_index", 0)
        agent_position = (current_agent_index % len(state["active_agents"])) + 1
        
        # 获取RAG上下文（优化后的版本）
        rag_context = get_rag_context_for_agent(agent_key, state["main_topic"], state)
        
        # 调用模型生成回复
        response = pipe.invoke({
            "role": agent_info["role"],
            "name": agent_info["name"],
            "bio": agent_info["bio"],
            "focus": agent_info["focus"],
            "perspective": agent_info["perspective"],
            "speaking_style": agent_info["speaking_style"],
            "main_topic": state["main_topic"],
            "current_round": state.get("current_round", 1),
            "agent_position": agent_position,
            "other_participants": other_participants,
            "rag_context": rag_context,
            "history": history,
        })
        
        # 清理并格式化响应
        response = response.strip()
        if not response.startswith(agent_info["name"]):
            response = f"{agent_info['name']}: {response}"
        
        print(f"🗣️ {response}")
        
        # 计算新的状态
        new_total_messages = state.get("total_messages", 0) + 1
        new_agent_index = state.get("current_agent_index", 0) + 1
        new_round = ((new_total_messages - 1) // len(state["active_agents"])) + 1
        
        # 更新状态，保持缓存信息
        update_data = {
            "messages": [AIMessage(content=response)],
            "total_messages": new_total_messages,
            "current_agent_index": new_agent_index,
            "current_round": new_round,
        }
        
        # 如果在第一轮完成了RAG检索，更新缓存状态
        current_round = state.get("current_round", 1)
        if current_round == 1:
            agent_paper_cache = state.get("agent_paper_cache", {})
            first_round_rag_completed = state.get("first_round_rag_completed", [])
            
            # 如果该专家的缓存已更新，同步到状态
            if agent_key in first_round_rag_completed:
                update_data["agent_paper_cache"] = agent_paper_cache
                update_data["first_round_rag_completed"] = first_round_rag_completed
        
        return update_data
        
    except Exception as e:
        error_msg = f"{AVAILABLE_ROLES[agent_key]['name']}: 抱歉，我现在无法发言。技术问题：{str(e)}"
        print(f"❌ {agent_key} 生成回复时出错: {e}")
        return {
            "messages": [AIMessage(content=error_msg)],
            "total_messages": state.get("total_messages", 0) + 1,
            "current_agent_index": state.get("current_agent_index", 0) + 1,
        }


def create_agent_node_function(agent_key: str):
    """
    为指定Agent创建节点函数
    """
    def agent_node(state: MultiAgentDebateState) -> Command:
        # 生成回复
        update_data = _generate_agent_response(state, agent_key)
        
        # 确定下一个节点
        current_round = update_data.get("current_round", state.get("current_round", 1))
        current_agent_index = update_data.get("current_agent_index", 0)
        active_agents = state["active_agents"]
        max_rounds = state.get("max_rounds", 3)
        
        # 检查是否应该结束辩论
        if current_round > max_rounds:
            next_node = END
        else:
            # 确定下一个发言者
            next_agent_index = current_agent_index % len(active_agents)
            next_agent_key = active_agents[next_agent_index]
            next_node = next_agent_key
        
        return Command(update=update_data, goto=next_node)
    
    return agent_node


def create_multi_agent_graph(active_agents: List[str], rag_enabled: bool = True) -> StateGraph:
    """
    创建多角色辩论图（增强版）
    
    Args:
        active_agents: 活跃Agent列表
        rag_enabled: 是否启用RAG功能
        
    Returns:
        StateGraph: 编译后的图
    """
    if len(active_agents) < 3:
        raise ValueError("至少需要3个Agent参与辩论")
    
    if len(active_agents) > 6:
        raise ValueError("最多支持6个Agent参与辩论")
    
    # 验证所有Agent都存在
    for agent_key in active_agents:
        if agent_key not in AVAILABLE_ROLES:
            raise ValueError(f"未知的Agent: {agent_key}")
    
    # 创建图构建器
    builder = StateGraph(MultiAgentDebateState)
    
    # 为每个活跃Agent添加节点
    for agent_key in active_agents:
        agent_function = create_agent_node_function(agent_key)
        builder.add_node(agent_key, agent_function)
    
    # 设置起始边
    first_agent = active_agents[0]
    builder.add_edge(START, first_agent)
    
    # 输出创建信息
    rag_status = "✅ 已启用（第一轮检索+缓存）" if rag_enabled and rag_module else "❌ 未启用"
    print(f"✅ 创建优化版多角色辩论图成功")
    print(f"👥 参与者: {[AVAILABLE_ROLES[k]['name'] for k in active_agents]}")
    print(f"📚 RAG学术检索: {rag_status}")
    
    return builder.compile()


def test_enhanced_multi_agent_debate(topic: str = "人工智能对教育的影响", 
                                   rounds: int = 2, 
                                   agents: List[str] = None,
                                   enable_rag: bool = True):
    """
    测试增强版多角色辩论功能
    """
    if agents is None:
        agents = ["tech_expert", "sociologist", "ethicist"]
    
    print(f"🎯 开始测试优化版多角色辩论: {topic}")
    print(f"👥 参与者: {[AVAILABLE_ROLES[k]['name'] for k in agents]}")
    print(f"📊 辩论轮数: {rounds}")
    print(f"📚 RAG检索: {'启用（第一轮检索+缓存）' if enable_rag else '禁用'}")
    print("=" * 70)
    
    try:
        test_graph = create_multi_agent_graph(agents, rag_enabled=enable_rag)
        
        inputs = {
            "main_topic": topic,
            "messages": [],
            "max_rounds": rounds,
            "active_agents": agents,
            "current_round": 0,
            "current_agent_index": 0,
            "total_messages": 0,
            "rag_enabled": enable_rag,
            "rag_sources": ["arxiv", "crossref"],
            "collected_references": [],
            "agent_paper_cache": {},  # 新增
            "first_round_rag_completed": []  # 新增
        }
        
        for i, output in enumerate(test_graph.stream(inputs, stream_mode="updates"), 1):
            print(f"消息 {i}: {output}")
            
        print("=" * 70)
        print("✅ 优化版多角色辩论测试完成!")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")


# 工具函数：预热RAG系统
def warmup_rag_system(test_topic: str = "artificial intelligence"):
    """预热RAG系统，预加载常见主题的检索结果"""
    if rag_module:
        print("🔥 预热RAG系统...")
        try:
            rag_module.search_academic_sources(test_topic, max_results_per_source=2)
            print("✅ RAG系统预热完成")
        except Exception as e:
            print(f"⚠️ RAG系统预热失败: {e}")


# 主程序入口
if __name__ == "__main__":
    # 检查环境变量
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("❌ 警告: DEEPSEEK_API_KEY 环境变量未设置")
    else:
        print("✅ 环境变量配置正确")
        
        # 预热RAG系统
        warmup_rag_system()
        
        # 运行测试
        test_enhanced_multi_agent_debate(
            topic="人工智能是否会威胁人类就业？",
            rounds=2,
            agents=["tech_expert", "economist", "sociologist"],
            enable_rag=True
        )