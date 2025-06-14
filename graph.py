"""
多角色AI辩论系统核心逻辑 - 学术API集成版本
支持3-6个不同角色的智能辩论，基于学术API的学术资料检索
增强版：强调辩论连贯性和回应性
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

# 导入基于学术API的RAG模块
from rag_module import initialize_rag_module, get_rag_module, DynamicRAGModule

# 加载环境变量
load_dotenv(find_dotenv())

# 全局变量
deepseek = None
rag_module = None

# 初始化DeepSeek模型和基于学术API的RAG模块
try:
    deepseek = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0.8,        # 稍微提高温度增加观点多样性
        max_tokens=2000,        # 增加token限制以容纳RAG内容
        timeout=60,
        max_retries=3,
    )
    print("✅ DeepSeek模型初始化成功")
    
    # 初始化基于学术API的RAG模块
    rag_module = initialize_rag_module(deepseek)
    if rag_module:
        print("✅ 学术检索模块初始化成功")
    else:
        print("⚠️ 学术检索模块初始化失败，将使用传统模式")
    
except Exception as e:
    print(f"❌ 模型初始化失败: {e}")
    deepseek = None
    rag_module = None


class MultiAgentDebateState(MessagesState):
    """多角色辩论状态管理（支持用户RAG配置）"""
    main_topic: str = "人工智能的发展前景"
    current_round: int = 0              # 当前轮次
    max_rounds: int = 3                 # 最大轮次
    active_agents: List[str] = []       # 活跃的Agent列表
    current_agent_index: int = 0        # 当前发言Agent索引
    total_messages: int = 0             # 总消息数
    rag_enabled: bool = True            # RAG功能开关
    rag_sources: List[str] = ["kimi"]   # RAG数据源
    collected_references: List[Dict] = [] # 收集的参考文献
    
    # 用户RAG配置支持
    max_refs_per_agent: int = 3         # 每个专家的最大参考文献数（用户设置）
    max_results_per_source: int = 2     # 每个数据源的最大检索数（可选配置）
    
    # 专家缓存相关
    agent_paper_cache: Dict[str, str] = {}  # 格式: {agent_key: rag_context}
    first_round_rag_completed: List[str] = []  # 已完成第一轮RAG检索的专家列表
    
    # 增强辩论连贯性的新字段（默认最高级别）
    agent_positions: Dict[str, List[str]] = {}  # 每个专家在各轮的立场记录
    key_points_raised: List[str] = []  # 已提出的关键论点
    controversial_points: List[str] = []  # 存在争议的观点


# 定义所有可用的角色
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
        "search_keywords": "环境保护 气候变化 可持续发展 生态影响 环境科学"
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
        "search_keywords": "经济影响 成本效益 市场分析 经济政策 宏观经济"
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
        "search_keywords": "政策制定 监管措施 治理框架 实施策略 公共政策"
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
        "search_keywords": "技术创新 技术可行性 技术发展 技术影响 前沿科技"
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
        "search_keywords": "社会影响 社会变化 社群效应 社会公平 社会学研究"
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
        "search_keywords": "伦理道德 道德责任 价值观念 伦理框架 道德哲学"
    }
}


# 增强版多角色辩论提示词模板（强调连贯性和回应性，默认最高级别）
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
当前轮次：第 {current_round} 轮（共 {max_rounds} 轮）
你的发言顺序：第 {agent_position} 位
参与者：{other_participants}

【基于学术检索的参考资料】
{rag_context}

【对话历史与关键争议点】
{history}

【关键争议点追踪】
已提出的主要论点：{key_points}
存在争议的观点：{controversial_points}
你之前的立场：{your_previous_positions}

【连贯性发言要求（最高级别）】
🎯 **第{current_round}轮发言重点**：
1. **详细回应前轮内容**：针对前面发言者的具体观点进行深入回应或反驳
2. **深度发展你的论证**：基于你在前几轮的立场，进一步深化或完善你的观点
3. **积极寻找争议点**：从你的专业角度主动识别和挑战其他专家的观点
4. **建设性深度辩论**：既要坚持自己的立场，也要承认对方合理的部分，并提出更深层次的观点

🔄 **严格避免重复**：
- 绝对不要重复你在前几轮已经表达过的观点
- 避免使用与前轮相同的论证逻辑
- 如需重申立场，必须用全新的角度或证据

💡 **发言策略（最高连贯性级别）**：
- 如果是第1轮：阐述你的基本立场和核心关切
- 如果是第2轮及以后：必须详细回应其他专家的观点，并深化你的论证
- 积极寻找与其他专家的分歧点，提出针对性的反驳
- 提出综合性解决方案时要展现专业深度

【发言格式】
请直接发表你的观点，无需加名字前缀。控制在3-4句话内，确保：
- 明确且详细地回应至少一位其他专家的具体观点
- 充分体现你的专业特色和角色定位
- 积极推进辩论向更深层次发展
- 如引用学术资料，请简洁说明（如"根据最新研究..."）

现在请基于以上要求发表你在第{current_round}轮的观点：
"""


def create_enhanced_chat_template():
    """创建增强版聊天模板"""
    return ChatPromptTemplate.from_messages([
        ("system", ENHANCED_MULTI_AGENT_DEBATE_TEMPLATE),
        ("user", "请基于以上背景发表你的专业观点，注意保持辩论的最高级别连贯性"),
    ])


def format_agent_history(messages: List, active_agents: List[str], current_agent: str, current_round: int) -> str:
    """格式化对话历史，突出上一轮的核心观点"""
    if not messages:
        return "这是辩论的开始，你是本轮第一个发言的人。请阐述你的基本立场。"
    
    formatted_history = []
    
    # 计算要显示的消息范围（优先显示最近的轮次）
    messages_per_round = len(active_agents)
    if len(messages) <= messages_per_round:
        # 只有一轮，显示全部
        start_idx = 0
    else:
        # 显示上一轮的所有发言和本轮已有的发言
        start_idx = max(0, len(messages) - messages_per_round * 2)
    
    recent_messages = messages[start_idx:]
    
    for i, message in enumerate(recent_messages):
        # 确定发言者
        global_msg_idx = start_idx + i
        agent_index = global_msg_idx % len(active_agents)
        agent_key = active_agents[agent_index]
        agent_name = AVAILABLE_ROLES[agent_key]["name"]
        
        # 计算这条消息属于哪一轮
        msg_round = (global_msg_idx // len(active_agents)) + 1
        
        # 获取消息内容
        if hasattr(message, 'content'):
            message_content = message.content
        elif isinstance(message, str):
            message_content = message
        else:
            message_content = str(message)
        
        # 清理消息内容
        clean_message = message_content.replace(f"{agent_name}:", "").strip()
        
        # 标记轮次信息
        round_label = f"第{msg_round}轮" if msg_round < current_round else "本轮"
        formatted_history.append(f"[{round_label}] {agent_name}: {clean_message}")
    
    # 如果当前专家在之前轮次有发言，特别标注
    your_previous_messages = []
    for i, message in enumerate(messages):
        agent_index = i % len(active_agents)
        agent_key = active_agents[agent_index]
        if agent_key == current_agent:
            msg_round = (i // len(active_agents)) + 1
            if msg_round < current_round:
                if hasattr(message, 'content'):
                    content = message.content
                else:
                    content = str(message)
                clean_content = content.replace(f"{AVAILABLE_ROLES[current_agent]['name']}:", "").strip()
                your_previous_messages.append(f"第{msg_round}轮: {clean_content}")
    
    result = "\n".join(formatted_history)
    if your_previous_messages:
        result += f"\n\n[你的历史立场]\n" + "\n".join(your_previous_messages)
    
    return result


def extract_key_points_and_controversies(messages: List, active_agents: List[str]) -> tuple:
    """提取关键论点和争议点"""
    key_points = []
    controversial_points = []
    
    if len(messages) < 2:
        return [], []
    
    # 简单的关键词提取和争议点识别
    keywords_by_agent = {}
    
    for i, message in enumerate(messages):
        agent_index = i % len(active_agents)
        agent_key = active_agents[agent_index]
        agent_name = AVAILABLE_ROLES[agent_key]["name"]
        
        if hasattr(message, 'content'):
            content = message.content.lower()
        else:
            content = str(message).lower()
        
        # 识别关键观点标志词
        opinion_markers = ["认为", "相信", "主张", "强调", "建议", "反对", "支持", "担心", "关注"]
        for marker in opinion_markers:
            if marker in content:
                # 提取观点句子
                sentences = content.split('。')
                for sentence in sentences:
                    if marker in sentence and len(sentence.strip()) > 10:
                        key_points.append(f"{agent_name}: {sentence.strip()[:50]}...")
        
        # 识别争议性表达
        controversy_markers = ["但是", "然而", "不同意", "质疑", "反驳", "挑战", "问题是"]
        for marker in controversy_markers:
            if marker in content:
                controversial_points.append(f"{agent_name}对此有不同看法")
    
    # 去重和限制数量
    key_points = list(dict.fromkeys(key_points))[-5:]  # 最多5个关键点
    controversial_points = list(dict.fromkeys(controversial_points))[-3:]  # 最多3个争议点
    
    return key_points, controversial_points


def get_agent_previous_positions(messages: List, active_agents: List[str], current_agent: str) -> List[str]:
    """获取某个专家之前的立场"""
    positions = []
    
    for i, message in enumerate(messages):
        agent_index = i % len(active_agents)
        agent_key = active_agents[agent_index]
        
        if agent_key == current_agent:
            round_num = (i // len(active_agents)) + 1
            if hasattr(message, 'content'):
                content = message.content
            else:
                content = str(message)
            
            # 提取核心立场（简化处理）
            clean_content = content.replace(f"{AVAILABLE_ROLES[current_agent]['name']}:", "").strip()
            if len(clean_content) > 20:
                positions.append(f"第{round_num}轮: {clean_content[:80]}...")
    
    return positions


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
    为Agent获取RAG上下文（支持用户设置）
    第一轮：使用学术检索并缓存论文
    后续轮次：使用缓存的论文
    """
    
    # 检查RAG是否启用
    if not state.get("rag_enabled", True) or not rag_module:
        return "当前未启用学术资料检索功能。"
    
    # 从状态读取用户设置的参考文献数量
    max_refs_per_agent = state.get("max_refs_per_agent", 3)
    max_results_per_source = state.get("max_results_per_source", 2)
    
    print(f"🔍 为{AVAILABLE_ROLES[agent_key]['name']}检索学术资料，设置最大文献数为 {max_refs_per_agent} 篇")
    
    # 检查当前轮次
    current_round = state.get("current_round", 1)
    agent_paper_cache = state.get("agent_paper_cache", {})
    first_round_rag_completed = state.get("first_round_rag_completed", [])
    
    try:
        # 如果是第一轮且该专家还未检索过，进行学术检索并缓存
        if current_round == 1 and agent_key not in first_round_rag_completed:
            print(f"🔍 第一轮：为{AVAILABLE_ROLES[agent_key]['name']}使用学术检索...")
            
            # 使用用户设置的数量而不是硬编码
            context = rag_module.get_rag_context_for_agent(
                agent_role=agent_key,
                debate_topic=debate_topic,
                max_sources=max_refs_per_agent,  # 使用用户设置
                max_results_per_source=max_results_per_source,
                force_refresh=True  # 强制刷新确保最新资料
            )
            
            # 将结果缓存到状态中
            if context and context.strip() != "暂无相关学术资料。":
                agent_paper_cache[agent_key] = context
                first_round_rag_completed.append(agent_key)
                
                actual_ref_count = context.count('参考资料')
                print(f"✅ 学术检索成功：{AVAILABLE_ROLES[agent_key]['name']}获得{actual_ref_count}篇资料")
                
                return context
            else:
                print(f"⚠️ {AVAILABLE_ROLES[agent_key]['name']}未找到相关学术资料")
                return "暂未找到直接相关的最新学术研究，请基于你的专业知识发表观点。"
        
        # 如果不是第一轮或该专家已检索过，使用缓存
        elif agent_key in agent_paper_cache:
            cached_context = agent_paper_cache[agent_key]
            actual_ref_count = cached_context.count('参考资料')
            print(f"📚 使用缓存：{AVAILABLE_ROLES[agent_key]['name']}获得{actual_ref_count}篇缓存资料")
            return cached_context
        
        # 兜底情况
        else:
            return "暂未找到直接相关的最新学术研究，请基于你的专业知识发表观点。"
        
    except Exception as e:
        print(f"❌ 获取{agent_key}的学术检索上下文失败: {e}")
        return "学术资料检索遇到技术问题，请基于你的专业知识发表观点。"


def _generate_agent_response(state: MultiAgentDebateState, agent_key: str) -> Dict[str, Any]:
    """
    生成指定Agent的回复（增强连贯性版本，默认最高级别）
    
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
        
        # 计算当前轮次和位置信息
        current_total_messages = state.get("total_messages", 0)
        active_agents_count = len(state["active_agents"])
        current_round = (current_total_messages // active_agents_count) + 1
        agent_position_in_round = (current_total_messages % active_agents_count) + 1
        
        # 格式化对话历史（增强版）
        history = format_agent_history(state["messages"], state["active_agents"], agent_key, current_round)
        
        # 获取其他参与者信息
        other_participants = get_other_participants(state["active_agents"], agent_key)
        
        # 提取关键论点和争议点
        key_points, controversial_points = extract_key_points_and_controversies(state["messages"], state["active_agents"])
        
        # 获取该专家之前的立场
        previous_positions = get_agent_previous_positions(state["messages"], state["active_agents"], agent_key)
        
        # 获取学术检索上下文（支持用户配置）
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
            "current_round": current_round,
            "max_rounds": state.get("max_rounds", 3),
            "agent_position": agent_position_in_round,
            "other_participants": other_participants,
            "rag_context": rag_context,
            "history": history,
            "key_points": "，".join(key_points) if key_points else "尚未提出明确论点",
            "controversial_points": "，".join(controversial_points) if controversial_points else "暂无明显争议",
            "your_previous_positions": "\n".join(previous_positions) if previous_positions else "这是你的首次发言"
        })
        
        # 清理并格式化响应
        response = response.strip()
        if not response.startswith(agent_info["name"]):
            response = f"{agent_info['name']}: {response}"
        
        print(f"🗣️ 第{current_round}轮 {agent_info['name']}: {response}")
        
        # 计算新的状态
        new_total_messages = current_total_messages + 1
        new_agent_index = state.get("current_agent_index", 0) + 1
        new_round = (new_total_messages // active_agents_count) + 1
        
        # 更新状态，保持缓存信息和用户配置
        update_data = {
            "messages": [AIMessage(content=response)],
            "total_messages": new_total_messages,
            "current_agent_index": new_agent_index,
            "current_round": new_round,
        }
        
        # 更新关键论点和争议点追踪
        updated_key_points, updated_controversies = extract_key_points_and_controversies(
            state["messages"] + [AIMessage(content=response)], 
            state["active_agents"]
        )
        update_data["key_points_raised"] = updated_key_points
        update_data["controversial_points"] = updated_controversies
        
        # 如果在第一轮完成了学术检索，更新缓存状态
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
    """为指定Agent创建节点函数（增强连贯性版本）"""
    def agent_node(state: MultiAgentDebateState) -> Command:
        try:
            # 首先检查是否应该结束辩论
            current_total_messages = state.get("total_messages", 0)
            active_agents = state.get("active_agents", [])
            max_rounds = state.get("max_rounds", 3)
            
            # 安全检查：确保活跃agents列表不为空
            if not active_agents:
                print("❌ 活跃agents列表为空，辩论结束")
                return Command(
                    update={"messages": []},
                    goto=END
                )
            
            # 检查是否已经完成所有轮次
            total_expected_messages = max_rounds * len(active_agents)
            if current_total_messages >= total_expected_messages:
                print(f"🏁 辩论结束：已完成 {max_rounds} 轮，共 {current_total_messages} 条发言")
                return Command(
                    update={"messages": []},
                    goto=END
                )
            
            # 计算当前应该是第几轮
            current_round = (current_total_messages // len(active_agents)) + 1
            
            # 确认当前应该发言的专家
            expected_agent_index = current_total_messages % len(active_agents)
            expected_agent = active_agents[expected_agent_index]
            
            # 如果当前节点不是应该发言的专家，跳转到正确的专家
            if agent_key != expected_agent:
                print(f"🔄 跳转到正确的发言者：{expected_agent}")
                return Command(
                    update={"messages": []},
                    goto=expected_agent
                )
            
            # 生成回复
            try:
                update_data = _generate_agent_response(state, agent_key)
                
                # 安全检查：确保update_data包含必要的键
                if not update_data or "messages" not in update_data:
                    print(f"❌ {agent_key} 生成的回复数据无效")
                    update_data = {
                        "messages": [AIMessage(content=f"{AVAILABLE_ROLES[agent_key]['name']}: 抱歉，我现在无法发言。")],
                        "total_messages": current_total_messages + 1,
                        "current_agent_index": state.get("current_agent_index", 0) + 1,
                        "current_round": current_round,
                    }
                
                # 确定下一个节点
                new_total_messages = update_data.get("total_messages", current_total_messages + 1)
                total_expected_messages = max_rounds * len(active_agents)
                
                # 检查辩论是否应该结束
                if new_total_messages >= total_expected_messages:
                    print(f"🏁 辩论结束：已完成 {max_rounds} 轮，共 {new_total_messages} 条发言")
                    next_node = END
                else:
                    # 确定下一个发言者
                    next_agent_index = new_total_messages % len(active_agents)
                    next_agent_key = active_agents[next_agent_index]
                    next_node = next_agent_key
                    
                    new_round = (new_total_messages // len(active_agents)) + 1
                    print(f"📊 轮次状态：第 {new_round} 轮，总发言 {new_total_messages} 条，下一位：{AVAILABLE_ROLES[next_agent_key]['name']}")
                
                return Command(update=update_data, goto=next_node)
                
            except Exception as e:
                print(f"❌ 专家 {agent_key} 发言失败: {e}")
                error_update = {
                    "messages": [AIMessage(content=f"{AVAILABLE_ROLES[agent_key]['name']}: 抱歉，技术问题导致无法发言。")],
                    "total_messages": current_total_messages + 1,
                    "current_agent_index": state.get("current_agent_index", 0) + 1,
                    "current_round": current_round,
                }
                return Command(update=error_update, goto=END)
        
        except Exception as e:
            print(f"❌ 专家节点 {agent_key} 处理失败: {e}")
            # 最终兜底：确保总是返回有效的update
            safe_update = {
                "messages": [AIMessage(content=f"系统错误：{agent_key} 无法处理")],
                "total_messages": state.get("total_messages", 0) + 1,
                "current_agent_index": state.get("current_agent_index", 0) + 1,
            }
            return Command(update=safe_update, goto=END)
    
    return agent_node


def create_multi_agent_graph(active_agents: List[str], rag_enabled: bool = True) -> StateGraph:
    """
    创建多角色辩论图（增强连贯性版本）
    
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
    rag_status = "✅ 已启用" if rag_enabled and rag_module else "❌ 未启用"
    print(f"✅ 创建多角色辩论图成功")
    print(f"👥 参与者: {[AVAILABLE_ROLES[k]['name'] for k in active_agents]}")
    print(f"📚 学术检索: {rag_status}")
    print(f"🔄 连贯性增强: 已启用最高级别")
    
    return builder.compile()


def test_enhanced_multi_agent_debate(topic: str = "人工智能对教育的影响", 
                                   rounds: int = 3, 
                                   agents: List[str] = None,
                                   enable_rag: bool = True,
                                   max_refs_per_agent: int = 3):
    """测试增强连贯性版多角色辩论功能"""
    if agents is None:
        agents = ["tech_expert", "sociologist", "ethicist"]
    
    print(f"🎯 开始测试多角色辩论: {topic}")
    print(f"👥 参与者: {[AVAILABLE_ROLES[k]['name'] for k in agents]}")
    print(f"📊 辩论轮数: {rounds}")
    print(f"📚 学术检索: {'启用' if enable_rag else '禁用'}")
    print(f"📄 每专家文献数: {max_refs_per_agent} 篇")
    print("=" * 70)
    
    try:
        test_graph = create_multi_agent_graph(agents, rag_enabled=enable_rag)
        
        # 测试用户配置传递
        inputs = {
            "main_topic": topic,
            "messages": [],
            "max_rounds": rounds,
            "active_agents": agents,
            "current_round": 0,
            "current_agent_index": 0,
            "total_messages": 0,
            "rag_enabled": enable_rag,
            "rag_sources": ["kimi"],  # 使用学术API作为数据源
            "collected_references": [],
            "max_refs_per_agent": max_refs_per_agent,
            "max_results_per_source": 2,
            "agent_paper_cache": {},
            "first_round_rag_completed": [],
            # 连贯性追踪字段（默认最高级别）
            "agent_positions": {},
            "key_points_raised": [],
            "controversial_points": []
        }
        
        print(f"🔍 测试配置：每专家{max_refs_per_agent}篇参考文献，连贯性追踪已启用最高级别")
        
        for i, output in enumerate(test_graph.stream(inputs, stream_mode="updates"), 1):
            print(f"消息 {i}: {output}")
            
        print("=" * 70)
        print("✅ 多角色辩论测试完成!")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")


# 工具函数：预热学术检索系统
def warmup_rag_system(test_topic: str = "人工智能"):
    """预热学术检索系统，测试API连接"""
    if rag_module:
        print("🔥 预热学术检索系统...")
        try:
            # 测试一个简单的检索请求
            test_results = rag_module.search_academic_sources(test_topic, max_results_per_source=1)
            if test_results:
                print("✅ 学术检索系统预热完成，API连接正常")
            else:
                print("⚠️ 学术检索系统预热完成，但未检索到测试结果")
        except Exception as e:
            print(f"⚠️ 学术检索系统预热失败: {e}")


# 主程序入口
if __name__ == "__main__":
    # 检查环境变量
    missing_keys = []
    if not os.getenv("DEEPSEEK_API_KEY"):
        missing_keys.append("DEEPSEEK_API_KEY")
    if not os.getenv("KIMI_API_KEY"):
        missing_keys.append("KIMI_API_KEY")
    
    if missing_keys:
        print(f"❌ 警告: {', '.join(missing_keys)} 环境变量未设置")
        print("请设置以下环境变量：")
        for key in missing_keys:
            print(f"export {key}=your_api_key")
    else:
        print("✅ 环境变量配置正确")
        
        # 预热学术检索系统
        warmup_rag_system()
        
        # 测试增强连贯性版辩论
        test_enhanced_multi_agent_debate(
            topic="ChatGPT对教育的影响",
            rounds=3,
            agents=["tech_expert", "sociologist", "ethicist"],
            enable_rag=True,
            max_refs_per_agent=3
        )