import streamlit as st
from graph import AVAILABLE_ROLES, create_multi_agent_graph, warmup_rag_system
from rag_module import get_rag_module
import time
import threading

def display_agent_message(agent_key, message, agent_info, round_num=None, is_latest=False):
    """
    显示Agent消息
    
    Args:
        agent_key (str): Agent标识符
        message (str): 消息内容 
        agent_info (dict): Agent信息
        round_num (int): 轮次编号
        is_latest (bool): 是否为最新消息
    """
    icon = agent_info["icon"]
    color = agent_info["color"]
    name = agent_info["name"]
    
    # 为最新消息添加特殊样式
    border_style = f"border-left: 5px solid {color}; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" if is_latest else f"border-left: 4px solid {color};"
    
    # 轮次标识
    round_label = f" 第{round_num}轮" if round_num else ""
    
    # 使用自定义样式显示消息
    st.markdown(f"""
    <div style="
        {border_style}
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: {'rgba(255,255,255,0.08)' if is_latest else 'rgba(255,255,255,0.05)'};
        border-radius: 5px;
        transition: all 0.3s ease;
    ">
        <div style="
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.5rem;
            font-weight: bold;
            color: {color};
        ">
            <span>{icon} {name}</span>
            <span style="font-size: 0.8rem; opacity: 0.7;">{round_label}</span>
        </div>
        <div style="margin-left: 1.5rem; {'font-weight: 500;' if is_latest else ''}">
            {message.replace(f'{name}:', '').strip()}
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_rag_status(rag_enabled, max_refs_per_agent=3):
    """显示联网搜索状态信息"""
    if rag_enabled:
        st.success(f"🌐 Kimi联网搜索已启用 | 每专家最多 {max_refs_per_agent} 篇参考文献")
    else:
        st.info("🌐 联网搜索已禁用，将基于内置知识辩论")

def preload_rag_for_all_agents(selected_agents, debate_topic, rag_config):
    """
    在第一轮开始前为所有专家预加载联网搜索资料
    
    Args:
        selected_agents (list): 选中的专家列表
        debate_topic (str): 辩论主题
        rag_config (dict): RAG配置，包含用户设置
        
    Returns:
        dict: 预加载结果状态
    """
    if not rag_config.get('enabled', True):
        return {"success": False, "message": "联网搜索未启用"}
    
    rag_module = get_rag_module()
    if not rag_module:
        return {"success": False, "message": "联网搜索模块未初始化"}
    
    max_refs_per_agent = rag_config.get('max_refs_per_agent', 3)
    
    try:
        # 显示预加载进度
        preload_progress = st.progress(0)
        preload_status = st.empty()
        
        total_agents = len(selected_agents)
        
        st.info(f"🔍 正在为 {total_agents} 位专家进行联网搜索...")
        
        preload_results = {}
        
        for i, agent_key in enumerate(selected_agents, 1):
            agent_name = AVAILABLE_ROLES[agent_key]["name"]
            
            # 更新进度
            progress = i / total_agents
            preload_progress.progress(progress)
            preload_status.text(f"🌐 正在为专家 {i}/{total_agents} ({agent_name}) 进行联网搜索...")
            
            # 为该专家进行联网搜索并缓存结果
            context = rag_module.get_rag_context_for_agent(
                agent_role=agent_key,
                debate_topic=debate_topic,
                max_sources=max_refs_per_agent,
                max_results_per_source=2,
                force_refresh=True
            )
            
            # 记录搜索结果
            if context and context.strip() != "暂无相关学术资料。":
                actual_ref_count = context.count('参考资料')
                preload_results[agent_key] = {
                    'success': True,
                    'ref_count': actual_ref_count,
                    'context_preview': context[:200] + "..."
                }
            else:
                preload_results[agent_key] = {
                    'success': False,
                    'ref_count': 0,
                    'context_preview': "未找到相关资料"
                }
            
            # 避免API限制
            if i < total_agents:
                time.sleep(3)
        
        # 完成预加载
        preload_progress.progress(1.0)
        preload_status.success(f"✅ 所有专家的联网搜索资料预加载完成！")
        
        return {"success": True, "message": "预加载完成", "results": preload_results}
        
    except Exception as e:
        st.error(f"❌ 预加载联网搜索资料失败: {str(e)}")
        return {"success": False, "message": f"预加载失败: {str(e)}"}

def generate_response(input_text, max_rounds, selected_agents, rag_config):
    """
    生成多Agent辩论响应
    
    Args:
        input_text (str): 辩论主题
        max_rounds (int): 最大辩论轮数
        selected_agents (list): 选中的Agent列表
        rag_config (dict): RAG配置，包含用户的所有设置
    """
    # 验证输入参数
    if not selected_agents:
        st.error("❌ 没有选择任何角色")
        return
    
    if len(selected_agents) < 3:
        st.error("❌ 至少需要选择3个角色")
        return
    
    if len(selected_agents) > 6:
        st.error("❌ 最多支持6个角色")
        return
    
    # 提取用户RAG设置
    max_refs_user_set = rag_config.get('max_refs_per_agent', 3)
    rag_sources = rag_config.get('sources', ['web_search'])
    rag_enabled = rag_config.get('enabled', True)
    
    # 动态创建适合当前角色组合的图
    try:
        current_graph = create_multi_agent_graph(selected_agents, rag_enabled=rag_enabled)
        st.success(f"✅ 成功创建{len(selected_agents)}角色辩论图")
    except Exception as e:
        st.error(f"❌ 创建辩论图失败: {str(e)}")
        return
    
    # 联网搜索状态显示
    display_rag_status(rag_enabled, max_refs_user_set)
    
    # 显示参与者信息
    st.subheader("🎭 本轮辩论参与者")
    cols = st.columns(len(selected_agents))
    for i, agent_key in enumerate(selected_agents):
        agent_info = AVAILABLE_ROLES[agent_key]
        with cols[i]:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; border-radius: 10px; background-color: rgba(255,255,255,0.1);">
                <div style="font-size: 2rem;">{agent_info['icon']}</div>
                <div style="font-weight: bold; color: {agent_info['color']};">{agent_info['name']}</div>
                <div style="font-size: 0.8rem; opacity: 0.8;">{agent_info['role']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 如果启用联网搜索，进行预加载
    if rag_enabled:
        st.subheader("🌐 联网搜索资料预加载")
        
        preload_result = preload_rag_for_all_agents(selected_agents, input_text, rag_config)
        
        if not preload_result["success"]:
            st.error(f"❌ 预加载失败: {preload_result['message']}")
            if st.button("🚀 继续辩论（不使用联网搜索）"):
                rag_config['enabled'] = False
                rag_enabled = False
            else:
                return
        else:
            st.success("🎯 所有专家已准备就绪，开始正式辩论！")
            st.markdown("---")
    
    # 初始化状态
    inputs = {
        "main_topic": input_text, 
        "messages": [], 
        "max_rounds": max_rounds,
        "active_agents": selected_agents,
        "current_round": 0,
        "rag_enabled": rag_enabled,
        "rag_sources": rag_sources,
        "collected_references": [],
        "max_refs_per_agent": max_refs_user_set,
        "max_results_per_source": 2,
        "agent_paper_cache": {},
        "first_round_rag_completed": [],
        # 简化版只保留基本字段
        "agent_positions": {},
        "key_points_raised": [],
        "controversial_points": []
    }
    
    # 简化的进度显示
    st.subheader("💬 辩论实况")
    progress_placeholder = st.empty()
    
    total_expected_messages = max_rounds * len(selected_agents)
    message_count = 0
    current_round = 1
    
    # 开始辩论流
    try:
        for update in current_graph.stream(inputs, {"recursion_limit": 200}, stream_mode="updates"):
            if not update:
                continue
                
            # 检查每个可能的Agent节点
            for agent_key in selected_agents:
                if agent_key in update and update[agent_key] is not None:
                    agent_update = update[agent_key]
                    
                    # 确保agent_update包含messages键
                    if not isinstance(agent_update, dict) or "messages" not in agent_update:
                        print(f"⚠️ {agent_key} 的更新数据格式无效: {agent_update}")
                        continue
                    
                    messages = agent_update["messages"]
                    
                    # 确保messages不为空
                    if not messages or len(messages) == 0:
                        print(f"⚠️ {agent_key} 的消息列表为空")
                        continue
                    
                    # 安全获取消息对象
                    try:
                        message_obj = messages[0]
                    except (IndexError, TypeError) as e:
                        print(f"⚠️ 无法获取 {agent_key} 的消息: {e}")
                        continue
                    
                    agent_info = AVAILABLE_ROLES.get(agent_key)
                    if not agent_info:
                        print(f"⚠️ 未找到 {agent_key} 的角色信息")
                        continue
                    
                    # 获取消息内容
                    if hasattr(message_obj, 'content'):
                        message = message_obj.content
                    else:
                        message = str(message_obj)
                    
                    # 确保消息不为空
                    if not message or message.strip() == "":
                        print(f"⚠️ {agent_key} 的消息内容为空")
                        continue
                    
                    # 更新计数器
                    message_count += 1
                    current_round = ((message_count - 1) // len(selected_agents)) + 1
                    
                    # 显示消息
                    is_latest = True  # 新消息总是最新的
                    display_agent_message(agent_key, message, agent_info, current_round, is_latest)
                    
                    # 简化的进度显示
                    with progress_placeholder:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("当前轮次", f"{current_round}/{max_rounds}")
                        with col2:
                            st.metric("总发言数", f"{message_count}")
                        with col3:
                            progress = message_count / total_expected_messages
                            st.metric("进度", f"{int(progress * 100)}%")
                    
                    # 添加小延迟增强观感
                    time.sleep(0.8)
                    
    except Exception as e:
        st.error(f"辩论过程中出现错误: {str(e)}")
        st.error("详细错误信息：")
        st.code(str(e))
        print(f"❌ 辩论流程错误: {e}")
        return
    
    # 完成提示
    st.success("🎉 辩论圆满结束！")

# 页面配置
st.set_page_config(
    page_title="🎭 多角色AI辩论平台",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 2rem 0;
    background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4, #FFEAA7, #D63031);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 3rem;
    font-weight: bold;
    margin-bottom: 2rem;
}

.feature-badge {
    background: linear-gradient(45deg, #667eea, #764ba2);
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 15px;
    font-size: 0.9rem;
    font-weight: bold;
    display: inline-block;
    margin: 0.2rem;
}

.agent-card {
    border: 2px solid #e0e0e0;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    transition: all 0.3s ease;
}

.agent-card:hover {
    border-color: #4ECDC4;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.stSelectbox > div > div {
    background-color: rgba(255,255,255,0.1);
}
</style>
""", unsafe_allow_html=True)

# 主标题
st.markdown("""
<h1 class="main-header">🎭 多角色AI辩论平台</h1>
<div style="text-align: center; margin-bottom: 2rem;">
    <span class="feature-badge">🌐 Kimi联网搜索</span>
    <span class="feature-badge">🚀 智能缓存</span>
    <span class="feature-badge">🎯 实时辩论</span>
</div>
""", unsafe_allow_html=True)

# 侧边栏配置
with st.sidebar:
    st.header("🎛️ 辩论配置")
    
    # 联网搜索设置区域
    st.subheader("🌐 Kimi联网搜索设置")
    
    rag_enabled = st.checkbox(
        "🔍 启用Kimi智能联网搜索",
        value=True,
        help="为每位专家进行实时联网搜索相关资料"
    )
    
    if rag_enabled:
        # 用户可配置的参考文献数量
        max_refs_per_agent = st.slider(
            "每角色最大参考文献数",
            min_value=1,
            max_value=5,
            value=3,
            help="设置每个专家在联网搜索中获取的最大资料数量"
        )
        
        st.success("⚡ Kimi联网搜索已启用")
        
        # 缓存管理
        if st.button("🗑️ 清理缓存", help="清理所有缓存的联网搜索资料"):
            rag_module = get_rag_module()
            if rag_module:
                rag_module.clear_all_caches()
                st.success("✅ 缓存已清理")
            
    else:
        max_refs_per_agent = 0
        st.warning("⚠️ 禁用联网搜索后，专家将仅基于预训练知识发言")
    
    st.markdown("---")
    
    # Agent选择
    st.subheader("👥 选择参与者")
    st.markdown("请选择3-6个不同角色参与辩论：")
    
    selected_agents = []
    for agent_key, agent_info in AVAILABLE_ROLES.items():
        if st.checkbox(
            f"{agent_info['icon']} {agent_info['name']}",
            value=(agent_key in ['environmentalist', 'economist', 'policy_maker']),  # 默认选中前3个
            key=f"select_{agent_key}"
        ):
            selected_agents.append(agent_key)
    
    # 验证选择
    if len(selected_agents) < 3:
        st.warning("⚠️ 请至少选择3个角色")
    elif len(selected_agents) > 6:
        st.warning("⚠️ 最多支持6个角色同时辩论")
    else:
        st.success(f"✅ 已选择 {len(selected_agents)} 个角色")
    
    st.markdown("---")
    
    # 显示角色信息
    st.subheader("🎭 角色说明")
    for agent_key in selected_agents:
        if agent_key in AVAILABLE_ROLES:
            agent = AVAILABLE_ROLES[agent_key]
            with st.expander(f"{agent['icon']} {agent['name']}"):
                st.markdown(f"**角色定位**: {agent['role']}")
                st.markdown(f"**关注重点**: {agent['focus']}")
                st.markdown(f"**典型观点**: {agent['perspective']}")
                if rag_enabled and agent_key in selected_agents:
                    st.markdown(f"**联网搜索**: {max_refs_per_agent} 篇资料")

# 主要内容区域
col1, col2 = st.columns([2, 1])

with col1:
    # 辩论话题输入
    st.subheader("📝 设置辩论话题")
    
    # 预设话题选择
    preset_topics = [
        "自定义话题...",
        "ChatGPT等生成式AI对教育系统的影响是正面还是负面？",
        "CRISPR基因编辑技术应该被允许用于人类胚胎吗？",
        "碳税vs碳交易：哪个更能有效应对气候变化？",
        "人工智能是否会威胁人类就业？",
        "核能发电是解决气候变化的最佳方案吗？",
        "远程工作对社会经济的长期影响",
        "数字货币能否取代传统货币？",
        "基因编辑技术的伦理边界在哪里？",
        "全民基本收入制度是否可行？",
        "太空探索的优先级vs地球环境保护",
        "人工肉类能否完全替代传统畜牧业？",
        "社交媒体监管的必要性与界限",
        "自动驾驶汽车的安全性与责任问题",
        "量子计算对网络安全的影响",
        "mRNA疫苗技术在传染病防控中的未来应用",
        "元宇宙技术对社会交往模式的改变",
        "人工智能在医疗诊断中的应用前景与风险"
    ]
    
    selected_topic = st.selectbox("选择或自定义话题：", preset_topics)
    
    if selected_topic == "自定义话题...":
        topic_text = st.text_area(
            "请输入自定义辩论话题：",
            placeholder="例如：人工智能在教育领域的应用前景...",
            height=100
        )
    else:
        topic_text = st.text_area(
            "辩论话题：",
            value=selected_topic,
            height=100
        )

with col2:
    st.subheader("⚙️ 辩论参数")
    
    # 辩论轮数
    max_rounds = st.slider(
        "辩论轮数",
        min_value=2,
        max_value=8,
        value=3,
        help="每轮所有选中的角色都会发言一次"
    )
    
    # 预估信息
    if len(selected_agents) >= 3:
        total_messages = max_rounds * len(selected_agents)
        
        st.metric("总发言数", f"{total_messages} 条")
        st.metric("参与角色", f"{len(selected_agents)} 个")
        
        if rag_enabled:
            total_refs = len(selected_agents) * max_refs_per_agent
            st.success("⚡ Kimi联网搜索已启用")
            st.info(f"总资料数：{total_refs} 篇")

# 辩论控制区域
st.markdown("---")
st.subheader("🚀 开始辩论")

# 开始辩论按钮
can_start = (
    len(selected_agents) >= 3 and 
    len(selected_agents) <= 6 and 
    topic_text.strip() != ""
)

if not can_start:
    if len(selected_agents) < 3:
        st.error("❌ 请至少选择3个角色参与辩论")
    elif len(selected_agents) > 6:
        st.error("❌ 最多支持6个角色同时辩论")
    elif not topic_text.strip():
        st.error("❌ 请输入辩论话题")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    button_text = f"🎭 开始辩论（{max_rounds}轮）"
    
    start_debate = st.button(
        button_text,
        disabled=not can_start,
        use_container_width=True,
        type="primary"
    )

# 执行辩论
if start_debate and can_start:
    # 构建完整的RAG配置
    rag_config = {
        'enabled': rag_enabled,
        'sources': ['web_search'] if rag_enabled else [],
        'max_refs_per_agent': max_refs_per_agent if rag_enabled else 0,
    }
    
    st.success(f"🎯 辩论话题: {topic_text}")
    st.info(f"👥 参与角色: {', '.join([AVAILABLE_ROLES[key]['name'] for key in selected_agents])}")
    
    feature_list = []
    if rag_enabled:
        feature_list.append(f"🌐 Kimi联网搜索 (每专家{max_refs_per_agent}篇)")
    
    if feature_list:
        st.info(f"✨ 启用特性: {' | '.join(feature_list)}")
    
    st.markdown("---")
    
    # 开始辩论
    generate_response(topic_text, max_rounds, selected_agents, rag_config)
    
    # 辩论结束
    st.balloons()

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; opacity: 0.7;'>
    🎭 多角色AI辩论平台<br>
    🔗 Powered by <a href='https://platform.deepseek.com/'>DeepSeek</a> & <a href='https://www.moonshot.cn/'>Kimi</a> & <a href='https://streamlit.io/'>Streamlit</a>
</div>
""", unsafe_allow_html=True)