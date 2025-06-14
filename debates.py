import streamlit as st
from graph import AVAILABLE_ROLES, create_multi_agent_graph, warmup_rag_system
from rag_module import get_rag_module
import time
import threading

def display_agent_message(agent_key, message, agent_info):
    """
    显示Agent消息
    
    Args:
        agent_key (str): Agent标识符
        message (str): 消息内容 
        agent_info (dict): Agent信息
    """
    icon = agent_info["icon"]
    color = agent_info["color"]
    name = agent_info["name"]
    
    # 使用自定义样式显示消息
    st.markdown(f"""
    <div style="
        border-left: 4px solid {color};
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: rgba(255,255,255,0.05);
        border-radius: 5px;
    ">
        <div style="
            display: flex;
            align-items: center;
            margin-bottom: 0.5rem;
            font-weight: bold;
            color: {color};
        ">
            {icon} {name}
        </div>
        <div style="margin-left: 1.5rem;">
            {message.replace(f'{name}:', '').strip()}
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_rag_status(rag_enabled, rag_sources, max_refs_per_agent=3):
    """显示Kimi RAG状态信息"""
    if rag_enabled:
        sources_text = "Kimi API" if "kimi" in rag_sources else " + ".join(rag_sources)
        st.success(f"🤖 Kimi学术检索已启用: {sources_text}")
        st.info(f"📄 每专家最多 {max_refs_per_agent} 篇参考文献")
    else:
        st.info("🤖 Kimi学术检索已禁用，将基于内置知识辩论")

def display_retrieved_references(references):
    """显示检索到的参考文献"""
    if not references:
        return
    
    with st.expander(f"🤖 本轮Kimi检索到的参考文献 ({len(references)} 篇)", expanded=False):
        for i, ref in enumerate(references, 1):
            st.markdown(f"""
            **{i}. {ref.get('title', '无标题')}**
            - 📝 作者: {', '.join(ref.get('authors', [])[:3])}
            - 🏛️ 来源: {ref.get('source', 'Kimi检索')} ({ref.get('published_date', 'N/A')})
            - 🔗 链接: [{ref.get('url', '#')}]({ref.get('url', '#')})
            - ⭐ 相关性: {ref.get('relevance_score', 'N/A')}/10
            """)

def preload_rag_for_all_agents(selected_agents, debate_topic, rag_config):
    """
    在第一轮开始前为所有专家预加载Kimi学术资料
    
    Args:
        selected_agents (list): 选中的专家列表
        debate_topic (str): 辩论主题
        rag_config (dict): RAG配置，包含用户设置
        
    Returns:
        dict: 预加载结果状态
    """
    if not rag_config.get('enabled', True):
        return {"success": False, "message": "Kimi RAG未启用"}
    
    rag_module = get_rag_module()
    if not rag_module:
        return {"success": False, "message": "Kimi RAG模块未初始化"}
    
    max_refs_per_agent = rag_config.get('max_refs_per_agent', 3)
    
    try:
        # 显示预加载进度
        preload_progress = st.progress(0)
        preload_status = st.empty()
        preload_details = st.empty()
        
        total_agents = len(selected_agents)
        
        st.info(f"🔍 正在为 {total_agents} 位专家检索Kimi学术资料，每人最多 {max_refs_per_agent} 篇...")
        
        for i, agent_key in enumerate(selected_agents, 1):
            agent_name = AVAILABLE_ROLES[agent_key]["name"]
            
            # 更新进度
            progress = i / total_agents
            preload_progress.progress(progress)
            preload_status.text(f"🤖 正在为专家 {i}/{total_agents} ({agent_name}) 检索Kimi学术资料...")
            
            # 为该专家检索并缓存学术资料
            context = rag_module.get_rag_context_for_agent(
                agent_role=agent_key,
                debate_topic=debate_topic,
                max_sources=max_refs_per_agent,
                max_results_per_source=2,
                force_refresh=True
            )
            
            # 显示检索结果
            if context and context.strip() != "暂无相关学术资料。":
                actual_ref_count = context.count('参考资料')
                with preload_details:
                    st.success(f"✅ {agent_name}: 获取到 {actual_ref_count} 篇相关学术文献")
            else:
                with preload_details:
                    st.warning(f"⚠️ {agent_name}: 未找到直接相关的学术文献")
            
            # 避免API限制
            if i < total_agents:
                time.sleep(3)
        
        # 完成预加载
        preload_progress.progress(1.0)
        preload_status.success(f"✅ 所有专家的Kimi学术资料预加载完成！")
        
        return {"success": True, "message": "Kimi预加载完成"}
        
    except Exception as e:
        st.error(f"❌ Kimi预加载学术资料失败: {str(e)}")
        return {"success": False, "message": f"Kimi预加载失败: {str(e)}"}

def generate_response(input_text, max_rounds, selected_agents, rag_config):
    """
    生成多Agent辩论响应（Kimi版）
    
    Args:
        input_text (str): 辞论主题
        max_rounds (int): 最大辞论轮数
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
    rag_sources = rag_config.get('sources', ['kimi'])
    rag_enabled = rag_config.get('enabled', True)
    
    # 动态创建适合当前角色组合的图
    try:
        current_graph = create_multi_agent_graph(selected_agents, rag_enabled=rag_enabled)
        st.success(f"✅ 成功创建{len(selected_agents)}角色辩论图")
    except Exception as e:
        st.error(f"❌ 创建辩论图失败: {str(e)}")
        return
    
    # Kimi RAG状态显示
    display_rag_status(rag_enabled, rag_sources, max_refs_user_set)
    
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
    
    # 如果启用Kimi RAG，进行预加载
    if rag_enabled:
        st.subheader("🤖 Kimi学术资料预加载")
        st.info(f"🔍 正在为所有专家预加载Kimi学术资料（每人最多{max_refs_user_set}篇）...")
        
        preload_result = preload_rag_for_all_agents(selected_agents, input_text, rag_config)
        
        if not preload_result["success"]:
            st.error(f"❌ Kimi预加载失败: {preload_result['message']}")
            if st.button("🚀 继续辞论（不使用Kimi RAG）"):
                rag_config['enabled'] = False
                rag_enabled = False
            else:
                return
        else:
            st.success("🎯 所有专家已准备就绪，Kimi学术资料已缓存，开始正式辩论！")
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
        "first_round_rag_completed": []
    }
    
    # 创建进度显示容器
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        round_info = st.empty()
    
    total_expected_messages = max_rounds * len(selected_agents)
    message_count = 0
    current_round = 1
    
    # Kimi RAG使用统计
    rag_stats = {
        "agents_with_refs": 0,
        "total_refs_retrieved": 0,
        "cache_hits": 0
    }
    
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
                    
                    # 显示消息
                    display_agent_message(agent_key, message, agent_info)
                    
                    # Kimi RAG使用统计
                    if rag_enabled and current_round == 1:
                        if "参考资料" in message or "研究表明" in message or "根据" in message:
                            rag_stats["agents_with_refs"] += 1
                    
                    # 更新进度
                    message_count += 1
                    progress = min(message_count / total_expected_messages, 1.0)
                    progress_bar.progress(progress)
                    
                    # 更新状态文本
                    if message_count % len(selected_agents) == 0:
                        current_round = message_count // len(selected_agents)
                    
                    status_text.text(f"进行中... ({message_count}/{total_expected_messages})")
                    round_info.info(f"第 {current_round} 轮 / 共 {max_rounds} 轮")
                    
                    # 第一轮结束后显示缓存状态
                    if rag_enabled and current_round == 1 and message_count == len(selected_agents):
                        st.info("✅ 第一轮完成！所有专家的Kimi学术资料已缓存，后续轮次将快速响应")
                        
                        # 显示Kimi RAG使用统计
                        if rag_stats["agents_with_refs"] > 0:
                            st.success(f"📊 {rag_stats['agents_with_refs']}/{len(selected_agents)} 位专家引用了Kimi检索的学术资料")
                    
                    # 添加小延迟增强观感
                    time.sleep(0.5)
                    
    except Exception as e:
        st.error(f"辩论过程中出现错误: {str(e)}")
        st.error("详细错误信息：")
        st.code(str(e))
        print(f"❌ 辩论流程错误: {e}")
        return
    
    # 完成提示
    progress_bar.progress(1.0)
    status_text.success("辩论完成！")
    round_info.success(f"总计 {message_count} 条发言")
    
    # 显示辩论总结
    if rag_enabled:
        st.success("🎉 Kimi辩论圆满结束！")
        st.info("📊 本次辩论采用了Kimi API学术检索，提供了权威性的学术支撑！")
        
        # 显示使用统计
        rag_module = get_rag_module()
        if rag_module:
            with st.expander("📈 Kimi RAG使用统计", expanded=False):
                total_expected_refs = len(selected_agents) * max_refs_user_set
                st.markdown(f"""
                **配置信息**：
                - **每专家文献数**：{max_refs_user_set} 篇
                - **参与专家**：{len(selected_agents)} 位
                - **预期总文献数**：{total_expected_refs} 篇
                
                **系统表现**：
                - **第一轮**：为 {len(selected_agents)} 位专家检索了专属学术资料
                - **后续轮次**：使用缓存，响应速度提升
                - **学术检索引擎**：Kimi API
                """)

# 页面配置
st.set_page_config(
    page_title="🎭 多角色AI辩论平台 (Kimi版)",
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

.kimi-badge {
    background: linear-gradient(45deg, #6c5ce7, #a29bfe);
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
    <span class="kimi-badge">🤖 Kimi API集成版</span>
    <span class="kimi-badge">🔍 第一轮检索+缓存</span>
    <span class="kimi-badge">🚀 响应速度优化</span>
    <span class="kimi-badge">🔧 支持用户自定义配置</span>
</div>
""", unsafe_allow_html=True)

# 侧边栏配置
with st.sidebar:
    st.header("🎛️ 辩论配置")
    
    # Kimi RAG设置区域
    st.subheader("🤖 Kimi学术检索设置")
    
    rag_enabled = st.checkbox(
        "🔍 启用Kimi智能学术检索",
        value=True,
        help="第一轮为每位专家使用Kimi检索专属资料并缓存，后续轮次快速响应"
    )
    
    if rag_enabled:
        # Kimi作为唯一数据源
        rag_sources = ["kimi"]
        st.info("📡 数据源：Kimi API（智能学术检索）")
        
        # 用户可配置的参考文献数量
        max_refs_per_agent = st.slider(
            "每角色最大参考文献数",
            min_value=1,
            max_value=5,
            value=3,
            help="设置每个专家在Kimi检索中获取的最大学术资料数量"
        )
        
        st.success("⚡ 策略：Kimi第一轮检索+缓存")
        st.info(f"""
        💡 **Kimi配置说明**：
        - **每专家文献数**：{max_refs_per_agent} 篇（用户可调）
        - **第一轮**：为每位专家使用Kimi检索专属学术资料
        - **后续轮次**：使用缓存，响应速度提升
        - **检索引擎**：Kimi API（权威学术能力）
        """)
        
        # 缓存管理
        if st.button("🗑️ 清理Kimi缓存", help="清理所有缓存的Kimi学术资料"):
            rag_module = get_rag_module()
            if rag_module:
                rag_module.clear_all_caches()
                st.success("✅ Kimi缓存已清理")
            
    else:
        rag_sources = []
        max_refs_per_agent = 0
        st.warning("⚠️ 禁用Kimi RAG后，专家将仅基于预训练知识发言")
    
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
                    st.markdown(f"**Kimi检索关键词**: {agent.get('kimi_keywords', 'general research')}")
                    st.markdown(f"**专属文献数**: {max_refs_per_agent} 篇")

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
    
    # Kimi预览功能
    if rag_enabled and topic_text and len(topic_text.strip()) > 10:
        if st.button("🤖 预览Kimi学术检索结果", help="提前查看各专家角色的Kimi相关学术文献"):
            if len(selected_agents) >= 3:
                with st.spinner("正在为各专家角色使用Kimi检索相关学术文献..."):
                    try:
                        rag_module = get_rag_module()
                        if rag_module:
                            st.info(f"🔍 预览配置：每专家 {max_refs_per_agent} 篇文献")
                            
                            # 为每个选中的专家预览检索结果
                            for agent_key in selected_agents[:3]:  # 限制预览前3个角色
                                agent_name = AVAILABLE_ROLES[agent_key]["name"]
                                
                                preview_context = rag_module.get_rag_context_for_agent(
                                    agent_role=agent_key,
                                    debate_topic=topic_text.strip(),
                                    max_sources=max_refs_per_agent,
                                    max_results_per_source=2,
                                    force_refresh=False
                                )
                                
                                if preview_context and preview_context.strip() != "暂无相关学术资料。":
                                    ref_count = preview_context.count('参考资料')
                                    with st.expander(f"🤖 {agent_name} 的Kimi相关文献 ({ref_count} 篇)"):
                                        st.markdown(preview_context[:500] + "...")
                                else:
                                    st.warning(f"⚠️ {agent_name}: Kimi未找到直接相关的学术文献")
                                
                            if len(selected_agents) > 3:
                                st.info(f"📝 预览显示前3位专家，另外 {len(selected_agents)-3} 位专家的Kimi资料将在正式辩论时检索")
                        else:
                            st.error("Kimi RAG模块未正确初始化")
                    except Exception as e:
                        st.error(f"Kimi预览检索失败: {e}")
            else:
                st.warning("请先选择至少3个专家角色")

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
        base_time = total_messages * 8  # 基础时间
        
        if rag_enabled:
            # Kimi RAG时间计算
            first_round_time = len(selected_agents) * (15 + max_refs_per_agent * 5)
            later_rounds_time = (total_messages - len(selected_agents)) * 3
            estimated_time = base_time + first_round_time + later_rounds_time
        else:
            estimated_time = base_time
        
        st.metric("总发言数", f"{total_messages} 条")
        st.metric("预估时长", f"{estimated_time//60}分{estimated_time%60}秒")
        st.metric("参与角色", f"{len(selected_agents)} 个")
        
        if rag_enabled:
            total_refs = len(selected_agents) * max_refs_per_agent
            st.success("⚡ Kimi RAG：首轮检索，后续缓存")
            st.info(f"""
            **Kimi配置**：
            - 总文献数：{total_refs} 篇
            - 每专家：{max_refs_per_agent} 篇
            - 第一轮：{first_round_time//60}分{first_round_time%60}秒（Kimi检索）
            - 后续轮次：约{later_rounds_time//60}分（缓存）
            """)

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
    button_text = f"🎭 开始Kimi辩论（{max_refs_per_agent}篇/专家）" if rag_enabled else "🎭 开始传统辩论"
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
        'sources': rag_sources if rag_enabled else [],
        'max_refs_per_agent': max_refs_per_agent if rag_enabled else 0
    }
    
    st.success(f"🎯 辩论话题: {topic_text}")
    st.info(f"👥 参与角色: {', '.join([AVAILABLE_ROLES[key]['name'] for key in selected_agents])}")
    
    if rag_enabled:
        st.info(f"🤖 Kimi RAG: 学术检索 (第一轮检索，每专家{max_refs_per_agent}篇，后续缓存)")
    
    st.markdown("---")
    st.subheader("💬 辩论实况")
    
    # 开始辩论
    generate_response(topic_text, max_rounds, selected_agents, rag_config)
    
    # 辩论结束
    st.balloons()
    if rag_enabled:
        st.success("🎉 Kimi辩论圆满结束！各位专家基于Kimi检索的学术研究的精彩论证令人印象深刻！")
        st.info("⚡ 本次辩论采用Kimi API学术检索策略，在保证学术权威性的同时提升了响应速度！")
    else:
        st.success("🎉 辩论圆满结束！感谢各位的精彩发言！")

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; opacity: 0.7;'>
    🎭 多角色AI辩论平台 (Kimi版) | 支持用户自定义配置，Kimi第一轮检索+缓存策略<br>
    🔗 Powered by <a href='https://platform.deepseek.com/'>DeepSeek</a> & <a href='https://streamlit.io/'>Streamlit</a> & <a href='https://kimi.moonshot.cn/'>Kimi API</a><br>
    🤖 学术检索: Kimi API | 🤖 智能分析: LangChain + RAG | ⚡ 策略: 缓存机制
</div>
""", unsafe_allow_html=True)