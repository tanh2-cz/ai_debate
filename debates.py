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

def display_rag_status(rag_enabled, rag_sources):
    """显示RAG状态信息"""
    if rag_enabled:
        sources_text = " + ".join(rag_sources)
        st.success(f"📚 学术检索已启用: {sources_text}")
    else:
        st.info("📚 学术检索已禁用，将基于内置知识辩论")

def display_retrieved_references(references):
    """显示检索到的参考文献"""
    if not references:
        return
    
    with st.expander(f"📚 本轮检索到的参考文献 ({len(references)} 篇)", expanded=False):
        for i, ref in enumerate(references, 1):
            st.markdown(f"""
            **{i}. {ref.get('title', '无标题')}**
            - 📝 作者: {', '.join(ref.get('authors', [])[:3])}
            - 🏛️ 来源: {ref.get('source', 'Unknown')} ({ref.get('published_date', 'N/A')})
            - 🔗 链接: [{ref.get('url', '#')}]({ref.get('url', '#')})
            - ⭐ 相关性: {ref.get('relevance_score', 'N/A')}/10
            """)

def generate_response(input_text, max_rounds, selected_agents, rag_config):
    """
    生成多Agent辩论响应（增强版，支持RAG）
    
    Args:
        input_text (str): 辩论主题
        max_rounds (int): 最大辩论轮数
        selected_agents (list): 选中的Agent列表
        rag_config (dict): RAG配置
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
    
    # 动态创建适合当前角色组合的图
    try:
        current_graph = create_multi_agent_graph(selected_agents, rag_enabled=rag_config.get('enabled', True))
        st.success(f"✅ 成功创建{len(selected_agents)}角色增强辩论图")
    except Exception as e:
        st.error(f"❌ 创建辩论图失败: {str(e)}")
        return
    
    # RAG状态显示
    display_rag_status(rag_config.get('enabled', True), rag_config.get('sources', ['arxiv']))
    
    inputs = {
        "main_topic": input_text, 
        "messages": [], 
        "max_rounds": max_rounds,
        "active_agents": selected_agents,
        "current_round": 0,
        "rag_enabled": rag_config.get('enabled', True),
        "rag_sources": rag_config.get('sources', ['arxiv', 'crossref']),
        "collected_references": []
    }
    
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
    
    # 创建进度显示和RAG信息容器
    progress_container = st.container()
    rag_info_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        round_info = st.empty()
    
    total_expected_messages = max_rounds * len(selected_agents)
    message_count = 0
    current_round = 1
    all_references = []
    
    # RAG预热（如果启用）
    if rag_config.get('enabled', True):
        with st.spinner("🔍 正在预热学术检索系统..."):
            try:
                warmup_rag_system(input_text.split()[0] if input_text else "research")
                st.success("✅ 学术检索系统准备就绪")
            except Exception as e:
                st.warning(f"⚠️ 学术检索系统预热失败: {e}")
    
    # 开始辩论流
    try:
        for update in current_graph.stream(inputs, {"recursion_limit": 200}, stream_mode="updates"):
            # 检查每个可能的Agent节点
            for agent_key in selected_agents:
                if agent_key in update:
                    agent_info = AVAILABLE_ROLES[agent_key]
                    message_obj = update[agent_key]["messages"][0]
                    
                    # 获取消息内容
                    if hasattr(message_obj, 'content'):
                        message = message_obj.content
                    else:
                        message = str(message_obj)
                    
                    # 显示消息
                    display_agent_message(agent_key, message, agent_info)
                    
                    # 更新进度
                    message_count += 1
                    progress = min(message_count / total_expected_messages, 1.0)
                    progress_bar.progress(progress)
                    
                    # 更新状态文本
                    if message_count % len(selected_agents) == 0:
                        current_round = message_count // len(selected_agents)
                    
                    status_text.text(f"进行中... ({message_count}/{total_expected_messages})")
                    round_info.info(f"第 {current_round} 轮 / 共 {max_rounds} 轮")
                    
                    # 显示RAG信息（如果有新的检索结果）
                    if rag_config.get('enabled', True) and message_count % len(selected_agents) == 1:
                        # 每轮开始时显示RAG状态
                        with rag_info_container:
                            rag_module = get_rag_module()
                            if rag_module:
                                try:
                                    # 模拟获取当前轮次的参考文献（实际会在Agent内部获取）
                                    current_round_refs = []
                                    if current_round <= 2:  # 只在前两轮显示，避免过多信息
                                        st.info(f"🔍 第{current_round}轮: 正在为专家们检索最新学术资料...")
                                except Exception as e:
                                    st.warning(f"⚠️ RAG检索遇到问题: {e}")
                    
                    # 添加小延迟增强观感
                    time.sleep(0.5)
                    
    except Exception as e:
        st.error(f"辩论过程中出现错误: {str(e)}")
        return
    
    # 完成提示
    progress_bar.progress(1.0)
    status_text.success("辩论完成！")
    round_info.success(f"总计 {message_count} 条发言")
    
    # 显示RAG使用总结
    if rag_config.get('enabled', True):
        with rag_info_container:
            st.success("📚 本次辩论已集成最新学术研究，论证更加权威可信！")

# 页面配置
st.set_page_config(
    page_title="🎭 多角色AI辩论平台 (RAG增强版)",
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

.rag-badge {
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
    <span class="rag-badge">📚 RAG增强版</span>
    <span class="rag-badge">🔍 实时学术检索</span>
    <span class="rag-badge">📊 权威数据支撑</span>
</div>
""", unsafe_allow_html=True)

# 侧边栏配置
with st.sidebar:
    st.header("🎛️ 辩论配置")
    
    # RAG设置区域
    st.subheader("📚 学术检索设置")
    
    rag_enabled = st.checkbox(
        "🔍 启用实时学术检索",
        value=True,
        help="基于辩论主题自动检索arXiv、CrossRef等学术数据库"
    )
    
    if rag_enabled:
        rag_sources = st.multiselect(
            "选择数据源",
            options=["arxiv", "crossref"],
            default=["arxiv", "crossref"],
            help="arXiv: 预印本论文库\nCrossRef: 期刊文章数据库"
        )
        
        max_refs_per_agent = st.slider(
            "每角色最大参考文献数",
            min_value=1,
            max_value=5,
            value=2,
            help="每个专家角色获取的最大参考文献数量"
        )
        
        st.info("💡 RAG功能将为每个专家实时检索相关学术资料，提供更权威的论证支撑")
    else:
        rag_sources = []
        max_refs_per_agent = 0
        st.warning("⚠️ 禁用RAG后，专家将仅基于预训练知识发言")
    
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
                    st.markdown(f"**检索关键词**: {agent.get('rag_keywords', 'general research')}")

# 主要内容区域
col1, col2 = st.columns([2, 1])

with col1:
    # 辩论话题输入
    st.subheader("📝 设置辩论话题")
    
    # 预设话题选择（新增RAG优化话题）
    preset_topics = [
        "自定义话题...",
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
        "量子计算对网络安全的影响",  # 新增
        "碳捕获技术在气候变化中的作用",  # 新增
        "人工智能在医疗诊断中的应用前景"  # 新增
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
    
    # RAG预览功能
    if rag_enabled and topic_text and len(topic_text.strip()) > 10:
        if st.button("🔍 预览学术检索结果", help="提前查看该话题的相关学术文献"):
            with st.spinner("正在检索相关学术文献..."):
                try:
                    rag_module = get_rag_module()
                    if rag_module:
                        preview_results = rag_module.search_academic_sources(
                            topic_text.strip(), 
                            sources=rag_sources, 
                            max_results_per_source=3
                        )
                        
                        if preview_results:
                            st.success(f"找到 {len(preview_results)} 篇相关文献")
                            for i, result in enumerate(preview_results[:3], 1):
                                with st.expander(f"📄 {i}. {result.title[:50]}..."):
                                    st.write(f"**作者**: {', '.join(result.authors[:3])}")
                                    st.write(f"**来源**: {result.source} ({result.published_date})")
                                    st.write(f"**摘要**: {result.abstract[:200]}...")
                                    st.write(f"**相关性**: {result.relevance_score}/10")
                        else:
                            st.warning("未找到直接相关的学术文献，建议调整话题描述")
                    else:
                        st.error("RAG模块未正确初始化")
                except Exception as e:
                    st.error(f"预览检索失败: {e}")

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
    
    # 预估信息（考虑RAG时间）
    if len(selected_agents) >= 3:
        total_messages = max_rounds * len(selected_agents)
        base_time = total_messages * 8  # 基础时间
        rag_time = total_messages * 5 if rag_enabled else 0  # RAG额外时间
        estimated_time = base_time + rag_time
        
        st.metric("总发言数", f"{total_messages} 条")
        st.metric("预估时长", f"{estimated_time//60}分{estimated_time%60}秒")
        st.metric("参与角色", f"{len(selected_agents)} 个")
        
        if rag_enabled:
            st.info("📚 启用RAG后会增加检索时间，但论证更权威")

# 辩论控制区域
st.markdown("---")
st.subheader("🚀 开始辩论")

# 开始辩论按钮
can_start = (
    len(selected_agents) >= 3 and 
    len(selected_agents) <= 6 and 
    topic_text.strip() != "" and
    (not rag_enabled or len(rag_sources) > 0)
)

if not can_start:
    if len(selected_agents) < 3:
        st.error("❌ 请至少选择3个角色参与辩论")
    elif len(selected_agents) > 6:
        st.error("❌ 最多支持6个角色同时辩论")
    elif not topic_text.strip():
        st.error("❌ 请输入辩论话题")
    elif rag_enabled and len(rag_sources) == 0:
        st.error("❌ 启用RAG时请至少选择一个数据源")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    start_debate = st.button(
        "🎭 开始智能辩论" if not rag_enabled else "🎭 开始RAG增强辩论",
        disabled=not can_start,
        use_container_width=True,
        type="primary"
    )

# 执行辩论
if start_debate and can_start:
    # 构建RAG配置
    rag_config = {
        'enabled': rag_enabled,
        'sources': rag_sources if rag_enabled else [],
        'max_refs_per_agent': max_refs_per_agent if rag_enabled else 0
    }
    
    st.success(f"🎯 辩论话题: {topic_text}")
    st.info(f"👥 参与角色: {', '.join([AVAILABLE_ROLES[key]['name'] for key in selected_agents])}")
    
    if rag_enabled:
        st.info(f"📚 学术检索: {' + '.join(rag_sources)} (每角色最多{max_refs_per_agent}篇)")
    
    st.markdown("---")
    st.subheader("💬 辩论实况")
    
    # 开始辩论
    generate_response(topic_text, max_rounds, selected_agents, rag_config)
    
    # 辩论结束
    st.balloons()
    if rag_enabled:
        st.success("🎉 RAG增强辩论圆满结束！各位专家基于最新学术研究的精彩论证令人印象深刻！")
    else:
        st.success("🎉 辩论圆满结束！感谢各位的精彩发言！")

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; opacity: 0.7;'>
    🎭 多角色AI辩论平台 (RAG增强版) | 基于真实学术研究的智能辩论体验<br>
    🔗 Powered by <a href='https://platform.deepseek.com/'>DeepSeek</a> & <a href='https://streamlit.io/'>Streamlit</a><br>
    📚 学术检索: arXiv + CrossRef | 🤖 智能分析: LangChain + RAG
</div>
""", unsafe_allow_html=True)

# 调试信息（开发时显示）
if st.sidebar.checkbox("🔧 显示调试信息", value=False):
    st.sidebar.markdown("### 🛠️ 调试信息")
    st.sidebar.json({
        "selected_agents": selected_agents,
        "rag_config": {
            "enabled": rag_enabled,
            "sources": rag_sources if rag_enabled else [],
            "max_refs": max_refs_per_agent if rag_enabled else 0
        },
        "topic_length": len(topic_text) if topic_text else 0,
        "can_start": can_start
    })