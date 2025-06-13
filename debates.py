import streamlit as st
from graph import graph

def generate_response(input_text, max_count):
    """
    生成AI辩论响应
    
    Args:
        input_text (str): 辩论主题
        max_count (int): 最大辩论轮数
    """
    inputs = {"main_topic": input_text, "messages": [], "max_count": max_count}
    
    # 使用LangGraph流式处理辩论
    for update in graph.stream(inputs, {"recursion_limit": 100}, stream_mode="updates"):
        # 显示埃隆·马斯克的回复
        if "🚀Elon" in update:
            st.info(update["🚀Elon"]["messages"][0], icon="🚀")
        
        # 显示萨姆·奥特曼的回复
        if "🧑Sam" in update:
            st.info(update["🧑Sam"]["messages"][0], icon="🧑")

# Streamlit应用主界面
st.title("🦜🔗 AI辩论场!")
st.markdown("---")
st.markdown("### 欢迎来到AI辩论场！观看埃隆·马斯克和萨姆·奥特曼的精彩辩论")

# 侧边栏说明
with st.sidebar:
    st.header("📖 使用说明")
    st.markdown("""
    **辩论角色：**
    - 🚀 **埃隆·马斯克**: 对AGI持谨慎态度
    - 🧑 **萨姆·奥特曼**: AGI技术的乐观推动者
    
    **操作步骤：**
    1. 输入你想讨论的话题
    2. 设置辩论轮数
    3. 点击开始辩论
    4. 观看精彩对话！
    """)
    
    st.header("🤖 技术栈")
    st.markdown("""
    - **AI模型**: DeepSeek-Chat
    - **前端**: Streamlit
    - **工作流**: LangGraph
    """)

# 主要表单区域
with st.form("debate_form"):
    st.subheader("🎯 设置辩论参数")
    
    # 辩论话题输入
    text = st.text_area(
        "辩论话题:",
        value="AGI会取代人类吗?",
        height=100,
        help="输入你想让两位AI大佬辩论的话题"
    )
    
    # 辩论轮数设置
    col1, col2 = st.columns(2)
    with col1:
        max_count = st.number_input(
            "辩论轮数", 
            min_value=5, 
            max_value=50, 
            value=10,
            help="设置辩论的消息轮数"
        )
    
    with col2:
        st.metric("预估时间", f"{max_count * 10}秒")
    
    # 提交按钮
    submitted = st.form_submit_button(
        "🚀 开始辩论", 
        use_container_width=True,
        type="primary"
    )
    
    # 处理表单提交
    if submitted:
        if not text.strip():
            st.error("请输入辩论话题！")
        else:
            st.success(f"开始辩论: {text}")
            st.markdown("---")
            
            # 显示辩论进度
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 生成辩论响应
            with st.container():
                st.subheader("💬 辩论实况")
                generate_response(text, max_count)
            
            # 完成提示
            progress_bar.progress(100)
            status_text.success("辩论结束！")
            st.balloons()

# 页脚
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🔗 Powered by <a href='https://platform.deepseek.com/'>DeepSeek</a> & <a href='https://streamlit.io/'>Streamlit</a></p>
    </div>
    """, 
    unsafe_allow_html=True
)