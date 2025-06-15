#  多角色AI辩论平台

一个基于Streamlit的多角色AI辩论系统，支持3-6个不同专业角色就特定话题进行智能辩论。集成DeepSeek大语言模型和Kimi联网搜索功能，为每个角色提供实时的学术资料支持。并通过Siliconflow的API实现语音播放。

## 主要特性

- **多角色辩论**：支持环保主义者、经济学家、政策制定者、技术专家、社会学家、伦理学家等6种专业角色
- **智能联网搜索**：集成Kimi API，为每个角色提供最新的学术资料和研究报告
- **实时辩论展示**：流式展示辩论过程，支持多轮深度讨论
- **缓存机制**：智能缓存搜索结果，提高响应速度
- **灵活配置**：用户可自定义参与角色、辩论轮数、参考文献数量等参数
- **语音播放**：通过Siliconflow的API实现语音播放
- **多线程**：通过多线程的方式，实现语音文字生成与语音播放异步进行，从而避免等待，实现流畅连续的辩论

## 环境要求

- DeepSeek API密钥
- Kimi API密钥
- Siliconflow API密钥(如需要语音播放)

## 安装步骤

   仿照 `.env.example` 创建 `.env` 文件并添加以下内容：
   ```env
   DEEPSEEK_API_KEY=your_deepseek_api_key
   KIMI_API_KEY=your_kimi_api_key
   SILICONCLOUD_API_KEY=your_siliconflow_api_key
   ```

## 使用方法
1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **启动应用**
   ```bash
   streamlit run debates.py
   ```
3. **配置辩论**
   - 在侧边栏选择3-6个参与角色
   - 设置是否启用Kimi联网搜索
   - 配置每个角色的参考文献数量（1-5篇）
   - 设置辩论轮数（2-8轮）

4. **选择话题**
   - 从预设话题中选择，或自定义辩论话题
   - 话题可以是任何具有争议性的现实问题

5. **开始辩论**
   - 点击"开始辩论"按钮
   - 系统将自动为每个角色搜索相关资料（如启用）
   - 观看AI专家们的实时辩论过程

## 配置说明

### API密钥获取

- **DeepSeek API**：访问 [DeepSeek平台](https://platform.deepseek.com/) 注册获取
- **Kimi API**：访问 [Moonshot AI](https://www.moonshot.cn/) 注册获取
- **Siliconflow API**：访问 [硅基流动平台](https://siliconflow.cn/) 注册获取

### 角色说明

| 角色 | 专业领域 | 关注重点 |
|------|---------|----------|
| 环保主义者 | 环境科学 | 生态平衡与可持续发展 |
| 经济学家 | 市场经济 | 成本效益与市场机制 |
| 政策制定者 | 公共管理 | 政策可行性与社会治理 |
| 技术专家 | 科技研发 | 技术创新与实现路径 |
| 社会学家 | 社会研究 | 社会影响与人文关怀 |
| 伦理学家 | 道德哲学 | 伦理道德与价值判断 |

## 文件结构

```
├── debates.py        # 主应用文件（Streamlit界面）
├── graph.py          # 多智能体辩论逻辑
├── rag_module.py     # Kimi联网搜索模块
├── tts_module.py     # 文本转语音模块
├── .env              # 环境变量配置
├── requirements.txt  # 主要依赖
└── README.md         # 项目说明文档
```

## 注意事项
Kimi API累计充值少于50的账户，每分钟请求限制为3次，如果请求过多可能造成too many requests的报错。