"""
动态RAG模块 - 基于Kimi API联网搜索的学术文献检索
使用Kimi API的$web_search工具进行实时联网学术文献检索和分析
集成JSON Mode功能，获得结构化的搜索结果
支持基于专家角色的缓存机制和联网搜索
"""

import os
import asyncio
import aiohttp
import requests
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import time
import re

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_deepseek import ChatDeepSeek
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# 配置
RAG_CONFIG = {
    "max_results_per_source": 5,
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "similarity_threshold": 0.7,
    "cache_duration_hours": 24,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    # 专家缓存过期时间（小时）
    "agent_cache_duration_hours": 6,
    # Kimi API配置（使用联网搜索）
    "api_url": "https://api.moonshot.cn/v1/chat/completions",
    "api_model": "moonshot-v1-auto",
    "api_timeout": 60
}

@dataclass
class SearchResult:
    """检索结果数据类"""
    title: str
    authors: List[str]
    abstract: str
    url: str
    published_date: str
    source: str
    relevance_score: float = 0.0
    key_findings: str = ""

class RAGCache:
    """RAG结果缓存管理（支持专家角色缓存）"""
    
    def __init__(self, cache_dir: str = "./rag_cache"):
        self.cache_dir = cache_dir
        self.agent_cache_dir = os.path.join(cache_dir, "agent_cache")
        
        try:
            os.makedirs(cache_dir, exist_ok=True)
            os.makedirs(self.agent_cache_dir, exist_ok=True)
        except Exception as e:
            print(f"⚠️ 缓存目录创建失败: {e}")
    
    def _get_cache_key(self, query: str, sources: List[str]) -> str:
        """生成缓存键"""
        try:
            key_string = f"{query}_{'-'.join(sorted(sources))}"
            return hashlib.md5(key_string.encode()).hexdigest()
        except Exception as e:
            print(f"⚠️ 缓存键生成失败: {e}")
            return f"fallback_{hash(query)}"
    
    def _get_agent_cache_key(self, agent_role: str, debate_topic: str) -> str:
        """生成专家角色特定的缓存键"""
        try:
            key_string = f"agent_{agent_role}_{debate_topic}"
            return hashlib.md5(key_string.encode()).hexdigest()
        except Exception as e:
            print(f"⚠️ 专家缓存键生成失败: {e}")
            return f"agent_fallback_{agent_role}_{hash(debate_topic)}"
    
    def get_cached_results(self, query: str, sources: List[str]) -> Optional[List[SearchResult]]:
        """获取缓存的检索结果"""
        try:
            cache_key = self._get_cache_key(query, sources)
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            
            if not os.path.exists(cache_file):
                return None
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # 检查是否过期
            cache_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cache_time > timedelta(hours=RAG_CONFIG['cache_duration_hours']):
                os.remove(cache_file)
                return None
            
            # 重构SearchResult对象
            results = []
            for item in cache_data['results']:
                try:
                    results.append(SearchResult(**item))
                except Exception as e:
                    print(f"⚠️ 缓存结果解析失败: {e}")
                    continue
            
            return results
            
        except Exception as e:
            print(f"❌ 缓存读取错误: {e}")
            return None
    
    def cache_results(self, query: str, sources: List[str], results: List[SearchResult]):
        """缓存检索结果"""
        try:
            cache_key = self._get_cache_key(query, sources)
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'sources': sources,
                'results': [result.__dict__ for result in results]
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"❌ 缓存写入错误: {e}")
    
    def get_agent_cached_context(self, agent_role: str, debate_topic: str) -> Optional[str]:
        """获取专家角色特定的缓存上下文"""
        try:
            cache_key = self._get_agent_cache_key(agent_role, debate_topic)
            cache_file = os.path.join(self.agent_cache_dir, f"{cache_key}.json")
            
            if not os.path.exists(cache_file):
                return None
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # 检查是否过期
            cache_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cache_time > timedelta(hours=RAG_CONFIG['agent_cache_duration_hours']):
                os.remove(cache_file)
                return None
            
            return cache_data['context']
            
        except Exception as e:
            print(f"❌ 专家缓存读取错误: {e}")
            return None
    
    def cache_agent_context(self, agent_role: str, debate_topic: str, context: str):
        """缓存专家角色特定的上下文"""
        try:
            cache_key = self._get_agent_cache_key(agent_role, debate_topic)
            cache_file = os.path.join(self.agent_cache_dir, f"{cache_key}.json")
            
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'agent_role': agent_role,
                'debate_topic': debate_topic,
                'context': context
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
            print(f"✅ 已缓存专家 {agent_role} 的学术资料")
                
        except Exception as e:
            print(f"❌ 专家缓存写入错误: {e}")
    
    def clear_agent_cache(self, agent_role: str = None):
        """清理专家缓存（可选择特定角色）"""
        try:
            if agent_role:
                # 清理特定角色的缓存
                for filename in os.listdir(self.agent_cache_dir):
                    if filename.startswith(f"agent_{agent_role}_"):
                        try:
                            os.remove(os.path.join(self.agent_cache_dir, filename))
                        except Exception as e:
                            print(f"⚠️ 删除缓存文件失败: {filename}, {e}")
                print(f"✅ 已清理专家 {agent_role} 的缓存")
            else:
                # 清理所有专家缓存
                for filename in os.listdir(self.agent_cache_dir):
                    try:
                        os.remove(os.path.join(self.agent_cache_dir, filename))
                    except Exception as e:
                        print(f"⚠️ 删除缓存文件失败: {filename}, {e}")
                print("✅ 已清理所有缓存")
        except Exception as e:
            print(f"❌ 清理缓存失败: {e}")

class WebSearchTool:
    """基于Kimi API的$web_search工具实现 (集成JSON Mode)"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("KIMI_API_KEY")
        self.api_url = RAG_CONFIG["api_url"]
        self.model = RAG_CONFIG["api_model"]
        self.session = requests.Session()
        
        if not self.api_key:
            print("⚠️ 警告: KIMI_API_KEY 环境变量未设置")
        else:
            print("✅ Kimi联网搜索工具初始化成功")
    
    def web_search_impl(self, arguments: Dict[str, Any]) -> Any:
        """实现web_search工具的具体逻辑"""
        # 在使用Kimi提供的$web_search工具的场合，只需要原封不动返回arguments即可
        # 不需要额外的处理逻辑
        return arguments
    
    def search_with_web_tool(self, query: str, agent_role: str = "") -> str:
        """使用Kimi的$web_search工具进行联网搜索 (启用JSON Mode)"""
        if not self.api_key:
            print("❌ Kimi API Key 未配置")
            return "联网搜索功能不可用，API密钥未设置。"
        
        try:
            # 构建搜索提示词 (JSON Mode)
            search_prompt = self._build_web_search_prompt_json(query, agent_role)
            
            print(f"🔍 正在使用Kimi联网搜索 (JSON Mode): {query}")
            
            # 调用Kimi API with $web_search tool and JSON Mode
            response = self._call_kimi_with_web_search_json(search_prompt)
            
            if response:
                return response
            else:
                return "联网搜索未返回有效结果。"
                
        except Exception as e:
            print(f"❌ 联网搜索失败: {e}")
            return f"联网搜索遇到技术问题: {str(e)}"
    
    def _build_web_search_prompt_json(self, query: str, agent_role: str = "") -> str:
        """构建使用JSON Mode的联网搜索提示词"""
        role_context = ""
        if agent_role:
            role_mapping = {
                "environmentalist": "环保主义者、环境科学专家",
                "economist": "经济学家、市场分析师",
                "policy_maker": "政策制定者、公共管理专家",
                "tech_expert": "技术专家、科技研究者",
                "sociologist": "社会学家、社会影响研究专家",
                "ethicist": "伦理学家、道德哲学研究者"
            }
            role_context = f"特别关注{role_mapping.get(agent_role, agent_role)}的视角，"
        
        prompt = f"""请使用联网搜索功能，{role_context}帮我搜索关于"{query}"的最新信息和学术资料。

请使用如下JSON格式输出搜索结果：

{{
  "search_results": [
    {{
      "title": "资料标题",
      "source": "来源网站或期刊",
      "published_date": "发布时间",
      "key_findings": "核心观点和发现",
      "relevance_score": 8,
      "url": "链接地址"
    }}
  ]
}}

搜索要求：
1. 寻找权威的学术文献、研究报告和最新资讯
2. 优先选择近期发表的高质量内容
3. 包含中英文资源，关注学术期刊和研究机构
4. 提取关键信息并整理为上述JSON格式
5. relevance_score为1-10的相关性评分

现在请为我搜索关于"{query}"的信息并以JSON格式返回："""
        
        return prompt
    
    def _call_kimi_with_web_search_json(self, prompt: str) -> Optional[str]:
        """调用Kimi API并支持$web_search工具和JSON Mode"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # 初始请求消息
            messages = [
                {"role": "system", "content": "你是Kimi。由Moonshot AI提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会寻求回应用户使用JSON格式的要求，如用户要求JSON格式输出。"},
                {"role": "user", "content": prompt}
            ]
            
            # 循环处理可能的工具调用
            finish_reason = None
            while finish_reason is None or finish_reason == "tool_calls":
                data = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.3,
                    "response_format": {"type": "json_object"},  # 启用JSON Mode
                    "tools": [
                        {
                            "type": "builtin_function",
                            "function": {
                                "name": "$web_search",
                            },
                        },
                    ]
                }
                
                response = self.session.post(
                    self.api_url,
                    headers=headers,
                    json=data,
                    timeout=RAG_CONFIG["api_timeout"]
                )
                
                response.raise_for_status()
                result = response.json()
                
                if "choices" not in result or len(result["choices"]) == 0:
                    print("❌ Kimi API 响应格式异常")
                    return None
                
                choice = result["choices"][0]
                finish_reason = choice["finish_reason"]  # 修复：使用字典访问方式
                
                # 添加助手回复到消息历史
                messages.append(choice["message"])  # 修复：使用字典访问方式
                
                # 判断当前返回内容是否包含tool_calls
                if finish_reason == "tool_calls":
                    # 处理工具调用
                    tool_calls = choice["message"].get("tool_calls", [])
                    for tool_call in tool_calls:
                        tool_call_name = tool_call["function"]["name"]
                        tool_call_arguments = json.loads(tool_call["function"]["arguments"])
                        
                        if tool_call_name == "$web_search":
                            # 使用联网搜索过程中，由Kimi大模型根据联网搜索结果生成的Tokens打印出来
                            search_content_total_tokens = tool_call_arguments.get("usage", {}).get("total_tokens", 0)
                            print(f"search_content_total_tokens: {search_content_total_tokens}")
                            
                            tool_result = self.web_search_impl(tool_call_arguments)
                        else:
                            tool_result = f"Error: unable to find tool by name '{tool_call_name}'"
                        
                        # 使用函数执行结果构造一个 role=tool 的 message，以此来向模型展示工具调用的结果
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": tool_call_name,
                            "content": json.dumps(tool_result),
                        })
                
                # 如果finish_reason不是tool_calls，说明模型已完成响应
                if finish_reason != "tool_calls":
                    return choice["message"]["content"]  # 修复：使用字典访问方式
            
            return None
                
        except requests.exceptions.Timeout:
            print("❌ Kimi API 请求超时")
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Kimi API 请求错误: {e}")
            return None
        except Exception as e:
            print(f"❌ Kimi API 调用失败: {e}")
            return None

class AcademicSearcher:
    """基于Kimi API联网搜索的学术文献检索器 (集成JSON Mode)"""
    
    def __init__(self, api_key: str = None):
        self.web_tool = WebSearchTool(api_key)
        
        if not self.web_tool.api_key:
            print("⚠️ 警告: KIMI_API_KEY 环境变量未设置")
        else:
            print("✅ 学术联网搜索器初始化成功")
    
    def search(self, query: str, max_results: int = 5, agent_role: str = "") -> List[SearchResult]:
        """使用Kimi联网搜索进行学术文献检索 (JSON Mode)"""
        try:
            print(f"🔍 正在使用Kimi联网搜索学术文献 (JSON Mode): {query} (最多{max_results}篇)")
            
            # 使用联网搜索 (JSON Mode)
            search_response = self.web_tool.search_with_web_tool(query, agent_role)
            
            if search_response and search_response != "联网搜索未返回有效结果。":
                # 解析JSON响应
                results = self._parse_json_search_response(search_response, query, max_results)
                return results
            else:
                print("⚠️ 联网搜索未返回有效结果")
                return []
                
        except Exception as e:
            print(f"❌ 学术联网搜索失败: {e}")
            return []
    
    def _parse_json_search_response(self, response: str, query: str, max_results: int) -> List[SearchResult]:
        """解析JSON格式的联网搜索响应"""
        results = []
        
        try:
            # 解析JSON响应
            json_data = json.loads(response)
            
            # 获取搜索结果数组
            search_results = json_data.get("search_results", [])
            
            for item in search_results[:max_results]:
                try:
                    # 提取信息
                    title = item.get("title", "").strip()
                    source = item.get("source", "").strip()
                    published_date = item.get("published_date", "").strip()
                    key_findings = item.get("key_findings", "").strip()
                    url = item.get("url", "").strip()
                    relevance_score = float(item.get("relevance_score", 7.0))
                    
                    # 基本验证
                    if title and len(title) > 5:
                        result = SearchResult(
                            title=title,
                            authors=["联网搜索获取"],
                            abstract=key_findings[:300] if key_findings else "通过联网搜索获得的资料",
                            url=url or "通过联网搜索获得",
                            published_date=published_date or datetime.now().strftime('%Y-%m-%d'),
                            source=source or "联网搜索",
                            relevance_score=min(max(relevance_score, 1.0), 10.0),
                            key_findings=key_findings
                        )
                        results.append(result)
                        
                except Exception as e:
                    print(f"⚠️ 解析单个JSON搜索结果失败: {e}")
                    continue
            
            print(f"✅ JSON联网搜索解析得到 {len(results)} 篇文献")
            return results
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON解析失败，尝试文本解析: {e}")
            return self._fallback_text_extraction(response, query, max_results)
        except Exception as e:
            print(f"❌ JSON搜索响应解析失败: {e}")
            return self._fallback_text_extraction(response, query, max_results)
    
    def _fallback_text_extraction(self, response: str, query: str, max_results: int) -> List[SearchResult]:
        """备用文本提取方法"""
        results = []
        
        try:
            # 简单的文本分段和信息提取
            sentences = response.split('。')
            current_title = ""
            current_content = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # 简单的标题识别
                if len(sentence) < 100 and ('研究' in sentence or '论文' in sentence or '报告' in sentence):
                    if current_title and current_content:
                        # 保存前一个结果
                        result = SearchResult(
                            title=current_title,
                            authors=["联网搜索获取"],
                            abstract=current_content[:200],
                            url="通过联网搜索获得",
                            published_date=datetime.now().strftime('%Y-%m-%d'),
                            source="联网搜索",
                            relevance_score=6.0,
                            key_findings=current_content[:150]
                        )
                        results.append(result)
                        
                        if len(results) >= max_results:
                            break
                    
                    current_title = sentence
                    current_content = ""
                else:
                    current_content += sentence + "。"
            
            # 处理最后一个
            if current_title and current_content and len(results) < max_results:
                result = SearchResult(
                    title=current_title,
                    authors=["联网搜索获取"],
                    abstract=current_content[:200],
                    url="通过联网搜索获得",
                    published_date=datetime.now().strftime('%Y-%m-%d'),
                    source="联网搜索",
                    relevance_score=6.0,
                    key_findings=current_content[:150]
                )
                results.append(result)
            
            # 如果还是没有结果，创建一个包含原始响应的结果
            if not results:
                result = SearchResult(
                    title=f"关于{query}的联网搜索结果",
                    authors=["联网搜索获取"],
                    abstract=response[:300],
                    url="通过联网搜索获得",
                    published_date=datetime.now().strftime('%Y-%m-%d'),
                    source="联网搜索",
                    relevance_score=5.0,
                    key_findings=response[:200]
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ 备用文本提取失败: {e}")
            return []

class RAGEnhancer:
    """RAG增强器 - 处理检索结果并生成洞察"""
    
    def __init__(self, llm: ChatDeepSeek):
        self.llm = llm
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个学术研究分析专家。基于给定的联网搜索结果，提取和总结关键发现，为特定角色的辩论提供支撑。

你的任务：
1. 分析搜索结果的核心观点和发现
2. 提取与辩论主题和指定角色相关的关键证据
3. 简洁地总结主要论点（2-3句话）
4. 评估信息的可信度和相关性

专家角色：{agent_role}
搜索结果信息：
标题：{title}
来源：{source}
核心内容：{abstract}
主要发现：{key_findings}
发布时间：{published_date}

辩论主题：{debate_topic}

请特别关注与{agent_role}专业领域相关的内容，提供：
1. 关键发现（核心观点和证据）
2. 与辩论主题的相关性评分（1-10分）
3. 建议该角色在辩论中如何引用这项信息"""),
            ("user", "请基于搜索结果分析并提供关键洞察")
        ])
    
    def enhance_results(self, results: List[SearchResult], debate_topic: str, agent_role: str = "") -> List[SearchResult]:
        """增强检索结果，提取关键洞察（针对特定角色优化）"""
        enhanced_results = []
        
        for result in results:
            try:
                # 如果结果已经有key_findings，可以用LLM进一步分析
                if self.llm and result.key_findings:
                    analysis = self._analyze_result(result, debate_topic, agent_role)
                    if analysis.get('enhanced_findings'):
                        result.key_findings = analysis['enhanced_findings']
                    if analysis.get('relevance_score'):
                        result.relevance_score = analysis['relevance_score']
                
                enhanced_results.append(result)
                
                # 避免API限制
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ 搜索结果分析失败 {result.title}: {e}")
                # 即使分析失败也保留原始结果
                enhanced_results.append(result)
        
        # 按相关性评分排序
        try:
            enhanced_results.sort(key=lambda x: x.relevance_score, reverse=True)
        except Exception as e:
            print(f"⚠️ 结果排序失败: {e}")
        
        return enhanced_results
    
    def _analyze_result(self, result: SearchResult, debate_topic: str, agent_role: str = "") -> dict:
        """分析单个搜索结果（针对特定角色）"""
        try:
            if not self.llm:
                return {
                    'enhanced_findings': result.key_findings,
                    'relevance_score': result.relevance_score or 5.0
                }
            
            pipe = self.analysis_prompt | self.llm | StrOutputParser()
            
            response = pipe.invoke({
                'agent_role': agent_role,
                'title': result.title,
                'source': result.source,
                'abstract': result.abstract[:1000],
                'key_findings': result.key_findings[:500] if result.key_findings else "",
                'published_date': result.published_date,
                'debate_topic': debate_topic
            })
            
            # 简单解析响应
            lines = response.strip().split('\n')
            enhanced_findings = ""
            relevance_score = result.relevance_score or 5.0
            
            for line in lines:
                if '关键发现' in line or '核心观点' in line:
                    enhanced_findings = line.split('：', 1)[-1].strip()
                elif '相关性' in line and '分' in line:
                    try:
                        import re
                        score_match = re.search(r'(\d+)', line)
                        if score_match:
                            relevance_score = float(score_match.group(1))
                    except:
                        pass
            
            return {
                'enhanced_findings': enhanced_findings or response[:200],
                'relevance_score': min(max(relevance_score, 1.0), 10.0)
            }
            
        except Exception as e:
            print(f"❌ LLM分析错误: {e}")
            return {
                'enhanced_findings': result.key_findings,
                'relevance_score': result.relevance_score or 5.0
            }

class DynamicRAGModule:
    """动态RAG主模块（基于Kimi API联网搜索的学术文献检索，集成JSON Mode）"""
    
    def __init__(self, llm: ChatDeepSeek):
        self.llm = llm
        self.cache = RAGCache()
        self.academic_searcher = AcademicSearcher()
        self.enhancer = RAGEnhancer(llm) if llm else None
        
        # 初始化向量存储（可选，用于更复杂的相似性检索）
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=RAG_CONFIG["embedding_model"]
            )
            print("✅ 嵌入模型加载成功")
        except Exception as e:
            print(f"⚠️ 嵌入模型加载失败: {e}")
            self.embeddings = None
    
    def search_academic_sources(self, 
                              topic: str, 
                              sources: List[str] = ["web_search"],
                              max_results_per_source: int = None,
                              agent_role: str = "") -> List[SearchResult]:
        """
        搜索学术数据源（使用联网搜索，集成JSON Mode）
        
        Args:
            topic: 搜索主题
            sources: 数据源列表
            max_results_per_source: 每个数据源的最大结果数（用户可配置）
            agent_role: 专家角色（用于定制化分析）
        """
        
        if max_results_per_source is None:
            max_results_per_source = RAG_CONFIG["max_results_per_source"]
        
        print(f"🔍 JSON Mode联网学术搜索配置：最多{max_results_per_source}篇，角色定制：{agent_role}")
        
        # 参数安全检查
        if not topic or not topic.strip():
            print("⚠️ 搜索主题为空")
            return []
        
        if not sources:
            sources = ["web_search"]  # 默认使用联网搜索
        
        # 检查缓存
        try:
            cached_results = self.cache.get_cached_results(topic, sources)
            if cached_results:
                print(f"✅ 使用缓存结果: {len(cached_results)} 篇文献")
                # 如果有角色信息，重新排序以适合该角色
                if agent_role and self.enhancer:
                    try:
                        cached_results = self.enhancer.enhance_results(cached_results, topic, agent_role)
                    except Exception as e:
                        print(f"⚠️ 缓存结果增强失败: {e}")
                return cached_results
        except Exception as e:
            print(f"⚠️ 缓存检查失败: {e}")
        
        all_results = []
        
        # 联网搜索 (JSON Mode)
        if "web_search" in sources or "kimi" in sources:  # 兼容旧的标识
            try:
                search_results = self.academic_searcher.search(topic, max_results_per_source, agent_role)
                all_results.extend(search_results)
                print(f"🌐 JSON Mode联网搜索找到 {len(search_results)} 篇文献（设置上限：{max_results_per_source}篇）")
                
            except Exception as e:
                print(f"❌ JSON Mode联网搜索出错: {e}")
        
        # 使用LLM增强结果（考虑专家角色）
        if all_results and self.enhancer:
            try:
                print(f"🤖 使用AI分析搜索相关性{'（为' + agent_role + '定制）' if agent_role else ''}...")
                all_results = self.enhancer.enhance_results(all_results, topic, agent_role)
            except Exception as e:
                print(f"⚠️ LLM增强失败，使用原始结果: {e}")
        
        # 缓存结果
        if all_results:
            try:
                self.cache.cache_results(topic, sources, all_results)
                print(f"💾 缓存了 {len(all_results)} 篇文献")
            except Exception as e:
                print(f"⚠️ 缓存写入失败: {e}")
        
        return all_results
    
    def get_rag_context_for_agent(self, 
                                 agent_role: str, 
                                 debate_topic: str, 
                                 max_sources: int = 3,
                                 max_results_per_source: int = 2,
                                 force_refresh: bool = False) -> str:
        """
        为特定角色获取基于联网搜索的RAG上下文 (JSON Mode)
        
        Args:
            agent_role: 专家角色
            debate_topic: 辩论主题
            max_sources: 最大参考文献数（来自用户设置）
            max_results_per_source: 每个数据源的最大检索数
            force_refresh: 是否强制刷新（忽略缓存）
        """
        
        print(f"🔍 为专家{agent_role}JSON Mode联网搜索学术资料，最大文献数{max_sources}篇")
        
        # 参数安全检查
        if not agent_role or not debate_topic:
            print("⚠️ 专家角色或辩论主题为空")
            return "暂无相关学术资料。"
        
        if max_sources <= 0:
            print("⚠️ 最大文献数设置无效")
            return "暂无相关学术资料。"
        
        # 如果不强制刷新，先检查专家缓存
        if not force_refresh:
            try:
                cached_context = self.cache.get_agent_cached_context(agent_role, debate_topic)
                if cached_context:
                    cached_ref_count = cached_context.count('参考资料')
                    print(f"📚 使用专家 {agent_role} 的缓存学术资料：{cached_ref_count}篇")
                    
                    # 如果缓存的数量不符合用户当前设置，重新检索
                    if cached_ref_count != max_sources:
                        print(f"🔄 缓存文献数({cached_ref_count})与用户设置({max_sources})不符，重新搜索...")
                    else:
                        return cached_context
            except Exception as e:
                print(f"⚠️ 缓存检查失败: {e}")
        
        # 基于角色调整搜索查询
        try:
            role_focused_query = self._create_role_focused_query(agent_role, debate_topic)
        except Exception as e:
            print(f"⚠️ 查询生成失败，使用原始主题: {e}")
            role_focused_query = debate_topic
        
        # 使用用户设置的数量进行联网搜索 (JSON Mode)
        try:
            results = self.search_academic_sources(
                role_focused_query, 
                sources=["web_search"],  # 使用联网搜索作为数据源
                max_results_per_source=max_sources,  # 直接使用用户设置
                agent_role=agent_role
            )
        except Exception as e:
            print(f"❌ JSON Mode联网搜索失败: {e}")
            return "联网搜索遇到技术问题，请基于你的专业知识发表观点。"
        
        if not results:
            context = "暂无相关学术资料。"
        else:
            try:
                # 选择用户设置数量的文献
                top_results = results[:max_sources]
                
                print(f"📊 JSON Mode联网搜索结果处理：为专家 {agent_role} 实际搜索到 {len(results)} 篇，按用户设置选择前 {len(top_results)} 篇")
                
                # 构建上下文
                context_parts = []
                for i, result in enumerate(top_results, 1):
                    try:
                        context_part = f"""
参考资料 {i}:
标题: {result.title}
来源: {result.source} ({result.published_date})
关键发现: {result.key_findings or result.abstract[:200]}
相关性: {result.relevance_score}/10
链接: {result.url}
"""
                        context_parts.append(context_part.strip())
                    except Exception as e:
                        print(f"⚠️ 处理第{i}篇文献失败: {e}")
                        continue
                
                context = "\n\n".join(context_parts)
                
                # 验证最终结果
                final_ref_count = context.count('参考资料')
                print(f"✅ JSON Mode联网搜索上下文构建完成：{final_ref_count}篇参考文献")
                
            except Exception as e:
                print(f"❌ 上下文构建失败: {e}")
                context = "联网搜索资料处理遇到技术问题，请基于你的专业知识发表观点。"
        
        # 缓存结果
        if context and context != "暂无相关学术资料。":
            try:
                self.cache.cache_agent_context(agent_role, debate_topic, context)
            except Exception as e:
                print(f"⚠️ 上下文缓存失败: {e}")
        
        return context
    
    def _create_role_focused_query(self, agent_role: str, debate_topic: str) -> str:
        """基于角色创建针对性查询"""
        try:
            role_keywords = {
                "environmentalist": "环境保护 气候变化 可持续发展 生态影响",
                "economist": "经济影响 成本效益 市场分析 经济政策",
                "policy_maker": "政策制定 监管措施 治理框架 实施策略",
                "tech_expert": "技术创新 技术可行性 技术发展 技术影响",
                "sociologist": "社会影响 社会变化 社群效应 社会公平",
                "ethicist": "伦理道德 道德责任 价值观念 伦理框架"
            }
            
            keywords = role_keywords.get(agent_role, "")
            focused_query = f"{debate_topic} {keywords}".strip()
            print(f"🎯 为{agent_role}定制JSON Mode联网搜索查询：{focused_query}")
            return focused_query
        except Exception as e:
            print(f"⚠️ 角色查询生成失败: {e}")
            return debate_topic
    
    def preload_agent_contexts(self, agent_roles: List[str], debate_topic: str, max_refs_per_agent: int = 3):
        """
        预加载所有专家的联网搜索上下文 (JSON Mode)
        
        Args:
            agent_roles: 专家角色列表
            debate_topic: 辩论主题
            max_refs_per_agent: 每个专家的最大参考文献数（用户设置）
        """
        
        if not agent_roles or not debate_topic:
            print("⚠️ 专家角色列表或辩论主题为空")
            return
        
        print(f"🚀 开始为 {len(agent_roles)} 位专家预加载JSON Mode联网搜索资料...")
        print(f"📊 用户配置：每专家最多 {max_refs_per_agent} 篇参考文献")
        
        for agent_role in agent_roles:
            try:
                print(f"🔍 为专家 {agent_role} 使用JSON Mode联网搜索...")
                context = self.get_rag_context_for_agent(
                    agent_role=agent_role,
                    debate_topic=debate_topic,
                    max_sources=max_refs_per_agent,  # 使用用户设置
                    max_results_per_source=2,
                    force_refresh=True  # 强制刷新确保最新资料
                )
                
                if context and context != "暂无相关学术资料。":
                    actual_count = context.count('参考资料')
                    print(f"✅ 专家 {agent_role} 的JSON Mode联网搜索资料已准备就绪：{actual_count}篇")
                else:
                    print(f"⚠️ 专家 {agent_role} 未找到相关资料")
                
                # 避免API限制
                time.sleep(3)
                
            except Exception as e:
                print(f"❌ 为专家 {agent_role} 预加载资料失败: {e}")
                continue
        
        print("✅ 所有专家的JSON Mode联网搜索资料预加载完成")
    
    def clear_all_caches(self):
        """清理所有缓存"""
        try:
            self.cache.clear_agent_cache()
            # 清理通用缓存
            for filename in os.listdir(self.cache.cache_dir):
                if filename.endswith('.json') and not filename.startswith('agent_'):
                    try:
                        os.remove(os.path.join(self.cache.cache_dir, filename))
                    except Exception as e:
                        print(f"⚠️ 删除缓存文件失败: {filename}, {e}")
            print("✅ 已清理所有缓存")
        except Exception as e:
            print(f"❌ 清理缓存失败: {e}")
    
    def test_integration(self, 
                        agent_role: str = "tech_expert", 
                        debate_topic: str = "人工智能对教育的影响",
                        test_configs: List[int] = [1, 3, 5]):
        """
        测试JSON Mode联网搜索集成
        
        Args:
            agent_role: 测试专家角色
            debate_topic: 测试辩论主题  
            test_configs: 测试的参考文献数量列表
        """
        print("🧪 开始测试Kimi JSON Mode联网搜索集成...")
        
        for max_refs in test_configs:
            print(f"\n📋 测试配置：每专家{max_refs}篇参考文献")
            
            # 清理缓存确保重新检索
            try:
                self.cache.clear_agent_cache(agent_role)
            except Exception as e:
                print(f"⚠️ 缓存清理失败: {e}")
            
            try:
                context = self.get_rag_context_for_agent(
                    agent_role=agent_role,
                    debate_topic=debate_topic,
                    max_sources=max_refs,  # 测试用户设置
                    force_refresh=True
                )
                
                if context and context != "暂无相关学术资料。":
                    actual_count = context.count('参考资料')
                    status = "✅" if actual_count == max_refs else "❌"
                    print(f"{status} 结果：实际{actual_count}篇，期望{max_refs}篇")
                    
                    if actual_count != max_refs:
                        print(f"⚠️ 配置不生效！请检查代码")
                    
                    # 显示简短内容预览
                    print(f"📄 内容预览：{context[:200]}...")
                else:
                    print(f"⚠️ 未找到学术资料")
                    
            except Exception as e:
                print(f"❌ 测试失败: {e}")
        
        print("\n🎉 Kimi JSON Mode联网搜索测试完成！")

# 全局RAG实例（将在graph.py中初始化）
rag_module = None

def initialize_rag_module(llm: ChatDeepSeek) -> DynamicRAGModule:
    """初始化RAG模块（基于Kimi API联网搜索，集成JSON Mode）"""
    global rag_module
    try:
        rag_module = DynamicRAGModule(llm)
        print("🔍 RAG模块已初始化，使用Kimi API联网搜索功能 (JSON Mode)")
        return rag_module
    except Exception as e:
        print(f"❌ JSON Mode联网搜索RAG模块初始化失败: {e}")
        return None

def get_rag_module() -> Optional[DynamicRAGModule]:
    """获取RAG模块实例"""
    return rag_module

# 测试函数
def test_rag_module():
    """测试基于Kimi JSON Mode联网搜索的RAG模块功能"""
    print("🧪 开始测试基于Kimi JSON Mode联网搜索的RAG模块...")
    
    # 检查环境变量
    if not os.getenv("KIMI_API_KEY"):
        print("❌ 警告: KIMI_API_KEY 环境变量未设置")
        print("请设置环境变量：export KIMI_API_KEY=your_api_key")
        return
    
    # 创建测试LLM（需要有效的API密钥）
    try:
        from langchain_deepseek import ChatDeepSeek
        test_llm = ChatDeepSeek(model="deepseek-chat", temperature=0.3)
        
        # 初始化RAG模块
        rag = initialize_rag_module(test_llm)
        
        if not rag:
            print("❌ JSON Mode联网搜索RAG模块初始化失败")
            return
        
        # 测试专家角色联网搜索
        test_topic = "人工智能对就业的影响"
        test_roles = ["tech_expert", "economist", "sociologist"]
        
        print("🔍 测试基于Kimi JSON Mode联网搜索的专家角色文献检索...")
        for role in test_roles:
            # 测试不同的用户配置
            for max_refs in [1, 3]:
                print(f"\n📊 测试：{role} 获取 {max_refs} 篇文献")
                try:
                    context = rag.get_rag_context_for_agent(
                        agent_role=role, 
                        debate_topic=test_topic,
                        max_sources=max_refs,  # 测试用户设置
                        force_refresh=True
                    )
                    
                    if context and context != "暂无相关学术资料。":
                        actual_count = context.count('参考资料')
                        status = "✅" if actual_count == max_refs else "❌"
                        print(f"{status} 结果：期望{max_refs}篇，实际{actual_count}篇")
                        print(f"前100字符：{context[:100]}...")
                    else:
                        print("⚠️ 未找到学术资料")
                except Exception as e:
                    print(f"❌ 测试出错: {e}")
        
        # 专门的JSON Mode联网搜索测试
        print("\n🔧 专门测试JSON Mode联网搜索功能...")
        try:
            rag.test_integration()
        except Exception as e:
            print(f"❌ JSON Mode联网搜索测试失败: {e}")
            
    except Exception as e:
        print(f"❌ JSON Mode联网搜索RAG模块测试失败: {e}")

if __name__ == "__main__":
    test_rag_module()