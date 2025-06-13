"""
动态RAG模块 - 实时检索权威数据库
支持arXiv、CrossRef等学术数据源的真实检索
修复：支持用户自定义的每专家最大参考文献数设置
优化：支持基于专家角色的缓存机制
"""

import os
import asyncio
import aiohttp
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import time

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
    # 新增：专家缓存过期时间（小时）
    "agent_cache_duration_hours": 6
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
    """RAG结果缓存管理（增强版，支持专家角色缓存）"""
    
    def __init__(self, cache_dir: str = "./rag_cache"):
        self.cache_dir = cache_dir
        self.agent_cache_dir = os.path.join(cache_dir, "agent_cache")
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(self.agent_cache_dir, exist_ok=True)
    
    def _get_cache_key(self, query: str, sources: List[str]) -> str:
        """生成缓存键"""
        key_string = f"{query}_{'-'.join(sorted(sources))}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_agent_cache_key(self, agent_role: str, debate_topic: str) -> str:
        """生成专家角色特定的缓存键"""
        key_string = f"agent_{agent_role}_{debate_topic}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_cached_results(self, query: str, sources: List[str]) -> Optional[List[SearchResult]]:
        """获取缓存的检索结果"""
        cache_key = self._get_cache_key(query, sources)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if not os.path.exists(cache_file):
            return None
        
        try:
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
                results.append(SearchResult(**item))
            
            return results
            
        except Exception as e:
            print(f"❌ 缓存读取错误: {e}")
            return None
    
    def cache_results(self, query: str, sources: List[str], results: List[SearchResult]):
        """缓存检索结果"""
        cache_key = self._get_cache_key(query, sources)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
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
        cache_key = self._get_agent_cache_key(agent_role, debate_topic)
        cache_file = os.path.join(self.agent_cache_dir, f"{cache_key}.json")
        
        if not os.path.exists(cache_file):
            return None
        
        try:
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
        cache_key = self._get_agent_cache_key(agent_role, debate_topic)
        cache_file = os.path.join(self.agent_cache_dir, f"{cache_key}.json")
        
        try:
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
                        os.remove(os.path.join(self.agent_cache_dir, filename))
                print(f"✅ 已清理专家 {agent_role} 的缓存")
            else:
                # 清理所有专家缓存
                for filename in os.listdir(self.agent_cache_dir):
                    os.remove(os.path.join(self.agent_cache_dir, filename))
                print("✅ 已清理所有专家缓存")
        except Exception as e:
            print(f"❌ 清理缓存失败: {e}")

class ArxivSearcher:
    """arXiv学术论文检索器"""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.session = requests.Session()
    
    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """检索arXiv论文"""
        try:
            # 构建查询参数
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            print(f"🔍 正在arXiv检索: {query} (最多{max_results}篇)")
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            return self._parse_arxiv_response(response.text)
            
        except Exception as e:
            print(f"❌ arXiv检索失败: {e}")
            return []
    
    def _parse_arxiv_response(self, xml_content: str) -> List[SearchResult]:
        """解析arXiv API响应"""
        results = []
        
        try:
            root = ET.fromstring(xml_content)
            namespace = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('atom:entry', namespace):
                title = entry.find('atom:title', namespace)
                title = title.text.strip() if title is not None else "无标题"
                
                # 作者信息
                authors = []
                for author in entry.findall('atom:author', namespace):
                    name = author.find('atom:name', namespace)
                    if name is not None:
                        authors.append(name.text.strip())
                
                # 摘要
                summary = entry.find('atom:summary', namespace)
                abstract = summary.text.strip() if summary is not None else "无摘要"
                
                # URL
                link = entry.find('atom:id', namespace)
                url = link.text.strip() if link is not None else ""
                
                # 发布时间
                published = entry.find('atom:published', namespace)
                published_date = published.text[:10] if published is not None else ""
                
                result = SearchResult(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    url=url,
                    published_date=published_date,
                    source="arXiv"
                )
                
                results.append(result)
                
        except ET.ParseError as e:
            print(f"❌ arXiv响应解析错误: {e}")
        
        return results

class CrossRefSearcher:
    """CrossRef期刊文章检索器"""
    
    def __init__(self):
        self.base_url = "https://api.crossref.org/works"
        self.session = requests.Session()
        # 设置用户代理避免被限制
        self.session.headers.update({
            'User-Agent': 'Multi-Agent-Debate-Platform/1.0 (mailto:admin@example.com)'
        })
    
    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """检索期刊文章"""
        try:
            params = {
                'query': query,
                'rows': max_results,
                'sort': 'relevance',
                'filter': 'type:journal-article',
                'select': 'title,author,abstract,URL,published-print,container-title'
            }
            
            print(f"🔍 正在CrossRef检索: {query} (最多{max_results}篇)")
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            return self._parse_crossref_response(response.json())
            
        except Exception as e:
            print(f"❌ CrossRef检索失败: {e}")
            return []
    
    def _parse_crossref_response(self, data: dict) -> List[SearchResult]:
        """解析CrossRef API响应"""
        results = []
        
        try:
            items = data.get('message', {}).get('items', [])
            
            for item in items:
                # 标题
                title_list = item.get('title', [])
                title = title_list[0] if title_list else "无标题"
                
                # 作者
                authors = []
                author_list = item.get('author', [])
                for author in author_list:
                    given = author.get('given', '')
                    family = author.get('family', '')
                    if given and family:
                        authors.append(f"{given} {family}")
                    elif family:
                        authors.append(family)
                
                # 摘要（CrossRef通常不提供完整摘要）
                abstract = item.get('abstract', '摘要信息需要访问原文')
                
                # URL
                url = item.get('URL', '')
                
                # 发布时间
                published_date = ""
                date_parts = item.get('published-print', {}).get('date-parts', [])
                if date_parts and len(date_parts[0]) >= 3:
                    year, month, day = date_parts[0][:3]
                    published_date = f"{year}-{month:02d}-{day:02d}"
                elif date_parts and len(date_parts[0]) >= 1:
                    published_date = str(date_parts[0][0])
                
                result = SearchResult(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    url=url,
                    published_date=published_date,
                    source="CrossRef"
                )
                
                results.append(result)
                
        except Exception as e:
            print(f"❌ CrossRef响应解析错误: {e}")
        
        return results

class RAGEnhancer:
    """RAG增强器 - 处理检索结果并生成洞察"""
    
    def __init__(self, llm: ChatDeepSeek):
        self.llm = llm
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个学术研究分析专家。基于给定的学术论文信息，提取和总结关键发现，为特定角色的辩论提供支撑。

你的任务：
1. 分析论文的核心观点和发现
2. 提取与辩论主题和指定角色相关的关键证据
3. 简洁地总结主要论点（2-3句话）
4. 评估研究的可信度和相关性

专家角色：{agent_role}
论文信息：
标题：{title}
作者：{authors}
摘要：{abstract}
发布时间：{published_date}
来源：{source}

辩论主题：{debate_topic}

请特别关注与{agent_role}专业领域相关的内容，提供：
1. 关键发现（核心观点和证据）
2. 与辩论主题的相关性评分（1-10分）
3. 建议该角色在辩论中如何引用这项研究"""),
            ("user", "请分析这篇论文并提供关键洞察")
        ])
    
    def enhance_results(self, results: List[SearchResult], debate_topic: str, agent_role: str = "") -> List[SearchResult]:
        """增强检索结果，提取关键洞察（针对特定角色优化）"""
        enhanced_results = []
        
        for result in results:
            try:
                # 使用LLM分析论文（考虑专家角色）
                analysis = self._analyze_paper(result, debate_topic, agent_role)
                result.key_findings = analysis.get('key_findings', '')
                result.relevance_score = analysis.get('relevance_score', 5.0)
                enhanced_results.append(result)
                
                # 避免API限制
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ 论文分析失败 {result.title}: {e}")
                # 即使分析失败也保留原始结果
                enhanced_results.append(result)
        
        # 按相关性评分排序
        enhanced_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return enhanced_results
    
    def _analyze_paper(self, result: SearchResult, debate_topic: str, agent_role: str = "") -> dict:
        """分析单篇论文（针对特定角色）"""
        try:
            pipe = self.analysis_prompt | self.llm | StrOutputParser()
            
            response = pipe.invoke({
                'agent_role': agent_role,
                'title': result.title,
                'authors': ', '.join(result.authors[:3]),
                'abstract': result.abstract[:1000],
                'published_date': result.published_date,
                'source': result.source,
                'debate_topic': debate_topic
            })
            
            # 简单解析响应
            lines = response.strip().split('\n')
            key_findings = ""
            relevance_score = 5.0
            
            for line in lines:
                if '关键发现' in line or '核心观点' in line:
                    key_findings = line.split('：', 1)[-1].strip()
                elif '相关性' in line and '分' in line:
                    try:
                        import re
                        score_match = re.search(r'(\d+)', line)
                        if score_match:
                            relevance_score = float(score_match.group(1))
                    except:
                        pass
            
            return {
                'key_findings': key_findings or response[:200],
                'relevance_score': min(max(relevance_score, 1.0), 10.0)
            }
            
        except Exception as e:
            print(f"❌ LLM分析错误: {e}")
            return {
                'key_findings': result.abstract[:150] + "...",
                'relevance_score': 5.0
            }

class DynamicRAGModule:
    """动态RAG主模块（修复版，支持用户自定义配置）"""
    
    def __init__(self, llm: ChatDeepSeek):
        self.llm = llm
        self.cache = RAGCache()
        self.arxiv_searcher = ArxivSearcher()
        self.crossref_searcher = CrossRefSearcher()
        self.enhancer = RAGEnhancer(llm)
        
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
                              sources: List[str] = ["arxiv", "crossref"],
                              max_results_per_source: int = None,
                              agent_role: str = "") -> List[SearchResult]:
        """
        搜索学术数据源（修复版，支持用户自定义结果数量）
        
        Args:
            topic: 搜索主题
            sources: 数据源列表
            max_results_per_source: 每个数据源的最大结果数（用户可配置）
            agent_role: 专家角色（用于定制化分析）
        """
        
        if max_results_per_source is None:
            max_results_per_source = RAG_CONFIG["max_results_per_source"]
        
        # 🔧 验证用户配置
        print(f"🔧 RAG检索配置：每源最多{max_results_per_source}篇，角色定制：{agent_role}")
        
        # 检查缓存
        cached_results = self.cache.get_cached_results(topic, sources)
        if cached_results:
            print(f"✅ 使用缓存结果: {len(cached_results)} 篇论文")
            # 如果有角色信息，重新排序以适合该角色
            if agent_role and self.llm:
                cached_results = self.enhancer.enhance_results(cached_results, topic, agent_role)
            return cached_results
        
        all_results = []
        
        # arXiv检索
        if "arxiv" in sources:
            arxiv_results = self.arxiv_searcher.search(topic, max_results_per_source)
            all_results.extend(arxiv_results)
            print(f"📚 arXiv找到 {len(arxiv_results)} 篇论文（设置上限：{max_results_per_source}篇）")
        
        # CrossRef检索
        if "crossref" in sources:
            crossref_results = self.crossref_searcher.search(topic, max_results_per_source)
            all_results.extend(crossref_results)
            print(f"📚 CrossRef找到 {len(crossref_results)} 篇论文（设置上限：{max_results_per_source}篇）")
        
        # 使用LLM增强结果（考虑专家角色）
        if all_results and self.llm:
            print(f"🤖 使用AI分析论文相关性{'（为' + agent_role + '定制）' if agent_role else ''}...")
            all_results = self.enhancer.enhance_results(all_results, topic, agent_role)
        
        # 缓存结果
        if all_results:
            self.cache.cache_results(topic, sources, all_results)
        
        return all_results
    
    def get_rag_context_for_agent(self, 
                                 agent_role: str, 
                                 debate_topic: str, 
                                 max_sources: int = 3,
                                 max_results_per_source: int = 2,
                                 force_refresh: bool = False) -> str:
        """
        为特定角色获取RAG上下文（修复版，完全支持用户自定义配置）
        
        Args:
            agent_role: 专家角色
            debate_topic: 辩论主题
            max_sources: 最大参考文献数（来自用户设置）- 🔧 关键修复点
            max_results_per_source: 每个数据源的最大检索数
            force_refresh: 是否强制刷新（忽略缓存）
        """
        
        # 🔧 关键验证：确认接收到用户设置
        print(f"🔧 RAG上下文配置确认：专家{agent_role}，最大文献数{max_sources}篇（用户设置），每源{max_results_per_source}篇")
        
        # 如果不强制刷新，先检查专家缓存
        if not force_refresh:
            cached_context = self.cache.get_agent_cached_context(agent_role, debate_topic)
            if cached_context:
                # 🔧 验证缓存中的文献数量是否符合用户设置
                cached_ref_count = cached_context.count('参考资料')
                print(f"📚 使用专家 {agent_role} 的缓存学术资料：{cached_ref_count}篇")
                
                # 如果缓存的数量不符合用户当前设置，重新检索
                if cached_ref_count != max_sources:
                    print(f"🔄 缓存文献数({cached_ref_count})与用户设置({max_sources})不符，重新检索...")
                else:
                    return cached_context
        
        # 基于角色调整搜索查询
        role_focused_query = self._create_role_focused_query(agent_role, debate_topic)
        
        # 🔧 关键修复：使用用户设置的数量进行检索
        results = self.search_academic_sources(
            role_focused_query, 
            max_results_per_source=max_results_per_source,  # 每源检索数
            agent_role=agent_role
        )
        
        if not results:
            context = "暂无相关学术资料。"
        else:
            # 🔧 关键修复：选择用户设置数量的文献，而不是硬编码
            top_results = results[:max_sources]  # 使用用户设置！
            
            print(f"📊 检索结果处理：为专家 {agent_role} 实际检索到 {len(results)} 篇，按用户设置选择前 {len(top_results)} 篇")
            
            # 构建上下文
            context_parts = []
            for i, result in enumerate(top_results, 1):
                context_part = f"""
参考资料 {i}:
标题: {result.title}
作者: {', '.join(result.authors[:2])}
来源: {result.source} ({result.published_date})
关键发现: {result.key_findings or result.abstract[:200]}
相关性: {result.relevance_score}/10
"""
                context_parts.append(context_part.strip())
            
            context = "\n\n".join(context_parts)
            
            # 🔧 验证最终结果
            final_ref_count = context.count('参考资料')
            print(f"✅ 上下文构建完成：{final_ref_count}篇参考文献（用户设置：{max_sources}篇）")
        
        # 缓存结果
        if context and context != "暂无相关学术资料。":
            self.cache.cache_agent_context(agent_role, debate_topic, context)
        
        return context
    
    def _create_role_focused_query(self, agent_role: str, debate_topic: str) -> str:
        """基于角色创建针对性查询"""
        role_keywords = {
            "environmentalist": "environment climate sustainability ecology",
            "economist": "economic cost benefit market analysis",
            "policy_maker": "policy governance regulation implementation",
            "tech_expert": "technology innovation technical feasibility",
            "sociologist": "social impact society community effects",
            "ethicist": "ethics moral responsibility values"
        }
        
        keywords = role_keywords.get(agent_role, "")
        focused_query = f"{debate_topic} {keywords}".strip()
        print(f"🎯 为{agent_role}定制查询：{focused_query}")
        return focused_query
    
    def preload_agent_contexts(self, agent_roles: List[str], debate_topic: str, max_refs_per_agent: int = 3):
        """
        预加载所有专家的上下文（修复版，支持用户配置）
        
        Args:
            agent_roles: 专家角色列表
            debate_topic: 辩论主题
            max_refs_per_agent: 每个专家的最大参考文献数（用户设置）
        """
        print(f"🚀 开始为 {len(agent_roles)} 位专家预加载学术资料...")
        print(f"📊 用户配置：每专家最多 {max_refs_per_agent} 篇参考文献")
        
        for agent_role in agent_roles:
            try:
                print(f"🔍 为专家 {agent_role} 检索学术资料...")
                context = self.get_rag_context_for_agent(
                    agent_role=agent_role,
                    debate_topic=debate_topic,
                    max_sources=max_refs_per_agent,  # 🔧 使用用户设置
                    max_results_per_source=2,
                    force_refresh=True  # 强制刷新确保最新资料
                )
                
                if context and context != "暂无相关学术资料。":
                    actual_count = context.count('参考资料')
                    print(f"✅ 专家 {agent_role} 的学术资料已准备就绪：{actual_count}篇（设置：{max_refs_per_agent}篇）")
                else:
                    print(f"⚠️ 专家 {agent_role} 未找到相关学术资料")
                
                # 避免API限制
                time.sleep(2)
                
            except Exception as e:
                print(f"❌ 为专家 {agent_role} 预加载资料失败: {e}")
        
        print("✅ 所有专家的学术资料预加载完成")
    
    def clear_all_caches(self):
        """清理所有缓存"""
        try:
            self.cache.clear_agent_cache()
            # 清理通用缓存
            for filename in os.listdir(self.cache.cache_dir):
                if filename.endswith('.json') and not filename.startswith('agent_'):
                    os.remove(os.path.join(self.cache.cache_dir, filename))
            print("✅ 已清理所有缓存")
        except Exception as e:
            print(f"❌ 清理缓存失败: {e}")
    
    def test_user_config_support(self, agent_role: str = "tech_expert", 
                                debate_topic: str = "artificial intelligence education",
                                test_configs: List[int] = [1, 3, 5]):
        """
        测试用户配置支持（验证修复效果）
        
        Args:
            agent_role: 测试专家角色
            debate_topic: 测试辩论主题  
            test_configs: 测试的参考文献数量列表
        """
        print("🧪 开始测试用户RAG配置支持...")
        
        for max_refs in test_configs:
            print(f"\n📋 测试配置：每专家{max_refs}篇参考文献")
            
            # 清理缓存确保重新检索
            self.cache.clear_agent_cache(agent_role)
            
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
                        print(f"⚠️ 配置不生效！请检查代码修复")
                else:
                    print(f"⚠️ 未找到学术资料")
                    
            except Exception as e:
                print(f"❌ 测试失败: {e}")
        
        print("\n🎉 用户RAG配置测试完成！")

# 全局RAG实例（将在graph.py中初始化）
rag_module = None

def initialize_rag_module(llm: ChatDeepSeek) -> DynamicRAGModule:
    """初始化RAG模块"""
    global rag_module
    rag_module = DynamicRAGModule(llm)
    return rag_module

def get_rag_module() -> Optional[DynamicRAGModule]:
    """获取RAG模块实例"""
    return rag_module

# 测试函数
def test_rag_module():
    """测试RAG模块功能（包括用户配置支持）"""
    print("🧪 开始测试修复版RAG模块...")
    
    # 创建测试LLM（需要有效的API密钥）
    try:
        from langchain_deepseek import ChatDeepSeek
        test_llm = ChatDeepSeek(model="deepseek-chat", temperature=0.3)
        
        # 初始化RAG模块
        rag = initialize_rag_module(test_llm)
        
        # 测试专家角色检索
        test_topic = "artificial intelligence employment impact"
        test_roles = ["tech_expert", "economist", "sociologist"]
        
        print("🔍 测试专家角色检索...")
        for role in test_roles:
            # 🔧 测试不同的用户配置
            for max_refs in [1, 3, 5]:
                print(f"\n📊 测试：{role} 获取 {max_refs} 篇文献")
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
                else:
                    print("⚠️ 未找到学术资料")
        
        # 专门的用户配置测试
        print("\n🔧 专门测试用户配置支持...")
        rag.test_user_config_support()
            
    except Exception as e:
        print(f"❌ RAG模块测试失败: {e}")

if __name__ == "__main__":
    test_rag_module()