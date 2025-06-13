"""
动态RAG模块 - 实时检索权威数据库
支持arXiv、CrossRef等学术数据源的真实检索
集成中文查询智能翻译功能
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
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
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
    """RAG结果缓存管理"""
    
    def __init__(self, cache_dir: str = "./rag_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, query: str, sources: List[str]) -> str:
        """生成缓存键"""
        key_string = f"{query}_{'-'.join(sorted(sources))}"
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

class QueryTranslator:
    """查询翻译器 - 将中文辩论主题转换为适合学术检索的英文查询"""
    
    def __init__(self, llm: ChatDeepSeek):
        self.llm = llm
        self.translation_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的学术检索查询翻译专家。你的任务是将中文辩论主题转换为适合在arXiv、CrossRef等国际学术数据库中搜索的英文关键词。

翻译要求：
1. 提取核心学术概念，转换为标准英文学术术语
2. 去除辩论性语言，保留客观研究主题
3. 使用简洁的关键词组合，不要完整句子
4. 优先使用在学术界通用的英文术语
5. 考虑同义词和相关概念

示例：
中文："人工智能是否会威胁人类就业？"
英文："artificial intelligence employment impact automation job displacement"

中文："核能发电是解决气候变化的最佳方案吗？"
英文："nuclear power climate change mitigation renewable energy policy"

请为以下中文主题提供最佳的英文学术检索查询："""),
            ("user", "中文主题：{chinese_topic}\n\n请提供3-6个核心英文关键词，用空格分隔：")
        ])
    
    def translate_to_academic_query(self, chinese_topic: str) -> str:
        """将中文主题翻译为英文学术查询"""
        try:
            # 首先检查是否已经是英文
            if self._is_english(chinese_topic):
                return self._extract_keywords(chinese_topic)
            
            pipe = self.translation_prompt | self.llm | StrOutputParser()
            
            english_query = pipe.invoke({"chinese_topic": chinese_topic})
            
            # 清理响应，提取关键词
            keywords = self._clean_query_response(english_query)
            
            print(f"🔤 查询翻译: '{chinese_topic}' -> '{keywords}'")
            return keywords
            
        except Exception as e:
            print(f"❌ 查询翻译失败: {e}")
            # 回退到简单翻译
            return self._fallback_translation(chinese_topic)
    
    def _is_english(self, text: str) -> bool:
        """检查文本是否主要为英文"""
        english_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        total_chars = sum(1 for c in text if c.isalpha())
        return total_chars > 0 and english_chars / total_chars > 0.7
    
    def _extract_keywords(self, english_text: str) -> str:
        """从英文文本中提取关键词"""
        # 移除常见停用词和疑问词
        stop_words = {'is', 'are', 'will', 'can', 'should', 'does', 'do', 'the', 'a', 'an', 
                     'how', 'what', 'why', 'where', 'when', 'which', 'whether', 'if'}
        
        words = english_text.lower().replace('?', '').replace('!', '').split()
        keywords = [word.strip('.,!?()[]{}') for word in words 
                   if word.lower() not in stop_words and len(word) > 2]
        
        return ' '.join(keywords[:6])  # 限制关键词数量
    
    def _clean_query_response(self, response: str) -> str:
        """清理LLM响应，提取关键词"""
        # 移除可能的解释文字，只保留关键词
        lines = response.strip().split('\n')
        for line in lines:
            if ':' in line:
                # 如果有冒号，取冒号后的部分
                keywords = line.split(':', 1)[1].strip()
            else:
                keywords = line.strip()
            
            # 验证是否是关键词格式
            if len(keywords.split()) <= 8 and not keywords.startswith('中文') and not keywords.startswith('英文'):
                return keywords
        
        # 如果没找到合适的行，返回整个响应的前6个词
        words = response.replace('\n', ' ').split()
        return ' '.join(words[:6])
    
    def _fallback_translation(self, chinese_topic: str) -> str:
        """回退翻译方案"""
        # 简单的词汇映射
        translation_map = {
            '人工智能': 'artificial intelligence',
            '机器学习': 'machine learning',
            '深度学习': 'deep learning',
            '就业': 'employment',
            '工作': 'job work',
            '核能': 'nuclear power',
            '气候变化': 'climate change',
            '环境保护': 'environmental protection',
            '经济': 'economics economy',
            '政策': 'policy',
            '技术': 'technology',
            '社会': 'society social',
            '伦理': 'ethics',
            '道德': 'moral',
            '可持续发展': 'sustainable development',
            '能源': 'energy',
            '教育': 'education',
            '医疗': 'healthcare medical',
            '自动化': 'automation',
            '机器人': 'robotics',
            '生物技术': 'biotechnology',
            '基因编辑': 'gene editing CRISPR',
            '区块链': 'blockchain',
            '量子计算': 'quantum computing',
            '自动驾驶': 'autonomous driving',
            '虚拟现实': 'virtual reality',
            '增强现实': 'augmented reality'
        }
        
        keywords = []
        for chinese, english in translation_map.items():
            if chinese in chinese_topic:
                keywords.extend(english.split())
        
        if not keywords:
            # 如果没有匹配，使用通用研究词汇
            keywords = ['research', 'analysis', 'study', 'technology', 'society']
        
        return ' '.join(keywords[:6])

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
            
            print(f"🔍 正在arXiv检索: {query}")
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
            
            print(f"🔍 正在CrossRef检索: {query}")
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
            ("system", """你是一个学术研究分析专家。基于给定的学术论文信息，提取和总结关键发现，为辩论提供支撑。

你的任务：
1. 分析论文的核心观点和发现
2. 提取与辩论主题相关的关键证据
3. 简洁地总结主要论点（2-3句话）
4. 评估研究的可信度和相关性

论文信息：
标题：{title}
作者：{authors}
摘要：{abstract}
发布时间：{published_date}
来源：{source}

辩论主题：{debate_topic}

请提供：
1. 关键发现（核心观点和证据）
2. 与辩论主题的相关性评分（1-10分）
3. 建议在辩论中如何引用这项研究"""),
            ("user", "请分析这篇论文并提供关键洞察")
        ])
    
    def enhance_results(self, results: List[SearchResult], debate_topic: str) -> List[SearchResult]:
        """增强检索结果，提取关键洞察"""
        enhanced_results = []
        
        for result in results:
            try:
                # 使用LLM分析论文
                analysis = self._analyze_paper(result, debate_topic)
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
    
    def _analyze_paper(self, result: SearchResult, debate_topic: str) -> dict:
        """分析单篇论文"""
        try:
            pipe = self.analysis_prompt | self.llm | StrOutputParser()
            
            response = pipe.invoke({
                'title': result.title,
                'authors': ', '.join(result.authors[:3]),  # 限制作者数量
                'abstract': result.abstract[:1000],  # 限制摘要长度
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
                        # 尝试提取数字
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
    """动态RAG主模块"""
    
    def __init__(self, llm: ChatDeepSeek):
        self.llm = llm
        self.cache = RAGCache()
        self.arxiv_searcher = ArxivSearcher()
        self.crossref_searcher = CrossRefSearcher()
        self.enhancer = RAGEnhancer(llm)
        self.query_translator = QueryTranslator(llm)  # 新增查询翻译器
        
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
                              max_results_per_source: int = None) -> List[SearchResult]:
        """搜索学术数据源（支持中文查询自动翻译）"""
        
        if max_results_per_source is None:
            max_results_per_source = RAG_CONFIG["max_results_per_source"]
        
        # 将中文主题翻译为英文学术查询
        english_query = self.query_translator.translate_to_academic_query(topic)
        
        # 检查缓存（使用英文查询作为缓存键）
        cached_results = self.cache.get_cached_results(english_query, sources)
        if cached_results:
            print(f"✅ 使用缓存结果: {len(cached_results)} 篇论文")
            return cached_results
        
        all_results = []
        
        # arXiv检索
        if "arxiv" in sources:
            print(f"🔍 arXiv检索中文主题: {topic}")
            print(f"🔍 arXiv使用英文查询: {english_query}")
            arxiv_results = self.arxiv_searcher.search(english_query, max_results_per_source)
            all_results.extend(arxiv_results)
            print(f"📚 arXiv找到 {len(arxiv_results)} 篇论文")
        
        # CrossRef检索
        if "crossref" in sources:
            print(f"🔍 CrossRef检索中文主题: {topic}")
            print(f"🔍 CrossRef使用英文查询: {english_query}")
            crossref_results = self.crossref_searcher.search(english_query, max_results_per_source)
            all_results.extend(crossref_results)
            print(f"📚 CrossRef找到 {len(crossref_results)} 篇论文")
        
        # 使用LLM增强结果（使用原始中文主题进行相关性评估）
        if all_results and self.llm:
            print("🤖 使用AI分析论文相关性...")
            all_results = self.enhancer.enhance_results(all_results, topic)  # 使用中文主题评估相关性
        
        # 缓存结果（使用英文查询作为缓存键）
        if all_results:
            self.cache.cache_results(english_query, sources, all_results)
        
        return all_results
    
    def get_rag_context_for_agent(self, 
                                 agent_role: str, 
                                 debate_topic: str, 
                                 max_sources: int = 3) -> str:
        """为特定角色获取RAG上下文（优化版，支持中文主题）"""
        
        # 基于角色调整搜索查询（现在支持中文翻译）
        role_focused_query = self._create_role_focused_query(agent_role, debate_topic)
        
        # 检索相关文献
        results = self.search_academic_sources(role_focused_query, max_results_per_source=2)
        
        if not results:
            print(f"⚠️ 未找到{agent_role}相关的学术资料")
            return "暂无相关学术资料。"
        
        # 选择最相关的几篇
        top_results = results[:max_sources]
        
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
        print(f"📚 为{agent_role}准备了{len(top_results)}篇参考文献")
        return context
    
    def _create_role_focused_query(self, agent_role: str, debate_topic: str) -> str:
        """基于角色创建针对性查询（优化版，支持中英文混合）"""
        
        # 基础英文关键词映射（根据你原有的角色定义）
        role_keywords = {
            "environmentalist": "environment climate sustainability ecology conservation renewable",
            "economist": "economic cost benefit market analysis finance policy",
            "policy_maker": "policy governance regulation implementation law public administration",
            "tech_expert": "technology innovation technical feasibility artificial intelligence",
            "sociologist": "social impact society community effects inequality demographics",
            "ethicist": "ethics moral responsibility values philosophy bioethics"
        }
        
        # 获取角色专业关键词
        role_english_keywords = role_keywords.get(agent_role, "research analysis")
        
        # 将中文主题翻译为英文
        english_topic = self.query_translator.translate_to_academic_query(debate_topic)
        
        # 组合查询：主题关键词 + 角色关键词
        combined_query = f"{english_topic} {role_english_keywords}"
        
        # 去重并限制关键词数量
        unique_keywords = []
        seen = set()
        for keyword in combined_query.split():
            if keyword.lower() not in seen and len(keyword) > 2:
                unique_keywords.append(keyword)
                seen.add(keyword.lower())
        
        final_query = ' '.join(unique_keywords[:8])  # 限制在8个关键词以内
        
        print(f"🎯 {agent_role}专用查询: {final_query}")
        return final_query

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
    """测试RAG模块功能（包括中文查询翻译）"""
    print("🧪 开始测试RAG模块...")
    
    # 创建测试LLM（需要有效的API密钥）
    try:
        from langchain_deepseek import ChatDeepSeek
        test_llm = ChatDeepSeek(model="deepseek-chat", temperature=0.3)
        
        # 初始化RAG模块
        rag = initialize_rag_module(test_llm)
        
        # 测试中文查询翻译
        print("\n🔤 测试查询翻译功能:")
        chinese_topics = [
            "人工智能对就业的影响",
            "核能发电与气候变化",
            "基因编辑技术的伦理问题"
        ]
        
        for topic in chinese_topics:
            english_query = rag.query_translator.translate_to_academic_query(topic)
            print(f"  中文: {topic}")
            print(f"  英文: {english_query}\n")
        
        # 测试检索
        test_topic = "人工智能对就业的影响"
        print(f"🔍 测试检索中文主题: {test_topic}")
        results = rag.search_academic_sources(test_topic, sources=["arxiv"])
        
        print(f"✅ 检索到 {len(results)} 篇相关论文")
        for i, result in enumerate(results[:2], 1):
            print(f"\n论文 {i}:")
            print(f"  标题: {result.title}")
            print(f"  相关性: {result.relevance_score}/10")
            print(f"  关键发现: {result.key_findings[:100]}...")
        
        # 测试角色专用查询
        print(f"\n🎭 测试角色专用查询:")
        for role in ["tech_expert", "economist", "sociologist"]:
            context = rag.get_rag_context_for_agent(role, test_topic, max_sources=1)
            print(f"  {role}: {len(context)} 字符的上下文")
            
    except Exception as e:
        print(f"❌ RAG模块测试失败: {e}")

def test_query_translation():
    """单独测试查询翻译功能"""
    print("🔤 测试查询翻译功能...")
    
    try:
        from langchain_deepseek import ChatDeepSeek
        test_llm = ChatDeepSeek(model="deepseek-chat", temperature=0.3)
        translator = QueryTranslator(test_llm)
        
        test_cases = [
            "人工智能是否会威胁人类就业？",
            "核能发电是解决气候变化的最佳方案吗？",
            "基因编辑技术的伦理边界在哪里？",
            "远程工作对社会经济的长期影响",
            "artificial intelligence employment impact",  # 已经是英文
            "AI在医疗诊断中的应用前景"
        ]
        
        print("查询翻译测试结果:")
        print("-" * 50)
        for topic in test_cases:
            english_query = translator.translate_to_academic_query(topic)
            print(f"原文: {topic}")
            print(f"英文: {english_query}")
            print()
            
    except Exception as e:
        print(f"❌ 查询翻译测试失败: {e}")

if __name__ == "__main__":
    # 检查环境变量
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("❌ 警告: DEEPSEEK_API_KEY 环境变量未设置")
        print("💡 可以先测试查询翻译的基础逻辑...")
        # test_query_translation()
    else:
        print("✅ 环境变量配置正确")
        test_rag_module()
        print("\n" + "="*50)
        test_query_translation()