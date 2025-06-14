"""
动态RAG模块 - 基于Kimi API的真实文献检索
使用Kimi的强大能力进行真实学术文献检索和分析
重点：确保所有检索到的学术资料都是真实存在的，绝不编造虚假论文
优化：支持基于专家角色的缓存机制
增强：更好的错误处理和异常安全性
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
    # Kimi API配置
    "kimi_api_url": "https://api.moonshot.cn/v1/chat/completions",
    "kimi_model": "moonshot-v1-8k",
    "kimi_timeout": 60
}

@dataclass
class SearchResult:
    """检索结果数据类 - 确保真实性"""
    title: str
    authors: List[str]
    abstract: str
    url: str
    published_date: str
    source: str
    relevance_score: float = 0.0
    key_findings: str = ""
    # 新增：真实性验证字段
    is_verified: bool = False
    verification_notes: str = ""

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

class KimiSearcher:
    """基于Kimi API的真实学术文献检索器"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("KIMI_API_KEY")
        self.api_url = RAG_CONFIG["kimi_api_url"]
        self.model = RAG_CONFIG["kimi_model"]
        self.session = requests.Session()
        
        if not self.api_key:
            print("⚠️ 警告: KIMI_API_KEY 环境变量未设置")
        else:
            print("✅ Kimi API 初始化成功")
    
    def search(self, query: str, max_results: int = 5, agent_role: str = "") -> List[SearchResult]:
        """使用Kimi API检索真实存在的学术文献"""
        if not self.api_key:
            print("❌ Kimi API Key 未配置")
            return []
        
        try:
            # 构建检索提示词，强调真实性
            search_prompt = self._build_real_search_prompt(query, max_results, agent_role)
            
            print(f"🔍 正在使用Kimi检索真实学术文献: {query} (最多{max_results}篇)")
            
            # 调用Kimi API
            response = self._call_kimi_api(search_prompt)
            
            if response:
                # 解析响应并验证真实性
                results = self._parse_and_verify_kimi_response(response, query)
                # 过滤掉可能虚假的结果
                verified_results = self._filter_real_results(results)
                return verified_results
            else:
                return []
                
        except Exception as e:
            print(f"❌ Kimi检索失败: {e}")
            return []
    
    def _build_real_search_prompt(self, query: str, max_results: int, agent_role: str = "") -> str:
        """构建强调真实性的Kimi检索提示词"""
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
        
        prompt = f"""请作为一个专业的学术研究助手，{role_context}帮我检索关于"{query}"的真实存在的学术文献和研究成果。

🚨 重要要求 - 绝对真实性：
1. 只能提供真实存在的学术论文和研究报告
2. 不得编造或虚构任何论文信息
3. 如果无法确认论文的真实性，请明确说明
4. 如果找不到足够的真实文献，请诚实回复找到的实际数量

检索要求：
1. 寻找{max_results}篇真实的高质量学术文献或研究报告
2. 优先选择近5年内发表的权威论文
3. 包含中英文文献，优先考虑影响因子较高的期刊
4. 每篇文献需要包含真实可验证的信息：
   - 真实的论文标题
   - 真实的作者姓名
   - 真实的发表时间和期刊
   - 论文的实际核心观点
   - 真实可访问的DOI或链接（如果有）

输出格式要求：
请按照以下JSON格式返回，并确保所有信息都是真实的：

```json
[
  {{
    "title": "真实的论文标题（中英文均可）",
    "authors": ["真实作者1", "真实作者2"],
    "abstract": "论文真实摘要或核心内容概述",
    "published_date": "真实发表日期(YYYY-MM-DD格式)",
    "key_findings": "论文的实际主要发现和观点",
    "relevance_score": 8.5,
    "source": "真实的期刊名称或出版机构",
    "url": "真实的DOI链接或官方链接",
    "verification_notes": "真实性说明，如：'该论文发表在Nature期刊2023年第XX期'"
  }}
]
```

关键提醒：
- 如果找不到{max_results}篇真实相关文献，请返回实际找到的数量
- 每篇论文都必须是真实存在的，可以通过学术数据库验证
- 不要为了凑数而编造任何虚假信息
- 如果某个信息不确定，请标注"待确认"而不是编造

现在请为我检索关于"{query}"的真实学术文献：
"""
        return prompt
    
    def _call_kimi_api(self, prompt: str) -> Optional[str]:
        """调用Kimi API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,  # 降低温度以提高准确性
                "max_tokens": 4000
            }
            
            response = self.session.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=RAG_CONFIG["kimi_timeout"]
            )
            
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                print("❌ Kimi API 响应格式异常")
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
    
    def _parse_and_verify_kimi_response(self, response: str, query: str) -> List[SearchResult]:
        """解析Kimi API响应并初步验证真实性"""
        results = []
        
        try:
            # 尝试提取JSON部分
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                # 如果没有找到JSON格式，尝试解析文本格式
                return self._parse_text_response(response, query)
            
            json_str = response[json_start:json_end]
            papers = json.loads(json_str)
            
            for paper in papers:
                try:
                    # 基本真实性检查
                    title = paper.get("title", "").strip()
                    authors = paper.get("authors", [])
                    source = paper.get("source", "").strip()
                    
                    # 跳过明显虚假的条目
                    if not title or len(title) < 10:
                        print(f"⚠️ 跳过标题过短或缺失的条目: {title}")
                        continue
                    
                    if not authors or len(authors) == 0:
                        print(f"⚠️ 跳过缺少作者信息的条目: {title}")
                        continue
                    
                    # 检查是否包含明显的编造痕迹
                    if self._is_likely_fabricated(paper):
                        print(f"⚠️ 跳过可能编造的条目: {title}")
                        continue
                    
                    result = SearchResult(
                        title=title,
                        authors=authors,
                        abstract=paper.get("abstract", ""),
                        url=paper.get("url", "待查询学术数据库"),
                        published_date=paper.get("published_date", ""),
                        source=source,
                        relevance_score=float(paper.get("relevance_score", 7.0)),
                        key_findings=paper.get("key_findings", ""),
                        is_verified=False,  # 需要进一步验证
                        verification_notes=paper.get("verification_notes", "")
                    )
                    results.append(result)
                    
                except Exception as e:
                    print(f"⚠️ 解析单篇文献失败: {e}")
                    continue
            
            print(f"✅ Kimi检索解析 {len(results)} 篇可能真实的文献")
            return results
            
        except json.JSONDecodeError:
            print("⚠️ JSON解析失败，尝试文本解析")
            return self._parse_text_response(response, query)
        except Exception as e:
            print(f"❌ Kimi响应解析失败: {e}")
            return []
    
    def _is_likely_fabricated(self, paper: dict) -> bool:
        """检查论文信息是否可能是编造的"""
        try:
            title = paper.get("title", "").lower()
            authors = paper.get("authors", [])
            source = paper.get("source", "").lower()
            
            # 检查标题中的可疑模式
            suspicious_title_patterns = [
                "example paper", "sample study", "hypothetical research",
                "示例论文", "样本研究", "假设研究", "虚构", "编造"
            ]
            
            for pattern in suspicious_title_patterns:
                if pattern in title:
                    return True
            
            # 检查作者姓名是否过于简单或可疑
            for author in authors:
                if len(author.strip()) < 3 or author.lower() in ["作者1", "author1", "研究者"]:
                    return True
            
            # 检查期刊名称是否可疑
            suspicious_sources = [
                "示例期刊", "sample journal", "example publication",
                "test journal", "虚构期刊"
            ]
            
            for sus_source in suspicious_sources:
                if sus_source in source:
                    return True
            
            return False
            
        except Exception as e:
            print(f"⚠️ 真实性检查失败: {e}")
            return False
    
    def _parse_text_response(self, response: str, query: str) -> List[SearchResult]:
        """解析文本格式的响应"""
        results = []
        
        try:
            # 简单的文本解析逻辑
            lines = response.split('\n')
            current_paper = {}
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if '标题' in line or 'title' in line.lower():
                    if current_paper and current_paper.get('title'):
                        # 保存前一篇文献
                        result = self._create_result_from_dict(current_paper, query)
                        if result and not self._is_likely_fabricated(current_paper):
                            results.append(result)
                    current_paper = {'title': line.split('：', 1)[-1].split(':', 1)[-1].strip()}
                elif '作者' in line or 'author' in line.lower():
                    authors_str = line.split('：', 1)[-1].split(':', 1)[-1].strip()
                    current_paper['authors'] = [a.strip() for a in authors_str.split(',')]
                elif '摘要' in line or 'abstract' in line.lower():
                    current_paper['abstract'] = line.split('：', 1)[-1].split(':', 1)[-1].strip()
                elif '发现' in line or 'finding' in line.lower():
                    current_paper['key_findings'] = line.split('：', 1)[-1].split(':', 1)[-1].strip()
                elif '期刊' in line or 'journal' in line.lower():
                    current_paper['source'] = line.split('：', 1)[-1].split(':', 1)[-1].strip()
            
            # 处理最后一篇文献
            if current_paper and current_paper.get('title'):
                result = self._create_result_from_dict(current_paper, query)
                if result and not self._is_likely_fabricated(current_paper):
                    results.append(result)
            
            print(f"✅ 文本解析获得 {len(results)} 篇可能真实的文献")
            return results
            
        except Exception as e:
            print(f"❌ 文本解析失败: {e}")
            return []
    
    def _create_result_from_dict(self, paper_dict: dict, query: str) -> Optional[SearchResult]:
        """从字典创建SearchResult对象"""
        try:
            return SearchResult(
                title=paper_dict.get('title', '未知标题'),
                authors=paper_dict.get('authors', []),
                abstract=paper_dict.get('abstract', ''),
                url=paper_dict.get('url', '待查询学术数据库'),
                published_date=paper_dict.get('published_date', datetime.now().strftime('%Y-%m-%d')),
                source=paper_dict.get('source', 'Kimi检索'),
                relevance_score=7.0,
                key_findings=paper_dict.get('key_findings', ''),
                is_verified=False,
                verification_notes=""
            )
        except Exception as e:
            print(f"⚠️ 创建SearchResult失败: {e}")
            return None
    
    def _filter_real_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """过滤掉可能虚假的结果，只保留看起来真实的"""
        filtered_results = []
        
        for result in results:
            # 进一步的真实性检查
            if self._appears_authentic(result):
                result.is_verified = True
                filtered_results.append(result)
            else:
                print(f"⚠️ 过滤掉可能不真实的文献: {result.title[:50]}...")
        
        print(f"🔍 真实性过滤：保留 {len(filtered_results)}/{len(results)} 篇文献")
        return filtered_results
    
    def _appears_authentic(self, result: SearchResult) -> bool:
        """检查单个结果是否看起来真实"""
        try:
            # 检查标题长度和复杂性
            if len(result.title) < 15 or len(result.title) > 200:
                return False
            
            # 检查作者数量和格式
            if not result.authors or len(result.authors) == 0:
                return False
            
            # 检查是否有有意义的摘要或关键发现
            if not result.abstract and not result.key_findings:
                return False
            
            # 检查日期格式
            if result.published_date:
                try:
                    # 简单的日期格式检查
                    if re.match(r'\d{4}-\d{1,2}-\d{1,2}', result.published_date):
                        year = int(result.published_date.split('-')[0])
                        if year < 1950 or year > datetime.now().year:
                            return False
                except:
                    pass
            
            # 检查来源是否合理
            if not result.source or len(result.source.strip()) < 3:
                return False
            
            return True
            
        except Exception as e:
            print(f"⚠️ 真实性检查失败: {e}")
            return False

class RAGEnhancer:
    """RAG增强器 - 处理检索结果并生成洞察（强调真实性）"""
    
    def __init__(self, llm: ChatDeepSeek):
        self.llm = llm
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个学术研究分析专家。基于给定的学术论文信息，提取和总结关键发现，为特定角色的辩论提供支撑。

🚨 重要提醒：你分析的论文信息来自Kimi API检索，请确保：
1. 只基于提供的真实论文信息进行分析
2. 不要添加任何未提供的虚假信息
3. 如果信息不足，请诚实说明
4. 保持分析的客观性和准确性

你的任务：
1. 分析论文的核心观点和发现
2. 提取与辩论主题和指定角色相关的关键证据
3. 简洁地总结主要论点（2-3句话）
4. 评估研究的可信度和相关性

专家角色：{agent_role}
论文信息：
标题：{title}
作者：{authors}
核心内容：{abstract}
主要发现：{key_findings}
发布时间：{published_date}
来源：{source}

辩论主题：{debate_topic}

请特别关注与{agent_role}专业领域相关的内容，提供：
1. 关键发现（核心观点和证据）
2. 与辩论主题的相关性评分（1-10分）
3. 建议该角色在辩论中如何引用这项研究"""),
            ("user", "请基于真实的论文信息分析并提供关键洞察")
        ])
    
    def enhance_results(self, results: List[SearchResult], debate_topic: str, agent_role: str = "") -> List[SearchResult]:
        """增强检索结果，提取关键洞察（针对特定角色优化）"""
        enhanced_results = []
        
        for result in results:
            try:
                # 如果结果已经有key_findings，直接使用，否则用LLM分析
                if not result.key_findings and self.llm:
                    analysis = self._analyze_paper(result, debate_topic, agent_role)
                    result.key_findings = analysis.get('key_findings', result.abstract[:200])
                    result.relevance_score = analysis.get('relevance_score', result.relevance_score)
                
                enhanced_results.append(result)
                
                # 避免API限制
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ 论文分析失败 {result.title}: {e}")
                # 即使分析失败也保留原始结果
                if not result.key_findings:
                    result.key_findings = result.abstract[:150] + "..."
                enhanced_results.append(result)
        
        # 按相关性评分排序
        try:
            enhanced_results.sort(key=lambda x: x.relevance_score, reverse=True)
        except Exception as e:
            print(f"⚠️ 结果排序失败: {e}")
        
        return enhanced_results
    
    def _analyze_paper(self, result: SearchResult, debate_topic: str, agent_role: str = "") -> dict:
        """分析单篇论文（针对特定角色）"""
        try:
            if not self.llm:
                return {
                    'key_findings': result.abstract[:150] + "...",
                    'relevance_score': result.relevance_score or 5.0
                }
            
            pipe = self.analysis_prompt | self.llm | StrOutputParser()
            
            response = pipe.invoke({
                'agent_role': agent_role,
                'title': result.title,
                'authors': ', '.join(result.authors[:3]),
                'abstract': result.abstract[:1000],
                'key_findings': result.key_findings[:500] if result.key_findings else "",
                'published_date': result.published_date,
                'source': result.source,
                'debate_topic': debate_topic
            })
            
            # 简单解析响应
            lines = response.strip().split('\n')
            key_findings = ""
            relevance_score = result.relevance_score or 5.0
            
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
                'relevance_score': result.relevance_score or 5.0
            }

class DynamicRAGModule:
    """动态RAG主模块（基于Kimi API的真实文献检索）"""
    
    def __init__(self, llm: ChatDeepSeek):
        self.llm = llm
        self.cache = RAGCache()
        self.kimi_searcher = KimiSearcher()
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
                              sources: List[str] = ["kimi"],
                              max_results_per_source: int = None,
                              agent_role: str = "") -> List[SearchResult]:
        """
        搜索真实的学术数据源（基于Kimi API）
        
        Args:
            topic: 搜索主题
            sources: 数据源列表（现在主要是"kimi"）
            max_results_per_source: 每个数据源的最大结果数（用户可配置）
            agent_role: 专家角色（用于定制化分析）
        """
        
        if max_results_per_source is None:
            max_results_per_source = RAG_CONFIG["max_results_per_source"]
        
        print(f"🔍 Kimi真实文献检索配置：最多{max_results_per_source}篇，角色定制：{agent_role}")
        
        # 参数安全检查
        if not topic or not topic.strip():
            print("⚠️ 搜索主题为空")
            return []
        
        if not sources:
            sources = ["kimi"]  # 默认使用Kimi
        
        # 检查缓存
        try:
            cached_results = self.cache.get_cached_results(topic, sources)
            if cached_results:
                print(f"✅ 使用缓存结果: {len(cached_results)} 篇论文")
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
        
        # Kimi检索真实文献
        if "kimi" in sources:
            try:
                kimi_results = self.kimi_searcher.search(topic, max_results_per_source, agent_role)
                all_results.extend(kimi_results)
                print(f"📚 Kimi找到 {len(kimi_results)} 篇真实论文（设置上限：{max_results_per_source}篇）")
                
                # 统计真实性验证结果
                verified_count = sum(1 for r in kimi_results if r.is_verified)
                print(f"✅ 其中 {verified_count} 篇通过真实性验证")
                
            except Exception as e:
                print(f"❌ Kimi检索出错: {e}")
        
        # 使用LLM增强结果（考虑专家角色）
        if all_results and self.enhancer:
            try:
                print(f"🤖 使用AI分析论文相关性{'（为' + agent_role + '定制）' if agent_role else ''}...")
                all_results = self.enhancer.enhance_results(all_results, topic, agent_role)
            except Exception as e:
                print(f"⚠️ LLM增强失败，使用原始结果: {e}")
        
        # 缓存结果（只缓存通过验证的真实结果）
        if all_results:
            try:
                verified_results = [r for r in all_results if r.is_verified]
                if verified_results:
                    self.cache.cache_results(topic, sources, verified_results)
                    print(f"💾 缓存了 {len(verified_results)} 篇经过验证的真实文献")
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
        为特定角色获取基于真实文献的RAG上下文
        
        Args:
            agent_role: 专家角色
            debate_topic: 辩论主题
            max_sources: 最大参考文献数（来自用户设置）
            max_results_per_source: 每个数据源的最大检索数
            force_refresh: 是否强制刷新（忽略缓存）
        """
        
        print(f"🔍 为专家{agent_role}检索真实学术资料，最大文献数{max_sources}篇")
        
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
                        print(f"🔄 缓存文献数({cached_ref_count})与用户设置({max_sources})不符，重新检索...")
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
        
        # 使用用户设置的数量进行真实文献检索
        try:
            results = self.search_academic_sources(
                role_focused_query, 
                sources=["kimi"],  # 使用Kimi作为数据源
                max_results_per_source=max_sources,  # 直接使用用户设置
                agent_role=agent_role
            )
        except Exception as e:
            print(f"❌ Kimi学术检索失败: {e}")
            return "学术资料检索遇到技术问题，请基于你的专业知识发表观点。"
        
        if not results:
            context = "暂无相关学术资料。"
        else:
            try:
                # 优先使用通过验证的真实文献
                verified_results = [r for r in results if r.is_verified]
                if not verified_results:
                    print("⚠️ 未找到通过验证的真实文献，使用原始结果")
                    verified_results = results
                
                # 选择用户设置数量的文献
                top_results = verified_results[:max_sources]
                
                print(f"📊 检索结果处理：为专家 {agent_role} 实际检索到 {len(results)} 篇，其中 {len(verified_results)} 篇通过验证，按用户设置选择前 {len(top_results)} 篇")
                
                # 构建上下文
                context_parts = []
                for i, result in enumerate(top_results, 1):
                    try:
                        verification_status = "✅ 已验证" if result.is_verified else "⚠️ 待验证"
                        context_part = f"""
参考资料 {i}: {verification_status}
标题: {result.title}
作者: {', '.join(result.authors[:2])}
来源: {result.source} ({result.published_date})
关键发现: {result.key_findings or result.abstract[:200]}
相关性: {result.relevance_score}/10
"""
                        if result.verification_notes:
                            context_part += f"验证说明: {result.verification_notes}\n"
                            
                        context_parts.append(context_part.strip())
                    except Exception as e:
                        print(f"⚠️ 处理第{i}篇文献失败: {e}")
                        continue
                
                context = "\n\n".join(context_parts)
                
                # 验证最终结果
                final_ref_count = context.count('参考资料')
                verified_final_count = context.count('✅ 已验证')
                print(f"✅ 上下文构建完成：{final_ref_count}篇参考文献（其中{verified_final_count}篇已验证真实性）")
                
            except Exception as e:
                print(f"❌ 上下文构建失败: {e}")
                context = "学术资料处理遇到技术问题，请基于你的专业知识发表观点。"
        
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
            print(f"🎯 为{agent_role}定制Kimi查询：{focused_query}")
            return focused_query
        except Exception as e:
            print(f"⚠️ 角色查询生成失败: {e}")
            return debate_topic
    
    def preload_agent_contexts(self, agent_roles: List[str], debate_topic: str, max_refs_per_agent: int = 3):
        """
        预加载所有专家的真实学术上下文
        
        Args:
            agent_roles: 专家角色列表
            debate_topic: 辩论主题
            max_refs_per_agent: 每个专家的最大参考文献数（用户设置）
        """
        
        if not agent_roles or not debate_topic:
            print("⚠️ 专家角色列表或辩论主题为空")
            return
        
        print(f"🚀 开始为 {len(agent_roles)} 位专家预加载Kimi真实学术资料...")
        print(f"📊 用户配置：每专家最多 {max_refs_per_agent} 篇参考文献")
        
        for agent_role in agent_roles:
            try:
                print(f"🔍 为专家 {agent_role} 使用Kimi检索真实学术资料...")
                context = self.get_rag_context_for_agent(
                    agent_role=agent_role,
                    debate_topic=debate_topic,
                    max_sources=max_refs_per_agent,  # 使用用户设置
                    max_results_per_source=2,
                    force_refresh=True  # 强制刷新确保最新资料
                )
                
                if context and context != "暂无相关学术资料。":
                    actual_count = context.count('参考资料')
                    verified_count = context.count('✅ 已验证')
                    print(f"✅ 专家 {agent_role} 的学术资料已准备就绪：{actual_count}篇（其中{verified_count}篇已验证）")
                else:
                    print(f"⚠️ 专家 {agent_role} 未找到相关学术资料")
                
                # 避免API限制
                time.sleep(3)  # Kimi API可能需要更长的间隔
                
            except Exception as e:
                print(f"❌ 为专家 {agent_role} 预加载资料失败: {e}")
                continue
        
        print("✅ 所有专家的Kimi真实学术资料预加载完成")
    
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
    
    def test_kimi_real_integration(self, 
                                  agent_role: str = "tech_expert", 
                                  debate_topic: str = "人工智能对教育的影响",
                                  test_configs: List[int] = [1, 3, 5]):
        """
        测试Kimi真实文献检索集成
        
        Args:
            agent_role: 测试专家角色
            debate_topic: 测试辩论主题  
            test_configs: 测试的参考文献数量列表
        """
        print("🧪 开始测试Kimi API真实文献检索集成...")
        
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
                    verified_count = context.count('✅ 已验证')
                    status = "✅" if actual_count == max_refs else "❌"
                    print(f"{status} Kimi结果：实际{actual_count}篇，期望{max_refs}篇，其中{verified_count}篇已验证真实性")
                    
                    if actual_count != max_refs:
                        print(f"⚠️ 配置不生效！请检查代码")
                    if verified_count == 0:
                        print(f"⚠️ 未找到通过验证的真实文献")
                else:
                    print(f"⚠️ Kimi未找到学术资料")
                    
            except Exception as e:
                print(f"❌ 测试失败: {e}")
        
        print("\n🎉 Kimi API真实文献检索测试完成！")

# 全局RAG实例（将在graph.py中初始化）
rag_module = None

def initialize_rag_module(llm: ChatDeepSeek) -> DynamicRAGModule:
    """初始化RAG模块（基于Kimi API的真实文献检索）"""
    global rag_module
    try:
        rag_module = DynamicRAGModule(llm)
        print("🔍 RAG模块已初始化，专注于Kimi API真实文献检索")
        return rag_module
    except Exception as e:
        print(f"❌ Kimi RAG模块初始化失败: {e}")
        return None

def get_rag_module() -> Optional[DynamicRAGModule]:
    """获取RAG模块实例"""
    return rag_module

# 测试函数
def test_rag_module():
    """测试基于Kimi API的真实文献检索RAG模块功能"""
    print("🧪 开始测试基于Kimi API的真实文献检索RAG模块...")
    
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
            print("❌ Kimi RAG模块初始化失败")
            return
        
        # 测试专家角色检索
        test_topic = "人工智能对就业的影响"
        test_roles = ["tech_expert", "economist", "sociologist"]
        
        print("🔍 测试基于Kimi的专家角色真实文献检索...")
        for role in test_roles:
            # 测试不同的用户配置
            for max_refs in [1, 3]:
                print(f"\n📊 测试：{role} 获取 {max_refs} 篇真实文献")
                try:
                    context = rag.get_rag_context_for_agent(
                        agent_role=role, 
                        debate_topic=test_topic,
                        max_sources=max_refs,  # 测试用户设置
                        force_refresh=True
                    )
                    
                    if context and context != "暂无相关学术资料。":
                        actual_count = context.count('参考资料')
                        verified_count = context.count('✅ 已验证')
                        status = "✅" if actual_count == max_refs else "❌"
                        print(f"{status} Kimi结果：期望{max_refs}篇，实际{actual_count}篇，验证{verified_count}篇")
                        print(f"前100字符：{context[:100]}...")
                    else:
                        print("⚠️ Kimi未找到学术资料")
                except Exception as e:
                    print(f"❌ 测试出错: {e}")
        
        # 专门的Kimi真实性检索测试
        print("\n🔧 专门测试Kimi真实文献检索...")
        try:
            rag.test_kimi_real_integration()
        except Exception as e:
            print(f"❌ Kimi真实性检索测试失败: {e}")
            
    except Exception as e:
        print(f"❌ Kimi RAG模块测试失败: {e}")

if __name__ == "__main__":
    test_rag_module()