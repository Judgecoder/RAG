"""
================================================================================
路由引擎模块 (route_engine.py)

【功能】
实现两阶段路由检索中的第一阶段——路由决策。
根据用户问题，自动判断最相关的知识库（Collection），
避免用户手动选择。

【核心组件】
1. DocumentRouterIndex - 路由索引，管理所有知识库的描述信息
2. 三种路由策略：LLM路由、向量路由、混合路由

【工作流程】
用户提问 → 路由引擎分析 → 选出Top-K个相关知识库 → 在这些库中精准检索

【与现有系统的关系】
- 不改变现有单库检索能力
- 当 auto_route=True 时，作为额外的前置阶段介入
================================================================================
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Tuple

from logger import logger
from models import get_ds_model_client, get_ali_embeddings


ROUTE_PROMPT_TEMPLATE = """你是一个知识库路由专家。请根据用户的问题，从以下知识库中选出最相关的 {top_k} 个。

每个知识库对应一个文档，其描述如下：

{descriptions}

【选择规则】
1. 仔细分析用户问题涉及的主题和关键词
2. 判断哪些知识库最可能包含答案
3. 如果不知道或无法判断，返回 "无法确定"
4. 只返回知识库名称，不要解释原因

【用户问题】
{user_query}

【输出格式】
只输出最相关的知识库名称，用英文逗号分隔，最多 {top_k} 个。
例如：人事管理流程_abc123,财务报销制度_def456"""


class DocumentRouterIndex:
    """
    文档路由索引

    职责：
    1. 为每个知识库（Collection）维护一段描述信息
    2. 根据用户问题，快速匹配最相关的知识库

    数据持久化：
    - 描述信息存储在 route_index.json 文件中
    - 向量缓存存储在内存中（服务重启后重新生成）

    使用方式：
        router = DocumentRouterIndex()
        # 构建索引
        router.build_description("coll_name", "file.docx", summaries)
        # 路由
        candidates = router.route("今年的招聘流程是什么？")
    """

    def __init__(self, index_file="route_index.json"):
        self.index_file = index_file
        self.descriptions: Dict[str, dict] = {}
        self._embeddings_cache: Dict[str, list] = {}
        self._llm_client = None
        self._embedding_model = None
        self._load()

    def _load(self):
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.descriptions = json.load(f)
                logger.info(f'【路由索引】已加载 {len(self.descriptions)} 个知识库描述')
            except Exception as e:
                logger.warning(f'【路由索引】加载失败: {e}，使用空索引')
                self.descriptions = {}
        else:
            logger.info('【路由索引】索引文件不存在，使用空索引')

    def _save(self):
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.descriptions, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f'【路由索引】保存失败: {e}')

    @property
    def llm_client(self):
        if self._llm_client is None:
            self._llm_client = get_ds_model_client()
        return self._llm_client

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = get_ali_embeddings()
        return self._embedding_model

    def build_description(self, collection_name: str, filepath: str, summaries: List[str]):
        """
        为指定知识库构建路由描述。

        【参数】
            collection_name: ChromaDB 集合名称
            filepath: 原始文件路径
            summaries: 文档各片段的摘要列表（由 DocumentProcessor 生成）
        """
        file_name = os.path.basename(filepath)

        # 将多个摘要合并为一段连贯的描述
        if summaries:
            combined = "\n".join(summaries)
            if len(combined) > 1000:
                combined = combined[:1000] + "..."

            # 用 LLM 精炼成更清晰的路由描述
            try:
                refined = self._refine_description(file_name, combined)
                summary_text, tags = refined
            except Exception as e:
                logger.warning(f'【路由索引】描述精炼失败: {e}，使用原始摘要')
                summary_text = combined
                tags = []
        else:
            summary_text = f"文档: {file_name}"
            tags = []

        self.descriptions[collection_name] = {
            "collection_name": collection_name,
            "file_name": file_name,
            "summary": summary_text,
            "tags": tags,
            "chunk_count": len(summaries),
            "created_at": datetime.now().isoformat()
        }

        self._embeddings_cache.pop(collection_name, None)
        self._save()
        logger.info(f'【路由索引】已构建知识库 "{file_name}" 的路由描述')

    def _refine_description(self, file_name: str, content: str) -> Tuple[str, List[str]]:
        """
        用 LLM 将文档摘要精炼为简短的路由描述和标签。
        返回 (summary, tags) 元组。
        """
        prompt = f"""基于以下文档内容，生成：

1. 一段简洁的文档概述（50字以内），说明文档的核心主题
2. 3-5个主题标签，用逗号分隔

文档名称：{file_name}
文档内容摘要：
{content[:2000]}

输出格式（严格按此格式）：
概述：xxx
标签：标签1,标签2,标签3"""

        response = self.llm_client.invoke(prompt)
        text = response.content if hasattr(response, 'content') else str(response)

        summary = ""
        tags = []

        for line in text.strip().split('\n'):
            line = line.strip()
            if line.startswith('概述：') or line.startswith('概述:'):
                summary = line.split('：', 1)[-1] if '：' in line else line.split(':', 1)[-1]
            elif line.startswith('标签：') or line.startswith('标签:'):
                tag_str = line.split('：', 1)[-1] if '：' in line else line.split(':', 1)[-1]
                tags = [t.strip() for t in tag_str.split(',') if t.strip()]

        if not summary:
            summary = content[:100]
        if not tags:
            tags = [file_name.replace('.', '_')]

        return summary, tags

    def remove_description(self, collection_name: str):
        """删除指定知识库的路由描述（文档删除时调用）。"""
        if collection_name in self.descriptions:
            del self.descriptions[collection_name]
            self._embeddings_cache.pop(collection_name, None)
            self._save()
            logger.info(f'【路由索引】已移除知识库 "{collection_name}" 的路由描述')

    def get_all_collections(self) -> List[str]:
        """获取所有已注册的知识库名称列表。"""
        return list(self.descriptions.keys())

    def route(self, user_query: str, strategy: str = "hybrid", top_k: int = 3) -> List[str]:
        """
        路由核心方法：给定用户问题，返回最相关的知识库名称列表。

        【参数】
            user_query: 用户提问
            strategy: 路由策略 ("llm", "vector", "hybrid")
            top_k: 返回最多几个候选知识库

        【返回值】
            候选知识库名称列表，按相关性降序排列
        """
        if not self.descriptions:
            logger.warning('【路由】路由索引为空，无法路由')
            return []

        available = list(self.descriptions.keys())
        actual_top_k = min(top_k, len(available))

        logger.info(f'【路由】策略={strategy}, 可用知识库={len(available)}个, 目标返回={actual_top_k}个')

        if strategy == "llm":
            return self._llm_route(user_query, actual_top_k)
        elif strategy == "vector":
            return self._vector_route(user_query, actual_top_k)
        else:
            return self._hybrid_route(user_query, actual_top_k)

    def _format_descriptions_for_prompt(self) -> str:
        """将路由描述格式化为 LLM Prompt 可读的文本。"""
        lines = []
        for coll_name, desc in self.descriptions.items():
            tags_str = ", ".join(desc.get("tags", []))
            tag_part = f" [{tags_str}]" if tags_str else ""
            lines.append(f'- {coll_name}: {desc["summary"]}{tag_part}')
        return "\n".join(lines)

    def _llm_route(self, user_query: str, top_k: int) -> List[str]:
        """策略A：LLM 路由——让大模型判断问题与哪些知识库相关。"""
        descriptions_text = self._format_descriptions_for_prompt()
        prompt = ROUTE_PROMPT_TEMPLATE.format(
            descriptions=descriptions_text,
            user_query=user_query,
            top_k=top_k
        )

        try:
            response = self.llm_client.invoke(prompt)
            text = response.content if hasattr(response, 'content') else str(response)
            text = text.strip()

            if not text or text == "无法确定":
                logger.warning('【路由-LLM】无法确定相关知识库')
                return []

            candidates = [name.strip() for name in text.split(',') if name.strip()]
            valid = [name for name in candidates if name in self.descriptions]

            logger.info(f'【路由-LLM】LLM 推荐: {candidates}, 有效: {valid}')
            return valid[:top_k]

        except Exception as e:
            logger.error(f'【路由-LLM】出错: {e}，回退到向量路由')
            return self._vector_route(user_query, top_k)

    def _vector_route(self, user_query: str, top_k: int) -> List[str]:
        """策略B：向量路由——通过向量相似度匹配。"""
        try:
            query_vector = self.embedding_model.embed_query(user_query)

            scores = []
            for coll_name in self.descriptions:
                if coll_name not in self._embeddings_cache:
                    desc_text = self.descriptions[coll_name]["summary"]
                    if self.descriptions[coll_name].get("tags"):
                        desc_text += " " + " ".join(self.descriptions[coll_name]["tags"])
                    self._embeddings_cache[coll_name] = self.embedding_model.embed_query(desc_text)

                desc_vector = self._embeddings_cache[coll_name]
                similarity = self._cosine_similarity(query_vector, desc_vector)
                scores.append((coll_name, similarity))

            scores.sort(key=lambda x: x[1], reverse=True)
            result = [name for name, _ in scores[:top_k]]

            logger.info(f'【路由-向量】候选: {[(n, f"{s:.3f}") for n, s in scores[:top_k]]}')
            return result

        except Exception as e:
            logger.error(f'【路由-向量】出错: {e}')
            return []

    def _hybrid_route(self, user_query: str, top_k: int) -> List[str]:
        """策略C：混合路由——融合 LLM 路由和向量路由的结果。"""
        llm_results = self._llm_route(user_query, top_k * 2)
        vector_results = self._vector_route(user_query, top_k * 2)

        # 加权融合：LLM 结果权重 0.6，向量结果权重 0.4
        score_map = {}
        for rank, name in enumerate(llm_results):
            score_map[name] = score_map.get(name, 0) + 0.6 * (1.0 - rank / (len(llm_results) or 1))
        for rank, name in enumerate(vector_results):
            score_map[name] = score_map.get(name, 0) + 0.4 * (1.0 - rank / (len(vector_results) or 1))

        merged = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        result = [name for name, _ in merged[:top_k]]

        logger.info(f'【路由-混合】候选: {result}')
        return result

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """计算两个向量的余弦相似度。"""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
