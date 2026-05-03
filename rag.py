"""
================================================================================
RAG核心逻辑文件 (rag.py)

【RAG的工作流程】
1. 用户上传文档 -> 文档被切分成小块 -> 存入向量数据库
2. 用户提问 -> 在数据库中搜索相关内容 -> 把相关内容+问题一起给AI -> AI生成答案

【这个文件的作用】
1. 处理文档上传：读取文档、切分、存入向量库
2. 处理用户提问：检索相关内容、构建提示词、调用AI生成答案

【依赖安装】
pip install pypinyin       # 中文转拼音
================================================================================
"""

# 从function_tools.py导入所有工具函数和类
# 包括：向量数据库类、文档读取函数、AI调用函数等
from function_tools import *
from logger import logger
from document_processor import DocumentProcessor
from route_engine import DocumentRouterIndex


class RAGService:
    """
    RAG服务类，封装了文档处理和智能问答功能
    """
    def __init__(self, 
                 collection_name='demo', 
                 n_results=10, 
                 hybrid_search=False, 
                 vector_weight=0.5, 
                 rerank=False, 
                 rerank_top_k=5,
                 auto_route=False,
                 redis_url="redis://localhost:6379"):
        """
        初始化RAG服务
        
        【参数】
            collection_name: 默认集合名称，默认是'demo'
            n_results: 默认返回最相关的结果数，默认是10条
            hybrid_search: 是否启用混合检索，默认是False
            vector_weight: 向量检索权重（0~1），默认是0.5
            rerank: 是否启用检索后重排序，默认是False
            rerank_top_k: 重排序后保留的结果数，默认是5条
            auto_route: 是否启用自动路由检索，默认是False
            redis_url: Redis连接URL，默认是"redis://localhost:6379"
        """
        # 配置参数
        self.collection_name = collection_name
        self.n_results = n_results
        self.hybrid_search = hybrid_search
        self.vector_weight = vector_weight
        self.rerank = rerank
        self.rerank_top_k = rerank_top_k
        self.auto_route = auto_route
        self.redis_url = redis_url
        
        # 创建文档处理器实例
        self.document_processor = DocumentProcessor()
        
        # 创建向量数据库实例
        self.vector_db = MyVectorDBConnector()
        
        # 路由索引（延迟初始化）
        self._route_index = None
    
    @property
    def route_index(self):
        if self._route_index is None:
            self._route_index = DocumentRouterIndex()
        return self._route_index
    
    def save_to_db(self, filepath, collection_name=None):
        """
        【功能】将文档存入向量数据库
        
        【参数】
            filepath: 文档的完整路径，比如 "uploads/合同.docx"
            collection_name: 集合名称（知识库的名字），默认使用类初始化时的集合名称
        
        【处理流程】
            1. 读取文档内容
            2. 将文档切分成小块（方便检索）
            3. 将小块存入向量数据库
        
        【返回值】
            str: 成功或错误信息
        """
        # 使用指定的集合名称或默认集合名称
        coll_name = collection_name or self.collection_name
        
        # 打印分隔线，让日志更清晰
        logger.info('-' * 100)
        logger.info(f'【文档上传】文件路径: {filepath}')
        logger.info(f'【文档上传】集合名称: {coll_name}')
        
        try:
            # 使用DocumentProcessor处理文档（加载、分割、生成摘要、存入向量数据库）
            # DocumentProcessor会自动检测文件类型并调用相应的处理器
            # 返回: 处理结果字典，包含split_docs、summaries和retriever
            result = self.document_processor.process_document(filepath, collection_name=coll_name)
            
            if result:
                success_msg = f'文档已成功存入向量数据库，共 {len(result["split_docs"])} 个片段'
                logger.info(f'【成功】{success_msg}')

                # 构建路由描述
                try:
                    self.route_index.build_description(
                        collection_name=coll_name,
                        filepath=filepath,
                        summaries=result.get('summaries', [])
                    )
                except Exception as route_error:
                    logger.warning(f'【路由描述构建警告】{route_error}')

                return success_msg
            else:
                error_msg = '文档处理失败'
                logger.error(f'【错误】{error_msg}')
                return error_msg
            
        except Exception as e:
            error_msg = f'处理文档时出错: {str(e)}'
            logger.error(f'【文档上传错误】{error_msg}')
            import traceback
            traceback.print_exc()  # 打印完整的堆栈跟踪
            return error_msg
    
    def chat(self, user_query, collection_name=None, session_id=None, auto_route=None):
        """
        【功能】RAG智能问答
        
        【参数】
            user_query: 用户的问题
            collection_name: 在哪个知识库中搜索，默认使用类初始化时的集合名称
            session_id: 用户工号，用于RedisChatMessageHistory的session_id
            auto_route: 是否启用自动路由检索，None表示使用实例默认值
        
        【返回值】
            response: AI生成的答案
            retrieved_docs: 检索到的相关文档片段列表
        """
        # =========================================================================
        # 阶段1：判断是否启用自动路由
        # =========================================================================
        use_route = auto_route if auto_route is not None else self.auto_route
        coll_name = collection_name or self.collection_name
        _rerank_applied = False

        logger.info('=' * 100)
        logger.info(f'【RAG问答】用户问题: {user_query}')
        logger.info(f'【RAG问答】自动路由: {"启用" if use_route else "关闭"}')
        logger.info(f'【RAG问答】目标集合: {"自动路由" if use_route else coll_name}')
        logger.info('=' * 100)

        retrieved_docs = []

        # =========================================================================
        # 阶段2：检索知识库
        # =========================================================================
        if use_route:
            # ---------------------------------------------------------------------
            # 两阶段路由检索
            # ---------------------------------------------------------------------
            logger.info('\n>>> 第1步：自动路由检索...')

            available = self.route_index.get_all_collections()
            if not available:
                logger.warning('【路由】没有已注册的知识库，回退到默认集合')
                search_results = self.vector_db.search(
                    user_query,
                    collection_name=coll_name,
                    n_results=self.n_results,
                    hybrid_search=self.hybrid_search,
                    vector_weight=self.vector_weight
                )
                retrieved_docs = search_results['documents'][0]
            else:
                # 1阶段：路由决策
                logger.info(f'【路由】可用知识库: {available}')
                candidates = self.route_index.route(user_query, strategy="hybrid", top_k=3)

                if not candidates:
                    logger.warning('【路由】未找到相关知识库，使用默认集合')
                    search_results = self.vector_db.search(
                        user_query,
                        collection_name=coll_name,
                        n_results=self.n_results,
                        hybrid_search=self.hybrid_search,
                        vector_weight=self.vector_weight
                    )
                    retrieved_docs = search_results['documents'][0]
                else:
                    logger.info(f'【路由】候选知识库: {candidates}')

                    # 2阶段：在候选知识库中精准检索
                    # 每个候选集合都检索 n_results 条，确保每个知识库的信息足够
                    all_docs = []
                    seen_content = set()

                    for candidate in candidates:
                        results = self.vector_db.search(
                            user_query,
                            collection_name=candidate,
                            n_results=self.n_results,
                            hybrid_search=self.hybrid_search,
                            vector_weight=self.vector_weight
                        )
                        candidate_docs = results['documents'][0]

                        # 每个候选集合独立重排序，各自保留 top_n 条
                        if self.rerank:
                            candidate_docs = rerank_documents(candidate_docs, user_query, top_n=self.rerank_top_k)

                        for doc in candidate_docs:
                            content_hash = hash(doc[:200])
                            if content_hash not in seen_content:
                                seen_content.add(content_hash)
                                all_docs.append(doc)

                    retrieved_docs = all_docs[:self.n_results * max(1, len(candidates))]
                    logger.info(f'【路由】从 {len(candidates)} 个知识库共检索到 {len(retrieved_docs)} 条内容')

                    # 标记已按集合独立重排序，跳过后续通用重排序
                    _rerank_applied = True
        else:
            # ---------------------------------------------------------------------
            # 传统单库检索
            # ---------------------------------------------------------------------
            logger.info(f'\n>>> 第1步：在向量数据库中检索相关内容（集合: {coll_name}）...')

            search_results = self.vector_db.search(
                user_query,
                collection_name=coll_name,
                n_results=self.n_results,
                hybrid_search=self.hybrid_search,
                vector_weight=self.vector_weight
            )

            retrieved_docs = search_results['documents'][0]
            logger.info(f'检索完成，找到 {len(retrieved_docs)} 条相关内容')

        # 打印检索到的文档内容
        logger.info('\n>>> 检索到的相关内容：')
        for i, doc in enumerate(retrieved_docs, 1):
            logger.info(f'\n[{i}] {doc[:500]}...')
        
        # -------------------------------------------------------------------------
        # 阶段3：重排序（Rerank）
        # -------------------------------------------------------------------------
        if self.rerank and not _rerank_applied:
            # 对检索到的文档进行重排序（非路由模式，或路由回退到默认集合的情况）
            logger.info('【重排序】开始重排序...')
            logger.info(f'【重排序】保留的结果数: {self.rerank_top_k}')
            retrieved_docs = rerank_documents(retrieved_docs, user_query, top_n=self.rerank_top_k)
            
            # 打印重排序后的文档
            logger.info('\n>>> 重排序后的相关内容：')
            for i, doc in enumerate(retrieved_docs, 1):
                logger.info(f'\n[{i}] {doc[:500]}...')  # 打印前500个字符，避免输出过长
        
        logger.info('-' * 100)

        # -------------------------------------------------------------------------
        # 阶段4：构建增强Prompt
        # -------------------------------------------------------------------------
        logger.info('\n>>> 第2步：构建Prompt（提示词）...')
        
        # 将检索到的文档片段用换行符连接成一个字符串
        # 这就是提供给AI的"已知信息"
        info = '\n'.join(retrieved_docs)
        
        # 处理聊天历史
        chat_history = None
        history_str = ''
        
        if session_id:
            try:
                # 尝试使用Redis存储
                try:
                    from langchain_community.chat_message_histories import RedisChatMessageHistory
                    chat_history = RedisChatMessageHistory(
                        session_id=session_id,
                        url=self.redis_url
                    )
                    
                    # 获取历史消息
                    history_messages = []
                    for msg in chat_history.messages:
                        if msg.type == "human":
                            history_messages.append(f"用户: {msg.content}")
                        else:
                            history_messages.append(f"助手: {msg.content}")
                    
                    # 只保留最近10条消息
                    history_str = "\n".join(history_messages[-10:])
                    logger.info(f'\n>>> 加载历史对话 {len(history_messages)} 条 (Redis)')
                except Exception as redis_error:
                    # Redis不可用，使用本地存储
                    logger.warning(f'\n>>> Redis不可用: {str(redis_error)}，使用本地存储')
                    import json
                    import os
                    
                    # 创建本地存储目录
                    history_dir = 'chat_history'
                    if not os.path.exists(history_dir):
                        os.makedirs(history_dir)
                    
                    # 历史文件路径
                    history_file = os.path.join(history_dir, f'{session_id}.json')
                    
                    # 读取历史消息
                    history_messages = []
                    if os.path.exists(history_file):
                        try:
                            with open(history_file, 'r', encoding='utf-8') as f:
                                messages = json.load(f)
                                for msg in messages:
                                    if msg['type'] == "human":
                                        history_messages.append(f"用户: {msg['content']}")
                                    else:
                                        history_messages.append(f"助手: {msg['content']}")
                                logger.info(f'\n>>> 加载历史对话 {len(history_messages)} 条 (本地存储)')
                        except Exception as e:
                            logger.error(f'\n>>> 读取本地历史失败: {str(e)}')
                    
                    # 只保留最近10条消息
                    history_str = "\n".join(history_messages[-10:])
                    
                    # 保存聊天历史的函数
                    def save_local_history(user_msg, ai_msg):
                        messages = []
                        if os.path.exists(history_file):
                            try:
                                with open(history_file, 'r', encoding='utf-8') as f:
                                    messages = json.load(f)
                            except Exception:
                                pass
                        
                        # 添加新消息
                        messages.append({"type": "human", "content": user_msg})
                        messages.append({"type": "ai", "content": ai_msg})
                        
                        # 只保留最近20条消息
                        messages = messages[-20:]
                        
                        # 保存到文件
                        try:
                            with open(history_file, 'w', encoding='utf-8') as f:
                                json.dump(messages, f, ensure_ascii=False, indent=2)
                            logger.info('\n>>> 保存对话历史成功 (本地存储)')
                        except Exception as e:
                            logger.error(f'\n>>> 保存对话历史失败: {str(e)}')
                    
                    # 替换chat_history对象
                    class LocalChatHistory:
                        def add_user_message(self, message):
                            pass
                        def add_ai_message(self, message):
                            pass
                    
                    chat_history = LocalChatHistory()
                    chat_history.add_user_message = lambda msg: None
                    chat_history.add_ai_message = lambda msg: save_local_history(user_query, msg)
                    
            except Exception as e:
                logger.error(f'\n>>> 加载历史对话失败: {str(e)}')
        
        # -------------------------------------------------------------------------
        # 阶段5：调用大语言模型生成答案
        # -------------------------------------------------------------------------
        logger.info('\n>>> 第3步：调用AI模型生成答案...')
        
        # 调用get_completion函数，把Prompt发给AI
        if coll_name == 'demo' and not use_route:
            # 对于demo集合且非路由模式，直接将用户问题交给LLM，不使用提示词模板
            client = get_huayan_model_client()
            response = client.invoke(user_query)
            if hasattr(response, 'content'):
                response = response.content
        else:
            # 对于其他集合或路由模式，使用正常的提示词模板（携带检索到的文档）
            response = get_completion(info, user_query, history=history_str)
        
        logger.info('AI回答生成完成')
        logger.info('=' * 100)
        
        # 保存对话历史
        if session_id and chat_history:
            try:
                chat_history.add_user_message(user_query)
                chat_history.add_ai_message(response)
                logger.info(f'\n>>> 保存对话历史成功')
            except Exception as e:
                logger.error(f'\n>>> 保存对话历史失败: {str(e)}')
        
        # 返回两个值：
        # 1. AI生成的答案
        # 2. 检索到的原始文档片段（用于展示给用户，增加可信度）
        return response, retrieved_docs


# 创建全局RAG服务实例
rag_service = RAGService()

# 保持向后兼容的函数 - 使用全局rag_service实例

def save_to_db(filepath, collection_name='demo'):
    """
    保持向后兼容的save_to_db函数
    """
    return rag_service.save_to_db(filepath, collection_name)

def rag_chat(user_query, collection_name='demo', n_results=10, hybrid_search=False, vector_weight=0.5, rerank=False, rerank_top_k=5, session_id=None, auto_route=False):
    """
    保持向后兼容的rag_chat函数
    """
    # 更新全局rag_service的配置
    rag_service.collection_name = collection_name
    rag_service.n_results = n_results
    rag_service.hybrid_search = hybrid_search
    rag_service.vector_weight = vector_weight
    rag_service.rerank = rerank
    rag_service.rerank_top_k = rerank_top_k
    rag_service.auto_route = auto_route
    # 调用chat方法
    return rag_service.chat(user_query, session_id=session_id, auto_route=auto_route)


# =============================================================================
# 测试代码
# =============================================================================
# 当直接运行这个文件时（不是被导入时），执行以下测试代码
if __name__ == '__main__':
    
    logger.info("=" * 100)
    logger.info("开始测试 RAG 系统")
    logger.info("=" * 100)
    
    # 创建RAG服务实例
    service = RAGService()
    
    # ------ 测试1：上传文档 ------
    logger.info("\n【测试1】上传文档到知识库...")
    
    # 调用save_to_db方法，将测试文档存入向量数据库
    service.save_to_db(
        filepath='uploads/人事管理流程.docx', 
        collection_name='人事管理流程.docx'
    )
    
    logger.info('-' * 100)
    
    # ------ 测试2：智能问答 ------
    logger.info("\n【测试2】测试智能问答功能...")
    
    # 定义测试问题
    user_query = "视为不符合录用条件的情形有哪些?"
    logger.info(f"测试问题: {user_query}")
    
    # 调用chat方法获取答案
    response, search_results = service.chat(
        user_query, 
        collection_name='人事管理流程.docx'
    )
    
    # 打印最终结果
    logger.info("\n" + "=" * 100)
    logger.info("【最终答案】")
    logger.info("=" * 100)
    logger.info(response)
    logger.info("=" * 100)
