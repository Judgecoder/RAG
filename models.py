"""
================================================================================
模型配置文件 (models.py)

【这个文件的作用】
这个文件是系统的"模型配置中心"，负责：
    1. 定义各种AI模型的名称和参数
    2. 提供获取模型客户端的函数

================================================================================
"""

# =============================================================================
# 第一部分：华言模型配置
# =============================================================================


HUAYAN_API_KEY_OS_VAR_NAME = "HUAYAN_API_KEY"
HUAYAN_URL = "https://yfmodelhub.dahuatech.com/v1/"

# 对话模型
HUAYAN_CODE_REASONING_MODEL = "text-chat-latest"

# 嵌入模型
HUAYAN_EMBEDDING_MODEL = "Qwen3-Embedding-8B"

# 重排模型
HUAYAN_RERANK_MODEL = "Qwen3-Reranker-4B"

API_KEY = "CodeAgentSharedKey"

# =============================================================================
# 第二部分：导入必要的库
# =============================================================================

from abc import ABC, abstractmethod

from langchain_openai import ChatOpenAI  # LangChain的OpenAI格式聊天模型
from langchain_openai import OpenAIEmbeddings  # LangChain的OpenAI格式嵌入模型


class BaseDocumentCompressor(ABC):
    @abstractmethod
    def compress_documents(self, documents, query, top_n=None):
        ...

    def invoke(self, documents, query, top_n=None):
        return self.compress_documents(documents, query, top_n)


# =============================================================================
# 第五部分：获取模型客户端的函数
# =============================================================================

def get_huayan_model_client(
    api_key=API_KEY,
    base_url=HUAYAN_URL,
    model=HUAYAN_CODE_REASONING_MODEL,
    temperature=0.7,
    verbose=False,
    debug=False
):
    """
    【功能】通过LangChain获取模型客户端
    
    【参数】
        api_key: API密钥
        base_url: API服务器地址
        model: 模型名称
        temperature: 温度参数（控制随机性）
        verbose: 是否输出调试信息
        debug: 是否输出详细调试信息
    
    【返回值】
        LangChain的ChatOpenAI对象
    """
    if verbose:
        print(f"【get_huayan_model_client】平台: {base_url}, 模型: {model}, 温度: {temperature}")

    if debug:
        print(f"【get_huayan_model_client】平台: {base_url}, 模型: {model}, 温度: {temperature}, Key: {api_key}")
    
    # 创建LangChain的ChatOpenAI对象
    # ChatOpenAI是LangChain对大语言模型的封装
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        extra_body={"enable_thinking": False}  # 关闭思考过程输出
    )


def get_huayan_embeddings(
    model=HUAYAN_EMBEDDING_MODEL,
    verbose=False,
    debug=False
):
    """
    【功能】获取华言文本向量化模型

    【参数】
        model: 向量化模型名称
        verbose: 是否输出调试信息
        debug: 是否输出详细调试信息

    【返回值】
        OpenAIEmbeddings对象（LangChain封装）
    """
    api_key = API_KEY
    base_url = HUAYAN_URL

    if verbose:
        print(f"【get_huayan_embeddings】连接平台: {base_url}, 模型: {model}")
    if debug:
        print(f"【get_huayan_embeddings】连接平台: {base_url}, 模型: {model}, Key: {api_key}")

    return OpenAIEmbeddings(
        api_key=api_key,
        base_url=base_url,
        model=model,
        check_embedding_ctx_length=False,
        allowed_special="all"
    )


class HuayanRerank(BaseDocumentCompressor):
    """
    【功能】华言重排序模型客户端

    实现 LangChain 的 BaseDocumentCompressor 接口，用于对检索结果重新排序。

    【使用示例】
        reranker = HuayanRerank(top_n=5)
        reranked_docs = reranker.compress_documents(documents, query="用户问题")
    """

    def __init__(self, model=HUAYAN_RERANK_MODEL, top_n=5):
        self.model = model
        self.top_n = top_n
        self.api_key = API_KEY
        self.base_url = HUAYAN_URL
        self.instruction = "Given a web search query, retrieve relevant passages that answer the query."

    def compress_documents(self, documents, query, top_n=None):
        """
        【核心方法】对文档进行重排序

        【参数】
            documents: Document对象列表
            query: 用户查询
            top_n: 保留前N条结果，默认使用初始化时设置的值

        【返回值】
            重排序后的Document对象列表
        """
        import requests

        n = top_n or self.top_n
        if not documents:
            return documents

        docs_text = [doc.page_content for doc in documents]

        payload = {
            "model": self.model,
            "query": query,
            "documents": docs_text,
            "instruction": self.instruction,
            "top_n": min(n, len(docs_text))
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            f"{self.base_url}rerank",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        result = response.json()

        indices = [item["index"] for item in result.get("results", [])]
        return [documents[i] for i in indices if i < len(documents)]


def get_huayan_rerank(top_n=5):
    """
    【功能】获取华言重排序模型

    【参数】
        top_n: 保留前N条结果

    【返回值】
        HuayanRerank对象

    【使用示例】
        reranker = get_huayan_rerank(top_n=5)
        reranked_docs = reranker.compress_documents(documents, query="用户问题")
    """
    return HuayanRerank(top_n=top_n)
