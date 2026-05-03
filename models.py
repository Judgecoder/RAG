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
# 第一部分：阿里云通义千问模型配置
# =============================================================================

# 环境变量名称，用于存储阿里云的API Key
# 使用环境变量而不是直接写在代码中，更加安全
ALI_TONGYI_API_KEY_OS_VAR_NAME = "DASHSCOPE_API_KEY"

# 阿里云百炼平台的API地址
# compatible-mode/v1 表示兼容OpenAI的API格式
ALI_TONGYI_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 各种模型的名称定义
# 这些名称是阿里云平台上模型的标识符

# 对话模型（用于回答问题）
ALI_TONGYI_MAX_MODEL = "qwen3.5-flash"           # 最大版本
ALI_TONGYI_TURBO_MODEL = "qwen3.5-flash"         # 轻量版（默认使用）

# DeepSeek系列模型（通过阿里云调用）
ALI_TONGYI_DEEPSEEK_R1 = "deepseek-r1"           # DeepSeek R1
ALI_TONGYI_DEEPSEEK_R10528 = "deepseek-r1-0528"  # DeepSeek R1 0528版本
ALI_TONGYI_DEEPSEEK_V3 = "deepseek-v3"           # DeepSeek V3

# 推理模型（擅长逻辑推理）
ALI_TONGYI_REASONER_MODEL = "qvq-max-latest"

# 文本向量化模型（用于生成Embedding）
ALI_TONGYI_EMBEDDING_V3 = "text-embedding-v3"    # V3版本
ALI_TONGYI_EMBEDDING_V4 = "text-embedding-v4"    # V4版本（默认使用）

# 重排序模型（用于对检索结果重新排序，提升准确性）
ALI_TONGYI_RERANK_MODEL = "gte-rerank-v2"           # 通用文本重排序模型


# =============================================================================
# 第二部分：DeepSeek模型配置
# =============================================================================

# DeepSeek官方平台的环境变量名
DEEPSEEK_API_KEY_OS_VAR_NAME = "DEEPSEEK_API_KEY"

# DeepSeek官方API地址
DEEPSEEK_URL = "https://api.deepseek.com/v1"

# DeepSeek模型名称
DEEPSEEK_CHAT_MODEL = "deepseek-chat"           # 对话模型
DEEPSEEK_REASONER_MODEL = "deepseek-reasoner"   # 推理模型

# =============================================================================
# 第四部分：腾讯混元模型配置（已注释）
# =============================================================================
"""
# 腾讯混元平台配置
TENCENT_HUNYUAN_API_KEY_OS_VAR_NAME = "HUNYUAN_API_KEY"
TENCENT_HUNYUAN_URL = "https://api.hunyuan.cloud.tencent.com/v1"
TENCENT_HUNYUAN_TURBO_MODEL = "hunyuan-turbos-latest"           # 轻量版
TENCENT_HUNYUAN_REASONER_MODEL = "hunyuan-t1-latest"            # 推理版
TENCENT_HUNYUAN_LONGCONTEXT_MODEL = "hunyuan-large-longcontext" # 长上下文版
"""


# =============================================================================
# 第四部分：导入必要的库
# =============================================================================

import os  # 用于读取环境变量

# LangChain相关库
# LangChain是一个AI应用开发框架，提供统一的接口调用各种模型
from langchain_openai import ChatOpenAI  # LangChain的OpenAI格式聊天模型
from langchain_openai import OpenAIEmbeddings  # LangChain的OpenAI格式嵌入模型
from openai import OpenAI  # OpenAI官方客户端（兼容格式）
import inspect  # 用于获取函数名（调试用）

# 阿里云Embedding模型（LangChain封装）
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_compressors.dashscope_rerank import DashScopeRerank


# =============================================================================
# 第五部分：获取模型客户端的函数
# =============================================================================

def get_normal_client(
    api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME),
    base_url=ALI_TONGYI_URL,
    verbose=False,
    debug=False
):
    """
    【功能】获取原生的OpenAI格式客户端
    
    【参数】
        api_key: API密钥，默认从环境变量读取
        base_url: API服务器地址
        verbose: 是否输出调试信息（基本信息）
        debug: 是否输出详细调试信息（包括API Key）
    
    【返回值】
        OpenAI客户端对象
    
    【使用示例】
        client = get_normal_client()
        # 或者指定其他平台
        client = get_normal_client(
            api_key="your-key",
            base_url="https://api.deepseek.com/v1"
        )
    
    【什么是OpenAI格式】
    很多国产AI平台都兼容OpenAI的API格式，这样我们可以用同一套代码
    调用不同平台的模型，只需要换API地址和Key就行。
    """
    # 获取当前函数名（用于调试输出）
    function_name = inspect.currentframe().f_code.co_name
    
    # 如果开启了verbose，输出平台信息
    if verbose:
        print(f"【{function_name}】连接平台: {base_url}")
    
    # 如果开启了debug，输出详细信息（注意：会暴露API Key！）
    if debug:
        print(f"【{function_name}】连接平台: {base_url}, API Key: {api_key}")
    
    # 创建并返回OpenAI客户端
    # api_key: 身份验证密钥
    # base_url: API服务器地址
    return OpenAI(api_key=api_key, base_url=base_url)


def get_lc_model_client(
    api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME),
    base_url=ALI_TONGYI_URL,
    model=ALI_TONGYI_TURBO_MODEL,
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

    【LangChain vs 原生API】
    - 原生API: 更灵活，直接控制请求参数
    - LangChain: 更方便，提供高级功能（如链式调用、记忆等）
    """
    function_name = inspect.currentframe().f_code.co_name
    
    if verbose:
        print(f"【{function_name}】平台: {base_url}, 模型: {model}, 温度: {temperature}")
    
    if debug:
        print(f"【{function_name}】平台: {base_url}, 模型: {model}, 温度: {temperature}, Key: {api_key}")
    
    # 创建LangChain的ChatOpenAI对象
    # ChatOpenAI是LangChain对大语言模型的封装
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        extra_body={"enable_thinking": False}  # 关闭思考过程输出
    )


def get_ali_model_client(
    model=ALI_TONGYI_DEEPSEEK_V3,
    temperature=0.7,
    verbose=False,
    debug=False
):
    """
    【功能】获取阿里云DeepSeek模型客户端（LangChain版）
    
    【参数】
        model: 模型名称，默认DeepSeek V3
        temperature: 温度参数
        verbose: 是否输出调试信息
        debug: 是否输出详细调试信息
    
    【使用场景】
    当你想使用阿里云平台上的DeepSeek模型时使用这个函数。
    """
    return get_lc_model_client(
        api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME),
        base_url=ALI_TONGYI_URL,
        model=model,
        temperature=temperature,
        verbose=verbose,
        debug=debug
    )


def get_ds_model_client(
    model=DEEPSEEK_CHAT_MODEL,
    temperature=0.7,
    verbose=False,
    debug=False
):
    """
    【功能】获取DeepSeek官方平台客户端（LangChain版）
    
    【参数】
        model: 模型名称，默认deepseek-chat
        temperature: 温度参数
        verbose: 是否输出调试信息
        debug: 是否输出详细调试信息
    
    【使用场景】
    当你想使用DeepSeek官方平台的模型时使用这个函数。
    """
    return get_lc_model_client(
        api_key=os.getenv(DEEPSEEK_API_KEY_OS_VAR_NAME),
        base_url=DEEPSEEK_URL,
        model=model,
        temperature=temperature,
        verbose=verbose,
        debug=debug
    )

def get_ali_embeddings(model=ALI_TONGYI_EMBEDDING_V3):
    """
    【功能】获取阿里云文本向量化模型
    
    【参数】
        model: 向量化模型名称，默认 text-embedding-v3
    
    【返回值】
        DashScopeEmbeddings对象（LangChain封装）
    
    【什么是Embedding模型】
    Embedding模型专门用于将文本转换为向量（数字表示）。
    这些向量可以表示文本的语义含义，用于相似度计算。
    
    【使用示例】
        embeddings = get_ali_embeddings()
        text_vectors = embeddings.embed_documents(["文本1", "文本2"])
    """
    return DashScopeEmbeddings(
        model=model,
        dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME)
    )

def get_ali_rerank(top_n=3):
    """
    通过LangChain获得一个阿里重排序模型的实例
    :return: 阿里通义千问嵌入模型的实例
    """
    return DashScopeRerank(
        model=ALI_TONGYI_RERANK_MODEL, 
        dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME),
        top_n=top_n
)

def get_ali_clients():
    """
    【功能】同时获取阿里云的大模型客户端和Embedding模型客户端
    
    【返回值】
        (chat_client, embedding_client) 元组
    
    【使用场景】
    当你同时需要对话能力和文本向量化能力时使用。
    比如构建一个完整的RAG系统。
    
    【使用示例】
        chat_client, embedding_client = get_ali_clients()
        # chat_client用于生成回答
        # embedding_client用于将文本转为向量
    """
    return get_ali_model_client(), get_ali_embeddings()


# =============================================================================
# 使用说明
# =============================================================================
"""
【快速开始】

1. 设置环境变量（在运行程序前执行）：
   Windows:
       set DASHSCOPE_API_KEY=你的阿里云API密钥
   Linux/Mac:
       export DASHSCOPE_API_KEY=你的阿里云API密钥

2. 在代码中导入并使用：
   from models import get_normal_client, ALI_TONGYI_TURBO_MODEL
   
   client = get_normal_client()
   # 现在可以使用client调用AI模型了

【切换平台】

如果想使用DeepSeek而不是阿里云：

    from models import get_normal_client, DEEPSEEK_URL, DEEPSEEK_API_KEY_OS_VAR_NAME
    import os
    
    client = get_normal_client(
        api_key=os.getenv(DEEPSEEK_API_KEY_OS_VAR_NAME),
        base_url=DEEPSEEK_URL
    )

【安全提示】

1. 永远不要将API Key直接写在代码中！
2. 使用环境变量存储敏感信息
3. 如果自己使用了.env，不要将.env文件提交到Git仓库
"""
