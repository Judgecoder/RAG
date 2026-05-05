"""
================================================================================
Web应用入口文件

【作用】
这个文件是整个系统的"大门"，负责：
1. 接收用户的网页请求
2. 处理文档上传
3. 处理用户提问
4. 返回网页给浏览器
================================================================================
"""

import os
import re
import hashlib
from typing import List, Optional

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from logger import logger
from rag import save_to_db, rag_chat
from models import HUAYAN_CODE_REASONING_MODEL, HUAYAN_EMBEDDING_MODEL, HUAYAN_RERANK_MODEL


def clean_collection_name(filename: str) -> str:
    """
    【功能】清理文件名或集合名，使其符合ChromaDB集合名称规范
    
    【ChromaDB集合名称规则】
    1. 长度: 3-63个字符
    2. 只能包含: 字母数字、下划线(_)、连字符(-)
    3. 必须以字母或数字开头和结尾
    4. 不能包含连续的两个点(..)
    5. 不能是有效的IPv4地址
    
    【处理策略】
    1. 首先检查输入是否已经是有效的集合名称（不包含点，符合规范）
    2. 如果是有效的集合名称，直接返回
    3. 否则，按文件名处理：移除扩展名，清理字符
    4. 添加短哈希值以确保唯一性
    
    【参数】
        filename: 原始文件名（如"2020-03-17__年度报告.pdf"）或集合名称
    
    【返回】
        符合ChromaDB规范的集合名称
    """
    import re
    
    # 1. 首先检查输入是否已经是有效的集合名称
    # 有效的集合名称不包含点（不是文件名），且符合所有规则
    if '.' not in filename:
        # 检查长度
        if 3 <= len(filename) <= 63:
            # 检查字符集
            if re.match(r'^[a-zA-Z0-9_-]+$', filename):
                # 检查开头和结尾
                if filename[0].isalnum() and filename[-1].isalnum():
                    # 检查连续两个点（不应出现，因为不包含点）
                    if '..' not in filename:
                        # 粗略检查IPv4地址（简化检查）
                        if not re.match(r'^\d{1,3}(\.\d{1,3}){3}$', filename):
                            # 已经是有效的集合名称，直接返回
                            return filename
    
    # 2. 如果是文件名，按文件名处理
    # 移除文件扩展名（如.pdf, .docx等）
    name_without_ext = os.path.splitext(filename)[0]
    
    # 只保留字母数字、下划线、连字符，其他字符替换为下划线
    cleaned = re.sub(r'[^a-zA-Z0-9_-]', '_', name_without_ext)
    
    # 如果清理后为空，使用默认名称
    if not cleaned:
        cleaned = 'doc'
    
    # 确保不以连字符或下划线开头或结尾
    cleaned = cleaned.strip('-_')
    
    # 如果清理后为空，再次使用默认名称
    if not cleaned:
        cleaned = 'doc'
    
    # 限制长度在57个字符以内（为哈希值留出空间）
    max_base_length = 57
    if len(cleaned) > max_base_length:
        # 保留开头部分，这样更易读
        cleaned = cleaned[:max_base_length]
    
    # 计算文件名的短哈希值（6个字符）以确保唯一性
    file_hash = hashlib.md5(filename.encode()).hexdigest()[:6]
    
    # 组合基础名称和哈希值
    result = f"{cleaned}_{file_hash}"
    
    # 最终长度检查（不超过63个字符）
    if len(result) > 63:
        # 如果仍然太长，进一步截断基础部分
        max_allowed = 63 - len(file_hash) - 1  # 减1是为了下划线
        cleaned = cleaned[:max_allowed]
        result = f"{cleaned}_{file_hash}"
    
    # 最终清理：确保以字母数字开头和结尾
    if not result[0].isalnum():
        result = 'doc_' + result[1:] if len(result) > 1 else 'doc'
    if not result[-1].isalnum():
        result = result[:-1] + 'x' if len(result) > 1 else 'doc'
    
    # 最终长度检查（至少3个字符）
    if len(result) < 3:
        result = 'doc_' + file_hash
    
    return result


# =============================================================================
# 第一步：创建FastAPI应用对象
# =============================================================================
app = FastAPI(title="Advance RAG 智能知识库系统", description="基于FastAPI的RAG系统")

# =============================================================================
# 第二步：配置模板和静态文件
# =============================================================================
templates = Jinja2Templates(directory="templates")

# 创建静态文件目录（如果不存在）
STATIC_FOLDER = 'static'
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

# 挂载静态文件服务
app.mount("/static", StaticFiles(directory=STATIC_FOLDER), name="static")

# =============================================================================
# 第三步：配置文件上传功能
# =============================================================================
UPLOAD_FOLDER = 'uploads'

# 检查uploads文件夹是否存在，如果不存在就创建
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 设置允许上传的文件类型
ALLOWED_EXTENSIONS = {'docx', 'pdf', 'md', 'xlsx', 'csv', 'txt', 'doc'}

def allowed_file(filename: str) -> bool:
    """
    【函数作用】检查文件是否允许上传
    
    【参数】
        filename: 文件名
    
    【返回值】
        True: 文件类型允许上传
        False: 文件类型不允许
    """
    # 检查文件名中是否有'.'
    if '.' not in filename:
        return False
    
    # 获取文件扩展名
    extension = filename.rsplit('.', 1)[1].lower()
    
    # 检查扩展名是否在允许的集合中
    return extension in ALLOWED_EXTENSIONS


# =============================================================================
# 第四步：定义数据模型
# =============================================================================
class ChatRequest(BaseModel):
    message: str
    top_k: int = 10
    hybrid_search: bool = False
    vector_weight: float = 0.5
    rerank: bool = False
    rerank_top_k: int = 5
    auto_route: bool = False
    session_id: str
    collection_name: Optional[str] = None

class CollectionRequest(BaseModel):
    collection_name: Optional[str] = None

class LoginRequest(BaseModel):
    employee_id: str
# =============================================================================
# 第六步：定义路由（处理用户请求）
# =============================================================================

# -----------------------------------------------------------------------------
# 路由1: Favicon图标
# -----------------------------------------------------------------------------
@app.get("/favicon.ico")
async def favicon():
    """
    【功能】提供网站图标
    
    【URL】/favicon.ico
    """
    response = FileResponse("static/favicon.ico")
    # 设置缓存控制头：缓存1年
    response.headers["Cache-Control"] = "public, max-age=31536000"
    return response

# -----------------------------------------------------------------------------
# 路由2: 设备图标
# -----------------------------------------------------------------------------
@app.get("/apple-touch-icon.png")
@app.get("/apple-touch-icon-precomposed.png")
async def apple_touch_icon():
    """
    【功能】提供苹果设备网站图标
    
    【URL】/apple-touch-icon.png 和 /apple-touch-icon-precomposed.png
    【说明】iOS Safari浏览器会自动请求这些图标用于主屏幕和书签
    """
    response = FileResponse("static/favicon.ico")
    # 设置缓存控制头：缓存1年
    response.headers["Cache-Control"] = "public, max-age=31536000"
    return response

# -----------------------------------------------------------------------------
# 路由3: 登录页面
# -----------------------------------------------------------------------------
@app.get("/login/", response_class=HTMLResponse)
async def login_page(request: Request):
    """
    【功能】显示登录页面
    
    【URL】/login/
    
    【请求方式】GET
    """
    return templates.TemplateResponse(request, "login.html")

# -----------------------------------------------------------------------------
# 路由4: 登录处理
# -----------------------------------------------------------------------------
@app.post("/login/")
async def login(login_request: LoginRequest):
    """
    【功能】处理登录请求
    
    【URL】/login/
    
    【请求方式】POST
    """
    employee_id = login_request.employee_id
    
    # 简单的登录验证（实际项目中可能需要更复杂的验证）
    if not employee_id:
        raise HTTPException(status_code=400, detail="请输入工号")
    
    # 设置会话
    session_id = employee_id
    
    # 创建JSONResponse并设置cookie
    response = JSONResponse({"message": "登录成功", "session_id": session_id})
    response.set_cookie(
        key="session_id",
        value=session_id,
        max_age=86400,  # 1天
        samesite="lax"
    )
    
    return response

# -----------------------------------------------------------------------------
# 路由5: 文档上传页面
# -----------------------------------------------------------------------------
@app.get("/document_upload/", response_class=HTMLResponse)
async def document_upload_page(request: Request):
    """
    【功能】显示文档上传页面
    
    【URL】/document_upload/
    
    【请求方式】GET
    """
    return templates.TemplateResponse(request, "document_upload.html")


@app.post("/document_upload/")
async def document_upload(request: Request, file: UploadFile = File(...)):
    """
    【功能】处理文档上传
    
    【URL】/document_upload/
    
    【请求方式】POST
    """
    global collection_name
    
    # 检查文件是否为空
    if file.filename == '':
        raise HTTPException(status_code=400, detail="没有选择文件")
    
    # 检查文件类型
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="不支持的文件类型，只支持 .docx、.pdf、.md、.xlsx、.csv、.txt 和 .doc 文件")
    
    # 获取文件名
    filename = file.filename
    logger.info(f"接收到文件: {filename}")
    
    # 构建完整的保存路径
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    logger.info(f"保存路径: {file_path}")
    
    # 保存文件到uploads文件夹
    with open(file_path, "wb") as buffer:   # 二进制写入的方式打开文件
        content = await file.read()  # 读取文件内容到内存
        buffer.write(content)    # 将内存中的文件内容写入对象缓冲区
        # with块结束时会自动关闭文件，触发flush()将数据持久化到磁盘
    
    # 使用正则表达式处理文件名，去除路径分隔符
    original_collection_name = re.split(r'[/\\]', filename)[-1]
    
    # 清理集合名称，使其符合ChromaDB规范
    collection_name = clean_collection_name(original_collection_name)
    logger.info(f"原始集合名称: {original_collection_name}")
    logger.info(f"清理后集合名称: {collection_name}")
    
    # 调用main.py中的save_to_db函数，将文档内容存入向量数据库
    result = save_to_db(file_path, collection_name=collection_name)
    
    # 检查处理结果
    if result and ("出错" in result or "错误" in result or "不支持" in result):
        # 处理失败，返回错误响应
        logger.error(f"文档处理失败: {result}")
        return JSONResponse({
            "message": f"文档处理失败: {result}",
            "filename": filename,
            "collection_name": collection_name,
            "error": True
        }, status_code=500)
    else:
        # 处理成功
        logger.info(f"文档已成功存入向量数据库，集合名称: {collection_name}")
        
        # 返回成功响应给前端
        return JSONResponse({
            "message": "文件上传成功",
            "filename": filename,
            "collection_name": collection_name,
            "detail": result  # 包含详细的处理结果
        }, status_code=200)


# -----------------------------------------------------------------------------
# 路由6: 聊天页面（首页）
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
@app.get("/chat/", response_class=HTMLResponse)
async def chat_page(request: Request):
    """
    【功能】显示聊天页面
    
    【URL】/ 或 /chat/
    
    【请求方式】GET
    """
    # 检查登录状态
    session_id = request.cookies.get("session_id")
    if not session_id:
        # 未登录，重定向到登录页面
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/login/")
    
    return templates.TemplateResponse(request, "chat.html", {"session_id": session_id})


@app.post("/chat/")
async def chat(chat_request: ChatRequest):
    """
    【功能】处理智能问答
    
    【URL】/chat/
    
    【请求方式】POST
    """
    message = chat_request.message
    logger.info(f"用户提问: {message}")
    
    # 检查问题是否为空
    if not message:
        raise HTTPException(status_code=400, detail="请输入问题")
    
    # 使用前端传递的 collection_name，清理后符合 ChromaDB 规范
    raw_collection = chat_request.collection_name
    if raw_collection and raw_collection != '__auto__':
        current_collection = clean_collection_name(raw_collection)
    else:
        current_collection = 'demo'
    logger.info(f"【聊天】原始知识库: {raw_collection}, 清理后: {current_collection}")
    
    # 调用rag.py中的rag_chat函数进行RAG问答
    response, retrieved_docs = rag_chat(
        message, 
        collection_name=current_collection, 
        n_results=chat_request.top_k,
        hybrid_search=chat_request.hybrid_search,
        vector_weight=chat_request.vector_weight,
        rerank=chat_request.rerank,
        rerank_top_k=chat_request.rerank_top_k,
        auto_route=chat_request.auto_route,
        session_id=chat_request.session_id
    )
    
    logger.info(f'AI回答: {response}')
    
    # 返回JSON格式的响应给前端
    return JSONResponse({
        'response': response,           # AI的回答
        'search_results': retrieved_docs  # 检索到的相关文档片段
    })


# -----------------------------------------------------------------------------
# 路由7: 切换当前使用的文档（集合）
# -----------------------------------------------------------------------------
@app.get("/collection/")
async def get_collection():
    """
    【功能】获取所有文档列表和当前使用的文档
    
    【URL】/collection/
    
    【请求方式】GET
    """
    # 获取uploads文件夹中的所有文件名
    name_list = os.listdir(UPLOAD_FOLDER)
    
    # 返回文件列表（知识库由前端维护状态）
    return JSONResponse({
        'name_list': name_list,
        'collection_name': 'demo'
    })
@app.post("/collection/")
async def switch_collection(collection_request: CollectionRequest):
    """
    【功能】切换当前使用的文档
    
    【URL】/collection/
    
    【请求方式】POST
    """
    new_collection = collection_request.collection_name
    
    # 清理集合名称，使其符合ChromaDB规范
    if new_collection and new_collection != 'demo':
        cleaned = clean_collection_name(new_collection)
        logger.info(f"切换集合: 原始名称={new_collection}, 清理后={cleaned}")
    else:
        cleaned = 'demo'
    
    logger.info(f'已切换到文档: {cleaned}')
    
    # 返回成功响应
    return JSONResponse({
        'message': '切换成功',
        'collection_name': cleaned
    })


@app.get("/documents/")
async def get_documents():
    """
    【功能】获取所有已上传的文档列表
    
    【URL】/documents/
    
    【请求方式】GET
    """
    try:
        # 读取uploads文件夹中的所有文件
        files = []
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(file_path):
                    files.append({
                        "name": filename,
                        "size": os.path.getsize(file_path)
                    })
        
        return JSONResponse({
            "message": "获取文档列表成功",
            "files": files
        })
    except Exception as e:
        logger.error(f"获取文档列表失败: {str(e)}")
        return JSONResponse({
            "message": "获取文档列表失败",
            "error": str(e)
        }, status_code=500)


# -----------------------------------------------------------------------------
# 路由8: 获取聊天历史
# -----------------------------------------------------------------------------
@app.get("/chat_history/")
async def get_chat_history(session_id: str):
    """
    【功能】获取指定用户的聊天历史记录
    
    【URL】/chat_history/
    
    【请求方式】GET
    
    【参数】
        session_id: 用户工号，用于标识聊天历史
    
    【返回】
        聊天历史消息列表
    """
    import json
    import os
    
    # 聊天历史存储目录
    history_dir = 'chat_history'
    
    if not os.path.exists(history_dir):
        return JSONResponse({
            "messages": [],
            "message": "暂无聊天历史"
        })
    
    # 构建历史文件路径
    history_file = os.path.join(history_dir, f'{session_id}.json')
    
    if not os.path.exists(history_file):
        return JSONResponse({
            "messages": [],
            "message": "暂无聊天历史"
        })
    
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            messages = json.load(f)
        
        return JSONResponse({
            "messages": messages,
            "message": "获取聊天历史成功",
            "count": len(messages)
        })
    except Exception as e:
        logger.error(f"读取聊天历史失败: {str(e)}")
        return JSONResponse({
            "messages": [],
            "message": "读取聊天历史失败",
            "error": str(e)
        }, status_code=500)


@app.delete("/document/")
async def delete_document(filename: str):
    """
    【功能】删除文档及其对应的向量数据
    
    【URL】/document/
    
    【请求方式】DELETE
    
    【参数】
        filename: 要删除的文件名
    """
    # 检查文件名是否为空
    if not filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    
    logger.info(f"接收到删除请求，文件名: {filename}")
    
    # 清理集合名称，使其符合ChromaDB规范
    collection_to_delete = clean_collection_name(filename)
    logger.info(f"原始文件名: {filename}")
    logger.info(f"清理后集合名称: {collection_to_delete}")
    
    # 构建文件路径
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    logger.info(f"文件路径: {file_path}")
    
    # 导入向量数据库实例
    from rag import rag_service
    vector_db = rag_service.vector_db
    
    # 尝试删除向量数据库中的集合
    delete_success = vector_db.delete_collection(collection_to_delete)
    
    # 从路由索引中删除对应的描述
    try:
        rag_service.route_index.remove_description(collection_to_delete)
        logger.info(f"路由索引中 {collection_to_delete} 的描述已删除")
    except Exception as e:
        logger.error(f"删除路由索引描述失败: {str(e)}")
    
    # 尝试删除本地文件
    file_deleted = False
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            file_deleted = True
            logger.info(f"本地文件 {filename} 删除成功")
        except Exception as e:
            logger.error(f"删除本地文件 {filename} 失败: {str(e)}")
    else:
        logger.info(f"本地文件 {filename} 不存在")
    
    # 返回删除结果
    if delete_success or file_deleted:
        return JSONResponse({
            "message": "文档删除成功",
            "filename": filename,
            "collection_name": collection_to_delete,
            "vector_db_deleted": delete_success,
            "file_deleted": file_deleted
        }, status_code=200)
    else:
        return JSONResponse({
            "message": "文档删除失败",
            "filename": filename,
            "collection_name": collection_to_delete,
            "vector_db_deleted": delete_success,
            "file_deleted": file_deleted
        }, status_code=500)




# -----------------------------------------------------------------------------
# 路由9: 获取模型名称（从 models.py 动态获取）
# -----------------------------------------------------------------------------
@app.get("/api/model-names")
async def get_model_names():
    """
    【功能】获取当前配置的模型名称
    
    【URL】/api/model-names
    
    【请求方式】GET
    
    【返回值】
        {
            "chat": "text-chat-latest",
            "embedding": "Qwen3-Embedding-8B",
            "rerank": "Qwen3-Reranker-4B",
            "mineru": "MinerU"
        }
    """
    return JSONResponse({
        'chat': HUAYAN_CODE_REASONING_MODEL,
        'embedding': HUAYAN_EMBEDDING_MODEL,
        'rerank': HUAYAN_RERANK_MODEL,
        'mineru': 'MinerU'
    })


# -----------------------------------------------------------------------------
# 路由10: 模型健康检测
# -----------------------------------------------------------------------------
@app.get("/api/model-health")
async def get_model_health(model_type: Optional[str] = None, test_type: str = "light"):
    """
    【功能】检测模型健康状态
    
    【URL】/api/model-health
    
    【请求方式】GET
    
    【参数】
        model_type: 模型类型 ('chat', 'embedding', 'rerank')，不指定则检测所有模型
        test_type: 检测类型 ('light', 'full')，默认轻量级检测
    """
    from function_tools import check_model_health, check_all_models_health
    
    try:
        if model_type:
            result = check_model_health(model_type, test_type)
        else:
            result = check_all_models_health(test_type)
        
        return JSONResponse({
            'status': 'success',
            'data': result
        })
    except Exception as e:
        return JSONResponse({
            'status': 'error',
            'message': str(e)
        }, status_code=500)

    # return JSONResponse({
    #     'status': 'error',
    #     'message': 'Model health check failed'
    # }, status_code=500)


# =============================================================================
# 第七步：启动应用（仅当直接运行时）
# =============================================================================
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, reload=False)